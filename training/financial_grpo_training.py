# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FINANCIAL_SYSTEM_PROMPT = (
    "你是一名金融数值推理助手。请严格依据题目材料作答。"
    "输出尽量保持结构化，优先包含‘问题分析：’、‘推理程序：’、‘最终答案：’。"
)
SECTION_ANCHORS = ["问题分析：", "推理程序：", "最终答案："]
PROGRAM_OPS = ["add", "subtract", "multiply", "divide", "greater", "table_max", "table_min"]
FINAL_ANSWER_RE = re.compile(r"最终答案：\s*(.+)", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass
class ScriptArguments:
    tokenizer_name_or_path: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "Directory containing local GRPO json/jsonl files."})
    train_samples: Optional[int] = field(default=-1)
    preprocessing_num_workers: Optional[int] = field(default=4)
    validation_split_percentage: Optional[int] = field(default=10)
    qlora: bool = field(default=False)
    answer_abs_tol: float = field(default=1e-4)
    answer_rel_tol: float = field(default=1e-4)
    format_weight: float = field(default=0.2)
    answer_weight: float = field(default=0.5)
    program_weight: float = field(default=0.3)


def get_checkpoint(training_args: GRPOConfig):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        cls = bnb.nn.Linear4bit if int4 else bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if 'lm_head' in name or 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def get_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content") or "").strip()
        return str(first).strip()
    if isinstance(completion, dict):
        return str(completion.get("content") or "").strip()
    return str(completion or "").strip()


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return (text or "").strip()


def normalize_number(text: str) -> Optional[float]:
    if not text:
        return None
    match = NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def reasoning_format_reward(completions, format_weight, **kwargs):
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        score = sum(1.0 for anchor in SECTION_ANCHORS if anchor in text) / len(SECTION_ANCHORS)
        rewards.append(score * format_weight)
    return rewards


def answer_correctness_reward(completions, answer, answer_weight, answer_abs_tol, answer_rel_tol, **kwargs):
    rewards = []
    for completion, gold in zip(completions, answer):
        pred_text = extract_final_answer(get_completion_text(completion))
        gold_text = str(gold or "").strip()
        pred_num = normalize_number(pred_text)
        gold_num = normalize_number(gold_text)
        if pred_num is not None and gold_num is not None:
            ok = abs(pred_num - gold_num) <= max(answer_abs_tol, abs(gold_num) * answer_rel_tol)
            rewards.append((1.0 if ok else 0.0) * answer_weight)
        else:
            rewards.append((1.0 if pred_text.strip() == gold_text.strip() else 0.0) * answer_weight)
    return rewards


def program_consistency_reward(completions, gold_program, program_weight, **kwargs):
    rewards = []
    for completion, gold in zip(completions, gold_program):
        gold = str(gold or "")
        if not gold.strip():
            rewards.append(0.0)
            continue
        text = get_completion_text(completion)
        pred_program = ""
        match = re.search(r"推理程序：\s*(.+)", text)
        if match:
            pred_program = match.group(1).strip()
        gold_ops = [op for op in PROGRAM_OPS if op in gold]
        if not gold_ops:
            rewards.append(0.0)
            continue
        hit = sum(1 for op in gold_ops if op in pred_program)
        rewards.append((hit / len(gold_ops)) * program_weight)
    return rewards


def grpo_train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    is_main_process = training_args.local_rank in [-1, 0]
    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(f"Training args: {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path or model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not script_args.train_file_dir or not os.path.exists(script_args.train_file_dir):
        raise ValueError("financial_grpo_training.py requires --train_file_dir with local GRPO json/jsonl files")

    dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    if script_args.train_samples and script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    with training_args.main_process_first(desc="Dataset preparation"):
        dataset = dataset.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                    {"role": "user", "content": x["prompt"]},
                ],
                "answer": x["answer"],
                "gold_program": x.get("gold_program", ""),
                "task_name": x.get("task_name", ""),
                "source_dataset": x.get("source_dataset", ""),
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing dataset" if is_main_process else None,
        )

    eval_ratio = max(1, min(50, int(script_args.validation_split_percentage))) / 100.0
    split = dataset.train_test_split(test_size=eval_ratio, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    torch_dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot both be set")

    quantization_config = None
    if model_args.load_in_4bit or model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
        device_map=None if int(os.environ.get("WORLD_SIZE", "1")) != 1 else "auto",
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if model_args.use_peft:
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    reward_funcs = [
        lambda completions, **kwargs: reasoning_format_reward(
            completions,
            format_weight=script_args.format_weight,
            **kwargs,
        ),
        lambda completions, answer, **kwargs: answer_correctness_reward(
            completions,
            answer,
            answer_weight=script_args.answer_weight,
            answer_abs_tol=script_args.answer_abs_tol,
            answer_rel_tol=script_args.answer_rel_tol,
            **kwargs,
        ),
        lambda completions, gold_program, **kwargs: program_consistency_reward(
            completions,
            gold_program,
            program_weight=script_args.program_weight,
            **kwargs,
        ),
    ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
    )

    last_checkpoint = get_checkpoint(training_args)
    if is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}")
        logger.info(f"Starting GRPO training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.model.config.save_pretrained(training_args.output_dir)


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()

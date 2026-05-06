from __future__ import annotations

import os
raw_omp_num_threads = os.environ.get("OMP_NUM_THREADS")
if raw_omp_num_threads is None or not raw_omp_num_threads.strip().isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

from evaluation.evaluate_financial_benchmarks import execute_prediction_program

PROGRAM_NUMERIC_REQUIRED = ["Evidence:", "Program:"]
PROGRAM_NUMERIC_FORBIDDEN = ["Reasoning:", "Answer:", "Normalized Answer:", "Program: N/A"]
PROGRAM_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
    "average",
    "sum",
    "max",
    "min",
]
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?")
ANCHORS = ["Evidence:", "Program:", "Reasoning:", "Answer:", "Normalized Answer:"]
SYSTEM_PROMPT = "You are a financial numerical reasoning assistant. Follow the requested schema exactly."


@dataclass
class ScriptArguments:
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer path. Defaults to model_name_or_path."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "Training json/jsonl file path."})
    valid_file: Optional[str] = field(default=None, metadata={"help": "Validation json/jsonl file path."})
    preprocessing_num_workers: int = field(default=4)
    reward_profile_expected: str = field(default="program_numeric")
    strict_schema_required: bool = field(default=True)
    answer_abs_tol: float = field(default=1e-4)
    answer_rel_tol: float = field(default=1e-4)
    format_reward_weight: float = field(default=0.03)
    program_executable_reward_weight: float = field(default=0.15)
    program_execution_closeness_reward_weight: float = field(default=0.30)
    program_structure_reward_weight: float = field(default=0.20)
    program_argument_coverage_reward_weight: float = field(default=0.12)
    program_step_count_reward_weight: float = field(default=0.08)
    program_exact_match_bonus_weight: float = field(default=0.05)
    evidence_reward_weight: float = field(default=0.04)
    brevity_reward_weight: float = field(default=0.02)
    log_prompt_completions: bool = field(default=False)
    tensorboard_logging_dir: Optional[str] = field(default=None)


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(first_text(v) for v in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return first_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return first_text(completion["content"])
        if "text" in completion:
            return first_text(completion["text"])
    if isinstance(completion, list) and completion:
        return completion_text(completion[0])
    return first_text(completion)


def extract_anchor(text: str, anchor: str) -> str:
    text = first_text(text)
    if not text or anchor not in text:
        return ""
    start = text.index(anchor) + len(anchor)
    end = len(text)
    for other in ANCHORS:
        if other == anchor:
            continue
        pos = text.find(other, start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end].strip()


def normalize_number(text: str) -> Optional[float]:
    text = first_text(text)
    if not text:
        return None
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    raw = matches[-1]
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1]
    try:
        value = float(raw)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def numeric_equal(pred: str, gold: str, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> bool:
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    if pred_num is None or gold_num is None:
        return first_text(pred).strip().lower() == first_text(gold).strip().lower()
    return abs(pred_num - gold_num) <= max(abs_tol, abs(gold_num) * rel_tol)


def numeric_closeness(pred: str, gold: str, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> float:
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    if pred_num is None or gold_num is None:
        return 1.0 if first_text(pred).strip().lower() == first_text(gold).strip().lower() else 0.0
    tolerance = max(abs_tol, abs(gold_num) * rel_tol)
    abs_error = abs(pred_num - gold_num)
    if abs_error <= tolerance:
        return 1.0
    scaled_error = abs_error / max(abs(gold_num), 1.0)
    return max(0.0, 1.0 - min(scaled_error, 2.0) / 2.0)


def program_ops(program: str) -> List[str]:
    program = first_text(program).lower()
    return [op for op in PROGRAM_OPS if re.search(rf"\b{re.escape(op)}\b", program)]


def parse_program_steps(program: str) -> List[tuple[str, List[str]]]:
    program = first_text(program)
    matches = re.findall(r"([A-Za-z_]+)\(([^()]*)\)", program)
    steps = []
    for op, raw_args in matches:
        args = [arg.strip().lower() for arg in raw_args.split(",") if arg.strip()]
        steps.append((op.lower(), args))
    return steps


def multiset_f1(pred_items: List[str], gold_items: List[str]) -> float:
    if not pred_items or not gold_items:
        return 0.0
    pred_counter = Counter(pred_items)
    gold_counter = Counter(gold_items)
    overlap = sum(min(pred_counter[item], gold_counter[item]) for item in pred_counter.keys() | gold_counter.keys())
    precision = overlap / max(len(pred_items), 1)
    recall = overlap / max(len(gold_items), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def program_similarity(pred_program: str, gold_program: str) -> float:
    pred_steps = parse_program_steps(pred_program)
    gold_steps = parse_program_steps(gold_program)
    if not pred_steps or not gold_steps:
        return 0.0

    pred_ops = [op for op, _ in pred_steps]
    gold_ops = [op for op, _ in gold_steps]
    op_score = multiset_f1(pred_ops, gold_ops)

    pred_args = [arg for _, args in pred_steps for arg in args]
    gold_args = [arg for _, args in gold_steps for arg in args]
    arg_score = multiset_f1(pred_args, gold_args)

    prefix_match = 0
    for pred_step, gold_step in zip(pred_steps, gold_steps):
        if pred_step[0] != gold_step[0]:
            break
        prefix_match += 1
    prefix_score = prefix_match / max(len(gold_steps), 1)

    step_count_gap = abs(len(pred_steps) - len(gold_steps)) / max(len(gold_steps), 1)
    length_score = max(0.0, 1.0 - min(step_count_gap, 1.0))
    return 0.45 * op_score + 0.25 * arg_score + 0.20 * prefix_score + 0.10 * length_score


def program_step_count_score(pred_program: str, gold_program: str) -> float:
    pred_steps = parse_program_steps(pred_program)
    gold_steps = parse_program_steps(gold_program)
    if not pred_steps or not gold_steps:
        return 0.0
    gap = abs(len(pred_steps) - len(gold_steps))
    return max(0.0, 1.0 - gap / max(len(gold_steps), 1))


def program_argument_coverage(pred_program: str, gold_program: str) -> float:
    pred_steps = parse_program_steps(pred_program)
    gold_steps = parse_program_steps(gold_program)
    pred_args = [arg for _, args in pred_steps for arg in args]
    gold_args = [arg for _, args in gold_steps for arg in args]
    return multiset_f1(pred_args, gold_args)


def evidence_overlap_score(evidence: str, prompt_text: str) -> float:
    ev_nums = set(NUMBER_RE.findall(first_text(evidence).replace(",", "")))
    prompt_nums = set(NUMBER_RE.findall(first_text(prompt_text).replace(",", "")))
    if not ev_nums or not prompt_nums:
        return 0.0
    overlap = len(ev_nums.intersection(prompt_nums))
    return min(overlap, 4) / 4.0


def length_shape_score(text: str, target_low: int = 80, target_high: int = 160, hard_cap: int = 220) -> float:
    word_count = len(first_text(text).split())
    if word_count == 0 or word_count > hard_cap:
        return 0.0
    if target_low <= word_count <= target_high:
        return 1.0
    if word_count < target_low:
        return max(0.0, word_count / max(target_low, 1))
    tail = hard_cap - target_high
    return max(0.0, 1.0 - ((word_count - target_high) / max(tail, 1)))


def prompt_text_from_any(prompt_value: Any) -> str:
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, list):
        parts = []
        for item in prompt_value:
            if isinstance(item, dict):
                parts.append(first_text(item.get("content")))
            else:
                parts.append(first_text(item))
        return "\n".join(p for p in parts if p)
    if isinstance(prompt_value, dict):
        return first_text(prompt_value.get("content") or prompt_value.get("text") or prompt_value.get("value"))
    return first_text(prompt_value)


def ensure_local_path(path_str: Optional[str], field_name: str) -> Path:
    if not path_str:
        raise ValueError(f"Missing required path for {field_name}")
    path = Path(path_str)
    if path.exists():
        return path
    raise FileNotFoundError(f"{field_name} does not exist: {path}")


def get_checkpoint(training_args: GRPOConfig):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def _normalize_report_to(report_to: Any) -> List[str]:
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [item.strip() for item in report_to.split(",") if item.strip()]
    if isinstance(report_to, (list, tuple, set)):
        return [str(item).strip() for item in report_to if str(item).strip()]
    return [str(report_to).strip()]


def configure_tensorboard_reporting(
    training_args: GRPOConfig,
    script_args: ScriptArguments,
    is_main_process: bool,
    run_label: str,
) -> Optional[Path]:
    logging_dir = (
        Path(script_args.tensorboard_logging_dir)
        if script_args.tensorboard_logging_dir
        else (Path(training_args.logging_dir) if training_args.logging_dir else None)
    )
    if logging_dir is None:
        return None

    logging_dir.mkdir(parents=True, exist_ok=True)
    training_args.logging_dir = str(logging_dir)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    report_to = _normalize_report_to(training_args.report_to)
    if script_args.tensorboard_logging_dir:
        report_to = [item for item in report_to if item.lower() != "none"]
        if not any(item.lower() == "tensorboard" for item in report_to):
            report_to.append("tensorboard")
        training_args.report_to = report_to
    if is_main_process:
        logger.info(
            f"{run_label} TensorBoard reporting: report_to={training_args.report_to}, "
            f"logging_dir={training_args.logging_dir}"
        )
    return logging_dir


def find_all_linear_names(peft_model, int4: bool = False, int8: bool = False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb

        cls = bnb.nn.Linear4bit if int4 else bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if "lm_head" in name or "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def audit_program_numeric_rows(rows: List[Dict[str, Any]], script_args: ScriptArguments) -> Dict[str, Any]:
    execute_fail = 0
    answer_mismatch = 0
    forbidden_schema = 0
    missing_program = 0
    missing_answer = 0
    bad_reward_profile = 0
    duplicate_prompt = 0
    seen_prompts = set()
    source_counts: Dict[str, int] = {}
    gold_program_step_hist: Dict[str, int] = {}

    for row in rows:
        prompt = first_text(row.get("input_prompt_raw"))
        if prompt in seen_prompts:
            duplicate_prompt += 1
        seen_prompts.add(prompt)

        reward_profile = first_text(row.get("reward_profile"))
        if reward_profile != script_args.reward_profile_expected:
            bad_reward_profile += 1
        source = first_text(row.get("source_dataset")) or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

        gold_answer = first_text(row.get("gold_answer"))
        gold_program = first_text(row.get("gold_program"))
        gold_step_count = len(parse_program_steps(gold_program))
        gold_program_step_hist[str(gold_step_count)] = gold_program_step_hist.get(str(gold_step_count), 0) + 1
        if not gold_answer:
            missing_answer += 1
        if not gold_program:
            missing_program += 1

        executed_value, _, _ = execute_prediction_program(gold_program)
        if executed_value is None:
            execute_fail += 1
        elif not numeric_equal(str(executed_value), gold_answer, script_args.answer_abs_tol, script_args.answer_rel_tol):
            answer_mismatch += 1

        if script_args.strict_schema_required:
            ref = first_text(row.get("reference_response"))
            if any(anchor in ref for anchor in PROGRAM_NUMERIC_FORBIDDEN):
                forbidden_schema += 1

    return {
        "rows": len(rows),
        "reward_profile_expected": script_args.reward_profile_expected,
        "source_counts": source_counts,
        "gold_program_step_hist": gold_program_step_hist,
        "program_execute_fail_rows": execute_fail,
        "program_answer_mismatch_rows": answer_mismatch,
        "forbidden_reference_schema_rows": forbidden_schema,
        "missing_program_rows": missing_program,
        "missing_answer_rows": missing_answer,
        "bad_reward_profile_rows": bad_reward_profile,
        "duplicate_prompt_rows": duplicate_prompt,
    }


def make_reward_funcs(script_args: ScriptArguments):
    def reward_program_executable(completions, **kwargs):
        rewards = []
        for completion in completions:
            program = extract_anchor(completion_text(completion), "Program:")
            if not program or program.strip().upper() == "N/A":
                rewards.append(-0.10)
                continue
            executed_value, _, _ = execute_prediction_program(program)
            reward = script_args.program_executable_reward_weight if executed_value is not None else -0.10
            rewards.append(reward)
        return rewards

    def reward_program_execution_closeness(completions, gold_answer=None, **kwargs):
        rewards = []
        gold_answer = gold_answer or [""] * len(completions)
        for completion, gold in zip(completions, gold_answer):
            program = extract_anchor(completion_text(completion), "Program:")
            executed_value, _, _ = execute_prediction_program(program)
            reward = 0.0
            if executed_value is not None:
                reward = script_args.program_execution_closeness_reward_weight * numeric_closeness(
                    str(executed_value), gold, script_args.answer_abs_tol, script_args.answer_rel_tol
                )
            rewards.append(reward)
        return rewards

    def reward_program_structure(completions, gold_program=None, **kwargs):
        rewards = []
        gold_program = gold_program or [""] * len(completions)
        for completion, gold_prog in zip(completions, gold_program):
            pred_program = extract_anchor(completion_text(completion), "Program:")
            if not pred_program or pred_program.strip().upper() == "N/A":
                rewards.append(0.0)
                continue
            rewards.append(
                script_args.program_structure_reward_weight * program_similarity(pred_program, gold_prog)
            )
        return rewards

    def reward_program_argument_coverage(completions, gold_program=None, **kwargs):
        rewards = []
        gold_program = gold_program or [""] * len(completions)
        for completion, gold_prog in zip(completions, gold_program):
            pred_program = extract_anchor(completion_text(completion), "Program:")
            if not pred_program or pred_program.strip().upper() == "N/A":
                rewards.append(0.0)
                continue
            rewards.append(
                script_args.program_argument_coverage_reward_weight
                * program_argument_coverage(pred_program, gold_prog)
            )
        return rewards

    def reward_program_step_count(completions, gold_program=None, **kwargs):
        rewards = []
        gold_program = gold_program or [""] * len(completions)
        for completion, gold_prog in zip(completions, gold_program):
            pred_program = extract_anchor(completion_text(completion), "Program:")
            if not pred_program or pred_program.strip().upper() == "N/A":
                rewards.append(0.0)
                continue
            rewards.append(
                script_args.program_step_count_reward_weight * program_step_count_score(pred_program, gold_prog)
            )
        return rewards

    def reward_program_exact_match_bonus(completions, gold_answer=None, **kwargs):
        rewards = []
        gold_answer = gold_answer or [""] * len(completions)
        for completion, gold in zip(completions, gold_answer):
            program = extract_anchor(completion_text(completion), "Program:")
            executed_value, _, _ = execute_prediction_program(program)
            if executed_value is not None and numeric_equal(
                str(executed_value), gold, script_args.answer_abs_tol, script_args.answer_rel_tol
            ):
                rewards.append(script_args.program_exact_match_bonus_weight)
            else:
                rewards.append(0.0)
        return rewards

    def reward_format_gate(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion_text(completion)
            anchor_hits = [anchor in text for anchor in PROGRAM_NUMERIC_REQUIRED]
            anchor_score = sum(anchor_hits) / len(PROGRAM_NUMERIC_REQUIRED)
            order_bonus = 0.0
            if all(anchor_hits):
                evidence_pos = text.find("Evidence:")
                program_pos = text.find("Program:")
                if 0 <= evidence_pos < program_pos:
                    order_bonus = 0.25
            penalty = sum(1 for anchor in PROGRAM_NUMERIC_FORBIDDEN if anchor in text) * 0.20
            schema_score = min(anchor_score + order_bonus, 1.0)
            rewards.append(schema_score * script_args.format_reward_weight - penalty)
        return rewards

    def reward_evidence_support(completions, input_prompt_raw=None, prompt=None, **kwargs):
        rewards = []
        input_prompt_raw = input_prompt_raw or prompt or [""] * len(completions)
        for completion, raw_prompt in zip(completions, input_prompt_raw):
            program = extract_anchor(completion_text(completion), "Program:")
            executed_value, _, _ = execute_prediction_program(program)
            if executed_value is None:
                rewards.append(0.0)
                continue
            evidence = extract_anchor(completion_text(completion), "Evidence:")
            raw_prompt_text = prompt_text_from_any(raw_prompt)
            overlap_score = evidence_overlap_score(evidence, raw_prompt_text)
            rewards.append(script_args.evidence_reward_weight * overlap_score)
        return rewards

    def reward_length_regularizer(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion_text(completion)
            rewards.append(script_args.brevity_reward_weight * length_shape_score(text))
        return rewards

    return [
        reward_program_executable,
        reward_program_execution_closeness,
        reward_program_structure,
        reward_program_argument_coverage,
        reward_program_step_count,
        reward_program_exact_match_bonus,
        reward_format_gate,
        reward_evidence_support,
        reward_length_regularizer,
    ]


def prepare_dataset(dataset: Dataset, script_args: ScriptArguments, is_main_process: bool) -> Dataset:
    def validate_row(row: Dict[str, Any]) -> Dict[str, Any]:
        reward_profile = first_text(row.get("reward_profile"))
        if reward_profile != script_args.reward_profile_expected:
            raise ValueError(f"Unexpected reward_profile: {reward_profile}")
        if not first_text(row.get("input_prompt_raw")):
            raise ValueError("Missing input_prompt_raw")
        if not first_text(row.get("gold_answer")):
            raise ValueError("Missing gold_answer")
        if not first_text(row.get("gold_program")):
            raise ValueError("Missing gold_program")
        executed_value, _, _ = execute_prediction_program(first_text(row.get("gold_program")))
        if executed_value is None:
            raise ValueError("gold_program is not executable")
        if not numeric_equal(
            str(executed_value),
            first_text(row.get("gold_answer")),
            script_args.answer_abs_tol,
            script_args.answer_rel_tol,
        ):
            raise ValueError("gold_program execution does not match gold_answer")
        input_prompt_raw = first_text(row["input_prompt_raw"])
        input_prompt_chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_prompt_raw},
        ]
        return {
            "prompt": input_prompt_chat,
            "input_prompt_chat": input_prompt_chat,
            "input_prompt_raw": input_prompt_raw,
            "gold_answer": first_text(row["gold_answer"]),
            "gold_program": first_text(row["gold_program"]),
            "reward_profile": reward_profile,
            "record_id": first_text(row.get("record_id")),
            "source_dataset": first_text(row.get("source_dataset")),
            "reference_response": first_text(row.get("reference_response")),
        }

    return dataset.map(
        validate_row,
        num_proc=script_args.preprocessing_num_workers,
        desc="Processing dataset" if is_main_process else None,
    )


def grpo_train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    is_main_process = training_args.local_rank in [-1, 0]
    train_file = ensure_local_path(script_args.train_file, "train_file")
    valid_file = ensure_local_path(script_args.valid_file, "valid_file")
    tokenizer_path = ensure_local_path(script_args.tokenizer_name_or_path or model_args.model_name_or_path, "tokenizer_name_or_path")
    model_path = ensure_local_path(model_args.model_name_or_path, "model_name_or_path")
    output_dir = Path(training_args.output_dir)
    logging_dir = configure_tensorboard_reporting(training_args, script_args, is_main_process, "GRPO")
    training_args.log_completions = bool(script_args.log_prompt_completions)
    training_args.multi_objective_aggregation = "normalize_then_sum"
    training_args.scale_rewards = "none"
    training_args.loss_type = "dapo"

    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model args: {model_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(f"Training args: {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    raw_eval_dataset = load_dataset("json", data_files=str(valid_file), split="train")

    with training_args.main_process_first(desc="Dataset preparation"):
        train_dataset = prepare_dataset(raw_train_dataset, script_args, is_main_process)
        eval_dataset = prepare_dataset(raw_eval_dataset, script_args, is_main_process)

    train_audit = audit_program_numeric_rows(list(raw_train_dataset), script_args)
    eval_audit = audit_program_numeric_rows(list(raw_eval_dataset), script_args)

    torch_dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot both be set")

    quantization_config = None
    if model_args.load_in_4bit or model_args.load_in_8bit:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )
    num_gpus = torch.cuda.device_count()
    if ddp:
        model_kwargs["device_map"] = None
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            usable_mem = int(gpu_props.total_memory * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)

    if model_args.use_peft:
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == "all" or (target_modules and "all" in target_modules):
            target_modules = find_all_linear_names(
                model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit
            )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in filter(lambda p: p.requires_grad, model.parameters()):
                param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        peft_config = None

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "model_args": asdict(model_args),
        "script_args": asdict(script_args),
        "training_args_summary": {
            "output_dir": training_args.output_dir,
            "learning_rate": training_args.learning_rate,
            "beta": training_args.beta,
            "max_steps": training_args.max_steps,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "num_generations": training_args.num_generations,
            "max_completion_length": training_args.max_completion_length,
            "report_to": training_args.report_to,
            "logging_dir": training_args.logging_dir,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "bf16": training_args.bf16,
            "use_vllm": training_args.use_vllm,
            "temperature": training_args.temperature,
            "top_p": training_args.top_p,
            "min_p": training_args.min_p,
            "multi_objective_aggregation": training_args.multi_objective_aggregation,
            "scale_rewards": training_args.scale_rewards,
            "loss_type": training_args.loss_type,
        },
        "train_audit": train_audit,
        "eval_audit": eval_audit,
        "tokenizer_path": str(tokenizer_path),
        "model_path": str(model_path),
        "tensorboard_logging_dir": str(logging_dir) if logging_dir else None,
        "created_at": datetime.now().isoformat(),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reward_funcs = make_reward_funcs(script_args)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
    )

    last_checkpoint = get_checkpoint(training_args)
    resume_checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None and is_main_process:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}")
        logger.info(
            f"Starting GRPO training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for max_steps={training_args.max_steps}"
        )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
    training_args.distributed_state.wait_for_everyone()
    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.model.config.save_pretrained(training_args.output_dir)
        logger.info(f"Training complete. Model saved to {training_args.output_dir}")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

try:
    from training.finqa_program_grpo import (
        ScriptArguments,
        asdict,
        audit_program_numeric_rows,
        configure_tensorboard_reporting,
        ensure_local_path,
        find_all_linear_names,
        get_checkpoint,
        get_peft_model,
        is_deepspeed_zero3_enabled,
        load_dataset,
        prepare_dataset,
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LoraConfig,
        TaskType,
    )
    BASE_MODULE = "training.finqa_program_grpo"
except ImportError:
    from finqa_program_grpo import (
        ScriptArguments,
        asdict,
        audit_program_numeric_rows,
        configure_tensorboard_reporting,
        ensure_local_path,
        find_all_linear_names,
        get_checkpoint,
        get_peft_model,
        is_deepspeed_zero3_enabled,
        load_dataset,
        prepare_dataset,
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LoraConfig,
        TaskType,
    )
    BASE_MODULE = "finqa_program_grpo"



class ProgramDiagnostics:
    def __init__(self):
        self.pending: Dict[str, List[float]] = {}

    def add(self, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        self.pending.setdefault(name, []).append(float(value))

    def pop_means(self) -> Dict[str, float]:
        metrics = {}
        for name, values in self.pending.items():
            valid = [value for value in values if value == value]
            if valid:
                metrics[name] = sum(valid) / len(valid)
        self.pending.clear()
        return metrics


class ProgramDiagnosticGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, program_diagnostics: Optional[ProgramDiagnostics] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.program_diagnostics = program_diagnostics

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        if self.importance_sampling_level == "sequence":
            for source, target in (
                ("clip_ratio/region_mean", "sequence/response_clipped_fraction"),
                ("clip_ratio/low_mean", "sequence/response_low_clipped_fraction"),
                ("clip_ratio/high_mean", "sequence/response_high_clipped_fraction"),
            ):
                if source in self._metrics[mode]:
                    self._metrics[mode][target].extend(self._metrics[mode][source])

        if self.program_diagnostics is not None:
            device = self.accelerator.device
            for name, local_mean in sorted(self.program_diagnostics.pop_means().items()):
                gathered = self.accelerator.gather(torch.tensor(local_mean, device=device))
                self._metrics[mode][name].append(gathered.nanmean().item())
        super().log(logs, start_time)


def evaluate_program_completion(base: Any, script_args: ScriptArguments, completion: Any, gold_ans: str, gold_prog: str) -> Dict[str, Any]:
    text = base.completion_text(completion)
    pred_program = base.extract_anchor(text, "Program:")
    result: Dict[str, Any] = {
        "text": text,
        "program": pred_program,
        "core_score": -0.10,
        "executable": 0.0,
        "exact_match": 0.0,
        "structure_score": 0.0,
        "argument_coverage": 0.0,
        "step_count_score": 0.0,
        "completion_words": float(len(text.split())),
    }
    if not pred_program or pred_program.strip().upper() == "N/A":
        return result

    executed_value, _, _ = base.execute_prediction_program(pred_program)
    if executed_value is None:
        return result

    closeness = base.numeric_closeness(str(executed_value), gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol)
    structure = base.program_similarity(pred_program, gold_prog)
    argument_coverage = base.program_argument_coverage(pred_program, gold_prog)
    step_count = base.program_step_count_score(pred_program, gold_prog)
    exact_match = base.numeric_equal(str(executed_value), gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol)

    core_score = script_args.program_executable_reward_weight
    core_score += script_args.program_execution_closeness_reward_weight * closeness
    core_score += script_args.program_structure_reward_weight * structure
    core_score += script_args.program_argument_coverage_reward_weight * argument_coverage
    core_score += script_args.program_step_count_reward_weight * step_count
    if exact_match:
        core_score += script_args.program_exact_match_bonus_weight

    result.update(
        {
            "core_score": core_score,
            "executable": 1.0,
            "exact_match": 1.0 if exact_match else 0.0,
            "structure_score": structure,
            "argument_coverage": argument_coverage,
            "step_count_score": step_count,
        }
    )
    return result

def make_gspo_reward_funcs(script_args: ScriptArguments):
    base = importlib.import_module(BASE_MODULE)
    diagnostics = ProgramDiagnostics()

    def reward_program_core(completions, gold_answer=None, gold_program=None, **kwargs):
        rewards = []
        gold_answer = gold_answer or [""] * len(completions)
        gold_program = gold_program or [""] * len(completions)

        for completion, gold_ans, gold_prog in zip(completions, gold_answer, gold_program):
            result = evaluate_program_completion(base, script_args, completion, gold_ans, gold_prog)
            rewards.append(result["core_score"])
            diagnostics.add("program/core_score", result["core_score"])
            diagnostics.add("program/executable_rate", result["executable"])
            diagnostics.add("program/exact_match_rate", result["exact_match"])
            diagnostics.add("program/structure_score", result["structure_score"])
            diagnostics.add("program/argument_coverage", result["argument_coverage"])
            diagnostics.add("program/step_count_score", result["step_count_score"])
            diagnostics.add("program/completion_words", result["completion_words"])

        programs = [
            base.extract_anchor(base.completion_text(completion), "Program:").strip()
            for completion in completions
        ]
        non_empty_programs = [program for program in programs if program and program.upper() != "N/A"]
        if completions:
            diagnostics.add("program/has_program_rate", len(non_empty_programs) / len(completions))
        if non_empty_programs:
            diagnostics.add("program/unique_program_ratio", len(set(non_empty_programs)) / len(non_empty_programs))
        return rewards

    def reward_format_gate(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = base.completion_text(completion)
            anchor_hits = [anchor in text for anchor in base.PROGRAM_NUMERIC_REQUIRED]
            anchor_score = sum(anchor_hits) / len(base.PROGRAM_NUMERIC_REQUIRED)
            order_bonus = 0.0
            if all(anchor_hits):
                evidence_pos = text.find("Evidence:")
                program_pos = text.find("Program:")
                if 0 <= evidence_pos < program_pos:
                    order_bonus = 0.25
            penalty = sum(1 for anchor in base.PROGRAM_NUMERIC_FORBIDDEN if anchor in text) * 0.20
            schema_score = min(anchor_score + order_bonus, 1.0)
            rewards.append(schema_score * script_args.format_reward_weight - penalty)
        return rewards

    def reward_evidence_support(completions, input_prompt_raw=None, prompt=None, **kwargs):
        rewards = []
        input_prompt_raw = input_prompt_raw or prompt or [""] * len(completions)
        for completion, raw_prompt in zip(completions, input_prompt_raw):
            program = base.extract_anchor(base.completion_text(completion), "Program:")
            executed_value, _, _ = base.execute_prediction_program(program)
            if executed_value is None:
                rewards.append(0.0)
                continue
            evidence = base.extract_anchor(base.completion_text(completion), "Evidence:")
            rewards.append(
                script_args.evidence_reward_weight
                * base.evidence_overlap_score(evidence, base.prompt_text_from_any(raw_prompt))
            )
        return rewards

    def reward_length_regularizer(completions, **kwargs):
        return [
            script_args.brevity_reward_weight * base.length_shape_score(base.completion_text(c))
            for c in completions
        ]

    reward_funcs = [
        reward_program_core,
        reward_format_gate,
        reward_evidence_support,
        reward_length_regularizer,
    ]
    return reward_funcs, diagnostics


def gspo_train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    is_main_process = training_args.local_rank in [-1, 0]
    train_file = ensure_local_path(script_args.train_file, "train_file")
    valid_file = ensure_local_path(script_args.valid_file, "valid_file")
    tokenizer_path = ensure_local_path(script_args.tokenizer_name_or_path or model_args.model_name_or_path, "tokenizer_name_or_path")
    model_path = ensure_local_path(model_args.model_name_or_path, "model_name_or_path")
    output_dir = Path(training_args.output_dir)
    logging_dir = configure_tensorboard_reporting(training_args, script_args, is_main_process, "GSPO")

    training_args.log_completions = bool(script_args.log_prompt_completions)
    training_args.multi_objective_aggregation = "normalize_then_sum"
    training_args.scale_rewards = "none"
    training_args.loss_type = "dapo"
    training_args.importance_sampling_level = "sequence"

    if is_main_process:
        logger.info("Running GSPO-compatible sequence-level training configuration.")

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
    quantization_config = None
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot both be set")
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

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
        device_map="auto",
    )

    if model_args.use_peft:
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == "all" or (target_modules and "all" in target_modules):
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
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in filter(lambda p: p.requires_grad, model.parameters()):
                param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

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
            "temperature": training_args.temperature,
            "top_p": training_args.top_p,
            "report_to": training_args.report_to,
            "logging_dir": training_args.logging_dir,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "bf16": training_args.bf16,
            "importance_sampling_level": training_args.importance_sampling_level,
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
    (output_dir / "experiment_config.json").write_text(json.dumps(experiment_config, ensure_ascii=False, indent=2), encoding="utf-8")

    reward_funcs, diagnostics = make_gspo_reward_funcs(script_args)
    trainer = ProgramDiagnosticGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        program_diagnostics=diagnostics,
    )

    last_checkpoint = get_checkpoint(training_args)
    resume_checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.model.config.save_pretrained(training_args.output_dir)


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    gspo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()

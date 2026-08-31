#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v34r23_prompt_processor import prepare_rows_like_grpo

FIXED_SMOKE_CONFIG = {
    "reward_mode": "frontier_execution_calibration",
    "num_generations": 8,
    "temperature": 0.72,
    "top_p": 0.95,
    "max_completion_length": 300,
    "learning_rate": 4e-8,
    "beta": 0.01,
    "seed": 34023,
    "data_seed": 34023,
    "bf16": True,
    "gradient_checkpointing": True,
    "max_steps": 5,
    "steps_per_generation": 1,
    "per_device_train_batch_size": 8,
    "bundle_ordered_sampling": False,
}

DEFAULT_BASE = "/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program_merged"
DEFAULT_ADAPTER = "/root/autodl-tmp/outputs/financial_reasoning_rl/v34r22_sft2_retention_aware_rs_sft_smoke50/checkpoint-20"
DEFAULT_DATA_DIR = "/root/autodl-tmp/data/financial_reasoning_v3/v34r23_rs_sft_frontier_grpo_increment/convfinqa_all1000_strict_program_only"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/outputs/financial_reasoning_rl/v34r23_convfinqa_strict_program_only_controlled_smoke5"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def training_bucket(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("v34r23_bucket") or row.get("training_bucket") or row.get("bucket") or "")


def is_history_dependent(row: Dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    return bool(meta.get("v34r23_requires_history") or meta.get("requires_history"))


def smoke_train_row_count(max_steps: int) -> int:
    return max_steps


def select_frontier_smoke_rows(rows: Sequence[Dict[str, Any]], max_steps: int, seed: int) -> List[Dict[str, Any]]:
    frontier = [row for row in rows if training_bucket(row) == "frontier"]
    needed = smoke_train_row_count(max_steps)
    if len(frontier) < needed:
        raise ValueError(f"not enough frontier rows for smoke: {len(frontier)} < {needed}")
    ordered = list(frontier)
    random.Random(seed).shuffle(ordered)
    return ordered[:needed]


def assert_controlled_smoke_config(config: Dict[str, Any]) -> None:
    if int(config.get("max_steps", 0)) > 5:
        raise ValueError("max_steps must be <= 5 for controlled smoke")
    for key, expected in FIXED_SMOKE_CONFIG.items():
        if key == "max_steps":
            continue
        actual = config.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise ValueError(f"{key} must remain {expected}; got {actual}")
        elif actual != expected:
            raise ValueError(f"{key} must remain {expected}; got {actual}")


def reward_group_summary(
    rewards: Sequence[float],
    completions: Sequence[str],
    diagnostics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    values = [float(value) for value in rewards]
    mean = sum(values) / max(len(values), 1)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / max(len(values), 1)) if values else 0.0
    correct = sum(1 for item in diagnostics if float(item.get("exact_match", 0.0)) >= 1.0 or item.get("answer_correct"))
    executable = sum(1 for item in diagnostics if float(item.get("executable", 0.0)) >= 1.0 or item.get("executable") is True)
    invalid = sum(1 for item in diagnostics if float(item.get("invalid", 0.0)) >= 1.0 or item.get("invalid") is True)
    wrong_exec = sum(1 for item in diagnostics if float(item.get("wrong_executable", 0.0)) >= 1.0 or item.get("wrong_executable") is True)
    programs = [str(item.get("program") or "") for item in diagnostics if str(item.get("program") or "")]
    return {
        "reward_mean": mean,
        "reward_std": std,
        "zero_std": std <= 1e-8,
        "all_correct": bool(diagnostics) and correct == len(diagnostics),
        "all_wrong": bool(diagnostics) and correct == 0,
        "mixed_reward": len(set(values)) > 1 and correct > 0 and correct < len(diagnostics),
        "sampled_correct_rate": correct / max(len(diagnostics), 1),
        "sampled_executable_rate": executable / max(len(diagnostics), 1),
        "invalid_rate": invalid / max(len(diagnostics), 1),
        "wrong_executable_rate": wrong_exec / max(len(diagnostics), 1),
        "unique_program_ratio": len(set(programs)) / max(len(programs), 1),
        "reasoning_marker_rate": sum(1 for text in completions if "Reasoning:" in text) / max(len(completions), 1),
        "answer_marker_rate": sum(1 for text in completions if "Answer:" in text) / max(len(completions), 1),
        "normalized_answer_marker_rate": sum(1 for text in completions if "Normalized Answer:" in text) / max(len(completions), 1),
        "multiple_program_rate": sum(1 for text in completions if text.count("Program:") != 1) / max(len(completions), 1),
    }


def prompt_hashes(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    processed = prepare_rows_like_grpo(rows, is_main_process=False)
    hashes = {}
    for row, proc in zip(rows, processed):
        prompt = "\n".join(message.get("content", "") for message in proc.get("prompt", []))
        hashes[str(row.get("record_id"))] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return hashes


def build_smoke_data(data_dir: Path, output_dir: Path, max_steps: int, seed: int) -> Dict[str, Any]:
    train_rows = read_jsonl(data_dir / "train.jsonl")
    valid_rows = [row for row in read_jsonl(data_dir / "valid.jsonl") if training_bucket(row) == "frontier"]
    selected = select_frontier_smoke_rows(train_rows, max_steps=max_steps, seed=seed)
    hashes = prompt_hashes(selected)
    train_file = output_dir / "controlled_smoke_train_frontier_only.jsonl"
    valid_file = output_dir / "controlled_smoke_valid_frontier_only.jsonl"
    write_jsonl(train_file, selected)
    write_jsonl(valid_file, valid_rows)
    planned = []
    for index, row in enumerate(selected, start=1):
        planned.append(
            {
                "step": index,
                "record_id": row.get("record_id"),
                "record_ids": [row.get("record_id")],
                "training_bucket": training_bucket(row),
                "training_buckets": [training_bucket(row)],
                "history_dependent": is_history_dependent(row),
                "history_dependent_values": [is_history_dependent(row)],
                "source_dataset": row.get("source_dataset"),
                "source_datasets": [row.get("source_dataset")],
                "prompt_sha256": hashes.get(str(row.get("record_id"))),
                "prompt_sha256_values": [hashes.get(str(row.get("record_id")))],
            }
        )
    (output_dir / "planned_steps.json").write_text(json.dumps(planned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"train_file": str(train_file), "valid_file": str(valid_file), "planned_steps": planned}


def load_trainer_logs(output_dir: Path) -> Dict[int, Dict[str, Any]]:
    path = output_dir / "trainer_state.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    logs: Dict[int, Dict[str, Any]] = {}
    for entry in data.get("log_history", []):
        step = entry.get("step")
        if isinstance(step, int):
            logs.setdefault(step, {}).update(entry)
    return logs


def compare_lora_adapters(initial_adapter: Path, output_dir: Path) -> Dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        return {"checked": False, "error": str(exc)}
    before_path = initial_adapter / "adapter_model.safetensors"
    after_path = output_dir / "adapter_model.safetensors"
    if not before_path.exists() or not after_path.exists():
        return {"checked": False, "before_exists": before_path.exists(), "after_exists": after_path.exists()}
    before = load_file(str(before_path), device="cpu")
    after = load_file(str(after_path), device="cpu")
    common = sorted(set(before) & set(after))
    changed = 0
    max_abs_delta = 0.0
    for key in common:
        delta = (after[key].float() - before[key].float()).abs()
        local = float(delta.max().item()) if delta.numel() else 0.0
        if local > 0.0:
            changed += 1
        max_abs_delta = max(max_abs_delta, local)
    return {
        "checked": True,
        "common_tensors": len(common),
        "changed_tensors": changed,
        "max_abs_delta": max_abs_delta,
        "updated": changed > 0 and max_abs_delta > 0.0,
    }


def run_training(args: argparse.Namespace, train_file: str, valid_file: str, reward_log: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["V34R23_CONTROLLED_SMOKE_REWARD_LOG"] = str(reward_log)
    env["V34R23_CONTROLLED_SMOKE_NUM_GENERATIONS"] = str(FIXED_SMOKE_CONFIG["num_generations"])
    cmd = [
        "/root/miniconda3/bin/python", "-m", "training.finqa_program_grpo",
        "--model_name_or_path", args.base_path,
        "--tokenizer_name_or_path", args.base_path,
        "--peft_path", args.adapter_path,
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--reward_profile_expected", "program_numeric",
        "--reward_mode", "frontier_execution_calibration",
        "--strict_schema_required", "True",
        "--schema_hard_gate", "True",
        "--outcome_first_reward", "False",
        "--dense_reward_shaping", "False",
        "--process_reward_enabled", "False",
        "--history_grounding_reward_weight", "0.0",
        "--evidence_reward_weight", "0.0",
        "--operation_prior_weight", "0.0",
        "--scale_rewards", "none",
        "--loss_type", "dapo",
        "--bundle_ordered_sampling", "False",
        "--output_dir", str(args.output_dir),
        "--run_name", "v34r23_convfinqa_strict_program_only_controlled_smoke5",
        "--report_to", "tensorboard",
        "--logging_dir", str(args.output_dir / "tensorboard"),
        "--do_train", "True",
        "--eval_strategy", "no",
        "--save_strategy", "no",
        "--max_steps", str(args.max_steps),
        "--learning_rate", "4e-8",
        "--beta", "0.01",
        "--lr_scheduler_type", "constant",
        "--warmup_steps", "0",
        "--per_device_train_batch_size", "8",
        "--gradient_accumulation_steps", "1",
        "--num_generations", "8",
        "--steps_per_generation", "1",
        "--max_completion_length", "300",
        "--temperature", "0.72",
        "--top_p", "0.95",
        "--bf16", "True",
        "--gradient_checkpointing", "True",
        "--use_vllm", "False",
        "--use_peft", "True",
        "--dtype", "bfloat16",
        "--lora_target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
        "--lora_r", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--seed", "34023",
        "--data_seed", "34023",
        "--logging_steps", "1",
        "--logging_first_step", "True",
        "--save_only_model", "True",
        "--remove_unused_columns", "False",
    ]
    (args.output_dir / "training_command.json").write_text(json.dumps(cmd, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def summarize_run(output_dir: Path, adapter_path: Path, max_steps: int) -> Dict[str, Any]:
    planned = json.loads((output_dir / "planned_steps.json").read_text(encoding="utf-8"))
    reward_rows = read_jsonl(output_dir / "step_reward_trace.jsonl") if (output_dir / "step_reward_trace.jsonl").exists() else []
    trainer_logs = load_trainer_logs(output_dir)
    steps = []
    for index, row in enumerate(reward_rows[:max_steps], start=1):
        merged = dict(row)
        merged["step"] = index
        for key, value in planned[index - 1].items():
            if not merged.get(key):
                merged[key] = value
        log = trainer_logs.get(index, {})
        merged["loss"] = log.get("loss")
        merged["grad_norm"] = log.get("grad_norm")
        merged["learning_rate"] = log.get("learning_rate")
        steps.append(merged)
    zero = sum(1 for row in steps if row.get("zero_std"))
    executable_values = [float(row.get("sampled_executable_rate", 0.0)) for row in steps]
    marker_max = max(
        [
            float(row.get("reasoning_marker_rate", 0.0))
            + float(row.get("answer_marker_rate", 0.0))
            + float(row.get("normalized_answer_marker_rate", 0.0))
            for row in steps
        ]
        or [0.0]
    )
    all_wrong_runs = 0
    current = 0
    for row in steps:
        if row.get("all_wrong"):
            current += 1
            all_wrong_runs = max(all_wrong_runs, current)
        else:
            current = 0
    lora = compare_lora_adapters(adapter_path, output_dir)
    gate = {
        "ran_full_5_steps": len(steps) == max_steps == 5,
        "reward_std_positive_steps": sum(1 for row in steps if float(row.get("reward_std", 0.0)) > 0.0),
        "mixed_reward_steps": sum(1 for row in steps if row.get("mixed_reward")),
        "zero_std_steps": zero,
        "frac_reward_zero_std": zero / max(len(steps), 1),
        "sampled_executable_rate_mean": sum(executable_values) / max(len(executable_values), 1),
        "max_consecutive_all_wrong": all_wrong_runs,
        "strict_marker_max_sum": marker_max,
        "lora_updated": bool(lora.get("updated")),
    }
    passed = (
        gate["ran_full_5_steps"]
        and gate["reward_std_positive_steps"] >= 2
        and gate["mixed_reward_steps"] >= 2
        and gate["frac_reward_zero_std"] < 0.80
        and gate["sampled_executable_rate_mean"] >= 0.70
        and gate["max_consecutive_all_wrong"] < 2
        and gate["strict_marker_max_sum"] == 0.0
        and gate["lora_updated"]
    )
    code = "convfinqa_controlled_smoke_passed_short_grpo_sweep_allowed_next_round" if passed else "controlled_smoke_online_variance_not_reproduced"
    if gate["max_consecutive_all_wrong"] >= 2:
        code = "controlled_smoke_history_frontier_variance_insufficient"
    if gate["sampled_executable_rate_mean"] < 0.70 or marker_max > 0.0:
        code = "controlled_smoke_execution_contract_regressed"
    summary = {"steps": steps, "trainer_logs": trainer_logs, "lora_update": lora, "gate": gate, "conclusion_code": code}
    (output_dir / "controlled_smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v34r23 ConvFinQA strict program-only controlled GRPO smoke.")
    parser.add_argument("--data_dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--base_path", default=DEFAULT_BASE)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=34023)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(FIXED_SMOKE_CONFIG, max_steps=args.max_steps, seed=args.seed, data_seed=args.seed)
    assert_controlled_smoke_config(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists and is not empty: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_smoke_data(args.data_dir, args.output_dir, args.max_steps, args.seed)
    snapshot = {"fixed_config": config, "data": data, "base_path": args.base_path, "adapter_path": args.adapter_path}
    (args.output_dir / "controlled_smoke_config.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return
    reward_log = args.output_dir / "step_reward_trace.jsonl"
    run_training(args, data["train_file"], data["valid_file"], reward_log)
    print(json.dumps(summarize_run(args.output_dir, Path(args.adapter_path), args.max_steps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

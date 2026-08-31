#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v34r23_prompt_processor import prepare_rows_like_grpo

FIXED_GRPO40_CONFIG = {
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
    "max_steps": 40,
    "save_steps": 10,
    "steps_per_generation": 1,
    "per_device_train_batch_size": 8,
    "bundle_ordered_sampling": False,
}

DEFAULT_BASE = "/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program_merged"
DEFAULT_ADAPTER = "/root/autodl-tmp/outputs/financial_reasoning_rl/v34r22_sft2_retention_aware_rs_sft_smoke50/checkpoint-20"
SMOKE_ADAPTER = "/root/autodl-tmp/outputs/financial_reasoning_rl/v34r23_convfinqa_strict_program_only_controlled_smoke5"
DEFAULT_DATA_DIR = "/root/autodl-tmp/data/financial_reasoning_v3/v34r23_rs_sft_frontier_grpo_increment/convfinqa_all1000_strict_program_only"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/outputs/financial_reasoning_rl/v34r23_convfinqa_strict_program_only_grpo40"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_bucket(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("v34r23_bucket") or row.get("training_bucket") or row.get("bucket") or "")


def is_history_dependent(row: Dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    return bool(meta.get("v34r23_requires_history") or meta.get("requires_history"))


def select_frontier_train_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = [row for row in rows if training_bucket(row) == "frontier"]
    if not selected:
        raise ValueError("frontier-only GRPO40 selected zero rows")
    return selected


def assert_grpo40_config(config: Dict[str, Any], adapter_path: str) -> None:
    if int(config.get("max_steps", 0)) > 40:
        raise ValueError("max_steps must be <= 40 for GRPO40 sweep")
    if int(config.get("save_steps", 0)) != 10:
        raise ValueError("save_steps must remain 10")
    if Path(adapter_path).resolve() == Path(SMOKE_ADAPTER).resolve():
        raise ValueError("GRPO40 must initialize from RS-SFT checkpoint-20, not the controlled smoke adapter")
    for key, expected in FIXED_GRPO40_CONFIG.items():
        actual = config.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise ValueError(f"{key} must remain {expected}; got {actual}")
        elif actual != expected:
            raise ValueError(f"{key} must remain {expected}; got {actual}")


def prompt_hashes(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    processed = prepare_rows_like_grpo(rows, is_main_process=False)
    hashes = {}
    for row, proc in zip(rows, processed):
        prompt = "\n".join(message.get("content", "") for message in proc.get("prompt", []))
        hashes[str(row.get("record_id"))] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return hashes


def build_frontier_only_data(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    train_rows = read_jsonl(data_dir / "train.jsonl")
    valid_rows = read_jsonl(data_dir / "valid.jsonl")
    frontier_rows = select_frontier_train_rows(train_rows)
    valid_frontier = [row for row in valid_rows if training_bucket(row) == "frontier"]
    hashes = prompt_hashes(frontier_rows)
    train_file = output_dir / "convfinqa_frontier_only_train.jsonl"
    valid_file = output_dir / "convfinqa_frontier_only_valid.jsonl"
    write_jsonl(train_file, frontier_rows)
    write_jsonl(valid_file, valid_frontier)
    manifest = {
        "source_train": str(data_dir / "train.jsonl"),
        "source_valid": str(data_dir / "valid.jsonl"),
        "source_train_sha256": file_sha256(data_dir / "train.jsonl"),
        "frontier_train_file": str(train_file),
        "frontier_train_sha256": file_sha256(train_file),
        "frontier_train_rows": len(frontier_rows),
        "frontier_valid_rows": len(valid_frontier),
        "bucket_counts": dict(Counter(training_bucket(row) for row in frontier_rows)),
        "history_rows": sum(1 for row in frontier_rows if is_history_dependent(row)),
        "nonhistory_rows": sum(1 for row in frontier_rows if not is_history_dependent(row)),
        "record_id_sha256": hashlib.sha256(
            "\n".join(str(row.get("record_id")) for row in frontier_rows).encode("utf-8")
        ).hexdigest(),
        "prompt_sha256_by_record_id": hashes,
    }
    (output_dir / "train_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"train_file": str(train_file), "valid_file": str(valid_file), "manifest": manifest}


def disk_budget(adapter_path: Path, output_dir: Path) -> Dict[str, Any]:
    stat = shutil.disk_usage(str(output_dir.parent if output_dir.parent.exists() else Path("/root/autodl-tmp")))
    adapter_size = sum(path.stat().st_size for path in adapter_path.rglob("*") if path.is_file())
    return {
        "available_bytes": stat.free,
        "available_gb": stat.free / (1024 ** 3),
        "single_adapter_checkpoint_bytes": adapter_size,
        "single_adapter_checkpoint_mb": adapter_size / (1024 ** 2),
        "four_checkpoint_estimate_mb": adapter_size * 4 / (1024 ** 2),
        "logs_and_final_adapter_estimate_mb": adapter_size * 2 / (1024 ** 2),
    }


def git_snapshot() -> Dict[str, Any]:
    def run(args: List[str]) -> str:
        return subprocess.check_output(args, cwd=str(REPO_ROOT), text=True).strip()

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "status_short": run(["git", "status", "--short"]),
        "diff_stat": run(["git", "diff", "--stat"]),
        "head": run(["git", "rev-parse", "HEAD"]),
    }


def run_training(args: argparse.Namespace, train_file: str, valid_file: str, reward_log: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["V34R23_CONTROLLED_SMOKE_REWARD_LOG"] = str(reward_log)
    env["V34R23_CONTROLLED_SMOKE_NUM_GENERATIONS"] = str(FIXED_GRPO40_CONFIG["num_generations"])
    env["V34R23_ASSERT_REWARD_GROUPS"] = "1"
    env["V34R23_GRPO_EARLY_STOP_ENABLED"] = "1"
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
        "--run_name", "v34r23_convfinqa_strict_program_only_grpo40",
        "--report_to", "tensorboard",
        "--logging_dir", str(args.output_dir / "tensorboard"),
        "--do_train", "True",
        "--eval_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "10",
        "--max_steps", "40",
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


def checkpoint_integrity(output_dir: Path) -> Dict[str, Any]:
    result = {}
    for step in (10, 20, 30, 40):
        ckpt = output_dir / f"checkpoint-{step}"
        files = sorted(path.name for path in ckpt.glob("*")) if ckpt.exists() else []
        result[f"checkpoint-{step}"] = {
            "exists": ckpt.exists(),
            "adapter_config": (ckpt / "adapter_config.json").exists(),
            "adapter_weights": (ckpt / "adapter_model.safetensors").exists(),
            "has_full_model_safetensors": (ckpt / "model.safetensors").exists(),
            "size_bytes": sum(path.stat().st_size for path in ckpt.rglob("*") if path.is_file()) if ckpt.exists() else 0,
            "files": files,
            "adapter_sha256": file_sha256(ckpt / "adapter_model.safetensors") if (ckpt / "adapter_model.safetensors").exists() else "",
        }
    return result


def summarize_windows(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    windows = {"overall": list(steps)}
    for start in (1, 11, 21, 31):
        windows[f"step_{start}_{start+9}"] = [row for row in steps if start <= int(row.get("step", 0)) <= start + 9]
    for name in ("history_frontier", "nonhistory_frontier"):
        want = name == "history_frontier"
        windows[name] = [row for row in steps if bool(row.get("history_dependent")) is want]

    summaries = {}
    for name, rows in windows.items():
        summaries[name] = {
            "steps": len(rows),
            "reward_std_mean": sum(float(row.get("reward_std", 0.0)) for row in rows) / max(len(rows), 1),
            "zero_std_ratio": sum(1 for row in rows if row.get("zero_std")) / max(len(rows), 1),
            "mixed_reward_ratio": sum(1 for row in rows if row.get("mixed_reward")) / max(len(rows), 1),
            "all_correct_ratio": sum(1 for row in rows if row.get("all_correct")) / max(len(rows), 1),
            "all_wrong_ratio": sum(1 for row in rows if row.get("all_wrong")) / max(len(rows), 1),
            "sampled_correct_rate": sum(float(row.get("sampled_correct_rate", 0.0)) for row in rows) / max(len(rows), 1),
            "sampled_executable_rate": sum(float(row.get("sampled_executable_rate", 0.0)) for row in rows) / max(len(rows), 1),
            "invalid_rate": sum(float(row.get("invalid_rate", 0.0)) for row in rows) / max(len(rows), 1),
            "wrong_executable_rate": sum(float(row.get("wrong_executable_rate", 0.0)) for row in rows) / max(len(rows), 1),
            "unique_program_ratio": sum(float(row.get("unique_program_ratio", 0.0)) for row in rows) / max(len(rows), 1),
            "reasoning_marker_rate": sum(float(row.get("reasoning_marker_rate", 0.0)) for row in rows) / max(len(rows), 1),
            "answer_marker_rate": sum(float(row.get("answer_marker_rate", 0.0)) for row in rows) / max(len(rows), 1),
            "normalized_answer_marker_rate": sum(float(row.get("normalized_answer_marker_rate", 0.0)) for row in rows) / max(len(rows), 1),
            "multiple_program_rate": sum(float(row.get("multiple_program_rate", 0.0)) for row in rows) / max(len(rows), 1),
        }
    return summaries


def summarize_run(output_dir: Path) -> Dict[str, Any]:
    reward_rows = read_jsonl(output_dir / "step_reward_trace.jsonl") if (output_dir / "step_reward_trace.jsonl").exists() else []
    trainer_logs = load_trainer_logs(output_dir)
    steps = []
    for index, row in enumerate(reward_rows, start=1):
        merged = dict(row)
        merged["step"] = index
        log = trainer_logs.get(index, {})
        for key in ("loss", "grad_norm", "learning_rate", "kl"):
            merged[key] = log.get(key)
        steps.append(merged)
    early = output_dir / "early_stop_reason.json"
    summary = {
        "ran_steps": len(steps),
        "early_stop": json.loads(early.read_text(encoding="utf-8")) if early.exists() else None,
        "windows": summarize_windows(steps),
        "checkpoints": checkpoint_integrity(output_dir),
        "trainer_logs": trainer_logs,
    }
    (output_dir / "grpo40_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v34r23 ConvFinQA strict program-only GRPO40 checkpoint sweep.")
    parser.add_argument("--data_dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--base_path", default=DEFAULT_BASE)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(FIXED_GRPO40_CONFIG)
    assert_grpo40_config(config, args.adapter_path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists and is not empty: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_frontier_only_data(args.data_dir, args.output_dir)
    snapshot = {
        "fixed_config": config,
        "data": data,
        "base_path": args.base_path,
        "adapter_path": args.adapter_path,
        "disk_budget": disk_budget(Path(args.adapter_path), args.output_dir),
        "git": git_snapshot(),
        "sampler": {
            "bundle_ordered_sampling": False,
            "trl_repeat_sampler_required": True,
            "group_size": 8,
            "steps_per_generation": 1,
            "per_device_train_batch_size": 8,
        },
    }
    (args.output_dir / "grpo40_config.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return
    reward_log = args.output_dir / "step_reward_trace.jsonl"
    run_training(args, data["train_file"], data["valid_file"], reward_log)
    print(json.dumps(summarize_run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

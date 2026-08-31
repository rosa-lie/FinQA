#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.finqa_program_grpo import (
    ScriptArguments,
    evaluate_program_completion,
    extract_anchor,
    first_text,
    make_reward_funcs,
)
from scripts.v34r23_prompt_processor import (
    dataset_row_for_prepare,
    is_history_dependent,
    prepare_rows_like_grpo,
    prompt_target_leakage_flags,
    training_bucket,
)

DEFAULT_BASE = "/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program_merged"
DEFAULT_ADAPTER = "/root/autodl-tmp/outputs/financial_reasoning_rl/v34r22_sft2_retention_aware_rs_sft_smoke50/checkpoint-20"
FORBIDDEN_TARGET_FIELDS = [
    "reference_response",
    "response",
    "target",
    "sampled_correct_winner",
    "completion",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(rows: Sequence[Dict[str, Any]], bucket: str, max_records: int, seed: int) -> List[Dict[str, Any]]:
    if bucket == "frontier":
        selected = [row for row in rows if training_bucket(row) == "frontier"]
    elif bucket == "retention":
        selected = [row for row in rows if training_bucket(row) == "retention_variance"]
    elif bucket == "history_frontier":
        selected = [row for row in rows if training_bucket(row) == "frontier" and is_history_dependent(row)]
    elif bucket == "nonhistory_frontier":
        selected = [row for row in rows if training_bucket(row) == "frontier" and not is_history_dependent(row)]
    elif bucket == "all":
        selected = list(rows)
    else:
        raise ValueError(f"unsupported bucket: {bucket}")
    if bucket != "all":
        rng = random.Random(seed)
        selected = list(selected)
        rng.shuffle(selected)
    return selected[:max_records] if max_records > 0 else selected


def prompt_text_from_processed(tokenizer: Any, processed_row: Dict[str, Any]) -> str:
    from evaluation.evaluate_financial_benchmarks import safe_apply_chat_template

    return safe_apply_chat_template(tokenizer, processed_row["prompt"])


def audit_prompt_leakage(raw_row: Dict[str, Any], processed_prompt_text: str) -> Dict[str, Any]:
    return prompt_target_leakage_flags(raw_row, processed_prompt_text)


def classify_reward_group(rewards: Sequence[float], exact_matches: Sequence[bool]) -> Dict[str, Any]:
    values = [float(value) for value in rewards]
    if not values:
        return {"reward_mean": 0.0, "reward_std": 0.0, "zero_std": True, "all_correct": False, "all_wrong": False, "mixed_reward": False}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    all_correct = bool(exact_matches) and all(bool(x) for x in exact_matches)
    all_wrong = bool(exact_matches) and not any(bool(x) for x in exact_matches)
    zero_std = std <= 1e-8
    return {
        "reward_mean": mean,
        "reward_std": std,
        "zero_std": zero_std,
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "mixed_reward": not zero_std and not all_correct and not all_wrong,
    }


def summarize_probe_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records = len(rows)
    total_completions = sum(len(row.get("completions") or []) for row in rows)
    correct = sum(sum(1 for item in row.get("scores", []) if item.get("answer_correct")) for row in rows)
    executable = sum(sum(1 for item in row.get("scores", []) if item.get("executable")) for row in rows)
    unique_programs = sum(len(set(first_text(item.get("program")) for item in row.get("scores", []) if first_text(item.get("program")))) for row in rows)
    program_count = sum(sum(1 for item in row.get("scores", []) if first_text(item.get("program"))) for row in rows)
    bucket_counts = Counter(row.get("training_bucket") for row in rows)
    history_rows = [row for row in rows if row.get("history_dependent")]
    nonhistory_rows = [row for row in rows if not row.get("history_dependent")]

    def ratio(predicate) -> float:
        return sum(1 for row in rows if predicate(row)) / max(records, 1)

    return {
        "records": records,
        "total_completions": total_completions,
        "bucket_counts": dict(bucket_counts),
        "zero_std_ratio": ratio(lambda row: row.get("zero_std")),
        "all_correct_ratio": ratio(lambda row: row.get("all_correct")),
        "all_wrong_ratio": ratio(lambda row: row.get("all_wrong")),
        "mixed_reward_ratio": ratio(lambda row: row.get("mixed_reward")),
        "sampled_correct_rate": correct / max(total_completions, 1),
        "sampled_executable_rate": executable / max(total_completions, 1),
        "unique_program_ratio": unique_programs / max(program_count, 1),
        "history_records": len(history_rows),
        "nonhistory_records": len(nonhistory_rows),
    }


def acquisition_summary_for_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_scores = sum(int((row.get("metadata") or {}).get("v34r23_sampled_score_count") or 0) for row in rows)
    correct = sum(int((row.get("metadata") or {}).get("v34r23_sampled_correct_count") or 0) for row in rows)
    wrong_exec = sum(int((row.get("metadata") or {}).get("v34r23_wrong_executable_count") or 0) for row in rows)
    all_correct = sum(1 for row in rows if int((row.get("metadata") or {}).get("v34r23_sampled_correct_count") or 0) == int((row.get("metadata") or {}).get("v34r23_sampled_score_count") or 0))
    all_wrong = sum(1 for row in rows if int((row.get("metadata") or {}).get("v34r23_sampled_correct_count") or 0) == 0)
    return {
        "records": len(rows),
        "sampled_correct_rate": correct / max(total_scores, 1),
        "sampled_executable_rate": (correct + wrong_exec) / max(total_scores, 1),
        "all_correct_ratio": all_correct / max(len(rows), 1),
        "all_wrong_ratio": all_wrong / max(len(rows), 1),
        "zero_std_ratio_estimate": (all_correct + all_wrong) / max(len(rows), 1),
    }


def config_diff(acquisition: Dict[str, Any], online: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    keys = sorted(set(acquisition) | set(online))
    return {
        key: {"acquisition": acquisition.get(key), "online_probe": online.get(key), "match": acquisition.get(key) == online.get(key)}
        for key in keys
    }


def load_model_and_tokenizer(base_path: str, adapter_path: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch_dtype,
        trust_remote_code=False,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


@torch.inference_mode()
def generate_completions(model: Any, tokenizer: Any, prompt_text: str, args: argparse.Namespace, seed: int) -> List[str]:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_generations,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
    return [tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True).strip() for output in outputs]


def score_completions(completions: List[str], processed_row: Dict[str, Any], script_args: ScriptArguments) -> List[Dict[str, Any]]:
    reward_func = make_reward_funcs(script_args)[0]
    rewards = reward_func(
        completions,
        gold_answer=[processed_row["gold_answer"]] * len(completions),
        gold_program=[processed_row["gold_program"]] * len(completions),
        input_prompt_raw=[processed_row["input_prompt_raw"]] * len(completions),
        source_dataset=[processed_row["source_dataset"]] * len(completions),
        metadata=[processed_row["metadata"]] * len(completions),
        requires_history=[processed_row["requires_history"]] * len(completions),
    )
    scores = []
    for completion, reward in zip(completions, rewards):
        diag = evaluate_program_completion(
            script_args,
            completion,
            processed_row["gold_answer"],
            processed_row["gold_program"],
            processed_row["input_prompt_raw"],
            source_dataset=processed_row["source_dataset"],
            metadata=processed_row["metadata"],
            requires_history=processed_row["requires_history"],
        )
        scores.append(
            {
                "program": first_text(extract_anchor(completion, "Program:")),
                "parse_status": bool(diag.get("has_program")),
                "executable": bool(diag.get("executable")),
                "executed_answer": diag.get("executed_answer"),
                "answer_correct": bool(diag.get("exact_match")),
                "contract_valid": not bool(diag.get("schema_hard_gate_violation_rate") or diag.get("prompt_contract_violation_rate")),
                "reward": float(reward),
                "invalid": bool(diag.get("invalid")),
                "wrong_executable": bool(diag.get("wrong_executable")),
            }
        )
    return scores


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    rows = select_rows(read_jsonl(Path(args.data_file)), args.bucket, args.max_records, args.seed)
    if not rows:
        raise ValueError("no rows selected")
    script_args = ScriptArguments(reward_mode=args.reward_mode, reward_profile_expected="program_numeric")
    processed = prepare_rows_like_grpo(rows, script_args=script_args, is_main_process=True)
    model, tokenizer = load_model_and_tokenizer(args.base_path, args.adapter_path, args.dtype)
    output_rows = []
    leakage_counts = Counter()
    for index, (raw_row, processed_row) in enumerate(tqdm(list(zip(rows, processed)), desc="Online variance probe")):
        prompt_text = prompt_text_from_processed(tokenizer, processed_row)
        leakage = audit_prompt_leakage(raw_row, prompt_text)
        leakage_counts.update({key: int(value) for key, value in leakage.items()})
        completions = generate_completions(model, tokenizer, prompt_text, args, args.seed + index)
        scores = score_completions(completions, processed_row, script_args)
        rewards = [item["reward"] for item in scores]
        exact = [item["answer_correct"] for item in scores]
        group = classify_reward_group(rewards, exact)
        output_rows.append(
            {
                "record_id": raw_row.get("record_id"),
                "training_bucket": training_bucket(raw_row),
                "history_dependent": is_history_dependent(raw_row),
                "prompt_leakage": leakage,
                "completions": completions,
                "scores": scores,
                "reward_mean": group["reward_mean"],
                "reward_std": group["reward_std"],
                "all_correct": group["all_correct"],
                "all_wrong": group["all_wrong"],
                "zero_std": group["zero_std"],
                "mixed_reward": group["mixed_reward"],
                "unique_completion_count": len(set(completions)),
                "unique_program_count": len(set(item["program"] for item in scores if item.get("program"))),
                "acquisition_sampled_correct_count": (raw_row.get("metadata") or {}).get("v34r23_sampled_correct_count"),
                "acquisition_wrong_executable_count": (raw_row.get("metadata") or {}).get("v34r23_wrong_executable_count"),
                "acquisition_sampled_score_count": (raw_row.get("metadata") or {}).get("v34r23_sampled_score_count"),
            }
        )
    write_jsonl(Path(args.output_file), output_rows)
    online_summary = summarize_probe_rows(output_rows)
    summary = {
        "version": "v34r23_online_reward_variance_probe",
        "data_file": args.data_file,
        "bucket": args.bucket,
        "base_path": args.base_path,
        "adapter_path": args.adapter_path,
        "reward_mode": args.reward_mode,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_length": args.max_completion_length,
        "seed": args.seed,
        "online_summary": online_summary,
        "history_summary": summarize_probe_rows([row for row in output_rows if row.get("history_dependent")]),
        "nonhistory_summary": summarize_probe_rows([row for row in output_rows if not row.get("history_dependent")]),
        "acquisition_summary_for_selected_rows": acquisition_summary_for_rows(rows),
        "prompt_leakage_counts": dict(leakage_counts),
        "config_diff": config_diff(
            {
                "base_path": args.base_path,
                "adapter_path": args.adapter_path,
                "num_generations": 8,
                "temperature": 0.72,
                "top_p": 0.95,
                "max_completion_length": 300,
                "seed": 34023,
                "reward_or_score": "executor_score_example",
                "prompt_processor": "safe_apply_chat_template(raw_input_prompt)",
            },
            {
                "base_path": args.base_path,
                "adapter_path": args.adapter_path,
                "num_generations": args.num_generations,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_completion_length": args.max_completion_length,
                "seed": args.seed,
                "reward_or_score": args.reward_mode,
                "prompt_processor": "prepare_dataset + safe_apply_chat_template",
            },
        ),
    }
    Path(args.output_file).with_suffix(Path(args.output_file).suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe v34r23 online GRPO reward variance without parameter updates.")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--bucket", choices=["frontier", "retention", "history_frontier", "nonhistory_frontier", "all"], default="frontier")
    parser.add_argument("--max_records", type=int, default=32)
    parser.add_argument("--seed", type=int, default=34023)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=300)
    parser.add_argument("--base_path", default=DEFAULT_BASE)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--reward_mode", default="frontier_execution_calibration")
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_probe(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

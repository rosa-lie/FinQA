#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_learnable_hard_buckets import row_is_noisy
from scripts.v34r23_prompt_processor import prepare_rows_like_grpo
from scripts.v34r21_common import (
    count_forbidden_markers,
    distribution,
    first_text,
    load_record_ids,
    program_family,
    read_jsonl,
    record_id,
    source_dataset,
    write_jsonl,
)

REQUIRED_FIELDS = ["input_prompt_raw", "gold_answer", "gold_program", "reward_profile", "source_dataset", "record_id"]


def score_correct(score: Dict[str, Any]) -> bool:
    return float(score.get("executed_answer_accuracy") or score.get("answer_correct") or 0.0) > 0.0


def score_executable(score: Dict[str, Any]) -> bool:
    return float(score.get("program_execution_rate") or 0.0) > 0.0


def score_program(score: Dict[str, Any]) -> str:
    return first_text(score.get("executed_program") or score.get("program") or score.get("prediction_program"))


def prediction_program(prediction: str) -> str:
    text = first_text(prediction)
    if "Program:" not in text:
        return ""
    return text.split("Program:", 1)[1].strip()


def clean_rollout_prediction(prediction: str) -> bool:
    text = first_text(prediction)
    if count_forbidden_markers(text):
        return False
    if text.count("Program:") != 1:
        return False
    program = prediction_program(text)
    if not program:
        return False
    if "=" in program:
        return False
    return True


def scored_predictions(diagnostic: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any], str]]:
    predictions = diagnostic.get("sampled_predictions") or []
    return [
        (idx, score, first_text(predictions[idx]) if idx < len(predictions) else "")
        for idx, score in enumerate(sampled_scores(diagnostic))
    ]


def sampled_scores(diagnostic: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [score for score in diagnostic.get("sampled_scores") or [] if isinstance(score, dict)]


def classify_diagnostic(diagnostic: Dict[str, Any]) -> str:
    if first_text(diagnostic.get("bucket")) == "noisy" or first_text(diagnostic.get("noisy_reason")):
        return "noisy"
    triples = scored_predictions(diagnostic)
    scores = [score for _idx, score, _prediction in triples]
    clean_correct = [score for _idx, score, prediction in triples if score_correct(score) and clean_rollout_prediction(prediction)]
    correct = [score for score in scores if score_correct(score)]
    wrong_exec = [score for score in scores if score_executable(score) and not score_correct(score)]
    invalid = [score for score in scores if not score_executable(score)]
    greedy_correct = bool(diagnostic.get("greedy_correct"))
    if scores and len(invalid) / max(len(scores), 1) > 0.5:
        return "invalid_prone"
    if greedy_correct and clean_correct and wrong_exec:
        return "retention_variance"
    if greedy_correct and scores and len(correct) == len(scores):
        return "all_correct"
    if greedy_correct:
        return "easy"
    if clean_correct and wrong_exec:
        return "frontier"
    if not clean_correct and not correct:
        return "all_wrong"
    if correct and not wrong_exec:
        return "all_correct"
    return "noisy"


def read_source_rows(path: Path, use_grpo_prompt_processor: bool = False) -> Dict[Tuple[str, str], Dict[str, Any]]:
    input_rows = read_jsonl(path)
    if use_grpo_prompt_processor:
        processed_rows = prepare_rows_like_grpo(input_rows, is_main_process=True)
        rewritten = []
        for raw_row, processed_row in zip(input_rows, processed_rows):
            row = dict(raw_row)
            row["input_prompt_raw"] = first_text(processed_row.get("input_prompt_raw"))
            row["reward_profile"] = first_text(processed_row.get("reward_profile"))
            row["source_dataset"] = first_text(processed_row.get("source_dataset"))
            row["gold_answer"] = first_text(processed_row.get("gold_answer"))
            row["gold_program"] = first_text(processed_row.get("gold_program"))
            meta = dict(raw_row.get("metadata") or {})
            meta.update(processed_row.get("metadata") or {})
            row["metadata"] = meta
            rewritten.append(row)
        input_rows = rewritten
    rows = {}
    for row in input_rows:
        key = (source_dataset(row), record_id(row))
        if key[0] and key[1] and key not in rows:
            rows[key] = row
    return rows


def valid_train_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    missing = [field for field in REQUIRED_FIELDS if not first_text(row.get(field))]
    if missing:
        return False, "missing_" + ",".join(missing)
    if count_forbidden_markers(first_text(row.get("reference_response"))):
        return False, "forbidden_reference_marker"
    noisy, reason, _canonical = row_is_noisy(row, 1e-4, 1e-4)
    if noisy:
        return False, reason
    return True, ""


def prompt_target_leakage_reason(row: Dict[str, Any]) -> str:
    prompt = first_text(row.get("input_prompt_raw"))
    meta = row.get("metadata") or {}
    checks = [
        ("reference_response", first_text(row.get("reference_response"))),
        ("winner_prediction", first_text(meta.get("v34r23_winner_prediction"))),
        ("hard_negative_prediction", first_text(meta.get("v34r23_hard_negative_prediction"))),
    ]
    leaked = [name for name, value in checks if value and value in prompt]
    return "target_leakage_" + "+".join(leaked) if leaked else ""


def first_candidate(diagnostic: Dict[str, Any], scores: Sequence[Dict[str, Any]], require_clean: bool = False) -> Tuple[str, Optional[int]]:
    all_scores = sampled_scores(diagnostic)
    predictions = diagnostic.get("sampled_predictions") or []
    if not scores or not all_scores:
        return "", None
    for target in scores:
        for idx, score in enumerate(all_scores):
            prediction = first_text(predictions[idx]) if idx < len(predictions) else ""
            if score is target and (not require_clean or clean_rollout_prediction(prediction)):
                return prediction, idx
    return "", None


def annotate(row: Dict[str, Any], diagnostic: Dict[str, Any], label: str) -> Dict[str, Any]:
    scores = sampled_scores(diagnostic)
    correct = [score for score in scores if score_correct(score)]
    wrong_exec = [score for score in scores if score_executable(score) and not score_correct(score)]
    programs = [score_program(score) for score in scores if score_program(score)]
    out = dict(row)
    clean_correct = [score for idx, score, prediction in scored_predictions(diagnostic) if score_correct(score) and clean_rollout_prediction(prediction)]
    winner_prediction, winner_index = first_candidate(diagnostic, clean_correct, require_clean=True)
    hard_negative_prediction, hard_negative_index = first_candidate(diagnostic, wrong_exec)
    if winner_prediction:
        out["reference_response"] = winner_prediction
    meta = dict(out.get("metadata") or {})
    meta.update(
        {
            "v34r23_bucket": label,
            "v34r23_winner_source": "current_rollout" if correct else "none",
            "v34r23_winner_sample_index": winner_index,
            "v34r23_hard_negative_type": "wrong_executable" if wrong_exec else "none",
            "v34r23_hard_negative_sample_index": hard_negative_index,
            "v34r23_sampled_score_count": len(scores),
            "v34r23_sampled_correct_count": len(correct),
            "v34r23_wrong_executable_count": len(wrong_exec),
            "v34r23_unique_candidate_program_count": len(set(programs)),
            "v34r23_program_family": program_family(row),
            "v34r23_manifest_phase": first_text(diagnostic.get("v34r23_manifest_phase")),
            "v34r23_winner_prediction": winner_prediction,
            "v34r23_hard_negative_prediction": hard_negative_prediction,
        }
    )
    out["metadata"] = meta
    return out


def split_valid(rows: Sequence[Dict[str, Any]], valid_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    if valid_ratio <= 0.0 or len(rows) <= 1:
        return rows, []
    valid_count = max(1, int(round(len(rows) * valid_ratio)))
    valid_count = min(valid_count, len(rows) - 1)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    valid_indices = set(indices[:valid_count])
    return [row for idx, row in enumerate(rows) if idx not in valid_indices], [row for idx, row in enumerate(rows) if idx in valid_indices]


def select_retention(rows: Sequence[Dict[str, Any]], frontier_count: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    if frontier_count <= 0 or not rows or float(args.frontier_ratio) <= 0:
        return []
    target = int(round(frontier_count * float(args.retention_ratio) / float(args.frontier_ratio)))
    target = min(target, len(rows))
    if target <= 0 and float(args.retention_ratio) > 0:
        target = min(1, len(rows))
    rng = random.Random(int(args.seed) + 17)
    ordered = list(rows)
    rng.shuffle(ordered)
    return ordered[:target]


def diagnostic_phase(path: str) -> str:
    lowered = first_text(path).lower()
    if "extension" in lowered:
        return "extension"
    if "pilot" in lowered or "current_policy_frontier_acquisition" in lowered:
        return "pilot"
    return "unknown"


def read_diagnostics_with_phase(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        phase = diagnostic_phase(path)
        for row in read_jsonl(Path(path)):
            out = dict(row)
            out["v34r23_manifest_phase"] = first_text(out.get("v34r23_manifest_phase") or phase)
            rows.append(out)
    return rows


def build_frontier_data(args: argparse.Namespace) -> Dict[str, Any]:
    diagnostic_files = args.diagnostics_file if isinstance(args.diagnostics_file, list) else [args.diagnostics_file]
    diagnostics = read_diagnostics_with_phase(diagnostic_files)
    source_rows = read_source_rows(Path(args.source_train_file), bool(getattr(args, "use_grpo_prompt_processor", False)))
    excluded: set[str] = set()
    for path in args.exclude_record_ids or []:
        if path:
            excluded.update(load_record_ids(Path(path)))

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "frontier": [],
        "retention_variance": [],
        "easy": [],
        "all_wrong": [],
        "all_correct": [],
        "invalid_prone": [],
        "noisy": [],
    }
    counters: Counter[str] = Counter()
    for diagnostic in diagnostics:
        rid = first_text(diagnostic.get("record_id"))
        source = source_dataset(diagnostic)
        if rid in excluded:
            counters["excluded_record_ids"] += 1
            continue
        key = (source, rid)
        row = source_rows.get(key)
        if row is None:
            counters["missing_source_row"] += 1
            continue
        ok, reason = valid_train_row(row)
        if not ok:
            counters[f"invalid_source_{reason}"] += 1
            continue
        label = classify_diagnostic(diagnostic)
        annotated = annotate(row, diagnostic, label)
        leakage_reason = prompt_target_leakage_reason(annotated)
        if leakage_reason:
            counters[f"invalid_source_{leakage_reason}"] += 1
            continue
        buckets.setdefault(label, []).append(annotated)

    frontier = buckets["frontier"][: int(args.per_record_cap) * len({record_id(row) for row in buckets["frontier"]})]
    retention = select_retention(buckets["retention_variance"], len(frontier), args)
    train, valid = split_valid(frontier + retention, float(args.valid_ratio), int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    write_jsonl(output_dir / "retention.jsonl", retention)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "valid.jsonl", valid)

    def unique_count(rows: Sequence[Dict[str, Any]]) -> int:
        return len({record_id(row) for row in rows})

    sample_total = sum(int((row.get("metadata") or {}).get("v34r23_sampled_score_count") or 0) for rows in buckets.values() for row in rows)
    sample_correct = sum(int((row.get("metadata") or {}).get("v34r23_sampled_correct_count") or 0) for rows in buckets.values() for row in rows)
    wrong_exec = sum(int((row.get("metadata") or {}).get("v34r23_wrong_executable_count") or 0) for rows in buckets.values() for row in rows)
    zero_std = sum(1 for d in diagnostics if classify_diagnostic(d) in {"all_wrong", "all_correct"})
    history_frontier = sum(1 for row in frontier if (row.get("metadata") or {}).get("v34r23_requires_history"))

    summary = {
        "version": "v34r23_frontier_grpo_data",
        "task": args.task,
        "diagnostics_file": diagnostic_files,
        "source_train_file": args.source_train_file,
        "use_grpo_prompt_processor": bool(getattr(args, "use_grpo_prompt_processor", False)),
        "diagnostic_rows": len(diagnostics),
        "diagnostic_unique_records": len({first_text(row.get("record_id")) for row in diagnostics}),
        "pilot_rows": sum(1 for row in diagnostics if first_text(row.get("v34r23_manifest_phase")) == "pilot"),
        "extension_rows": sum(1 for row in diagnostics if first_text(row.get("v34r23_manifest_phase")) == "extension"),
        "frontier_unique_records": unique_count(frontier),
        "retention_unique_records": unique_count(retention),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "bucket_counts": {name: unique_count(rows) for name, rows in buckets.items()},
        "greedy_wrong_sampled_correct_count": unique_count(buckets["frontier"]),
        "greedy_wrong_wrong_executable_count": unique_count(buckets["frontier"] + buckets["all_wrong"]),
        "group_all_wrong_ratio": len(buckets["all_wrong"]) / max(len(diagnostics), 1),
        "group_all_correct_ratio": len(buckets["all_correct"]) / max(len(diagnostics), 1),
        "zero_std_group_ratio_estimate": zero_std / max(len(diagnostics), 1),
        "sampled_correct_rate": sample_correct / max(sample_total, 1),
        "sampled_executable_rate": (sample_correct + wrong_exec) / max(sample_total, 1),
        "history_dependent_frontier_ratio": history_frontier / max(len(frontier), 1),
        "non_history_frontier_count": sum(1 for row in frontier if not (row.get("metadata") or {}).get("v34r23_requires_history")),
        "pilot_frontier_contribution": sum(1 for row in frontier if first_text((row.get("metadata") or {}).get("v34r23_manifest_phase")) == "pilot"),
        "extension_frontier_contribution": sum(1 for row in frontier if first_text((row.get("metadata") or {}).get("v34r23_manifest_phase")) == "extension"),
        "pilot_extension_duplicate_record_count": len(diagnostics) - len({first_text(row.get("record_id")) for row in diagnostics}),
        "distribution": distribution(frontier + retention),
        "counters": dict(counters),
        "recommend_stop_no_learnable_frontier": unique_count(frontier) < int(args.min_frontier_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v34r23 strict frontier GRPO data.")
    parser.add_argument("--task", choices=["finqa", "convfinqa"], required=True)
    parser.add_argument("--diagnostics_file", action="append", required=True)
    parser.add_argument("--source_train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclude_record_ids", action="append", default=[])
    parser.add_argument("--frontier_ratio", type=float, default=0.85)
    parser.add_argument("--retention_ratio", type=float, default=0.15)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--per_record_cap", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_frontier_records", type=int, default=64)
    parser.add_argument("--use_grpo_prompt_processor", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_frontier_data(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

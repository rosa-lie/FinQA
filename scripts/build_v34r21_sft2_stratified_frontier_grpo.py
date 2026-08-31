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
from scripts.v34r21_common import (
    count_forbidden_markers,
    distribution,
    first_text,
    load_record_ids,
    program_family,
    program_text,
    read_jsonl,
    record_id,
    source_dataset,
    write_jsonl,
)

REQUIRED_TRAIN_FIELDS = [
    "input_prompt_raw",
    "gold_answer",
    "gold_program",
    "reward_profile",
    "source_dataset",
    "record_id",
]


def row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (source_dataset(row), record_id(row))


def score_correct(score: Dict[str, Any]) -> bool:
    return float(score.get("executed_answer_accuracy") or 0.0) > 0.0


def score_executable(score: Dict[str, Any]) -> bool:
    return float(score.get("program_execution_rate") or 0.0) > 0.0


def diagnostic_greedy_correct(diagnostic: Dict[str, Any]) -> bool:
    if "greedy_correct" in diagnostic:
        return bool(diagnostic.get("greedy_correct"))
    score = diagnostic.get("greedy_score")
    return isinstance(score, dict) and score_correct(score)


def sampled_scores(diagnostic: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [score for score in diagnostic.get("sampled_scores") or [] if isinstance(score, dict)]


def sampled_programs(diagnostic: Dict[str, Any]) -> List[str]:
    values = diagnostic.get("sampled_programs")
    if isinstance(values, list):
        return [first_text(value) for value in values]
    responses = diagnostic.get("sampled_responses")
    if isinstance(responses, list):
        return [first_text(value) for value in responses]
    return []


def validate_train_row(row: Dict[str, Any], *, abs_tol: float, rel_tol: float) -> Tuple[bool, str]:
    missing = [field for field in REQUIRED_TRAIN_FIELDS if not first_text(row.get(field))]
    if missing:
        return False, "missing_" + ",".join(missing)
    noisy, noisy_reason, _canonical = row_is_noisy(row, abs_tol, rel_tol)
    if noisy:
        return False, noisy_reason
    return True, ""


def annotate_row(row: Dict[str, Any], diagnostic: Optional[Dict[str, Any]], bucket: str, source: str) -> Dict[str, Any]:
    out = dict(row)
    meta = dict(out.get("metadata") or {})
    meta["v34r21_bucket"] = bucket
    meta["v34r21_winner_source"] = source
    meta["v34r21_program_family"] = program_family(row)
    if diagnostic:
        scores = sampled_scores(diagnostic)
        correct = [score for score in scores if score_correct(score)]
        wrong_exec = [score for score in scores if score_executable(score) and not score_correct(score)]
        programs = sampled_programs(diagnostic)
        meta["v34r21_unique_candidate_program_count"] = len(set(programs))
        meta["v34r21_sampled_score_count"] = len(scores)
        meta["v34r21_sampled_correct_count"] = len(correct)
        meta["v34r21_wrong_executable_count"] = len(wrong_exec)
        meta["v34r21_expected_zero_std_group"] = float(len(correct) in {0, len(scores)}) if scores else 1.0
        meta["v34r21_hard_negative_type"] = "wrong_executable" if wrong_exec else "none"
    out["metadata"] = meta
    return out


def split_valid(rows: Sequence[Dict[str, Any]], valid_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    if valid_ratio <= 0 or len(rows) <= 1:
        return rows, []
    valid_count = max(1, int(round(len(rows) * valid_ratio)))
    valid_count = min(valid_count, len(rows) - 1)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    valid_indices = set(indices[:valid_count])
    return [row for idx, row in enumerate(rows) if idx not in valid_indices], [row for idx, row in enumerate(rows) if idx in valid_indices]


def select_retention(easy_rows: Sequence[Dict[str, Any]], frontier_count: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    if frontier_count <= 0 or not easy_rows:
        return []
    target = int(round(frontier_count * float(args.retention_ratio)))
    target = max(int(args.retention_min), target)
    target = min(int(args.retention_max), target, len(easy_rows))
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in easy_rows:
        by_family.setdefault(program_family(row), []).append(row)
    rng = random.Random(int(args.seed) + 17)
    for rows in by_family.values():
        rows.sort(key=record_id)
        rng.shuffle(rows)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    priority = ["direct_lookup", "sum", "percentage_change", "share_of_total", "ratio", "multi_step_divide"]
    for family in priority:
        if len(selected) >= target:
            break
        for row in by_family.get(family, []):
            rid = record_id(row)
            if rid not in selected_ids:
                selected.append(row)
                selected_ids.add(rid)
                break
    leftovers = [row for rows in by_family.values() for row in rows if record_id(row) not in selected_ids]
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max(target - len(selected), 0)])
    return selected[:target]


def build_data(args: argparse.Namespace) -> Dict[str, Any]:
    source_rows = [row for row in read_jsonl(Path(args.source_train_file)) if source_dataset(row) == "finqa"]
    diagnostics = [row for row in read_jsonl(Path(args.diagnostics_file)) if source_dataset(row) == "finqa"]
    manifest_ids = load_record_ids(Path(args.manifest_file)) if args.manifest_file else set()
    forbidden_ids: set[str] = set()
    for path in args.exclude_record_ids:
        if path:
            forbidden_ids.update(load_record_ids(Path(path)))
    rows_by_key = {row_key(row): row for row in source_rows if not manifest_ids or record_id(row) in manifest_ids}
    diagnostics_by_key = {row_key(row): row for row in diagnostics if not manifest_ids or record_id(row) in manifest_ids}

    frontier_rows: List[Dict[str, Any]] = []
    easy_rows: List[Dict[str, Any]] = []
    counters = Counter()
    noisy_reasons = Counter()
    forbidden_marker_counts = Counter()
    unique_program_ratios: List[float] = []
    zero_std_estimates: List[float] = []

    for key, diagnostic in sorted(diagnostics_by_key.items()):
        row = rows_by_key.get(key)
        if row is None:
            counters["missing_source_rows"] += 1
            continue
        valid, reason = validate_train_row(row, abs_tol=float(args.numeric_abs_tol), rel_tol=float(args.numeric_rel_tol))
        if not valid:
            noisy_reasons[reason] += 1
            continue
        rid = record_id(row)
        if rid in forbidden_ids:
            counters["dev_test_history_overlap_excluded"] += 1
            continue
        scores = sampled_scores(diagnostic)
        correct_scores = [score for score in scores if score_correct(score)]
        wrong_exec_scores = [score for score in scores if score_executable(score) and not score_correct(score)]
        programs = sampled_programs(diagnostic)
        if scores:
            zero_std_estimates.append(1.0 if len(correct_scores) in {0, len(scores)} else 0.0)
        if programs:
            unique_program_ratios.append(len(set(programs)) / max(len(programs), 1))
        for text in [first_text(diagnostic.get("greedy_response"))] + [first_text(item) for item in diagnostic.get("sampled_responses") or []]:
            forbidden_marker_counts.update(count_forbidden_markers(text))

        greedy_correct = diagnostic_greedy_correct(diagnostic)
        if greedy_correct:
            counters["easy"] += 1
            easy_rows.append(annotate_row(row, diagnostic, "retention_easy", "sft2_greedy_correct"))
            continue
        if correct_scores:
            counters["greedy_wrong_sampled_correct"] += 1
        if wrong_exec_scores:
            counters["greedy_wrong_wrong_executable"] += 1
        if not correct_scores:
            counters["group_all_wrong"] += 1
        if scores and len(correct_scores) == len(scores):
            counters["group_all_correct"] += 1
        if correct_scores and wrong_exec_scores:
            counters["learnable_hard"] += 1
            frontier_rows.append(annotate_row(row, diagnostic, "frontier", "current_sft2_rollout"))
        elif scores and not correct_scores:
            counters["hard"] += 1
        elif not scores:
            counters["invalid_prone"] += 1
        else:
            counters["noisy"] += 1

    retention_rows = select_retention(easy_rows, len(frontier_rows), args)
    mixed_rows = list(frontier_rows) + list(retention_rows)
    random.Random(int(args.seed)).shuffle(mixed_rows)
    train_rows, valid_rows = split_valid(mixed_rows, float(args.valid_ratio), int(args.seed) + 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "frontier.jsonl", frontier_rows)
    write_jsonl(output_dir / "retention.jsonl", retention_rows)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)

    source_ids = {record_id(row) for row in source_rows}
    summary = {
        "version": "v34r21_sft2_stratified_frontier_finqa_grpo",
        "diagnostics_file": args.diagnostics_file,
        "source_train_file": args.source_train_file,
        "manifest_file": args.manifest_file,
        "seed": int(args.seed),
        "retention_ratio": float(args.retention_ratio),
        "valid_ratio": float(args.valid_ratio),
        "source_finqa_rows": len(source_rows),
        "source_unique_records": len(source_ids),
        "diagnostic_rows": len(diagnostics),
        "diagnostic_unique_records": len(diagnostics_by_key),
        "total_rows": len(train_rows) + len(valid_rows),
        "unique_records": len({record_id(row) for row in train_rows + valid_rows}),
        "frontier_unique_records": len({record_id(row) for row in frontier_rows}),
        "retention_unique_records": len({record_id(row) for row in retention_rows}),
        "frontier_retention_ratio": len(frontier_rows) / max(len(retention_rows), 1),
        "per_record_repetition": 1.0,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "frontier_rows": len(frontier_rows),
        "retention_rows": len(retention_rows),
        "bucket_distribution": dict(counters),
        "noisy_filtered_rows": dict(noisy_reasons),
        "question_type_distribution": distribution(train_rows + valid_rows)["question_type"],
        "answer_scale_distribution": distribution(train_rows + valid_rows)["answer_scale"],
        "program_family_distribution": distribution(train_rows + valid_rows)["program_family"],
        "winner_source_distribution": dict(Counter((row.get("metadata") or {}).get("v34r21_winner_source") for row in frontier_rows + retention_rows)),
        "hard_negative_type_distribution": dict(Counter((row.get("metadata") or {}).get("v34r21_hard_negative_type") for row in frontier_rows)),
        "test_dev_overlap": counters["dev_test_history_overlap_excluded"],
        "forbidden_marker_count": dict(forbidden_marker_counts),
        "expected_zero_std_group_ratio": sum(zero_std_estimates) / len(zero_std_estimates) if zero_std_estimates else 0.0,
        "unique_candidate_program_ratio": sum(unique_program_ratios) / len(unique_program_ratios) if unique_program_ratios else 0.0,
        "recommend_stop_no_learnable_frontier": len(frontier_rows) < int(args.min_frontier_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v34r21 frontier/retention GRPO data.")
    parser.add_argument("--diagnostics_file", required=True)
    parser.add_argument("--source_train_file", required=True)
    parser.add_argument("--manifest_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclude_record_ids", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retention_ratio", type=float, default=0.2)
    parser.add_argument("--retention_min", type=int, default=8)
    parser.add_argument("--retention_max", type=int, default=128)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--min_frontier_records", type=int, default=32)
    parser.add_argument("--numeric_abs_tol", type=float, default=1e-4)
    parser.add_argument("--numeric_rel_tol", type=float, default=1e-4)
    args = parser.parse_args()
    if not 0 <= args.retention_ratio < 1:
        raise ValueError("--retention_ratio must be in [0, 1)")
    if not 0 <= args.valid_ratio < 1:
        raise ValueError("--valid_ratio must be in [0, 1)")
    return args


def main() -> None:
    print(json.dumps(build_data(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

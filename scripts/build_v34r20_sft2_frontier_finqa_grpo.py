#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_learnable_hard_buckets import first_text, row_is_noisy


REQUIRED_TRAIN_FIELDS = [
    "input_prompt_raw",
    "gold_answer",
    "gold_program",
    "reward_profile",
    "source_dataset",
    "record_id",
]

RETENTION_PRIORITY = [
    "direct_lookup",
    "sum",
    "growth_rate",
    "share_of_total",
    "multi_step_divide",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(source_dataset: Any, record_id: Any) -> Tuple[str, str]:
    return (first_text(source_dataset).lower(), first_text(record_id))


def is_finqa(item: Dict[str, Any]) -> bool:
    return first_text(item.get("source_dataset")).lower() == "finqa"


def score_correct(score: Dict[str, Any]) -> bool:
    return float(score.get("executed_answer_accuracy") or 0.0) > 0.0


def score_executable(score: Dict[str, Any]) -> bool:
    return float(score.get("program_execution_rate") or 0.0) > 0.0


def diagnostic_greedy_correct(diagnostic: Dict[str, Any]) -> bool:
    if "greedy_correct" in diagnostic:
        return bool(diagnostic.get("greedy_correct"))
    greedy_score = diagnostic.get("greedy_score")
    return isinstance(greedy_score, dict) and score_correct(greedy_score)


def diagnostic_has_sample_correct(diagnostic: Dict[str, Any]) -> bool:
    sampled_scores = diagnostic.get("sampled_scores") or []
    return any(isinstance(score, dict) and score_correct(score) for score in sampled_scores)


def diagnostic_has_wrong_executable_sample(diagnostic: Dict[str, Any]) -> bool:
    sampled_scores = diagnostic.get("sampled_scores") or []
    return any(
        isinstance(score, dict) and score_executable(score) and not score_correct(score)
        for score in sampled_scores
    )


def load_allowlist_record_ids(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    record_ids: set[str] = set()
    for row in read_jsonl(path):
        record_id = first_text(row.get("record_id"))
        if record_id:
            record_ids.add(record_id)
    return record_ids


def validate_train_row(row: Dict[str, Any], *, abs_tol: float, rel_tol: float) -> Tuple[bool, str]:
    missing = [field for field in REQUIRED_TRAIN_FIELDS if not first_text(row.get(field))]
    if missing:
        return False, "missing_" + ",".join(missing)
    noisy, noisy_reason, _canonical = row_is_noisy(row, abs_tol, rel_tol)
    if noisy:
        return False, noisy_reason
    return True, ""


def retention_bucket(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    question_type = first_text(metadata.get("question_type")).lower().replace("-", "_").replace(" ", "_")
    program = first_text(row.get("gold_program")).lower()
    ops = [first_text(item).lower() for item in (metadata.get("program_ops") or [])]
    if bool(metadata.get("direct_lookup")) or program.replace(".", "", 1).lstrip("-").isdigit():
        return "direct_lookup"
    if "sum" in ops or program.startswith("sum(") or program.startswith("add("):
        return "sum"
    if "growth" in question_type or ("subtract(" in program and "divide(" in program):
        return "growth_rate"
    if "share" in question_type or "percentage" in question_type:
        return "share_of_total"
    if "divide" in ops or program.startswith("divide("):
        return "multi_step_divide"
    return "other"


def select_retention_rows(
    easy_rows: Sequence[Dict[str, Any]],
    *,
    frontier_count: int,
    seed: int,
    retention_ratio: float,
    retention_min: int,
    retention_max: int,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    if frontier_count <= 0 or not easy_rows or retention_ratio <= 0:
        return [], 0, {}
    target = int(math.ceil(frontier_count * retention_ratio))
    if easy_rows:
        target = max(retention_min, target)
    target = min(retention_max, target, len(easy_rows))
    if target <= 0:
        return [], target, {}

    rows_by_bucket: Dict[str, List[Dict[str, Any]]] = {bucket: [] for bucket in RETENTION_PRIORITY + ["other"]}
    for row in easy_rows:
        rows_by_bucket.setdefault(retention_bucket(row), []).append(row)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    selected_keys: set[Tuple[str, str]] = set()

    for bucket in RETENTION_PRIORITY:
        if len(selected) >= target:
            break
        bucket_rows = rows_by_bucket.get(bucket) or []
        if not bucket_rows:
            continue
        row = bucket_rows.pop(0)
        key = row_key(row.get("source_dataset"), row.get("record_id"))
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)

    remaining = [
        row
        for bucket in RETENTION_PRIORITY + ["other"]
        for row in rows_by_bucket.get(bucket, [])
        if row_key(row.get("source_dataset"), row.get("record_id")) not in selected_keys
    ]
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= target:
            break
        selected.append(row)
        selected_keys.add(row_key(row.get("source_dataset"), row.get("record_id")))

    bucket_counts = Counter(retention_bucket(row) for row in selected)
    return selected, target, dict(bucket_counts)


def split_valid(rows: Sequence[Dict[str, Any]], *, valid_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows = list(rows)
    if valid_ratio <= 0 or not train_rows:
        return train_rows, []
    valid_count = int(round(len(train_rows) * valid_ratio))
    if len(train_rows) > 1 and valid_count == 0:
        valid_count = 1
    valid_count = min(valid_count, max(len(train_rows) - 1, 0))
    rng = random.Random(seed)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)
    valid_indices = set(indices[:valid_count])
    valid_rows = [row for idx, row in enumerate(train_rows) if idx in valid_indices]
    kept_train_rows = [row for idx, row in enumerate(train_rows) if idx not in valid_indices]
    return kept_train_rows, valid_rows


def build_v34r20_sft2_frontier_finqa_grpo_data(
    *,
    diagnostics_file: Path,
    source_train_file: Path,
    allowlist_file: Optional[Path],
    output_dir: Path,
    seed: int = 42,
    retention_ratio: float = 0.2,
    retention_min: int = 8,
    retention_max: int = 64,
    valid_ratio: float = 0.1,
    numeric_abs_tol: float = 1e-4,
    numeric_rel_tol: float = 1e-4,
) -> Dict[str, Any]:
    source_rows = [row for row in read_jsonl(source_train_file) if is_finqa(row)]
    diagnostics = [item for item in read_jsonl(diagnostics_file) if is_finqa(item)]
    allowlist_record_ids = load_allowlist_record_ids(allowlist_file)

    rows_by_key = {row_key(row.get("source_dataset"), row.get("record_id")): row for row in source_rows}
    diagnostics_by_key = {
        row_key(item.get("source_dataset"), item.get("record_id")): item
        for item in diagnostics
    }

    frontier_rows: List[Dict[str, Any]] = []
    easy_rows: List[Dict[str, Any]] = []
    noisy_reasons = Counter()
    excluded_allowlist = 0
    missing_source_rows = 0
    candidate_counts = Counter()

    for key, diagnostic in diagnostics_by_key.items():
        row = rows_by_key.get(key)
        if row is None:
            missing_source_rows += 1
            continue
        record_id = first_text(row.get("record_id"))
        valid, noisy_reason = validate_train_row(row, abs_tol=numeric_abs_tol, rel_tol=numeric_rel_tol)
        if not valid:
            noisy_reasons[noisy_reason] += 1
            continue

        greedy_correct = diagnostic_greedy_correct(diagnostic)
        sample_correct = diagnostic_has_sample_correct(diagnostic)
        wrong_executable = diagnostic_has_wrong_executable_sample(diagnostic)

        if greedy_correct:
            if record_id not in allowlist_record_ids:
                easy_rows.append(row)
            continue

        if sample_correct:
            candidate_counts["greedy_wrong_sample_correct"] += 1
        if wrong_executable:
            candidate_counts["greedy_wrong_wrong_executable"] += 1
        if sample_correct and wrong_executable:
            if record_id in allowlist_record_ids:
                excluded_allowlist += 1
            else:
                candidate_counts["greedy_wrong_sample_correct_wrong_executable"] += 1
                frontier_rows.append(row)

    retention_rows, retention_target, retention_bucket_counts = select_retention_rows(
        easy_rows,
        frontier_count=len(frontier_rows),
        seed=seed,
        retention_ratio=retention_ratio,
        retention_min=retention_min,
        retention_max=retention_max,
    )

    mixed_rows = list(frontier_rows) + list(retention_rows)
    random.Random(seed).shuffle(mixed_rows)
    train_rows, valid_rows = split_valid(mixed_rows, valid_ratio=valid_ratio, seed=seed + 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "frontier.jsonl", frontier_rows)
    write_jsonl(output_dir / "retention.jsonl", retention_rows)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)

    summary = {
        "version": "v34r20_sft2_frontier_finqa_step_sweep",
        "diagnostics_file": str(diagnostics_file),
        "source_train_file": str(source_train_file),
        "allowlist_file": str(allowlist_file) if allowlist_file is not None else "",
        "seed": seed,
        "retention_ratio": retention_ratio,
        "retention_min": retention_min,
        "retention_max": retention_max,
        "retention_target_rows": retention_target,
        "valid_ratio": valid_ratio,
        "source_finqa_rows": len(source_rows),
        "diagnostic_finqa_rows": len(diagnostics),
        "frontier_rows": len(frontier_rows),
        "retention_rows": len(retention_rows),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "allowlist_record_ids": len(allowlist_record_ids),
        "allowlist_excluded_frontier_candidates": excluded_allowlist,
        "missing_source_rows": missing_source_rows,
        "noisy_filtered_rows": dict(noisy_reasons),
        "frontier_diagnostics": dict(candidate_counts),
        "retention_bucket_counts": retention_bucket_counts,
        "recommend_pass16": len(frontier_rows) < 64,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v34r20 SFT2 frontier FinQA GRPO data from learnable-hard diagnostics.")
    parser.add_argument("--diagnostics_file", required=True)
    parser.add_argument("--source_train_file", required=True)
    parser.add_argument("--allowlist_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retention_ratio", type=float, default=0.2)
    parser.add_argument("--retention_min", type=int, default=8)
    parser.add_argument("--retention_max", type=int, default=64)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--numeric_abs_tol", type=float, default=1e-4)
    parser.add_argument("--numeric_rel_tol", type=float, default=1e-4)
    args = parser.parse_args()
    if not 0 <= args.retention_ratio < 1:
        raise ValueError("--retention_ratio must be in [0, 1)")
    if args.retention_min < 0 or args.retention_max < 0:
        raise ValueError("--retention_min/--retention_max must be non-negative")
    if args.retention_max and args.retention_min > args.retention_max:
        raise ValueError("--retention_min cannot exceed --retention_max")
    if not 0 <= args.valid_ratio < 1:
        raise ValueError("--valid_ratio must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    summary = build_v34r20_sft2_frontier_finqa_grpo_data(
        diagnostics_file=Path(args.diagnostics_file),
        source_train_file=Path(args.source_train_file),
        allowlist_file=Path(args.allowlist_file) if args.allowlist_file else None,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        retention_ratio=args.retention_ratio,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        valid_ratio=args.valid_ratio,
        numeric_abs_tol=args.numeric_abs_tol,
        numeric_rel_tol=args.numeric_rel_tol,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

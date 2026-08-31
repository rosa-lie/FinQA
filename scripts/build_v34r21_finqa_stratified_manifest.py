#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v34r21_common import (
    TARGET_FAMILIES,
    answer_scale,
    distribution,
    load_json_or_jsonl,
    load_record_ids,
    program_family,
    record_id,
    source_dataset,
    strat_key,
    write_jsonl,
)


def select_stratified(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_key: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[strat_key(row)].append(row)
    for bucket_rows in by_key.values():
        bucket_rows.sort(key=record_id)
        rng.shuffle(bucket_rows)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    for family in TARGET_FAMILIES:
        candidates = [row for row in rows if program_family(row) == family and record_id(row) not in selected_ids]
        candidates.sort(key=record_id)
        if candidates:
            row = rng.choice(candidates)
            selected.append(row)
            selected_ids.add(record_id(row))

    remaining_slots = max(sample_size - len(selected), 0)
    total_rows = len(rows)
    allocations: Dict[tuple[str, str, str], int] = {}
    remainders: List[tuple[float, tuple[str, str, str]]] = []
    for key, bucket_rows in by_key.items():
        already = sum(1 for row in bucket_rows if record_id(row) in selected_ids)
        ideal = remaining_slots * (len(bucket_rows) / max(total_rows, 1))
        count = min(max(int(ideal), 0), max(len(bucket_rows) - already, 0))
        allocations[key] = count
        remainders.append((ideal - int(ideal), key))
    assigned = sum(allocations.values())
    for _, key in sorted(remainders, reverse=True):
        if assigned >= remaining_slots:
            break
        available = len([row for row in by_key[key] if record_id(row) not in selected_ids])
        if allocations[key] < available:
            allocations[key] += 1
            assigned += 1

    for key in sorted(by_key):
        count = allocations.get(key, 0)
        if count <= 0:
            continue
        for row in by_key[key]:
            if count <= 0:
                break
            rid = record_id(row)
            if rid in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(rid)
            count -= 1

    if len(selected) < sample_size:
        leftovers = [row for row in rows if record_id(row) not in selected_ids]
        rng.shuffle(leftovers)
        for row in leftovers[: sample_size - len(selected)]:
            selected.append(row)
            selected_ids.add(record_id(row))

    selected.sort(key=record_id)
    return selected[:sample_size]


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    rows = [row for row in load_json_or_jsonl(Path(args.source_train_file)) if source_dataset(row) == "finqa"]
    rows_by_id = {record_id(row): row for row in rows if record_id(row)}
    rows = list(rows_by_id.values())
    excluded = set()
    for path in [args.exclude_allowlist, args.exclude_dev_file, args.exclude_test_file]:
        if path:
            excluded.update(load_record_ids(Path(path)))
    eligible = [row for row in rows if record_id(row) not in excluded]
    sample_size = min(int(args.sample_size), len(eligible))
    selected = select_stratified(eligible, sample_size, int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    selected_ids = {record_id(row) for row in selected}
    for idx, row in enumerate(selected):
        manifest_rows.append(
            {
                "manifest_index": idx,
                "record_id": record_id(row),
                "source_dataset": "finqa",
                "question_type": strat_key(row)[0],
                "answer_scale": answer_scale(row),
                "program_family": program_family(row),
                "stratification_key": "|".join(strat_key(row)),
            }
        )
    write_jsonl(output_dir / "acquisition_manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "acquisition_input.jsonl", selected)

    summary = {
        "version": "v34r21_sft2_stratified_frontier_finqa_grpo",
        "source_train_file": args.source_train_file,
        "original_finqa_train_rows": len(rows),
        "original_unique_records": len(rows_by_id),
        "eligible_records": len(eligible),
        "sample_size_requested": int(args.sample_size),
        "sample_size": len(selected),
        "seed": int(args.seed),
        "excluded_record_ids": len(excluded),
        "selected_unique_records": len(selected_ids),
        "selected_distribution": distribution(selected),
        "source_distribution": distribution(rows),
        "missing_target_families": [family for family in TARGET_FAMILIES if not any(program_family(row) == family for row in selected)],
        "record_id_manifest": str(output_dir / "acquisition_manifest.jsonl"),
        "acquisition_input_file": str(output_dir / "acquisition_input.jsonl"),
        "strata_count": len(Counter("|".join(strat_key(row)) for row in selected)),
    }
    (output_dir / "acquisition_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v34r21 stratified FinQA acquisition manifest.")
    parser.add_argument("--source_train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_allowlist", default="")
    parser.add_argument("--exclude_dev_file", default="")
    parser.add_argument("--exclude_test_file", default="")
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    return args


def main() -> None:
    print(json.dumps(build_manifest(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import evaluate_financial_benchmarks as eval_bench
from scripts.v34r21_common import distribution, load_record_ids, program_family, record_id, strat_key, write_jsonl


def unique_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = record_id(row)
        if rid and rid not in by_id:
            by_id[rid] = row
    return list(by_id.values())


def load_evaluator_rows(finqa_test_file: str) -> List[Dict[str, Any]]:
    eval_bench.PROCESSOR_ARGS.sft_variant = "program_executor_sft"
    eval_bench.PROCESSOR_ARGS.numeric_output_format = "reasoning_program_executor"
    examples = eval_bench.load_finqa_examples(finqa_test_file, max_samples=0)
    rows: List[Dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "record_id": example.record_id,
                "source_dataset": "finqa",
                "gold_answer": example.gold_answer,
                "gold_program": example.gold_program,
                "metadata": dict(example.metadata or {}),
            }
        )
    return unique_rows(rows)


def round_robin_stratified(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[strat_key(row)].append(row)
    for bucket_rows in grouped.values():
        bucket_rows.sort(key=record_id)
        rng.shuffle(bucket_rows)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    # First pass: ensure the major program families are represented when possible.
    for family in sorted({program_family(row) for row in rows}):
        candidates = [row for row in rows if program_family(row) == family and record_id(row) not in selected_ids]
        if candidates and len(selected) < sample_size:
            row = rng.choice(candidates)
            selected.append(row)
            selected_ids.add(record_id(row))

    # Second pass: rotate through strata so large buckets do not dominate the manifest.
    while len(selected) < sample_size:
        progressed = False
        for key in sorted(grouped):
            if len(selected) >= sample_size:
                break
            while grouped[key]:
                row = grouped[key].pop(0)
                rid = record_id(row)
                if rid not in selected_ids:
                    selected.append(row)
                    selected_ids.add(rid)
                    progressed = True
                    break
        if not progressed:
            break

    selected.sort(key=record_id)
    return selected[:sample_size]


def build_allowlist(args: argparse.Namespace) -> Dict[str, Any]:
    source_rows = load_evaluator_rows(args.finqa_test_file)
    excluded: set[str] = set()
    exclude_breakdown: Dict[str, int] = {}
    for path_text in args.exclude_file:
        ids = load_record_ids(Path(path_text))
        exclude_breakdown[path_text] = len(ids)
        excluded.update(ids)

    eligible = [row for row in source_rows if record_id(row) not in excluded]
    if len(eligible) < args.sample_size:
        raise ValueError(f"Only {len(eligible)} eligible rows remain; cannot build FinQA-{args.sample_size}")

    selected = round_robin_stratified(eligible, args.sample_size, args.seed)
    selected_ids = {record_id(row) for row in selected}
    overlap_after_selection = sorted(selected_ids & excluded)
    if overlap_after_selection:
        raise ValueError(f"Selected rows overlap excluded ids: {overlap_after_selection[:5]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowlist_path = output_dir / "v34r21_finqa64_allowlist.jsonl"
    manifest_path = output_dir / "v34r21_finqa64_manifest.jsonl"
    summary_path = output_dir / "v34r21_finqa64_allowlist.summary.json"

    allow_rows = [{"record_id": record_id(row), "source_dataset": "finqa_test", "gate": "v34r21_single_seed_finqa64"} for row in selected]
    manifest_rows = [
        {
            "manifest_index": idx,
            "record_id": record_id(row),
            "source_dataset": "finqa_test",
            "question_type": strat_key(row)[0],
            "answer_scale": strat_key(row)[1],
            "program_family": strat_key(row)[2],
            "stratification_key": "|".join(strat_key(row)),
        }
        for idx, row in enumerate(selected)
    ]
    write_jsonl(allowlist_path, allow_rows)
    write_jsonl(manifest_path, manifest_rows)

    summary = {
        "version": "v34r21_single_seed_finqa64_gate",
        "finqa_test_file": args.finqa_test_file,
        "seed": args.seed,
        "sample_size_requested": args.sample_size,
        "sample_size": len(selected),
        "source_unique_records": len(source_rows),
        "source_pool": "evaluation.load_finqa_examples(program_executor_sft, reasoning_program_executor)",
        "excluded_unique_record_ids": len(excluded),
        "exclude_breakdown": exclude_breakdown,
        "eligible_unique_records": len(eligible),
        "allowlist_file": str(allowlist_path),
        "manifest_file": str(manifest_path),
        "selected_distribution": distribution(selected),
        "eligible_distribution": distribution(eligible),
        "selected_strata_count": len(Counter("|".join(strat_key(row)) for row in selected)),
        "excluded_overlap_after_selection": len(overlap_after_selection),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed FinQA-64 allowlist for the v34r21 single-seed gate.")
    parser.add_argument("--finqa_test_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=34021)
    parser.add_argument("--exclude_file", action="append", default=[])
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    return args


def main() -> None:
    print(json.dumps(build_allowlist(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

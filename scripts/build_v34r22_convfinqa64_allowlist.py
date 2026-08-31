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

from evaluation import evaluate_financial_benchmarks as eval_bench
from scripts.v34r21_common import write_jsonl


def example_key(example: Any) -> tuple[str, str, str]:
    meta = dict(example.metadata or {})
    question_type = str(meta.get("question_type") or meta.get("question_family") or "unknown").lower()
    answer_scale = str(meta.get("answer_scale") or meta.get("answer_unit") or "unknown").lower()
    history = "history" if bool(meta.get("requires_history")) else "no_history"
    return question_type, answer_scale, history


def select_examples(examples: List[Any], sample_size: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    grouped: Dict[tuple[str, str, str], List[Any]] = defaultdict(list)
    for example in examples:
        grouped[example_key(example)].append(example)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.record_id)
        rng.shuffle(rows)
    selected: List[Any] = []
    selected_ids: set[str] = set()
    while len(selected) < sample_size:
        progressed = False
        for key in sorted(grouped):
            if len(selected) >= sample_size:
                break
            while grouped[key]:
                row = grouped[key].pop(0)
                if row.record_id not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row.record_id)
                    progressed = True
                    break
        if not progressed:
            break
    selected.sort(key=lambda item: item.record_id)
    return selected[:sample_size]


def build(args: argparse.Namespace) -> Dict[str, Any]:
    eval_bench.PROCESSOR_ARGS.sft_variant = "program_executor_sft"
    eval_bench.PROCESSOR_ARGS.numeric_output_format = "program_executor"
    examples = eval_bench.load_convfinqa_examples(args.convfinqa_dev_file, max_samples=0)
    if len(examples) < args.sample_size:
        raise ValueError(f"Only {len(examples)} examples available; cannot build ConvFinQA-{args.sample_size}")
    selected = select_examples(examples, args.sample_size, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowlist_path = output_dir / "v34r22_convfinqa64_allowlist.jsonl"
    manifest_path = output_dir / "v34r22_convfinqa64_manifest.jsonl"
    allow_rows = [{"record_id": item.record_id, "source_dataset": "convfinqa_dev", "gate": "v34r22_joint_baseline"} for item in selected]
    manifest_rows = [
        {
            "manifest_index": idx,
            "record_id": item.record_id,
            "source_dataset": "convfinqa_dev",
            "question_type": example_key(item)[0],
            "answer_scale": example_key(item)[1],
            "history_bucket": example_key(item)[2],
        }
        for idx, item in enumerate(selected)
    ]
    write_jsonl(allowlist_path, allow_rows)
    write_jsonl(manifest_path, manifest_rows)
    summary = {
        "version": "v34r22_convfinqa64_joint_eval_allowlist",
        "convfinqa_dev_file": args.convfinqa_dev_file,
        "seed": args.seed,
        "source_pool": "evaluation.load_convfinqa_examples(program_executor_sft, program_executor)",
        "source_examples": len(examples),
        "sample_size": len(selected),
        "allowlist_file": str(allowlist_path),
        "manifest_file": str(manifest_path),
        "history_distribution": dict(Counter(example_key(item)[2] for item in selected)),
        "question_type_distribution": dict(Counter(example_key(item)[0] for item in selected)),
        "answer_scale_distribution": dict(Counter(example_key(item)[1] for item in selected)),
    }
    (output_dir / "v34r22_convfinqa64_allowlist.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed ConvFinQA-64 allowlist for v34r22 joint evaluation.")
    parser.add_argument("--convfinqa_dev_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=34022)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

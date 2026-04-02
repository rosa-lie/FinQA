#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import DATASET_FAMILIES, infer_dataset_family, iter_records, load_source, parse_bool_arg
from .families import FAMILY_MODULES


def _process_family_records(records: List[Dict[str, Any]], family: str, task: str, args: Any) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    module = FAMILY_MODULES[family]
    family_stats: Dict[str, int] = {}

    if family == "convfinqa_turn" and parse_bool_arg(args.convfinqa_keep_final_only):
        records, dedupe_stats = module.dedupe_final_turn(records)
        family_stats.update(dedupe_stats)

    builder = module.build_sft_item if task == "sft" else module.build_dpo_item
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for rec in records:
        item = builder(rec, args)
        if item is None:
            skipped += 1
            continue
        rows.append(item)

    family_stats.setdefault("group_count", len(records) if family == "convfinqa_turn" else 0)
    family_stats.setdefault("dedup_dropped_rows", 0)
    family_stats.setdefault("fallback_selected_rows", 0)
    return rows, skipped, family_stats


def run_pipeline(task: str, args: Any) -> Dict[str, Any]:
    ds = load_source(args)
    input_records = list(iter_records(ds))

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if args.dataset_family == "auto":
        for rec in input_records:
            family = infer_dataset_family(rec)
            grouped[family].append(rec)
    else:
        grouped[args.dataset_family] = input_records

    output_rows: List[Dict[str, Any]] = []
    total_skipped = 0
    family_outputs: Dict[str, Dict[str, int]] = {}

    for family in ["convfinqa_turn", "finqa", "fineval", "fiqa_qa"]:
        records = grouped.get(family, [])
        if not records:
            continue
        rows, skipped, family_stats = _process_family_records(records, family, task, args)
        total_skipped += skipped
        output_rows.extend(rows)
        family_outputs[family] = {
            "input_rows": len(records),
            "saved_rows": len(rows),
            "skipped_rows": skipped,
            **family_stats,
        }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "task": task,
        "output_file": str(out_path),
        "dataset_family": args.dataset_family,
        "input_rows": len(input_records),
        "saved_rows": len(output_rows),
        "skipped_rows": total_skipped,
        "per_family": family_outputs,
    }

    conv_stats = family_outputs.get("convfinqa_turn")
    if conv_stats:
        summary["group_count"] = conv_stats.get("group_count", 0)
        summary["dedup_dropped_rows"] = conv_stats.get("dedup_dropped_rows", 0)
        summary["fallback_selected_rows"] = conv_stats.get("fallback_selected_rows", 0)

    return summary


def build_common_parser(default_task: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    if default_task is None:
        parser.add_argument("--task", type=str, required=True, choices=["sft", "dpo"])
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--dataset_family", type=str, default="auto", choices=sorted(DATASET_FAMILIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--convfinqa_keep_final_only", type=str, default="true")
    parser.add_argument("--max_history_turns", type=int, default=6)
    parser.add_argument("--max_context_items", type=int, default=6)
    parser.add_argument("--max_context_chars", type=int, default=400)
    parser.add_argument("--max_supporting_facts", type=int, default=6)
    parser.add_argument("--max_table_rows", type=int, default=20)
    parser.add_argument("--max_table_cols", type=int, default=12)
    parser.add_argument("--max_cell_chars", type=int, default=80)
    return parser


def run_cli(default_task: str | None = None) -> None:
    parser = build_common_parser(default_task=default_task)
    args = parser.parse_args()
    task = default_task or args.task
    summary = run_pipeline(task=task, args=args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_cli()

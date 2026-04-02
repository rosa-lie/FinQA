#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import DATASET_FAMILIES, infer_dataset_family, iter_records, load_source, parse_bool_arg, to_text
from .families import FAMILY_MODULES


DPO_STRUCTURED_ANCHORS = ["最终答案：", "结论：", "推理程序：", "推理：", "解释："]


def _check_dpo_row(row: Dict[str, Any]) -> str | None:
    required_fields = ["system", "history", "question", "response_chosen", "response_rejected", "source_dataset", "record_id", "metadata"]
    for field in required_fields:
        if field not in row:
            return "missing_required_field"

    if not isinstance(row.get("history"), list):
        return "invalid_history_type"
    if not isinstance(row.get("metadata"), dict):
        return "invalid_metadata_type"

    question = to_text(row.get("question"))
    chosen = to_text(row.get("response_chosen"))
    rejected = to_text(row.get("response_rejected"))

    if not question:
        return "empty_question"
    if not chosen:
        return "empty_chosen"
    if not rejected:
        return "empty_rejected"

    if chosen == rejected:
        return "identical_pair"

    # comparable but lower-quality: reject responses that are too short to compare.
    min_rejected_chars = max(24, int(len(chosen) * 0.2))
    if len(rejected) < min_rejected_chars:
        return "rejected_too_short"

    # If chosen uses a structured answer style, rejected should share at least one anchor.
    chosen_anchors = [anchor for anchor in DPO_STRUCTURED_ANCHORS if anchor in chosen]
    if chosen_anchors and not any(anchor in rejected for anchor in chosen_anchors):
        return "not_comparable"

    return None


def _postprocess_dpo_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept_rows: List[Dict[str, Any]] = []
    seen_keys = set()
    reject_counts: Dict[str, int] = defaultdict(int)

    for row in rows:
        reason = _check_dpo_row(row)
        if reason is not None:
            reject_counts[reason] += 1
            continue

        dedupe_key = (
            to_text(row.get("question")),
            to_text(row.get("response_chosen")),
            to_text(row.get("response_rejected")),
        )
        if dedupe_key in seen_keys:
            reject_counts["duplicate_pair"] += 1
            continue
        seen_keys.add(dedupe_key)
        kept_rows.append(row)

    stats: Dict[str, int] = {
        "post_filter_input_rows": len(rows),
        "post_filter_saved_rows": len(kept_rows),
        "post_filter_skipped_rows": sum(reject_counts.values()),
    }
    for reason, count in sorted(reject_counts.items()):
        stats[f"post_filter_{reason}_rows"] = count
    return kept_rows, stats


def _process_family_records(records: List[Dict[str, Any]], family: str, task: str, args: Any) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    module = FAMILY_MODULES[family]
    family_stats: Dict[str, int] = {}

    if family == "convfinqa_turn" and parse_bool_arg(args.convfinqa_keep_final_only):
        records, dedupe_stats = module.dedupe_final_turn(records)
        family_stats.update(dedupe_stats)

    builder = module.build_sft_item if task == "sft" else module.build_dpo_item
    rows: List[Dict[str, Any]] = []
    build_skipped = 0
    for rec in records:
        item = builder(rec, args)
        if item is None:
            build_skipped += 1
            continue
        rows.append(item)

    post_filter_skipped = 0
    if task == "dpo":
        rows, post_filter_stats = _postprocess_dpo_rows(rows)
        family_stats.update(post_filter_stats)
        post_filter_skipped = post_filter_stats.get("post_filter_skipped_rows", 0)

    family_stats.setdefault("group_count", len(records) if family == "convfinqa_turn" else 0)
    family_stats.setdefault("dedup_dropped_rows", 0)
    family_stats.setdefault("fallback_selected_rows", 0)
    family_stats["build_skipped_rows"] = build_skipped
    family_stats["post_filter_skipped_rows"] = post_filter_skipped

    skipped_total = build_skipped + post_filter_skipped
    return rows, skipped_total, family_stats


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

    if task == "dpo":
        dpo_post_filter_totals: Dict[str, int] = defaultdict(int)
        for family_stats in family_outputs.values():
            for key, value in family_stats.items():
                if key.startswith("post_filter_") and isinstance(value, int):
                    dpo_post_filter_totals[key] += value
        summary["dpo_post_filter"] = dict(sorted(dpo_post_filter_totals.items()))

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

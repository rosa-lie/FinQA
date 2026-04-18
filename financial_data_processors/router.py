#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import (
    DATASET_FAMILIES,
    build_audit_item,
    infer_dataset_family,
    iter_records,
    load_source,
    parse_bool_arg,
    to_text,
)
from .families import FAMILY_MODULES


DPO_STRUCTURED_ANCHORS = [
    "Answer:",
    "Program:",
    "Evidence:",
    "最终答案：",
    "结论：",
    "推理程序：",
    "推理：",
    "解释：",
]

NORMALIZED_ANSWER_RE = re.compile(r"(?m)^Normalized Answer:\s*(.*)$")
ANSWER_RE = re.compile(r"(?m)^Answer:\s*(.*)$")


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

    min_rejected_chars = max(24, int(len(chosen) * 0.2))
    if len(rejected) < min_rejected_chars:
        return "rejected_too_short"

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


def _has_json_like_evidence(item: Dict[str, Any]) -> bool:
    conversations = item.get("conversations") if isinstance(item.get("conversations"), list) else []
    if len(conversations) < 2 or not isinstance(conversations[1], dict):
        return False
    text = to_text(conversations[1].get("value"))
    return '{"text_' in text or '{"table_' in text


def _extract_target_label(item: Dict[str, Any]) -> str:
    conversations = item.get("conversations") if isinstance(item.get("conversations"), list) else []
    if len(conversations) < 2 or not isinstance(conversations[1], dict):
        return ""
    text = to_text(conversations[1].get("value"))
    match = NORMALIZED_ANSWER_RE.search(text)
    if match:
        return to_text(match.group(1))
    match = ANSWER_RE.search(text)
    if match:
        return to_text(match.group(1))
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    answer_norm = to_text(metadata.get("answer_norm"))
    program_canonical = to_text(metadata.get("program_canonical"))
    if answer_norm or program_canonical:
        return f"{answer_norm}\n{program_canonical}"
    return ""


def _postprocess_sft_rows(rows: List[Dict[str, Any]], args: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not parse_bool_arg(getattr(args, "filter_conflicting_prompts", "true")):
        return rows, {
            "conflict_prompt_groups": 0,
            "conflict_prompt_rows": 0,
        }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conversations = row.get("conversations") if isinstance(row.get("conversations"), list) else []
        prompt = to_text(conversations[0].get("value")) if conversations and isinstance(conversations[0], dict) else ""
        grouped[prompt].append(row)

    conflict_prompts = {
        prompt
        for prompt, items in grouped.items()
        if prompt and len({normalize_question_for_label(_extract_target_label(item)) for item in items if _extract_target_label(item)}) > 1
    }
    if not conflict_prompts:
        return rows, {
            "conflict_prompt_groups": 0,
            "conflict_prompt_rows": 0,
        }

    kept = []
    dropped = 0
    for row in rows:
        conversations = row.get("conversations") if isinstance(row.get("conversations"), list) else []
        prompt = to_text(conversations[0].get("value")) if conversations and isinstance(conversations[0], dict) else ""
        if prompt in conflict_prompts:
            dropped += 1
            continue
        kept.append(row)
    return kept, {
        "conflict_prompt_groups": len(conflict_prompts),
        "conflict_prompt_rows": dropped,
    }


def normalize_question_for_label(text: Any) -> str:
    return " ".join(to_text(text).lower().split())


def _inc_flag_stats(stats: Dict[str, int], norm: Dict[str, Any]) -> None:
    tier = to_text(norm.get("quality_tier") or "C").upper() or "C"
    stats[f"tier_{tier}_rows"] = stats.get(f"tier_{tier}_rows", 0) + 1
    if norm.get("answer_display"):
        stats["answer_display_rows"] = stats.get("answer_display_rows", 0) + 1
    flags = set(norm.get("semantic_audit_flags") or []) | set(norm.get("audit_flags") or [])
    for flag in [
        "question_semantic_risk",
        "question_text_suspicious",
        "history_answer_missing",
        "weak_table_evidence_rendering",
        "evidence_not_in_rendered_prompt",
        "duplicate_current_question_in_history",
        "current_target_leaked_in_history",
        "current_answer_leaked_in_history",
        "current_answer_repeated_in_history",
    ]:
        if flag in flags:
            stats[f"{flag}_rows"] = stats.get(f"{flag}_rows", 0) + 1
    if norm.get("evidence_visible_in_prompt"):
        stats["evidence_visible_in_prompt_rows"] = stats.get("evidence_visible_in_prompt_rows", 0) + 1
    if norm.get("table_evidence_column_pruned"):
        stats["table_evidence_column_pruned_rows"] = stats.get("table_evidence_column_pruned_rows", 0) + 1
    if norm.get("requires_history"):
        stats["requires_history_rows"] = stats.get("requires_history_rows", 0) + 1
    if norm.get("history_answer_missing"):
        stats["history_answer_missing_rows"] = stats.get("history_answer_missing_rows", 0) + 1
    history_turn_count = int(norm.get("history_turn_count") or 0)
    full_reasoning_turn_count = int(norm.get("history_full_reasoning_turn_count") or 0)
    question_only_turn_count = int(norm.get("history_question_only_turn_count") or 0)
    if history_turn_count:
        stats["multiturn_history_rows"] = stats.get("multiturn_history_rows", 0) + 1
    if full_reasoning_turn_count:
        stats["history_full_reasoning_rows"] = stats.get("history_full_reasoning_rows", 0) + 1
        stats["rendered_history_full_reasoning_turns"] = stats.get("rendered_history_full_reasoning_turns", 0) + full_reasoning_turn_count
    if question_only_turn_count:
        stats["history_question_only_rows"] = stats.get("history_question_only_rows", 0) + 1
        stats["rendered_history_question_only_turns"] = stats.get("rendered_history_question_only_turns", 0) + question_only_turn_count



def _process_family_records(
    records: List[Dict[str, Any]],
    family: str,
    task: str,
    args: Any,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    module = FAMILY_MODULES[family]
    family_stats: Dict[str, int] = {}
    normalized_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    if family == "convfinqa_turn":
        convfinqa_mode = getattr(args, "convfinqa_mode", "turn_level")
        if convfinqa_mode == "legacy_dedupe" or (
            convfinqa_mode == "turn_level" and parse_bool_arg(getattr(args, "convfinqa_keep_final_only", "false"))
        ):
            records, dedupe_stats = module.dedupe_final_turn(records)
            family_stats.update(dedupe_stats)
        elif convfinqa_mode == "final_turn_only":
            records, dedupe_stats = module.select_final_turn(records)
            family_stats.update(dedupe_stats)
        else:
            family_stats.update({
                "raw_turn_rows": len(records),
                "saved_turn_rows": len(records),
            })
    if family == "convfinqa_turn" and hasattr(module, "prepare_multiturn_records"):
        records, multiturn_stats = module.prepare_multiturn_records(records, args)
        family_stats.update(multiturn_stats)

    rows: List[Dict[str, Any]] = []
    build_skipped = 0

    if task == "sft" and hasattr(module, "normalize_record") and hasattr(module, "render_sft_item"):
        for rec in records:
            norm = module.normalize_record(rec, args)
            normalized_rows.append(norm)
            if norm.get("evidence_match_type") == "exact":
                family_stats["exact_evidence_alignment_rows"] = family_stats.get("exact_evidence_alignment_rows", 0) + 1
            if norm.get("answer_matches_program"):
                family_stats["program_answer_match_rows"] = family_stats.get("program_answer_match_rows", 0) + 1
            if norm.get("program_raw"):
                family_stats["raw_program_unchanged_rows"] = family_stats.get("raw_program_unchanged_rows", 0) + 1
            _inc_flag_stats(family_stats, norm)
            if norm.get("quality_tier") != "A":
                audit_rows.append(build_audit_item(norm))

            item = module.render_sft_item(norm, args)
            if item is None:
                build_skipped += 1
                if norm.get("quality_tier") == "A":
                    audit_rows.append(build_audit_item(norm))
                continue
            if _has_json_like_evidence(item):
                family_stats["json_like_evidence_rows"] = family_stats.get("json_like_evidence_rows", 0) + 1
            rows.append(item)
    else:
        builder = module.build_sft_item if task == "sft" else module.build_dpo_item
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
    family_stats.setdefault("exact_evidence_alignment_rows", 0)
    family_stats.setdefault("program_answer_match_rows", 0)
    family_stats.setdefault("raw_program_unchanged_rows", 0)
    family_stats.setdefault("json_like_evidence_rows", 0)
    for key in [
        "tier_A_rows", "tier_B_rows", "tier_C_rows", "question_semantic_risk_rows",
        "history_answer_missing_rows", "weak_table_evidence_rendering_rows",
        "question_text_suspicious_rows", "evidence_not_in_rendered_prompt_rows",
        "duplicate_current_question_in_history_rows", "current_target_leaked_in_history_rows", "current_answer_leaked_in_history_rows", "current_answer_repeated_in_history_rows",
        "evidence_visible_in_prompt_rows", "table_evidence_column_pruned_rows", "answer_display_rows", "requires_history_rows", "multiturn_history_rows",
        "history_full_reasoning_rows", "history_question_only_rows", "rendered_history_full_reasoning_turns", "rendered_history_question_only_turns",
        "conversation_count", "history_full_reasoning_turns", "history_question_only_turns",
    ]:
        family_stats.setdefault(key, 0)
    family_stats["normalized_rows"] = len(normalized_rows)
    family_stats["audit_rows"] = len(audit_rows)
    family_stats["build_skipped_rows"] = build_skipped
    family_stats["post_filter_skipped_rows"] = post_filter_skipped

    skipped_total = build_skipped + post_filter_skipped
    return rows, skipped_total, family_stats, normalized_rows, audit_rows


def _write_jsonl(path_text: str | None, rows: List[Dict[str, Any]]) -> str | None:
    if not path_text:
        return None
    out_path = Path(path_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(out_path)


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
    normalized_output_rows: List[Dict[str, Any]] = []
    audit_output_rows: List[Dict[str, Any]] = []
    total_skipped = 0
    family_outputs: Dict[str, Dict[str, int]] = {}

    for family in ["convfinqa_turn", "finqa", "fineval", "fiqa_qa"]:
        records = grouped.get(family, [])
        if not records:
            continue
        rows, skipped, family_stats, normalized_rows, audit_rows = _process_family_records(records, family, task, args)
        total_skipped += skipped
        output_rows.extend(rows)
        normalized_output_rows.extend(normalized_rows)
        audit_output_rows.extend(audit_rows)
        family_outputs[family] = {
            "input_rows": len(records),
            "saved_rows": len(rows),
            "skipped_rows": skipped,
            **family_stats,
        }

    sft_postprocess_stats: Dict[str, int] = {}
    if task == "sft":
        before_sft_postprocess = len(output_rows)
        output_rows, sft_postprocess_stats = _postprocess_sft_rows(output_rows, args)
        total_skipped += before_sft_postprocess - len(output_rows)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    normalized_path = _write_jsonl(getattr(args, "normalized_output_file", None), normalized_output_rows)
    audit_path = _write_jsonl(getattr(args, "audit_output_file", None), audit_output_rows)

    summary: Dict[str, Any] = {
        "task": task,
        "output_file": str(out_path),
        "normalized_output_file": normalized_path,
        "audit_output_file": audit_path,
        "dataset_family": args.dataset_family,
        "sft_variant": getattr(args, "sft_variant", "benchmark_sft"),
        "strict_tiers": getattr(args, "strict_tiers", "A"),
        "input_rows": len(input_records),
        "saved_rows": len(output_rows),
        "skipped_rows": total_skipped,
        "normalized_rows": len(normalized_output_rows),
        "audit_rows": len(audit_output_rows),
        "per_family": family_outputs,
    }
    if sft_postprocess_stats:
        summary["sft_post_filter"] = sft_postprocess_stats

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
    parser.add_argument("--normalized_output_file", type=str, default=None)
    parser.add_argument("--audit_output_file", type=str, default=None)
    parser.add_argument("--dataset_family", type=str, default="auto", choices=sorted(DATASET_FAMILIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--convfinqa_keep_final_only", type=str, default="false")
    parser.add_argument("--convfinqa_mode", type=str, default="turn_level", choices=["turn_level", "final_turn_only", "legacy_dedupe"])
    parser.add_argument("--filter_conflicting_prompts", type=str, default="true")
    parser.add_argument("--sft_variant", type=str, default="benchmark_sft", choices=["benchmark_sft", "assistant_sft", "dual_answer_sft", "program_executor_sft"])
    parser.add_argument("--strict_tiers", type=str, default="A")
    parser.add_argument("--max_history_turns", type=int, default=6)
    parser.add_argument("--max_context_items", type=int, default=6)
    parser.add_argument("--max_context_chars", type=int, default=400)
    parser.add_argument("--max_supporting_facts", type=int, default=3)
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

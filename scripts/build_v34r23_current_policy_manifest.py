#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v34r21_common import (
    answer_scale,
    distribution,
    first_text,
    load_json_or_jsonl,
    load_record_ids,
    program_family,
    record_id,
    source_dataset,
    strat_key,
    write_jsonl,
)

TASK_SOURCE = {"finqa": "finqa", "convfinqa": "convfinqa_turn"}


def conversation_value(row: Dict[str, Any], role: str) -> str:
    for turn in row.get("conversations") or []:
        if first_text(turn.get("from")).lower() == role:
            return first_text(turn.get("value"))
    return ""


def normalize_for_acquisition(row: Dict[str, Any], task: str) -> Dict[str, Any]:
    meta = dict(row.get("metadata") or {})
    if task == "convfinqa":
        prompt = first_text(row.get("input_prompt_raw")) or conversation_value(row, "human")
        reference = first_text(row.get("reference_response")) or conversation_value(row, "gpt")
        gold_program = first_text(row.get("gold_program") or meta.get("program_canonical") or meta.get("program_raw"))
        gold_answer = first_text(row.get("gold_answer") or meta.get("answer_norm") or meta.get("answer_raw") or meta.get("answer_exe"))
        source = "convfinqa_turn"
        meta["v34r23_requires_history"] = bool(
            meta.get("requires_history")
            or int(meta.get("history_turns") or 0) > 0
            or "Conversation history:" in prompt
            or "Previous question:" in prompt
        )
        meta["v34r23_history_dependency_type"] = first_text(meta.get("history_dependency_type") or "self_contained")
    else:
        prompt = first_text(row.get("input_prompt_raw"))
        reference = first_text(row.get("reference_response"))
        gold_program = first_text(row.get("gold_program") or meta.get("program_canonical") or meta.get("program_raw"))
        gold_answer = first_text(row.get("gold_answer") or meta.get("answer_norm") or meta.get("answer_raw") or meta.get("answer_exe"))
        source = "finqa"
        meta["v34r23_requires_history"] = False
        meta["v34r23_history_dependency_type"] = "none"

    out = dict(row)
    out["record_id"] = record_id(row)
    out["source_dataset"] = source
    out["input_prompt_raw"] = prompt
    out["reference_response"] = reference
    out["gold_answer"] = gold_answer
    out["gold_program"] = gold_program
    out["reward_profile"] = "program_numeric"
    out["metadata"] = meta
    return out


def dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = record_id(row)
        if rid and rid not in by_id:
            by_id[rid] = row
    return list(by_id.values())


def select_stratified(rows: Sequence[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[strat_key(row)].append(row)
    for bucket_rows in by_key.values():
        bucket_rows.sort(key=record_id)
        rng.shuffle(bucket_rows)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    total = max(len(rows), 1)
    allocations: Dict[Tuple[str, str, str], int] = {}
    remainders: List[Tuple[float, Tuple[str, str, str]]] = []
    for key, bucket_rows in by_key.items():
        ideal = sample_size * len(bucket_rows) / total
        count = min(int(ideal), len(bucket_rows))
        allocations[key] = count
        remainders.append((ideal - int(ideal), key))
    assigned = sum(allocations.values())
    for _remainder, key in sorted(remainders, reverse=True):
        if assigned >= sample_size:
            break
        if allocations[key] < len(by_key[key]):
            allocations[key] += 1
            assigned += 1

    for key in sorted(by_key):
        for row in by_key[key][: allocations.get(key, 0)]:
            rid = record_id(row)
            if rid not in selected_ids:
                selected.append(row)
                selected_ids.add(rid)

    if len(selected) < sample_size:
        leftovers = [row for row in rows if record_id(row) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: sample_size - len(selected)])

    selected.sort(key=record_id)
    return selected[:sample_size]


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    task = args.task.lower()
    if task not in TASK_SOURCE:
        raise ValueError(f"Unsupported task: {args.task}")
    raw_rows = load_json_or_jsonl(Path(args.source_file))
    normalized = [normalize_for_acquisition(row, task) for row in raw_rows]
    normalized = [row for row in normalized if source_dataset(row) == TASK_SOURCE[task]]
    rows = dedupe_rows(normalized)

    excluded: set[str] = set()
    for path in args.exclude_allowlist or []:
        if path:
            excluded.update(load_record_ids(Path(path)))
    eligible = [row for row in rows if record_id(row) and record_id(row) not in excluded]
    sample_size = min(int(args.sample_size), len(eligible))
    selected = select_stratified(eligible, sample_size, int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(selected):
        phase = "pilot" if index < int(args.pilot_size) else "extension"
        manifest_rows.append(
            {
                "manifest_index": index,
                "manifest_phase": phase,
                "record_id": record_id(row),
                "source_dataset": source_dataset(row),
                "question_type": strat_key(row)[0],
                "answer_scale": answer_scale(row),
                "program_family": program_family(row),
                "requires_history": bool((row.get("metadata") or {}).get("v34r23_requires_history")),
                "history_dependency_type": first_text((row.get("metadata") or {}).get("v34r23_history_dependency_type")),
                "stratification_key": "|".join(strat_key(row)),
            }
        )

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "acquisition_input.jsonl", selected)
    summary = {
        "version": "v34r23_current_policy_frontier_manifest",
        "task": task,
        "source_file": args.source_file,
        "seed": int(args.seed),
        "sample_size_requested": int(args.sample_size),
        "pilot_size": int(args.pilot_size),
        "source_rows": len(raw_rows),
        "normalized_unique_records": len(rows),
        "excluded_record_ids": len(excluded),
        "eligible_records": len(eligible),
        "selected_unique_records": len(selected),
        "pilot_records": min(int(args.pilot_size), len(selected)),
        "extension_records": max(len(selected) - int(args.pilot_size), 0),
        "distribution": distribution(selected),
        "history_distribution": dict(Counter("history" if (row.get("metadata") or {}).get("v34r23_requires_history") else "no_history" for row in selected)),
        "manifest_file": str(output_dir / "manifest.jsonl"),
        "acquisition_input_file": str(output_dir / "acquisition_input.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v34r23 current-policy frontier acquisition manifest.")
    parser.add_argument("--task", choices=sorted(TASK_SOURCE), required=True)
    parser.add_argument("--source_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--pilot_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=34023)
    parser.add_argument("--exclude_allowlist", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_manifest(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

from financial_data_processors.common import iter_records
from financial_data_processors.families import FAMILY_MODULES

FINAL_ANSWER_RE = re.compile(r"最终答案：\s*(.+)", re.DOTALL)


DEFAULT_ARGS = SimpleNamespace(
    max_history_turns=6,
    max_context_items=6,
    max_context_chars=400,
    max_supporting_facts=6,
    max_table_rows=20,
    max_table_cols=12,
    max_cell_chars=80,
)


def load_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for row in data:
            yield dict(row)
        return
    if isinstance(data, dict):
        for key in ["data", "records", "examples", "items"]:
            if isinstance(data.get(key), list):
                for row in data[key]:
                    yield dict(row)
                return
        yield dict(data)
        return
    raise ValueError(f"Unsupported JSON structure: {path}")


def parse_source_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --source_spec: {spec}")
    family, path = spec.split("=", 1)
    family = family.strip()
    path = Path(path.strip())
    if family not in FAMILY_MODULES:
        raise ValueError(f"Unsupported family: {family}")
    if not path.exists():
        raise FileNotFoundError(path)
    return family, path


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return (text or "").strip()


def build_processor_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        max_history_turns=args.max_history_turns,
        max_context_items=args.max_context_items,
        max_context_chars=args.max_context_chars,
        max_supporting_facts=args.max_supporting_facts,
        max_table_rows=args.max_table_rows,
        max_table_cols=args.max_table_cols,
        max_cell_chars=args.max_cell_chars,
    )


def build_distill_row(rec: Dict[str, Any], family: str, processor_args: SimpleNamespace) -> Dict[str, Any] | None:
    module = FAMILY_MODULES[family]
    item = module.build_sft_item(rec, processor_args)
    if item is None:
        return None
    conversations = item.get("conversations") or []
    if len(conversations) < 2:
        return None
    prompt = (conversations[0].get("value") or "").strip()
    gold_response = (conversations[1].get("value") or "").strip()
    gold_answer = extract_final_answer(gold_response)
    metadata = dict(item.get("metadata") or {})
    gold_program = str(metadata.get("program") or "").strip()
    if not prompt or not gold_answer:
        return None
    record_id = str(item.get("record_id") or rec.get("id") or rec.get("filename") or "").strip()
    if not record_id:
        record_id = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return {
        "prompt": prompt,
        "gold_response": gold_response,
        "gold_answer": gold_answer,
        "gold_program": gold_program,
        "task_name": family,
        "source_dataset": item.get("source_dataset", family),
        "record_id": record_id,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build financial distillation input dataset from raw sources.")
    parser.add_argument("--source_spec", action="append", required=True, help="family=/path/to/file.{json,jsonl}")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples_per_family", type=int, default=0)
    parser.add_argument("--convfinqa_keep_final_only", type=str, default="true")
    parser.add_argument("--max_history_turns", type=int, default=DEFAULT_ARGS.max_history_turns)
    parser.add_argument("--max_context_items", type=int, default=DEFAULT_ARGS.max_context_items)
    parser.add_argument("--max_context_chars", type=int, default=DEFAULT_ARGS.max_context_chars)
    parser.add_argument("--max_supporting_facts", type=int, default=DEFAULT_ARGS.max_supporting_facts)
    parser.add_argument("--max_table_rows", type=int, default=DEFAULT_ARGS.max_table_rows)
    parser.add_argument("--max_table_cols", type=int, default=DEFAULT_ARGS.max_table_cols)
    parser.add_argument("--max_cell_chars", type=int, default=DEFAULT_ARGS.max_cell_chars)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor_args = build_processor_args(args)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for spec in args.source_spec:
        family, path = parse_source_spec(spec)
        records = [dict(row) for row in iter_records(load_json_records(path))]
        if family == "convfinqa_turn" and str(args.convfinqa_keep_final_only).lower() in {"1", "true", "yes", "y", "on"}:
            records, dedupe_stats = FAMILY_MODULES[family].dedupe_final_turn(records)
        else:
            dedupe_stats = {}

        family_rows: List[Dict[str, Any]] = []
        build_skipped = 0
        for rec in records:
            row = build_distill_row(rec, family, processor_args)
            if row is None:
                build_skipped += 1
                continue
            family_rows.append(row)

        if args.max_samples_per_family and args.max_samples_per_family > 0:
            family_rows = family_rows[: args.max_samples_per_family]

        rows.extend(family_rows)
        summary.append({
            "family": family,
            "source_file": str(path),
            "rows": len(family_rows),
            "build_skipped_rows": build_skipped,
            **dedupe_stats,
        })

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output_file": str(out_path),
        "total_rows": len(rows),
        "families": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

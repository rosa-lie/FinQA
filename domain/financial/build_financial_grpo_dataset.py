# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

from domain.financial.processors.common import iter_records
from domain.financial.processors.families import FAMILY_MODULES

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


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return (text or "").strip()


def parse_source_spec(spec: str) -> tuple[str, Path]:
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


def build_grpo_row(rec: Dict[str, Any], family: str) -> Dict[str, Any] | None:
    module = FAMILY_MODULES[family]
    item = module.build_sft_item(rec, DEFAULT_ARGS)
    if item is None:
        return None
    prompt = item["conversations"][0]["value"].strip()
    chosen = item["conversations"][1]["value"].strip()
    answer = extract_final_answer(chosen)
    metadata = item.get("metadata", {}) or {}
    gold_program = str(metadata.get("program") or "").strip()
    if not prompt or not answer:
        return None
    return {
        "prompt": prompt,
        "answer": answer,
        "gold_program": gold_program,
        "task_name": family,
        "source_dataset": item.get("source_dataset", family),
        "record_id": item.get("record_id", ""),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build financial GRPO dataset from raw financial reasoning sources.")
    parser.add_argument("--source_spec", action="append", required=True, help="family=/path/to/file.{json,jsonl}")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples_per_family", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for spec in args.source_spec:
        family, path = parse_source_spec(spec)
        family_rows = []
        for rec in iter_records(load_json_records(path)):
            row = build_grpo_row(rec, family)
            if row is not None:
                family_rows.append(row)
        if args.max_samples_per_family and args.max_samples_per_family > 0:
            family_rows = family_rows[: args.max_samples_per_family]
        rows.extend(family_rows)
        summary.append({
            "family": family,
            "source_file": str(path),
            "rows": len(family_rows),
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert FinGPT-style data to MedicalGPT DPO pair format.

Output schema (compatible with dpo_training.py):
{
  "system": "",
  "history": [],
  "question": "...",
  "response_chosen": "...",
  "response_rejected": "..."
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset

QUESTION_KEYS = ["instruction", "question", "query", "prompt", "input", "context", "text"]
ANSWER_KEYS = ["output", "answer", "response", "completion", "label", "sentiment"]


def pick_first(rec: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in rec and rec[k] is not None:
            val = str(rec[k]).strip()
            if val:
                return val
    return ""


def parse_options(text: str) -> List[str]:
    # Very lightweight parser for prompts like "Options: positive, neutral, negative"
    m = re.search(r"(?i)options?\s*[:：]\s*(.+)", text)
    if not m:
        return []
    raw = m.group(1)
    cands = re.split(r"[,/|;；、]", raw)
    options = [c.strip() for c in cands if c.strip()]
    return options


def make_question(rec: Dict[str, Any]) -> str:
    instruction = str(rec.get("instruction", "")).strip()
    inp = str(rec.get("input", "")).strip()
    q = pick_first(rec, QUESTION_KEYS)
    if instruction and inp:
        return f"{instruction}\n\n{inp}"
    if instruction:
        return instruction
    return q


def make_pair(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    q = make_question(rec)

    # If already pairwise data, reuse directly
    chosen = str(rec.get("response_chosen", "")).strip()
    rejected = str(rec.get("response_rejected", "")).strip()
    if chosen and rejected:
        return q, chosen, rejected

    # Build chosen from single-response fields
    chosen = pick_first(rec, ANSWER_KEYS)
    if not q or not chosen:
        return "", "", ""

    # Try to build a stronger rejected answer from options
    options = parse_options(q)
    rejected = ""
    if options:
        lower_chosen = chosen.lower()
        for opt in options:
            if opt.lower() != lower_chosen:
                rejected = opt
                break

    # Fallback generic weak response
    if not rejected:
        rejected = "信息不足，无法给出可靠金融结论。"

    return q, chosen, rejected


def iter_records(ds: Dataset) -> Iterable[Dict[str, Any]]:
    for row in ds:
        yield dict(row)


def load_source(args: argparse.Namespace) -> Dataset:
    if args.source_file:
        ext = Path(args.source_file).suffix.lower()
        if ext not in {".json", ".jsonl"}:
            raise ValueError("--source_file currently supports .json/.jsonl only")
        return load_dataset("json", data_files=args.source_file, split="train")
    if not args.dataset_name:
        raise ValueError("Please provide --dataset_name or --source_file")
    return load_dataset(args.dataset_name, args.dataset_config, split=args.split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="FinGPT/fingpt-sentiment-train")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    ds = load_source(args)

    rows = []
    for rec in iter_records(ds):
        q, chosen, rejected = make_pair(rec)
        if not q or not chosen or not rejected:
            continue
        rows.append(
            {
                "system": "",
                "history": [],
                "question": q,
                "response_chosen": chosen,
                "response_rejected": rejected,
            }
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} DPO rows to {out_path}")


if __name__ == "__main__":
    main()

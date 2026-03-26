#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert FinGPT-style records to MedicalGPT SFT (ShareGPT) jsonl.

Examples:
python fin_to_sharegpt.py \
  --dataset_name FinGPT/fingpt-sentiment-train \
  --split train \
  --output_file data/fingpt/fingpt_sft_sharegpt.jsonl

python fin_to_sharegpt.py \
  --source_file data/fingpt/raw.jsonl \
  --output_file data/fingpt/fingpt_sft_sharegpt.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset, load_dataset

QUESTION_KEYS = ["question", "query", "prompt", "input", "context", "text"]
ANSWER_KEYS = ["output", "answer", "response", "completion", "label", "sentiment"]


def to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v).strip()
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def pick_first(rec: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in rec:
            val = to_text(rec.get(k))
            if val:
                return val
    return ""


def normalize_qa(rec: Dict[str, Any]) -> Dict[str, str]:
    instruction = to_text(rec.get("instruction"))
    inp = to_text(rec.get("input"))
    q = pick_first(rec, QUESTION_KEYS)
    a = pick_first(rec, ANSWER_KEYS)

    if instruction and inp:
        q = f"{instruction}\n\n{inp}"
    elif instruction and q and instruction != q:
        q = f"{instruction}\n\n{q}"
    elif instruction and not q:
        q = instruction

    if not q and "conversations" in rec:
        convs = rec.get("conversations") or []
        if len(convs) >= 2:
            q = to_text(convs[0].get("value"))
            a = to_text(convs[1].get("value"))

    return {"question": q.strip(), "answer": a.strip()}


def iter_records(ds: Dataset) -> Iterable[Dict[str, Any]]:
    for row in ds:
        yield dict(row)


def to_sharegpt(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in rows:
        pair = normalize_qa(rec)
        q, a = pair["question"], pair["answer"]
        if not q or not a:
            continue
        out.append(
            {
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a},
                ]
            }
        )
    return out


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
    rows = to_sharegpt(iter_records(ds))

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} SFT rows to {out_path}")


if __name__ == "__main__":
    main()

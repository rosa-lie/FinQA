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
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset

QUESTION_KEYS = ["question", "query", "prompt", "input", "context", "text"]
ANSWER_KEYS = ["output", "answer", "response", "completion", "label", "sentiment"]

SENTIMENT_CANDIDATES = ["positive", "neutral", "negative"]
CN_SENTIMENT_CANDIDATES = ["积极", "中性", "消极"]


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


def _split_options(raw: str) -> List[str]:
    cands = re.split(r"[,/|;；、]", raw)
    return [c.strip() for c in cands if c.strip()]


def parse_options(text: str) -> List[str]:
    # Pattern 1: "Options: a, b, c"
    m = re.search(r"(?i)options?\s*[:：]\s*(.+)", text)
    if m:
        options = _split_options(m.group(1))
        if options:
            return options

    # Pattern 2: "... from {negative/neutral/positive}"
    b = re.search(r"\{([^{}]+)\}", text)
    if b:
        options = _split_options(b.group(1))
        if options:
            return options

    # Pattern 3: "... from [negative, neutral, positive]"
    s = re.search(r"\[([^\[\]]+)\]", text)
    if s:
        options = _split_options(s.group(1))
        if options:
            return options
    return []


def make_question(rec: Dict[str, Any]) -> str:
    instruction = to_text(rec.get("instruction"))
    inp = to_text(rec.get("input"))
    q = pick_first(rec, QUESTION_KEYS)
    if instruction and inp:
        return f"{instruction}\n\n{inp}"
    if instruction and q and instruction != q:
        return f"{instruction}\n\n{q}"
    if instruction:
        return instruction
    return q


def choose_rejected(chosen: str, candidates: List[str]) -> str:
    if not candidates:
        return ""
    chosen_norm = chosen.strip().lower()
    pool = [c for c in candidates if c.strip().lower() != chosen_norm]
    if not pool:
        return ""
    return random.choice(pool)


def make_rejected_from_sentiment(chosen: str) -> str:
    rejected = choose_rejected(chosen, SENTIMENT_CANDIDATES)
    if rejected:
        return rejected
    rejected = choose_rejected(chosen, CN_SENTIMENT_CANDIDATES)
    if rejected:
        return rejected
    return ""


def make_pair(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    q = make_question(rec)

    chosen = to_text(rec.get("response_chosen"))
    rejected = to_text(rec.get("response_rejected"))
    if chosen and rejected:
        return q, chosen, rejected

    chosen = pick_first(rec, ANSWER_KEYS)
    if not q or not chosen:
        return "", "", ""

    options = parse_options(q)
    if options:
        rejected = choose_rejected(chosen, options)

    if not rejected:
        rejected = make_rejected_from_sentiment(chosen)

    if not rejected:
        pool = [
            "信息不足，无法给出可靠金融结论。",
            "该结论证据不足，建议补充财务与市场信息后再判断。",
        ]
        rejected = random.choice(pool)

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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
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

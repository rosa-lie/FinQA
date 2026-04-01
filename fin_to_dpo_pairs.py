#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build lightweight DPO pairs for financial reasoning datasets.

Chosen responses are built from dataset gold answers.
Rejected responses are heuristic negatives that violate one of:
- final answer correctness
- reasoning completeness / structure
- program consistency
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset

from fin_to_sharegpt import (
    DATASET_FAMILIES,
    build_convfinqa_item,
    build_fineval_item,
    build_finqa_item,
    build_fiqa_item,
    infer_dataset_family,
    load_source,
    to_text,
)


NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def mutate_numeric_answer(text: str) -> str:
    final_answer_match = re.search(r"(最终答案：\s*)(.+)", text)
    if final_answer_match:
        answer_text = final_answer_match.group(2)
        match = NUM_RE.search(answer_text)
        if match:
            token = match.group(0).replace(",", "")
            try:
                value = float(token)
                mutated = value + 1 if value >= 0 else value - 1
                replacement = str(int(mutated)) if mutated.is_integer() else f"{mutated:.4f}".rstrip("0").rstrip(".")
            except Exception:
                replacement = token + "1"
            new_answer = answer_text[:match.start()] + replacement + answer_text[match.end():]
            return text[:final_answer_match.start(2)] + new_answer + text[final_answer_match.end(2):]
        return text[:final_answer_match.end(1)] + "信息不足，暂不作答。"

    match = NUM_RE.search(text)
    if not match:
        return text + "\n最终答案：信息不足，暂不作答。"
    token = match.group(0).replace(",", "")
    try:
        value = float(token)
        mutated = value + 1 if value >= 0 else value - 1
        replacement = str(int(mutated)) if mutated.is_integer() else f"{mutated:.4f}".rstrip("0").rstrip(".")
    except Exception:
        replacement = token + "1"
    return text[:match.start()] + replacement + text[match.end():]


def remove_program_section(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("推理程序："):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def make_rejected(item: Dict[str, Any], family: str) -> str:
    chosen = item["conversations"][1]["value"]
    if family in {"convfinqa_turn", "finqa"}:
        rejected = remove_program_section(chosen)
        rejected = mutate_numeric_answer(rejected)
        if "推理程序：" not in rejected:
            rejected += "\n推理程序：未给出。"
        return rejected
    if family == "fineval":
        return "题目理解：这是一道金融考试题。\n推理：根据常识快速判断即可。\n最终答案：A"
    return "问题分析：这是一个金融问题。\n解释：信息有限，给出简短判断。\n结论：可能与市场波动有关。"


def normalize_item(rec: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    family = args.dataset_family
    if family == "auto":
        family = infer_dataset_family(rec)
    builder = {
        "convfinqa_turn": build_convfinqa_item,
        "finqa": build_finqa_item,
        "fineval": build_fineval_item,
        "fiqa_qa": build_fiqa_item,
    }[family]
    item = builder(rec, args)
    if item is None:
        return None
    question = item["conversations"][0]["value"]
    chosen = item["conversations"][1]["value"]
    rejected = make_rejected(item, family)
    return {
        "system": "",
        "history": [],
        "question": question,
        "response_chosen": chosen,
        "response_rejected": rejected,
        "source_dataset": item.get("source_dataset", family),
        "record_id": item.get("record_id", ""),
        "metadata": item.get("metadata", {}),
    }


def iter_records(ds: Dataset) -> Iterable[Dict[str, Any]]:
    for row in ds:
        yield dict(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--dataset_family", type=str, default="auto", choices=sorted(DATASET_FAMILIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history_turns", type=int, default=6)
    parser.add_argument("--max_context_items", type=int, default=6)
    parser.add_argument("--max_context_chars", type=int, default=400)
    parser.add_argument("--max_supporting_facts", type=int, default=6)
    parser.add_argument("--max_table_rows", type=int, default=20)
    parser.add_argument("--max_table_cols", type=int, default=12)
    parser.add_argument("--max_cell_chars", type=int, default=80)
    args = parser.parse_args()

    random.seed(args.seed)
    ds = load_source(args)
    rows = []
    skipped = 0
    for rec in iter_records(ds):
        item = normalize_item(rec, args)
        if item is None:
            skipped += 1
            continue
        rows.append(item)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output_file": str(out_path),
        "dataset_family": args.dataset_family,
        "saved_rows": len(rows),
        "skipped_rows": skipped,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

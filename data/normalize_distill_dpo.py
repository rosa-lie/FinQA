#!/usr/bin/env python3
"""Normalize and filter distill DPO jsonl data for training."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--drop_same_responses", action="store_true", default=True)
    parser.add_argument("--drop_missing_answer_tag", action="store_true", default=True)
    parser.add_argument("--min_question_chars", default=16, type=int)
    return parser.parse_args()


REQUIRED_KEYS = ["question", "response_chosen", "response_rejected"]


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def has_answer_tag(text: str) -> bool:
    return bool(ANSWER_RE.search(text or ""))


def has_nonempty_think(text: str) -> bool:
    match = THINK_RE.search(text or "")
    return bool(match and match.group(1).strip())


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    with input_file.open("r", encoding="utf-8") as rf, output_file.open("w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                stats["skip_blank_line"] += 1
                continue
            stats["raw_records"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["drop_bad_json"] += 1
                continue

            normalized: Dict[str, Any] = dict(obj)
            for key in REQUIRED_KEYS:
                normalized[key] = normalize_text(normalized.get(key))

            if any(not normalized[key] for key in REQUIRED_KEYS):
                stats["drop_missing_required_field"] += 1
                continue
            if len(normalized["question"]) < args.min_question_chars:
                stats["drop_short_question"] += 1
                continue
            if args.drop_missing_answer_tag and (
                not has_answer_tag(normalized["response_chosen"]) or not has_answer_tag(normalized["response_rejected"])
            ):
                stats["drop_missing_answer_tag"] += 1
                continue
            if args.drop_same_responses and normalized["response_chosen"] == normalized["response_rejected"]:
                stats["drop_same_responses"] += 1
                continue

            meta = normalized.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["distill_dpo_cleaning"] = {
                "chosen_has_nonempty_think": has_nonempty_think(normalized["response_chosen"]),
                "rejected_has_nonempty_think": has_nonempty_think(normalized["response_rejected"]),
            }
            normalized["metadata"] = meta

            wf.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            stats["kept_records"] += 1

    summary = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "stats": dict(stats),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

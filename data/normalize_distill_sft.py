#!/usr/bin/env python3
"""Normalize distill SFT jsonl into a training-friendly ShareGPT dataset."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


TAG_RE = {
    "think": re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL),
    "answer": re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument(
        "--assistant_mode",
        default="answer_only",
        choices=["answer_only", "answer_with_tag", "full"],
        help="How to rewrite assistant turns after parsing <think>/<answer>.",
    )
    parser.add_argument(
        "--drop_missing_answer",
        action="store_true",
        default=True,
        help="Drop samples whose assistant turn does not contain a usable <answer> block.",
    )
    parser.add_argument(
        "--keep_original_response",
        action="store_true",
        help="Store the original assistant text under metadata.distill_cleaning.original_response.",
    )
    return parser.parse_args()


def extract_tag(text: str, tag: str) -> str:
    match = TAG_RE[tag].search(text)
    if not match:
        return ""
    return match.group(1).strip()


def normalize_turn(turn: Dict[str, Any]) -> Dict[str, str] | None:
    if not isinstance(turn, dict):
        return None
    role = turn.get("from")
    value = turn.get("value")
    if role not in {"human", "gpt"} or not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return {"from": role, "value": value}


def rewrite_assistant(value: str, assistant_mode: str) -> Tuple[str, Dict[str, Any]]:
    think = extract_tag(value, "think")
    answer = extract_tag(value, "answer")
    info = {
        "has_think": bool(think),
        "has_answer": bool(answer),
        "original_len": len(value),
        "answer_len": len(answer),
    }
    if not answer:
        return "", info
    if assistant_mode == "answer_only":
        out = answer
    elif assistant_mode == "answer_with_tag":
        out = f"<answer>\n{answer}\n</answer>"
    else:
        out = value.strip()
    info["normalized_len"] = len(out)
    return out, info


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    with input_file.open("r", encoding="utf-8") as rf, output_file.open("w", encoding="utf-8") as wf:
        for line_no, line in enumerate(rf, 1):
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

            conv = obj.get("conversations")
            if not isinstance(conv, list) or not conv:
                stats["drop_missing_conversations"] += 1
                continue

            cleaned: List[Dict[str, str]] = []
            distill_info: Dict[str, Any] = {"assistant_mode": args.assistant_mode}
            bad_turn = False
            for idx, turn in enumerate(conv):
                normalized = normalize_turn(turn)
                if normalized is None:
                    stats["drop_invalid_turn"] += 1
                    bad_turn = True
                    break
                if normalized["from"] == "gpt":
                    rewritten, info = rewrite_assistant(normalized["value"], args.assistant_mode)
                    if not rewritten and args.drop_missing_answer:
                        stats["drop_missing_answer_tag"] += 1
                        bad_turn = True
                        break
                    if rewritten:
                        if args.keep_original_response:
                            info["original_response"] = normalized["value"]
                        normalized["value"] = rewritten
                        stats["rewritten_gpt_turns"] += 1
                    distill_info[f"assistant_turn_{idx}"] = info
                cleaned.append(normalized)
            if bad_turn:
                continue

            out = dict(obj)
            out["conversations"] = cleaned
            meta = out.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["distill_cleaning"] = distill_info
            out["metadata"] = meta
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["kept_records"] += 1

    summary = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "assistant_mode": args.assistant_mode,
        "stats": dict(stats),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

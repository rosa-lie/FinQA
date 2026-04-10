#!/usr/bin/env python3
"""Audit distill ShareGPT SFT data after normalization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


PAT_EVIDENCE_PLACEHOLDER = re.compile(r'"text_\d+"\s*:\s*"\s*1\s*\.?\s*"')
PAT_CONST = re.compile(r"\bconst_\d+\b")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--max_answer_chars", default=128, type=int)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_no"] = ln
            rows.append(obj)
    return rows


def safe_get_turn(obj: Dict[str, Any], idx: int) -> str:
    conv = obj.get("conversations", [])
    if not isinstance(conv, list) or len(conv) <= idx:
        return ""
    turn = conv[idx]
    if not isinstance(turn, dict):
        return ""
    val = turn.get("value")
    return val if isinstance(val, str) else ""


def short(text: str, n: int = 240) -> str:
    return text.replace("\n", " ").strip()[:n]


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalized_label(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_file)
    total = len(rows)
    stats = Counter()

    by_prompt: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for obj in rows:
        ln = obj["_line_no"]
        rid = str(obj.get("record_id", ""))
        human = safe_get_turn(obj, 0).strip()
        answer = extract_answer(safe_get_turn(obj, 1))
        by_prompt[human].append((ln, normalized_label(answer), rid))

    conflict_lines: Dict[int, Dict[str, Any]] = {}
    for prompt, items in by_prompt.items():
        uniq = sorted({label for _, label, _ in items if label})
        if len(uniq) > 1:
            related_lines = [ln for ln, _, _ in items]
            for ln, _, _ in items:
                conflict_lines[ln] = {
                    "conflict_group_size": len(items),
                    "conflict_unique_labels": len(uniq),
                    "conflict_related_lines": related_lines,
                    "conflict_record_ids": [rid for _, _, rid in items],
                    "conflict_prompt_md5": md5_text(prompt),
                    "conflict_prompt_head": short(prompt, 180),
                }

    review_rows: List[Dict[str, Any]] = []
    for obj in rows:
        ln = obj["_line_no"]
        rid = str(obj.get("record_id", ""))
        task_type = str(obj.get("task_type", ""))
        src = str(obj.get("source_dataset", ""))
        human = safe_get_turn(obj, 0).strip()
        gpt = safe_get_turn(obj, 1).strip()
        answer = extract_answer(gpt)

        reasons: List[str] = []
        severity = "medium"

        if not answer:
            reasons.append("empty_answer")
            severity = "high"
        if len(answer) > args.max_answer_chars:
            reasons.append("answer_too_long")
        if PAT_EVIDENCE_PLACEHOLDER.search(gpt):
            reasons.append("evidence_placeholder_text_1")
            severity = "high"
        if PAT_CONST.search(gpt):
            reasons.append("contains_const_placeholder")
            severity = "high"
        if ln in conflict_lines:
            reasons.append("same_prompt_conflicting_labels")
            severity = "high"

        if not reasons:
            continue

        stats["flagged_total"] += 1
        for reason in reasons:
            stats[f"reason::{reason}"] += 1
        stats[f"severity::{severity}"] += 1

        rec: Dict[str, Any] = {
            "line_no": ln,
            "record_id": rid,
            "task_type": task_type,
            "source_dataset": src,
            "severity": severity,
            "reasons": reasons,
            "human_len": len(human),
            "gpt_len": len(gpt),
            "answer_len": len(answer),
            "human_head": short(human),
            "gpt_head": short(gpt),
            "answer_head": short(answer, 120),
        }
        if ln in conflict_lines:
            rec.update(conflict_lines[ln])
        review_rows.append(rec)

    review_rows.sort(key=lambda x: (x["severity"] != "high", x["line_no"]))

    out_jsonl = output_dir / "dirty_samples_review.jsonl"
    out_csv = output_dir / "dirty_samples_review.csv"
    out_summary = output_dir / "dirty_samples_summary.json"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in review_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_fields = [
        "line_no",
        "record_id",
        "source_dataset",
        "task_type",
        "severity",
        "reasons",
        "human_len",
        "gpt_len",
        "answer_len",
        "human_head",
        "gpt_head",
        "answer_head",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in review_rows:
            flat = {k: rec.get(k, "") for k in csv_fields}
            if isinstance(flat.get("reasons"), list):
                flat["reasons"] = ";".join(flat["reasons"])
            writer.writerow(flat)

    summary = {
        "input_file": str(input_file),
        "total_records": total,
        "flagged_records": len(review_rows),
        "flagged_ratio": round((len(review_rows) / total) if total else 0.0, 6),
        "stats": dict(stats),
        "output_files": {
            "review_jsonl": str(out_jsonl),
            "review_csv": str(out_csv),
            "summary_json": str(out_summary),
        },
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

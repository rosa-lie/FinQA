#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FULL_TAGGED_RESPONSE_RE = re.compile(r"^\s*<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_raw_message_content(row: Dict[str, Any]) -> str:
    raw = row.get("raw_response") or {}
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def is_full_tagged_response(text: str) -> bool:
    return bool(FULL_TAGGED_RESPONSE_RE.match((text or "").strip()))


def select_clean_response(row: Dict[str, Any]) -> str:
    raw_content = extract_raw_message_content(row)
    if is_full_tagged_response(raw_content):
        return raw_content.strip()
    response = str(row.get("response") or "").strip()
    teacher_backend = str(row.get("teacher_backend") or "")
    if teacher_backend in {"gold", "copy_gold_final"} and is_full_tagged_response(response):
        return response
    return ""


def group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("source_dataset") or ""),
        str(row.get("task_name") or ""),
        str(row.get("record_id") or ""),
    )


def replace_think(text: str, new_think: str) -> str:
    if not text:
        return text
    match = THINK_TAG_RE.search(text)
    if not match:
        return text
    return text[:match.start(1)] + new_think + text[match.end(1):]


def replace_answer(text: str, new_answer: str) -> str:
    if not text:
        return text
    match = ANSWER_TAG_RE.search(text)
    if not match:
        return text
    return text[:match.start(1)] + new_answer + text[match.end(1):]


def perturb_number(token: str, factor: float) -> str:
    raw = token
    is_percent = raw.endswith("%")
    if raw.startswith("$"):
        raw = raw[1:]
    if is_percent:
        raw = raw[:-1]
    raw_clean = raw.replace(",", "")
    try:
        value = float(raw_clean)
    except ValueError:
        return token
    new_value = value * factor
    decimals = 0
    if "." in raw_clean:
        decimals = len(raw_clean.split(".", 1)[1])
    fmt = f"{{:.{decimals}f}}" if decimals > 0 else "{:.0f}"
    new_token = fmt.format(new_value)
    if is_percent:
        new_token += "%"
    return new_token


def perturb_answer_number(answer_text: str) -> Optional[str]:
    matches = list(NUMBER_RE.finditer(answer_text))
    if not matches:
        return None
    match = matches[-1]
    original = match.group(0)
    for factor in (1.01, 0.99, 1.005, 0.995):
        candidate = perturb_number(original, factor)
        if candidate != original:
            return answer_text[:match.start()] + candidate + answer_text[match.end():]
    return None


def build_synthetic_row(base: Dict[str, Any], response: str, synthetic_type: str, index: int) -> Dict[str, Any]:
    row = dict(base)
    row.update({
        "response": response,
        "raw_response": {},
        "teacher_backend": "copy_gold_final",
        "teacher_provider": "synthetic",
        "teacher_model": "synthetic",
        "teacher_temperature": 0.0,
        "candidate_index": f"synthetic_{synthetic_type}_{index}",
        "generation_key": f"synthetic_{row.get('record_id')}_{synthetic_type}_{index}",
    })
    metadata = dict(row.get("metadata") or {})
    metadata["synthetic_type"] = synthetic_type
    row["metadata"] = metadata
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic rejected candidates from scored audit rows.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--modes", type=str, default="answer_perturb,think_empty,think_program")
    parser.add_argument("--max_per_group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rows = load_jsonl(Path(args.input_file))

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    synthetic_rows: List[Dict[str, Any]] = []

    for group_index, (_, group) in enumerate(grouped.items()):
        # select best clean row by quality_score if available
        clean_rows = []
        for row in group:
            clean = row.get("clean_response")
            if not clean:
                clean = select_clean_response(row)
                if clean:
                    row["clean_response"] = clean
            if clean:
                clean_rows.append(row)
        if not clean_rows:
            continue
        chosen = sorted(clean_rows, key=lambda r: r.get("quality_score", 0.0), reverse=True)[0]
        base_response = str(chosen.get("clean_response") or "").strip()
        if not is_full_tagged_response(base_response):
            continue
        program_text = str(chosen.get("gold_program") or chosen.get("program") or "").strip()
        created = 0
        if modes:
            offset = group_index % len(modes)
            ordered_modes = modes[offset:] + modes[:offset]
        else:
            ordered_modes = []
        for mode in ordered_modes:
            if created >= args.max_per_group:
                break
            new_response = None
            if mode == "answer_perturb":
                answer_match = ANSWER_TAG_RE.search(base_response)
                if answer_match:
                    new_answer = perturb_answer_number(answer_match.group(1))
                    if new_answer:
                        new_response = replace_answer(base_response, new_answer)
            elif mode == "think_empty":
                new_response = replace_think(base_response, "")
            elif mode == "think_program":
                if program_text:
                    new_response = replace_think(base_response, program_text)
            if not new_response:
                continue
            if not is_full_tagged_response(new_response):
                continue
            synthetic_rows.append(build_synthetic_row(chosen, new_response, mode, created))
            created += 1

    save_jsonl(Path(args.output_file), synthetic_rows)
    print(json.dumps({"input_rows": len(rows), "output_rows": len(synthetic_rows), "modes": modes}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FINAL_ANSWER_RE = re.compile(r"(?:最终答案|答案|answer)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
PROGRAM_RE = re.compile(r"(?:推理程序|program)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
JSON_LIKE_RE = re.compile(r'\{\s*"(?:text|table|value|content|sentence|evidence|gold_ind|gold_inds)')
STRUCTURED_ANCHORS = ["问题分析：", "关键证据：", "推理程序：", "最终答案："]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_section(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text or "")
    if not match:
        return ""
    return match.group(1).strip().split("\n")[0].strip()


def normalize_program(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def normalize_answer_text(text: str) -> str:
    text = extract_section(text, FINAL_ANSWER_RE) or text
    text = (text or "").replace(" ", "").replace("，", ",").strip().lower()
    if text.startswith("$"):
        text = text[1:]
    return text


def parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    final_section = extract_section(text, FINAL_ANSWER_RE) or text
    matches = NUMBER_RE.findall(final_section.replace("，", ","))
    if not matches:
        return None
    token = matches[-1].replace(",", "").strip()
    is_percent = token.endswith("%")
    if token.startswith("$"):
        token = token[1:]
    if is_percent:
        token = token[:-1]
    try:
        number = float(token)
    except ValueError:
        return None
    return number / 100.0 if is_percent else number


def get_evidence_section(text: str) -> str:
    if "关键证据：" not in (text or ""):
        return ""
    tail = text.split("关键证据：", 1)[1]
    for marker in ["推理程序：", "最终答案："]:
        if marker in tail:
            tail = tail.split(marker, 1)[0]
    return tail.strip()


def json_like_hits(text: str) -> int:
    evidence = get_evidence_section(text)
    hits = len(JSON_LIKE_RE.findall(evidence or ""))
    hits += evidence.count('{"')
    return hits


def answer_correct(prediction: str, gold_answer: str, abs_tol: float, rel_tol: float) -> bool:
    pred_num = parse_number(prediction)
    gold_num = parse_number(gold_answer)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, abs_tol=abs_tol, rel_tol=rel_tol)
    return normalize_answer_text(prediction) == normalize_answer_text(gold_answer)


def score_candidate(row: Dict[str, Any], abs_tol: float, rel_tol: float, require_program_match_for_positive: bool) -> Dict[str, Any]:
    response = str(row.get("response") or "").strip()
    final_answer = extract_section(response, FINAL_ANSWER_RE)
    program_section = extract_section(response, PROGRAM_RE)
    structured = all(anchor in response for anchor in STRUCTURED_ANCHORS)
    ans_ok = answer_correct(response, str(row.get("gold_answer") or ""), abs_tol, rel_tol)
    gold_program = str(row.get("gold_program") or "").strip()
    program_ok = None
    if gold_program:
        program_ok = normalize_program(program_section) == normalize_program(gold_program)
    evidence_hits = json_like_hits(response)
    program_positive = (program_ok is True) if require_program_match_for_positive and gold_program else True
    eligible_sft = bool(structured and final_answer and ans_ok and evidence_hits == 0 and program_positive)

    quality_score = 0.0
    quality_score += 4.0 if ans_ok else 0.0
    quality_score += 1.5 if structured else 0.0
    quality_score += 0.5 if final_answer else 0.0
    quality_score += 1.0 if program_ok is True else 0.0
    quality_score -= min(2.0, evidence_hits * 0.5)
    quality_score -= max(0.0, len(response) - 1200) / 1200.0

    return {
        **row,
        "final_answer": final_answer,
        "program_section": program_section,
        "structured": structured,
        "answer_correct": ans_ok,
        "program_consistent": program_ok,
        "evidence_json_like_hits": evidence_hits,
        "response_chars": len(response),
        "eligible_sft": eligible_sft,
        "quality_score": round(quality_score, 6),
    }


def group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("source_dataset") or ""),
        str(row.get("task_name") or ""),
        str(row.get("record_id") or ""),
    )


def build_sft_row(best: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(best.get("metadata") or {})
    metadata.update({
        "teacher_model": best.get("teacher_model"),
        "teacher_backend": best.get("teacher_backend"),
        "teacher_candidate_index": best.get("candidate_index"),
        "distill_quality_score": best.get("quality_score"),
    })
    return {
        "source_dataset": best.get("source_dataset"),
        "task_type": f"distill_{best.get('task_name')}",
        "record_id": best.get("record_id"),
        "metadata": metadata,
        "conversations": [
            {"from": "human", "value": best.get("prompt", "")},
            {"from": "gpt", "value": best.get("response", "")},
        ],
    }


def build_dpo_row(chosen: Dict[str, Any], rejected: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(chosen.get("metadata") or {})
    metadata.update({
        "chosen_teacher_model": chosen.get("teacher_model"),
        "rejected_teacher_model": rejected.get("teacher_model"),
        "chosen_candidate_index": chosen.get("candidate_index"),
        "rejected_candidate_index": rejected.get("candidate_index"),
        "chosen_quality_score": chosen.get("quality_score"),
        "rejected_quality_score": rejected.get("quality_score"),
    })
    return {
        "system": "",
        "history": [],
        "question": chosen.get("prompt", ""),
        "response_chosen": chosen.get("response", ""),
        "response_rejected": rejected.get("response", ""),
        "source_dataset": chosen.get("source_dataset"),
        "record_id": chosen.get("record_id"),
        "metadata": metadata,
    }


def pick_best(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    positives = [row for row in rows if row.get("eligible_sft")]
    pool = positives or list(rows)
    if not pool:
        return None
    return sorted(pool, key=lambda row: (row.get("quality_score", 0.0), -(row.get("response_chars", 0))), reverse=True)[0]


def pick_rejected(rows: Sequence[Dict[str, Any]], chosen: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [
        row for row in rows
        if row.get("response") and row.get("response") != chosen.get("response")
    ]
    if not candidates:
        return None
    bad_first = sorted(
        candidates,
        key=lambda row: (
            row.get("answer_correct") is True,
            row.get("structured") is True,
            -(row.get("evidence_json_like_hits") or 0),
            row.get("quality_score", 0.0),
        ),
    )
    return bad_first[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score teacher candidates and emit training-ready SFT/DPO distill datasets.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--audit_output_file", type=str, required=True)
    parser.add_argument("--sft_output_file", type=str, required=True)
    parser.add_argument("--dpo_output_file", type=str, required=True)
    parser.add_argument("--summary_output_file", type=str, default="")
    parser.add_argument("--summary_csv_file", type=str, default="")
    parser.add_argument("--numeric_abs_tol", type=float, default=1e-4)
    parser.add_argument("--numeric_rel_tol", type=float, default=1e-4)
    parser.add_argument("--require_program_match_for_positive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_jsonl(Path(args.input_file))
    scored = [score_candidate(row, args.numeric_abs_tol, args.numeric_rel_tol, args.require_program_match_for_positive) for row in candidates]

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in scored:
        grouped.setdefault(group_key(row), []).append(row)

    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for key, rows in grouped.items():
        chosen = pick_best(rows)
        rejected = pick_rejected(rows, chosen) if chosen else None
        if chosen and chosen.get("eligible_sft"):
            sft_rows.append(build_sft_row(chosen))
        if chosen and rejected:
            dpo_rows.append(build_dpo_row(chosen, rejected))
        summary_rows.append({
            "source_dataset": key[0],
            "task_name": key[1],
            "record_id": key[2],
            "num_candidates": len(rows),
            "num_answer_correct": sum(1 for row in rows if row.get("answer_correct")),
            "num_structured": sum(1 for row in rows if row.get("structured")),
            "num_program_consistent": sum(1 for row in rows if row.get("program_consistent") is True),
            "num_eligible_sft": sum(1 for row in rows if row.get("eligible_sft")),
            "chosen_candidate_index": chosen.get("candidate_index") if chosen else None,
            "chosen_quality_score": chosen.get("quality_score") if chosen else None,
            "rejected_candidate_index": rejected.get("candidate_index") if rejected else None,
            "rejected_quality_score": rejected.get("quality_score") if rejected else None,
        })

    save_jsonl(Path(args.audit_output_file), scored)
    save_jsonl(Path(args.sft_output_file), sft_rows)
    save_jsonl(Path(args.dpo_output_file), dpo_rows)
    if args.summary_output_file:
        save_jsonl(Path(args.summary_output_file), summary_rows)
    if args.summary_csv_file:
        save_csv(Path(args.summary_csv_file), summary_rows)

    print(json.dumps({
        "input_rows": len(candidates),
        "audit_rows": len(scored),
        "grouped_records": len(grouped),
        "sft_rows": len(sft_rows),
        "dpo_rows": len(dpo_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

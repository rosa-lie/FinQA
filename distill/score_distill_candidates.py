#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)(?:\s*</answer>|$)", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"(?:最终答案|答案|answer)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
PROGRAM_RE = re.compile(r"(?:推理程序|program)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"<think>\s*(.*?)(?:\s*</think>|$)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
JSON_LIKE_RE = re.compile(r'\{\s*"(?:text|table|value|content|sentence|evidence|gold_ind|gold_inds)')
RUBRIC_FIELDS = [
    "internal_consistency",
    "instruction_alignment",
    "task_relevance",
    "logical_coherence",
    "evidence_quality",
    "reasoning_completeness",
    "content_diversity",
]
DEFAULT_JUDGE_PROMPT = Path("distill/prompts/financial_reasoning_judge.txt")


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


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)




def extract_answer_body(text: str) -> str:
    tag_match = ANSWER_TAG_RE.search(text or "")
    if tag_match:
        return tag_match.group(1).strip()
    return text or ""


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
    matches = NUMBER_RE.findall(extract_answer_body(final_section).replace("，", ","))
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


def program_consistent(program_section: str, gold_program: str) -> Optional[bool]:
    gold_program = (gold_program or "").strip()
    if not gold_program:
        return None
    return normalize_program(program_section) == normalize_program(gold_program)


def operator_sequence(program_text: str) -> List[str]:
    return re.findall(r"([A-Za-z_]+)\(", program_text or "")


def build_answer_check(row: Dict[str, Any], abs_tol: float, rel_tol: float) -> Dict[str, Any]:
    response = str(row.get("response") or "").strip()
    answer_text = extract_answer_body(response)
    final_answer = extract_section(answer_text, FINAL_ANSWER_RE) or answer_text.strip().split("\n")[0].strip()
    program_section = extract_section(response, PROGRAM_RE)
    has_think = bool(THINK_TAG_RE.search(response))
    has_answer = bool(ANSWER_TAG_RE.search(response))
    format_ok = bool(has_think and has_answer and final_answer)
    ans_ok = answer_correct(response, str(row.get("gold_answer") or ""), abs_tol, rel_tol)
    gold_program = str(row.get("gold_program") or "").strip()
    prog_ok = program_consistent(program_section, gold_program)
    evidence_hits = json_like_hits(response)
    operator_match = None
    if gold_program and program_section:
        operator_match = operator_sequence(program_section) == operator_sequence(gold_program)
    answer_check_pass = bool(format_ok and ans_ok and evidence_hits == 0)
    return {
        "final_answer": final_answer,
        "program_section": program_section,
        "has_think": has_think,
        "has_answer": has_answer,
        "format_ok": format_ok,
        "answer_correct": ans_ok,
        "program_consistent": prog_ok,
        "operator_sequence_match": operator_match,
        "evidence_json_like_hits": evidence_hits,
        "response_chars": len(response),
        "answer_check_pass": answer_check_pass,
    }


def heuristic_reasoning_scores(row: Dict[str, Any], answer_info: Dict[str, Any]) -> Dict[str, Any]:
    response = str(row.get("response") or "")
    think_match = THINK_TAG_RE.search(response)
    think_text = think_match.group(1).strip() if think_match else ""
    program_section = answer_info.get("program_section") or ""
    final_answer = answer_info.get("final_answer") or ""
    repeated_lines = len(set(response.splitlines())) < max(1, len(response.splitlines()) - 1)
    think_len = len(think_text)
    program_len = len(program_section)

    scores = {
        "internal_consistency": 2 if answer_info.get("answer_correct") else 1 if answer_info.get("format_ok") else 0,
        "instruction_alignment": 2 if answer_info.get("format_ok") and final_answer else 1 if answer_info.get("format_ok") else 0,
        "task_relevance": 2 if think_len >= 80 else 1 if think_len >= 20 else 0,
        "logical_coherence": 2 if answer_info.get("program_consistent") is True else 1 if think_len >= 80 or program_len > 0 else 0,
        "evidence_quality": 2 if answer_info.get("evidence_json_like_hits", 0) == 0 and think_len >= 20 else 1 if answer_info.get("evidence_json_like_hits", 0) == 0 else 0,
        "reasoning_completeness": 2 if think_len >= 80 and final_answer else 1 if final_answer else 0,
        "content_diversity": 2 if not repeated_lines and think_len >= 80 else 1 if think_len >= 20 else 0,
    }
    summary = "规则评分（轻量 <think>/<answer> 口径）"
    return {
        **scores,
        "summary": summary,
        "rubric_total": sum(scores.values()),
        "reasoning_selection_pass": bool(scores["instruction_alignment"] >= 1 and scores["task_relevance"] >= 1 and scores["evidence_quality"] >= 1 and sum(scores.values()) >= 8),
        "judge_backend": "heuristic",
    }


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def build_judge_user_prompt(row: Dict[str, Any]) -> str:
    return "\n\n".join([
        f"[PROMPT]\n{row.get('prompt', '')}",
        f"[GOLD_ANSWER]\n{row.get('gold_answer', '')}",
        f"[GOLD_PROGRAM]\n{row.get('gold_program', '')}",
        f"[CANDIDATE_RESPONSE]\n{row.get('response', '')}",
    ])


def create_judge_client(args: argparse.Namespace):
    from role_play_data.llm_client import create_llm_client
    client, model_name = create_llm_client(
        provider=args.judge_provider,
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        model=args.judge_model,
    )
    return client, model_name


def llm_reasoning_scores(row: Dict[str, Any], args: argparse.Namespace, judge_prompt: str, client_bundle: Tuple[Any, str]) -> Dict[str, Any]:
    client, model_name = client_bundle
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": build_judge_user_prompt(row)},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.judge_max_tokens,
    )
    content = response.choices[0].message.content or ""
    payload = parse_json_object(content) or {}
    scores: Dict[str, Any] = {}
    total = 0
    for key in RUBRIC_FIELDS:
        value = payload.get(key, 0)
        try:
            ivalue = max(0, min(2, int(value)))
        except Exception:
            ivalue = 0
        scores[key] = ivalue
        total += ivalue
    summary = str(payload.get("summary") or "")[:200]
    scores.update({
        "summary": summary,
        "rubric_total": total,
        "reasoning_selection_pass": bool(total >= args.reasoning_total_threshold and scores["instruction_alignment"] >= args.reasoning_min_instruction_alignment and scores["task_relevance"] >= args.reasoning_min_task_relevance and scores["logical_coherence"] >= args.reasoning_min_logical_coherence and scores["evidence_quality"] >= args.reasoning_min_evidence_quality),
        "judge_backend": "llm",
        "judge_model": model_name,
        "judge_raw_response": content,
    })
    return scores


def reasoning_selection(row: Dict[str, Any], answer_info: Dict[str, Any], args: argparse.Namespace, judge_prompt: str, client_bundle: Optional[Tuple[Any, str]]) -> Dict[str, Any]:
    merged = {**row, **answer_info}
    if args.enable_reasoning_judge:
        if client_bundle is None:
            raise ValueError("Reasoning judge enabled but judge client is unavailable.")
        return llm_reasoning_scores(merged, args, judge_prompt, client_bundle)
    return heuristic_reasoning_scores(merged, answer_info)


def final_quality(answer_info: Dict[str, Any], reasoning_info: Dict[str, Any]) -> float:
    score = 0.0
    score += 4.0 if answer_info.get("answer_correct") else 0.0
    score += 1.5 if answer_info.get("format_ok") else 0.0
    score += 0.5 if answer_info.get("final_answer") else 0.0
    score += 1.0 if answer_info.get("program_consistent") is True else 0.0
    score -= min(2.0, (answer_info.get("evidence_json_like_hits") or 0) * 0.5)
    score += float(reasoning_info.get("rubric_total") or 0) / 2.0
    score -= max(0.0, (answer_info.get("response_chars") or 0) - 1400) / 1400.0
    return round(score, 6)


def score_candidate(row: Dict[str, Any], args: argparse.Namespace, judge_prompt: str, client_bundle: Optional[Tuple[Any, str]]) -> Dict[str, Any]:
    answer_info = build_answer_check(row, args.numeric_abs_tol, args.numeric_rel_tol)
    if answer_info["answer_check_pass"] or args.run_reasoning_selection_on_failed_answer_check:
        reasoning_info = reasoning_selection(row, answer_info, args, judge_prompt, client_bundle)
    else:
        reasoning_info = {key: 0 for key in RUBRIC_FIELDS}
        reasoning_info.update({
            "summary": "跳过 reasoning selection：answer check 未通过",
            "rubric_total": 0,
            "reasoning_selection_pass": False,
            "judge_backend": "skipped",
        })
    eligible_sft = bool(answer_info["answer_check_pass"] and reasoning_info["reasoning_selection_pass"] and (answer_info.get("program_consistent") is True if args.require_program_match_for_positive and row.get("gold_program") else True))
    return {
        **row,
        **answer_info,
        **reasoning_info,
        "eligible_sft": eligible_sft,
        "quality_score": final_quality(answer_info, reasoning_info),
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
        "answer_check_pass": best.get("answer_check_pass"),
        "reasoning_selection_pass": best.get("reasoning_selection_pass"),
        "judge_backend": best.get("judge_backend"),
        "rubric_total": best.get("rubric_total"),
        "judge_scores": {key: best.get(key) for key in RUBRIC_FIELDS},
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
        "chosen_answer_check_pass": chosen.get("answer_check_pass"),
        "rejected_answer_check_pass": rejected.get("answer_check_pass"),
        "chosen_reasoning_selection_pass": chosen.get("reasoning_selection_pass"),
        "rejected_reasoning_selection_pass": rejected.get("reasoning_selection_pass"),
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
    return sorted(pool, key=lambda row: (row.get("quality_score", 0.0), -int(not row.get("answer_check_pass", False)), -int(not row.get("reasoning_selection_pass", False))), reverse=True)[0]


def pick_rejected(rows: Sequence[Dict[str, Any]], chosen: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if row.get("response") and row.get("response") != chosen.get("response")]
    if not candidates:
        return None
    bad_first = sorted(candidates, key=lambda row: (
        row.get("answer_check_pass") is True,
        row.get("reasoning_selection_pass") is True,
        row.get("answer_correct") is True,
        row.get("quality_score", 0.0),
    ))
    return bad_first[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage scoring for financial distillation candidates.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--audit_output_file", type=str, required=True)
    parser.add_argument("--sft_output_file", type=str, required=True)
    parser.add_argument("--dpo_output_file", type=str, required=True)
    parser.add_argument("--summary_output_file", type=str, default="")
    parser.add_argument("--summary_csv_file", type=str, default="")
    parser.add_argument("--numeric_abs_tol", type=float, default=1e-4)
    parser.add_argument("--numeric_rel_tol", type=float, default=1e-4)
    parser.add_argument("--require_program_match_for_positive", action="store_true")
    parser.add_argument("--run_reasoning_selection_on_failed_answer_check", action="store_true")
    parser.add_argument("--enable_reasoning_judge", action="store_true")
    parser.add_argument("--judge_provider", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_base_url", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--judge_prompt_file", type=str, default=str(DEFAULT_JUDGE_PROMPT))
    parser.add_argument("--judge_max_tokens", type=int, default=256)
    parser.add_argument("--reasoning_total_threshold", type=int, default=8)
    parser.add_argument("--reasoning_min_instruction_alignment", type=int, default=1)
    parser.add_argument("--reasoning_min_task_relevance", type=int, default=1)
    parser.add_argument("--reasoning_min_logical_coherence", type=int, default=1)
    parser.add_argument("--reasoning_min_evidence_quality", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_jsonl(Path(args.input_file))
    judge_prompt = Path(args.judge_prompt_file).read_text(encoding="utf-8").strip() if args.judge_prompt_file else ""
    client_bundle = create_judge_client(args) if args.enable_reasoning_judge else None
    scored = [score_candidate(row, args, judge_prompt, client_bundle) for row in candidates]

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in scored:
        grouped.setdefault(group_key(row), []).append(row)

    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    answer_check_pass_count = 0
    reasoning_pass_count = 0

    for key, rows in grouped.items():
        chosen = pick_best(rows)
        rejected = pick_rejected(rows, chosen) if chosen else None
        answer_check_pass_count += sum(1 for row in rows if row.get("answer_check_pass"))
        reasoning_pass_count += sum(1 for row in rows if row.get("reasoning_selection_pass"))
        if chosen and chosen.get("eligible_sft"):
            sft_rows.append(build_sft_row(chosen))
        if chosen and rejected:
            dpo_rows.append(build_dpo_row(chosen, rejected))
        summary_row = {
            "source_dataset": key[0],
            "task_name": key[1],
            "record_id": key[2],
            "num_candidates": len(rows),
            "num_answer_check_pass": sum(1 for row in rows if row.get("answer_check_pass")),
            "num_reasoning_selection_pass": sum(1 for row in rows if row.get("reasoning_selection_pass")),
            "num_answer_correct": sum(1 for row in rows if row.get("answer_correct")),
            "num_program_consistent": sum(1 for row in rows if row.get("program_consistent") is True),
            "num_eligible_sft": sum(1 for row in rows if row.get("eligible_sft")),
            "chosen_candidate_index": chosen.get("candidate_index") if chosen else None,
            "chosen_quality_score": chosen.get("quality_score") if chosen else None,
            "chosen_answer_check_pass": chosen.get("answer_check_pass") if chosen else None,
            "chosen_reasoning_selection_pass": chosen.get("reasoning_selection_pass") if chosen else None,
            "rejected_candidate_index": rejected.get("candidate_index") if rejected else None,
            "rejected_quality_score": rejected.get("quality_score") if rejected else None,
        }
        for field in RUBRIC_FIELDS:
            summary_row[f"chosen_{field}"] = chosen.get(field) if chosen else None
        summary_rows.append(summary_row)

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
        "answer_check_pass_rows": answer_check_pass_count,
        "reasoning_selection_pass_rows": reasoning_pass_count,
        "sft_rows": len(sft_rows),
        "dpo_rows": len(dpo_rows),
        "reasoning_judge_enabled": args.enable_reasoning_judge,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

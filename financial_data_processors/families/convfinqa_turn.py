#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..common import (
    build_context_sections,
    extract_numeric_text,
    safe_jsonable,
    summarize_evidence_blocks,
    summarize_history_questions,
    to_text,
)

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def build_sft_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else {}
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}

    history = rec.get("cur_dial") if isinstance(rec.get("cur_dial"), list) else []
    if not history:
        history = annotation.get("cur_dial") if isinstance(annotation.get("cur_dial"), list) else []

    current_question = to_text(qa.get("question") or (history[-1] if history else None))
    if not current_question:
        return None

    prior_questions = [to_text(x) for x in history[:-1] if to_text(x)]
    if not prior_questions:
        dialogue_break = annotation.get("dialogue_break") if isinstance(annotation.get("dialogue_break"), list) else []
        turn_ind = rec.get("turn_ind")
        if turn_ind is None:
            turn_ind = annotation.get("turn_ind")
        if dialogue_break and isinstance(turn_ind, int):
            prior_questions = [to_text(x) for x in dialogue_break[:turn_ind] if to_text(x)]
    prior_questions = summarize_history_questions(prior_questions, args.max_history_turns, args.max_context_chars)

    program = to_text(
        rec.get("cur_program")
        or qa.get("program_re")
        or qa.get("program")
        or annotation.get("cur_program")
        or annotation.get("original_program")
        or rec.get("program")
        or rec.get("program_re")
    )
    final_answer = to_text(
        qa.get("answer")
        or qa.get("exe_ans")
        or rec.get("exe_ans")
        or rec.get("answer")
    )
    if not final_answer:
        return None

    gold_ind = qa.get("gold_inds") or rec.get("gold_ind") or rec.get("gold_inds") or annotation.get("gold_ind") or []
    gold_facts = summarize_evidence_blocks(gold_ind, args.max_supporting_facts, args.max_context_chars)

    prompt_parts = [
        "你是一名金融数值推理助手。请结合对话历史、文本材料和表格，进行分步推理并给出最终数值答案。",
    ]
    if prior_questions:
        prompt_parts.append("历史对话：\n" + "\n".join(f"- {q}" for q in prior_questions))
    prompt_parts.extend(build_context_sections(rec, args))
    prompt_parts.append(f"当前问题：{current_question}")
    prompt_parts.append("请按以下结构作答：\n问题分析：...\n关键证据：\n- ...\n推理程序：...\n最终答案：...")

    answer_lines = [
        "问题分析：需要结合当前问题、历史对话和财务材料定位指标后再做数值计算。",
        "关键证据：",
    ]
    if gold_facts:
        answer_lines.extend([f"- {fact}" for fact in gold_facts])
    else:
        answer_lines.append("- 需要结合历史追问和当前表格中的相关财务指标完成计算。")
    answer_lines.append(f"推理程序：{program or '请依据材料逐步完成数值计算。'}")
    answer_lines.append(f"最终答案：{extract_numeric_text(final_answer)}")

    return {
        "source_dataset": "ConvFinQA",
        "task_type": "financial_conversational_numerical_reasoning",
        "record_id": to_text(rec.get("id")),
        "metadata": {
            "turn_ind": rec.get("turn_ind", annotation.get("turn_ind")),
            "cur_type": to_text(rec.get("cur_type") or annotation.get("cur_type")),
            "program": program,
            "gold_ind": safe_jsonable(gold_ind),
        },
        "conversations": [
            {"from": "human", "value": "\n\n".join(prompt_parts)},
            {"from": "gpt", "value": "\n".join(answer_lines)},
        ],
    }


def _parse_id_suffix(rec_id: str) -> Optional[int]:
    if not rec_id or "_" not in rec_id:
        return None
    tail = rec_id.rsplit("_", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def _turn_rank(rec: Dict[str, Any]) -> Tuple[int, int]:
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}
    turn_ind = rec.get("turn_ind")
    if not isinstance(turn_ind, int):
        turn_ind = annotation.get("turn_ind")
    if isinstance(turn_ind, int):
        return turn_ind, 1

    rec_id = to_text(rec.get("id"))
    suffix = _parse_id_suffix(rec_id)
    if suffix is not None:
        return suffix, 0
    return -1, -1


def _group_key(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else {}
    filename = to_text(rec.get("filename"))
    question = to_text(qa.get("question"))
    program = to_text(qa.get("program") or qa.get("program_re"))
    answer = to_text(qa.get("answer") if qa.get("answer") is not None else qa.get("exe_ans"))
    return (filename, question, program, answer)


def dedupe_final_turn(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    order: List[Tuple[str, str, str, str]] = []
    best_by_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    rank_by_key: Dict[Tuple[str, str, str, str], Tuple[int, int]] = {}

    for rec in records:
        key = _group_key(rec)
        rank = _turn_rank(rec)
        if key not in best_by_key:
            order.append(key)
            best_by_key[key] = rec
            rank_by_key[key] = rank
            continue
        if rank > rank_by_key[key]:
            best_by_key[key] = rec
            rank_by_key[key] = rank

    selected = [best_by_key[k] for k in order]
    fallback_selected_rows = sum(1 for k in order if rank_by_key[k][1] < 1)
    stats = {
        "group_count": len(order),
        "dedup_dropped_rows": max(0, len(records) - len(selected)),
        "fallback_selected_rows": fallback_selected_rows,
    }
    return selected, stats


def _mutate_numeric_answer(text: str) -> str:
    final_answer_match = re.search(r"(最终答案：\s*)(.+)", text)
    if final_answer_match:
        answer_text = final_answer_match.group(2)
        match = NUM_RE.search(answer_text)
        if match:
            token = match.group(0).replace(",", "")
            try:
                value = float(token)
                step = max(abs(value) * 0.12, 1.0)
                mutated = value + step if value >= 0 else value - step
                replacement = str(int(mutated)) if float(mutated).is_integer() else f"{mutated:.4f}".rstrip("0").rstrip(".")
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
        step = max(abs(value) * 0.12, 1.0)
        mutated = value + step if value >= 0 else value - step
        replacement = str(int(mutated)) if float(mutated).is_integer() else f"{mutated:.4f}".rstrip("0").rstrip(".")
    except Exception:
        replacement = token + "1"
    return text[:match.start()] + replacement + text[match.end():]


def _mutate_program_expr(program: str) -> str:
    p = program
    substitutions = [
        ("divide", "multiply"),
        ("subtract", "add"),
        ("减去", "加上"),
        ("同比", "环比"),
    ]
    for src, dst in substitutions:
        if src in p:
            p = p.replace(src, dst, 1)
            break

    fn_call = re.search(r"([A-Za-z_]+)\(([^,()]+),\s*([^)]+)\)", p)
    if fn_call and fn_call.group(1).lower() in {"subtract", "divide", "add", "multiply"}:
        fn, a, b = fn_call.group(1), fn_call.group(2).strip(), fn_call.group(3).strip()
        p = p[:fn_call.start()] + f"{fn}({b}, {a})" + p[fn_call.end():]

    if p == program:
        p = program + "；并忽略上轮筛选条件后再计算。"
    return p


def _mutate_program_section(text: str) -> str:
    lines = text.splitlines()
    out = []
    changed = False
    for line in lines:
        if line.startswith("推理程序："):
            prog = line[len("推理程序："):].strip()
            out.append("推理程序：" + _mutate_program_expr(prog))
            changed = True
        else:
            out.append(line)
    if not changed:
        out.append("推理程序：采用相邻年份近似估算，并忽略历史追问中的筛选条件。")
    return "\n".join(out)


def _mutate_evidence_section(text: str) -> str:
    lines = text.splitlines()
    try:
        start = lines.index("关键证据：")
    except ValueError:
        return text

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("推理程序：") or lines[i].startswith("最终答案："):
            end = i
            break

    mutated = ["- 误把上轮问题中的参考值当作本轮计算依据。"]
    return "\n".join(lines[: start + 1] + mutated + lines[end:])


def _mutate_analysis_section(text: str) -> str:
    lines = []
    changed = False
    for line in text.splitlines():
        if line.startswith("问题分析："):
            lines.append("问题分析：沿用上轮口径近似估算，并默认本轮筛选条件没有变化。")
            changed = True
        else:
            lines.append(line)
    if not changed:
        lines.insert(0, "问题分析：沿用上轮口径近似估算，并默认筛选条件不变。")
    return "\n".join(lines)


def _build_high_confusion_rejected(chosen: str) -> str:
    rejected = chosen
    rejected = _mutate_analysis_section(rejected)
    rejected = _mutate_evidence_section(rejected)
    rejected = _mutate_program_section(rejected)
    rejected = _mutate_numeric_answer(rejected)
    return rejected


def build_dpo_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    item = build_sft_item(rec, args)
    if item is None:
        return None

    chosen = item["conversations"][1]["value"]
    rejected = _build_high_confusion_rejected(chosen)

    return {
        "system": "",
        "history": [],
        "question": item["conversations"][0]["value"],
        "response_chosen": chosen,
        "response_rejected": rejected,
        "source_dataset": item.get("source_dataset", "convfinqa_turn"),
        "record_id": item.get("record_id", ""),
        "metadata": item.get("metadata", {}),
    }

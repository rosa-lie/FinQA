#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..common import (
    build_context_sections,
    extract_numeric_text,
    safe_jsonable,
    summarize_evidence_blocks,
    to_text,
)

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def build_sft_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else rec
    question = to_text(qa.get("question") or rec.get("question"))
    if not question:
        return None

    program = to_text(qa.get("program_re") or qa.get("program") or rec.get("program_re") or rec.get("program"))
    final_answer = to_text(qa.get("exe_ans") or rec.get("exe_ans") or qa.get("answer") or rec.get("answer"))
    if not final_answer:
        return None

    gold_inds = qa.get("gold_inds") or rec.get("gold_inds") or []
    supporting = summarize_evidence_blocks(gold_inds, args.max_supporting_facts, args.max_context_chars)

    prompt_parts = [
        "你是一名金融表文混合推理助手。请结合文本、表格和问题，给出可执行的推理程序与最终答案。",
    ]
    prompt_parts.extend(build_context_sections(rec, args))
    prompt_parts.append(f"问题：{question}")
    prompt_parts.append("请按以下结构作答：\n问题分析：...\n关键证据：\n- ...\n推理程序：...\n最终答案：...")

    answer_lines = [
        "问题分析：先定位题目涉及的财务指标，再根据给定程序完成数值计算。",
        "关键证据：",
    ]
    if supporting:
        answer_lines.extend([f"- {fact}" for fact in supporting])
    else:
        answer_lines.append("- 需要从题目相关的表格行、文本说明和财务指标中提取关键数值。")
    answer_lines.append(f"推理程序：{program or '请根据题意构造数值推理程序。'}")
    answer_lines.append(f"最终答案：{extract_numeric_text(final_answer)}")

    return {
        "source_dataset": "FinQA",
        "task_type": "financial_table_text_reasoning",
        "record_id": to_text(rec.get("id")),
        "metadata": {
            "program": program,
            "gold_inds": safe_jsonable(gold_inds),
        },
        "conversations": [
            {"from": "human", "value": "\n\n".join(prompt_parts)},
            {"from": "gpt", "value": "\n".join(answer_lines)},
        ],
    }


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
        p = program + "；并将关键取值替换为相邻年份口径。"
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
        out.append("推理程序：采用相邻年份近似估算，并把除法替换为乘法。")
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

    mutated_evidence = ["- 误将相邻年度和相邻财务科目作为核心证据。"]
    return "\n".join(lines[: start + 1] + mutated_evidence + lines[end:])


def _mutate_analysis_section(text: str) -> str:
    lines = []
    changed = False
    for line in text.splitlines():
        if line.startswith("问题分析："):
            lines.append("问题分析：为快速得到结果，按近似口径估算并默认关键条件不变。")
            changed = True
        else:
            lines.append(line)
    if not changed:
        lines.insert(0, "问题分析：按近似口径估算，默认关键条件不变。")
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
        "source_dataset": item.get("source_dataset", "finqa"),
        "record_id": item.get("record_id", ""),
        "metadata": item.get("metadata", {}),
    }

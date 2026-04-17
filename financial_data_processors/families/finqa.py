#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from ..common import (
    build_english_context_sections,
    build_reasoning_supervision,
    build_rejected_from_strict_response,
    finalize_prompt_audits,
    quality_tier_allowed,
    render_strict_target,
    safe_jsonable,
    to_text,
)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and to_text(value):
            return value
    return None


def maybe_rewrite_finqa_question(question: str, norm: Dict[str, Any]) -> str:
    evidence_text = " ".join(to_text(item.get("rendered_text")) for item in norm.get("aligned_evidence") or []).lower()
    q = to_text(question).lower()
    if "basis points" in evidence_text and "interest expense" in evidence_text and "libor" in evidence_text:
        if "change" not in q and "basis" not in q:
            return "What would be the annual interest expense change if LIBOR changes by 100 basis points?"
    return ""


def build_prompt(rec: Dict[str, Any], question: str, args: Any) -> str:
    if getattr(args, "sft_variant", "benchmark_sft") == "dual_answer_sft":
        prompt_parts = [
            "You are a financial table-and-text reasoning assistant.",
            f"Current question:\n{question}",
            "Output format:\n"
            "Evidence:\n"
            "- ...\n\n"
            "Program: ...\n"
            "Answer: ...\n"
            "Normalized Answer: ...",
            "Normalization rule:\n"
            "- For percentage questions, Normalized Answer must be a decimal ratio.\n"
            "- Answer may use natural units such as %, $, million, billion.",
        ]
        context_sections = build_english_context_sections(rec, args)
        if context_sections:
            prompt_parts.append("Report context:\n" + "\n\n".join(context_sections))
        return "\n\n".join(prompt_parts)

    prompt_parts = [
        "You are a financial table-and-text reasoning assistant.",
        "Use the report context to identify the supporting evidence, produce the executable program, and give the final answer.",
    ]
    prompt_parts.extend(build_english_context_sections(rec, args))
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append(
        "Respond exactly in this format:\n"
        "Evidence:\n"
        "- ...\n\n"
        "Program: ...\n"
        "Answer: ..."
    )
    return "\n\n".join(prompt_parts)


def normalize_record(rec: Dict[str, Any], args: Any) -> Dict[str, Any]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else rec
    question = to_text(qa.get("question") or rec.get("question"))
    program_re = _first_present(qa.get("program_re"), rec.get("program_re"))
    raw_answer = _first_present(qa.get("answer"), rec.get("answer"))
    exe_ans = qa.get("exe_ans") if qa.get("exe_ans") is not None else rec.get("exe_ans")
    gold_inds = qa.get("gold_inds") or rec.get("gold_inds") or {}

    norm = build_reasoning_supervision(
        rec,
        family="finqa",
        source_dataset="FinQA",
        task_type="financial_table_text_reasoning",
        record_id=to_text(rec.get("id")),
        question=question,
        program_re=program_re,
        raw_answer=raw_answer,
        exe_ans=exe_ans,
        gold_evidence=gold_inds,
        args=args,
        extra_metadata={
            "program": to_text(qa.get("program") or rec.get("program")),
            "program_re": to_text(program_re),
            "gold_inds": safe_jsonable(gold_inds),
            "answer": safe_jsonable(raw_answer),
            "exe_ans": safe_jsonable(exe_ans),
        },
    )
    question_rewritten = maybe_rewrite_finqa_question(question, norm)
    norm["question_raw"] = question
    norm["question_rewritten"] = question_rewritten
    prompt_question = question_rewritten or question
    norm["prompt"] = build_prompt(rec, prompt_question, args) if prompt_question else ""
    if norm["prompt"]:
        finalize_prompt_audits(norm, norm["prompt"])
    return norm


def render_sft_item(norm: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    if not norm.get("strict_ok") or not quality_tier_allowed(norm, args):
        return None
    prompt = to_text(norm.get("prompt"))
    if not prompt:
        return None
    return {
        "source_dataset": norm.get("source_dataset", "FinQA"),
        "task_type": norm.get("task_type", "financial_table_text_reasoning"),
        "record_id": norm.get("record_id", ""),
        "metadata": {
            "program_raw": norm.get("program_raw", ""),
            "program_canonical": norm.get("program_canonical", ""),
            "program_executable": norm.get("program_executable"),
            "answer_raw": norm.get("answer_raw", ""),
            "answer_exe": norm.get("answer_exe"),
            "answer_norm": norm.get("answer_norm", ""),
            "answer_display": norm.get("answer_display", ""),
            "answer_unit": norm.get("answer_unit", ""),
            "answer_scale": norm.get("answer_scale", ""),
            "answer_source": norm.get("answer_source", ""),
            "answer_matches_program": norm.get("answer_matches_program", False),
            "aligned_evidence": norm.get("aligned_evidence", []),
            "evidence_match_type": norm.get("evidence_match_type", ""),
            "audit_flags": norm.get("audit_flags", []),
            "semantic_audit_flags": norm.get("semantic_audit_flags", []),
            "quality_tier": norm.get("quality_tier", ""),
            "evidence_visible_in_prompt": norm.get("evidence_visible_in_prompt"),
            "table_evidence_column_pruned": norm.get("table_evidence_column_pruned", False),
            "question_raw": norm.get("question_raw", ""),
            "question_rewritten": norm.get("question_rewritten", ""),
            "raw_metadata": norm.get("metadata", {}),
        },
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": render_strict_target(norm, getattr(args, "sft_variant", "benchmark_sft"))},
        ],
    }


def build_sft_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    return render_sft_item(normalize_record(rec, args), args)


def build_dpo_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    item = build_sft_item(rec, args)
    if item is None:
        return None

    chosen = item["conversations"][1]["value"]
    rejected = build_rejected_from_strict_response(chosen)
    return {
        "system": "",
        "history": [],
        "question": item["conversations"][0]["value"],
        "response_chosen": chosen,
        "response_rejected": rejected,
        "source_dataset": item.get("source_dataset", "FinQA"),
        "record_id": item.get("record_id", ""),
        "metadata": item.get("metadata", {}),
    }

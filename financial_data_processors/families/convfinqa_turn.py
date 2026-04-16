#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..common import (
    build_english_context_sections,
    build_reasoning_supervision,
    build_rejected_from_strict_response,
    add_audit_flags,
    canonicalize_program_re,
    clean_question_text,
    choose_answer_display,
    choose_answer_norm,
    execute_program,
    finalize_prompt_audits,
    quality_tier_allowed,
    render_strict_target,
    normalize_question_for_comparison,
    safe_jsonable,
    summarize_history_questions,
    to_text,
)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and to_text(value):
            return value
    return None


def _parse_id_suffix(rec_id: str) -> Optional[int]:
    if not rec_id or "_" not in rec_id:
        return None
    tail = rec_id.rsplit("_", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def _conversation_id(rec: Dict[str, Any]) -> str:
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}
    for key in ["conversation_id", "dialogue_id", "dialog_id", "id"]:
        value = to_text(rec.get(key) or annotation.get(key))
        if value:
            return value.rsplit("_", 1)[0] if _parse_id_suffix(value) is not None else value
    filename = to_text(rec.get("filename") or annotation.get("filename"))
    return filename or to_text(rec.get("id"))


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


def _annotation_current_question(annotation: Dict[str, Any]) -> str:
    history = annotation.get("cur_dial") if isinstance(annotation.get("cur_dial"), list) else []
    if history:
        return clean_question_text(history[-1])
    dialogue_break = annotation.get("dialogue_break") if isinstance(annotation.get("dialogue_break"), list) else []
    turn_ind = annotation.get("turn_ind")
    if dialogue_break and isinstance(turn_ind, int) and 0 <= turn_ind < len(dialogue_break):
        return clean_question_text(dialogue_break[turn_ind])
    return ""


def _extract_history(rec: Dict[str, Any]) -> Tuple[str, List[str]]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else {}
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}

    history = annotation.get("cur_dial") if isinstance(annotation.get("cur_dial"), list) else []
    if not history:
        history = rec.get("cur_dial") if isinstance(rec.get("cur_dial"), list) else []

    current_question = clean_question_text(history[-1]) if history else ""
    prior_questions = [clean_question_text(x) for x in history[:-1] if clean_question_text(x)]

    dialogue_break = annotation.get("dialogue_break") if isinstance(annotation.get("dialogue_break"), list) else []
    turn_ind = rec.get("turn_ind")
    if turn_ind is None:
        turn_ind = annotation.get("turn_ind")
    if dialogue_break and isinstance(turn_ind, int):
        if not current_question and 0 <= turn_ind < len(dialogue_break):
            current_question = clean_question_text(dialogue_break[turn_ind])
        if not prior_questions:
            prior_questions = [clean_question_text(x) for x in dialogue_break[:turn_ind] if clean_question_text(x)]

    if not current_question:
        current_question = clean_question_text(rec.get("question") or qa.get("question"))
    return current_question, prior_questions


def _reasoning_inputs(rec: Dict[str, Any]) -> Tuple[str, Any, Any, Any, Any]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else {}
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}
    question, _ = _extract_history(rec)

    program_re = _first_present(annotation.get("cur_program"), rec.get("cur_program"), rec.get("program_re"), qa.get("program_re"))
    exe_ans = annotation.get("exe_ans")
    if exe_ans is None:
        exe_ans = rec.get("exe_ans")
    if exe_ans is None:
        exe_ans = qa.get("exe_ans")

    turn_ind = rec.get("turn_ind")
    if turn_ind is None:
        turn_ind = annotation.get("turn_ind")
    answer_list = annotation.get("answer_list") if isinstance(annotation.get("answer_list"), list) else []
    raw_answer = answer_list[turn_ind] if isinstance(turn_ind, int) and 0 <= turn_ind < len(answer_list) else None
    if to_text(raw_answer).upper().startswith("A") or raw_answer is None:
        raw_answer = exe_ans
    if raw_answer is None:
        raw_answer = _first_present(rec.get("answer"), qa.get("answer"))
    gold_ind = annotation.get("gold_ind") or rec.get("gold_ind") or rec.get("gold_inds") or qa.get("gold_inds") or {}
    return question, program_re, raw_answer, exe_ans, gold_ind


def _history_answer(rec: Dict[str, Any], sft_variant: str) -> str:
    question, program_re, raw_answer, exe_ans, _ = _reasoning_inputs(rec)
    program_value, _ = execute_program(canonicalize_program_re(program_re))
    answer_norm, _, _ = choose_answer_norm(raw_answer, exe_ans, program_value)
    answer_display = choose_answer_display(raw_answer, answer_norm, question)
    return answer_display if sft_variant == "assistant_sft" else answer_norm


def _raw_metadata(rec: Dict[str, Any], program_re: Any, raw_answer: Any, exe_ans: Any, gold_ind: Any) -> Dict[str, Any]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else {}
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}
    final_program = _first_present(qa.get("program_re"), qa.get("program"))
    return {
        "turn_ind": rec.get("turn_ind", annotation.get("turn_ind")),
        "cur_type": to_text(rec.get("cur_type") or annotation.get("cur_type")),
        "current_question": _annotation_current_question(annotation) or clean_question_text(rec.get("question")),
        "current_program": to_text(program_re),
        "current_answer": safe_jsonable(raw_answer),
        "current_exe_ans": safe_jsonable(exe_ans),
        "program_re": to_text(program_re),
        "program": to_text(program_re),
        "gold_ind": safe_jsonable(gold_ind),
        "answer": safe_jsonable(raw_answer),
        "exe_ans": safe_jsonable(exe_ans),
        "final_question": to_text(qa.get("question")),
        "final_program_re": to_text(qa.get("program_re")),
        "final_program": to_text(final_program),
        "final_answer": safe_jsonable(qa.get("answer")),
        "final_exe_ans": safe_jsonable(qa.get("exe_ans")),
    }

def _normalize_turn_for_history(rec: Dict[str, Any], args: Any) -> Dict[str, Any]:
    question, program_re, raw_answer, exe_ans, gold_ind = _reasoning_inputs(rec)
    return build_reasoning_supervision(
        rec,
        family="convfinqa_turn",
        source_dataset="ConvFinQA",
        task_type="financial_conversational_numerical_reasoning",
        record_id=to_text(rec.get("id")),
        question=question,
        program_re=program_re,
        raw_answer=raw_answer,
        exe_ans=exe_ans,
        gold_evidence=gold_ind,
        args=args,
        history_questions=[],
        extra_metadata=_raw_metadata(rec, program_re, raw_answer, exe_ans, gold_ind),
    )


def _history_reasoning_turn(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    norm = _normalize_turn_for_history(rec, args)
    if norm.get("quality_tier") != "A":
        return None
    if norm.get("evidence_match_type") != "exact":
        return None
    if norm.get("answer_matches_program") is not True:
        return None
    if not norm.get("program_canonical") or not norm.get("answer_norm"):
        return None
    return {
        "question": norm.get("question", ""),
        "target": render_strict_target(norm, getattr(args, "sft_variant", "benchmark_sft")),
        "quality_tier": norm.get("quality_tier", ""),
        "evidence_match_type": norm.get("evidence_match_type", ""),
        "answer_matches_program": norm.get("answer_matches_program", False),
    }

def _annotation_history_reasoning_by_question(
    rec: Dict[str, Any],
    args: Any,
    max_index: int,
) -> Dict[str, Dict[str, Any]]:
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}
    questions = annotation.get("dialogue_break") if isinstance(annotation.get("dialogue_break"), list) else []
    if not questions:
        questions = annotation.get("cur_dial") if isinstance(annotation.get("cur_dial"), list) else []
    programs = annotation.get("turn_program") if isinstance(annotation.get("turn_program"), list) else []
    if not programs:
        programs = annotation.get("turn_program_ori") if isinstance(annotation.get("turn_program_ori"), list) else []
    exe_answers = annotation.get("exe_ans_list") if isinstance(annotation.get("exe_ans_list"), list) else []
    raw_answers = annotation.get("answer_list") if isinstance(annotation.get("answer_list"), list) else []
    gold_ind = annotation.get("gold_ind") or {}

    out: Dict[str, Dict[str, Any]] = {}
    upper = min(max_index, len(questions), len(programs))
    for idx in range(upper):
        question = clean_question_text(questions[idx])
        q_norm = normalize_question_for_comparison(question)
        if not question or not q_norm:
            continue
        program_re = programs[idx]
        exe_ans = exe_answers[idx] if idx < len(exe_answers) else None
        raw_answer = raw_answers[idx] if idx < len(raw_answers) else None
        if to_text(raw_answer).upper().startswith("A") or raw_answer is None:
            raw_answer = exe_ans
        exe_ans_for_norm = raw_answer if raw_answer is not None else exe_ans
        norm = build_reasoning_supervision(
            rec,
            family="convfinqa_turn",
            source_dataset="ConvFinQA",
            task_type="financial_conversational_numerical_reasoning",
            record_id=f"{to_text(rec.get('id'))}::history_{idx}",
            question=question,
            program_re=program_re,
            raw_answer=raw_answer,
            exe_ans=exe_ans_for_norm,
            gold_evidence=gold_ind,
            args=args,
            history_questions=[],
            extra_metadata={
                "turn_ind": idx,
                "cur_type": "annotation_history_turn",
                "program_re": to_text(program_re),
                "program": to_text(program_re),
                "gold_ind": safe_jsonable(gold_ind),
                "answer": safe_jsonable(raw_answer),
                "exe_ans": safe_jsonable(exe_ans_for_norm),
            },
        )
        if norm.get("quality_tier") != "A":
            continue
        if norm.get("evidence_match_type") != "exact":
            continue
        if norm.get("answer_matches_program") is not True:
            continue
        if not norm.get("program_canonical") or not norm.get("answer_norm"):
            continue
        out[q_norm] = {
            "question": question,
            "target": render_strict_target(norm, getattr(args, "sft_variant", "benchmark_sft")),
            "quality_tier": norm.get("quality_tier", ""),
            "evidence_match_type": norm.get("evidence_match_type", ""),
            "answer_matches_program": norm.get("answer_matches_program", False),
            "history_source": "annotation_turn_program",
        }
    return out

def _requires_history(question: str, history_turns: List[Dict[str, str]]) -> Tuple[bool, str]:
    q = to_text(question).lower()
    if any(term in q for term in ["percentage change", "percent change", "percentage", "percent"]):
        return bool(history_turns), "follow_up_percentage_change" if history_turns else "self_contained"
    if "difference" in q or "change" in q:
        return bool(history_turns), "follow_up_difference" if history_turns else "self_contained"
    if any(term in q for term in ["what about", "how about"]):
        return True, "follow_up_lookup"
    if any(term in q for term in ["prior", "previous", "it", "that", "those"]):
        return bool(history_turns), "entity_carryover" if history_turns else "self_contained"
    return False, "self_contained"


def prepare_multiturn_records(records: Sequence[Dict[str, Any]], args: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(_conversation_id(rec), []).append(rec)

    prepared: List[Dict[str, Any]] = []
    history_question_only_turns = 0
    history_full_reasoning_turns = 0
    requires_history_rows = 0
    duplicate_current_question_rows = 0
    current_answer_leaked_rows = 0
    for conversation_id, group in grouped.items():
        sorted_group = sorted(group, key=lambda row: _turn_rank(row))
        prior_reasoning_by_question: Dict[str, Dict[str, Any]] = {}
        for rec in sorted_group:
            current_question, raw_prior_questions = _extract_history(rec)
            current_q_norm = normalize_question_for_comparison(current_question)
            history_questions = summarize_history_questions(
                raw_prior_questions,
                int(getattr(args, "max_history_turns", 6)),
                int(getattr(args, "max_context_chars", 400)),
            )
            annotation_reasoning_by_question = _annotation_history_reasoning_by_question(
                rec,
                args,
                max_index=len(raw_prior_questions),
            )
            usable_history: List[Dict[str, Any]] = []
            for question in history_questions:
                q_norm = normalize_question_for_comparison(question)
                if not q_norm:
                    continue
                if current_q_norm and q_norm == current_q_norm:
                    duplicate_current_question_rows += 1
                    continue
                full_turn = annotation_reasoning_by_question.get(q_norm) or prior_reasoning_by_question.get(q_norm)
                if full_turn:
                    usable_history.append(dict(full_turn))
                    history_full_reasoning_turns += 1
                else:
                    usable_history.append({"question": question})
                    history_question_only_turns += 1

            current_answer = _history_answer(rec, to_text(getattr(args, "sft_variant", "benchmark_sft")))
            if current_answer:
                for item in usable_history:
                    if to_text(item.get("target")) and f"Answer: {current_answer}" in to_text(item.get("target")):
                        current_answer_leaked_rows += 1
                        break

            requires_history, dependency_type = _requires_history(current_question, usable_history)
            if requires_history:
                requires_history_rows += 1

            copied = dict(rec)
            copied["__conversation_id"] = conversation_id
            copied["__turn_ind"] = _turn_rank(rec)[0]
            copied["__history_turns"] = usable_history
            copied["__history_question_only_turn_count"] = sum(1 for item in usable_history if not item.get("target"))
            copied["__history_full_reasoning_turn_count"] = sum(1 for item in usable_history if item.get("target"))
            copied["__history_answer_missing"] = bool(copied["__history_question_only_turn_count"])
            copied["__requires_history"] = requires_history
            copied["__history_dependency_type"] = dependency_type
            prepared.append(copied)

            history_turn = _history_reasoning_turn(rec, args)
            if history_turn and current_q_norm:
                prior_reasoning_by_question[current_q_norm] = history_turn

    return prepared, {
        "conversation_count": len(grouped),
        "history_question_only_turns": history_question_only_turns,
        "history_full_reasoning_turns": history_full_reasoning_turns,
        "requires_history_source_rows": requires_history_rows,
        "duplicate_current_question_in_history_source_rows": duplicate_current_question_rows,
        "current_answer_leaked_in_history_source_rows": current_answer_leaked_rows,
    }


def _history_text(history_turns: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in history_turns:
        if not isinstance(item, dict):
            continue
        parts.append(to_text(item.get("question")))
        parts.append(to_text(item.get("target")))
    return "\n".join(part for part in parts if part)


def _target_body_without_evidence(target: str) -> str:
    lines = []
    for line in to_text(target).splitlines():
        if line.startswith("Evidence:") or line.startswith("- "):
            continue
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def _finalize_history_leak_audits(norm: Dict[str, Any]) -> None:
    history_turns = norm.get("history_turns") if isinstance(norm.get("history_turns"), list) else []
    if not history_turns:
        return
    history_norm = normalize_question_for_comparison(_history_text(history_turns))
    question_norm = normalize_question_for_comparison(norm.get("question"))
    flags: List[str] = []
    semantic_flags: List[str] = []
    if question_norm and question_norm in history_norm:
        flags.append("duplicate_current_question_in_history")

    current_target = render_strict_target(norm, "benchmark_sft")
    target_norm = normalize_question_for_comparison(_target_body_without_evidence(current_target))
    if target_norm and target_norm in history_norm:
        flags.append("current_target_leaked_in_history")

    answer_text = to_text(norm.get("answer_norm"))
    if answer_text and any(f"Answer: {answer_text}" in to_text(item.get("target")) for item in history_turns if isinstance(item, dict)):
        semantic_flags.append("current_answer_repeated_in_history")

    if flags or semantic_flags:
        add_audit_flags(norm, audit_flags=flags, semantic_flags=semantic_flags)

def build_prompt(rec: Dict[str, Any], question: str, history_turns: List[Dict[str, Any]], args: Any) -> str:
    prompt_parts = [
        "You are a financial conversational reasoning assistant.",
        "Use the conversation history and report context to identify supporting evidence, produce the executable program, and give the final answer.",
    ]
    if history_turns:
        blocks: List[str] = []
        has_full_reasoning = any(to_text(item.get("target")) for item in history_turns)
        for item in history_turns:
            q = to_text(item.get("question"))
            target = to_text(item.get("target"))
            if not q:
                continue
            if target:
                blocks.append(f"Q: {q}\n{target}")
            elif has_full_reasoning:
                blocks.append(f"Q: {q}")
            else:
                blocks.append(f"- {q}")
        heading = "Conversation history:" if has_full_reasoning else "Conversation history questions:"
        prompt_parts.append(heading + "\n" + "\n\n".join(blocks))
    prompt_parts.extend(build_english_context_sections(rec, args))
    prompt_parts.append(f"Current question: {question}")
    prompt_parts.append(
        "Respond exactly in this format:\n"
        "Evidence:\n"
        "- ...\n\n"
        "Program: ...\n"
        "Answer: ..."
    )
    return "\n\n".join(prompt_parts)

def normalize_record(rec: Dict[str, Any], args: Any) -> Dict[str, Any]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else {}
    annotation = rec.get("annotation") if isinstance(rec.get("annotation"), dict) else {}
    current_question, prior_questions = _extract_history(rec)
    has_prepared_history = "__history_turns" in rec
    history_turns = rec.get("__history_turns") if isinstance(rec.get("__history_turns"), list) else []
    history_questions = [to_text(item.get("question")) for item in history_turns if isinstance(item, dict) and to_text(item.get("question"))]
    if not has_prepared_history and not history_questions:
        history_questions = summarize_history_questions(
            prior_questions,
            int(getattr(args, "max_history_turns", 6)),
            int(getattr(args, "max_context_chars", 400)),
        )
        history_turns = [{"question": q, "answer": ""} for q in history_questions]
    history_answers = []
    history_full_reasoning_turn_count = sum(1 for item in history_turns if isinstance(item, dict) and item.get("target"))
    history_question_only_turn_count = sum(1 for item in history_turns if isinstance(item, dict) and not item.get("target"))

    current_question, program_re, raw_answer, exe_ans, gold_ind = _reasoning_inputs(rec)

    norm = build_reasoning_supervision(
        rec,
        family="convfinqa_turn",
        source_dataset="ConvFinQA",
        task_type="financial_conversational_numerical_reasoning",
        record_id=to_text(rec.get("id")),
        question=current_question,
        program_re=program_re,
        raw_answer=raw_answer,
        exe_ans=exe_ans,
        gold_evidence=gold_ind,
        args=args,
        history_questions=history_questions,
        extra_metadata=_raw_metadata(rec, program_re, raw_answer, exe_ans, gold_ind),
    )
    norm["history_answer_missing"] = bool(rec.get("__history_answer_missing"))
    norm["conversation_id"] = to_text(rec.get("__conversation_id") or _conversation_id(rec))
    norm["turn_ind"] = rec.get("__turn_ind", _turn_rank(rec)[0])
    norm["history_turns"] = history_turns
    norm["history_answers"] = history_answers
    norm["history_turn_count"] = len(history_turns)
    norm["history_full_reasoning_turn_count"] = history_full_reasoning_turn_count
    norm["history_question_only_turn_count"] = history_question_only_turn_count
    norm["history_full_reasoning_ratio"] = round(history_full_reasoning_turn_count / len(history_turns), 6) if history_turns else 0.0
    norm["requires_history"] = bool(rec.get("__requires_history"))
    norm["history_dependency_type"] = to_text(rec.get("__history_dependency_type") or "self_contained")
    norm["sft_mode"] = "turn_level_multiturn"
    norm["prompt"] = build_prompt(rec, current_question, history_turns, args) if current_question else ""
    _finalize_history_leak_audits(norm)
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
        "source_dataset": norm.get("source_dataset", "ConvFinQA"),
        "task_type": norm.get("task_type", "financial_conversational_numerical_reasoning"),
        "record_id": norm.get("record_id", ""),
        "conversation_id": norm.get("conversation_id", ""),
        "turn_ind": norm.get("turn_ind"),
        "sft_mode": norm.get("sft_mode", "turn_level_multiturn"),
        "metadata": {
            "program_raw": norm.get("program_raw", ""),
            "program_canonical": norm.get("program_canonical", ""),
            "program_executable": norm.get("program_executable"),
            "answer_raw": norm.get("answer_raw", ""),
            "answer_exe": norm.get("answer_exe"),
            "answer_norm": norm.get("answer_norm", ""),
            "answer_display": norm.get("answer_display", ""),
            "answer_matches_program": norm.get("answer_matches_program", False),
            "aligned_evidence": norm.get("aligned_evidence", []),
            "evidence_match_type": norm.get("evidence_match_type", ""),
            "audit_flags": norm.get("audit_flags", []),
            "semantic_audit_flags": norm.get("semantic_audit_flags", []),
            "quality_tier": norm.get("quality_tier", ""),
            "evidence_visible_in_prompt": norm.get("evidence_visible_in_prompt"),
            "table_evidence_column_pruned": norm.get("table_evidence_column_pruned", False),
            "history_questions": norm.get("history_questions", []),
            "history_answers": norm.get("history_answers", []),
            "history_turns": norm.get("history_turn_count", 0),
            "history_full_reasoning_turns": norm.get("history_full_reasoning_turn_count", 0),
            "history_question_only_turns": norm.get("history_question_only_turn_count", 0),
            "history_full_reasoning_ratio": norm.get("history_full_reasoning_ratio", 0.0),
            "history_answer_missing": norm.get("history_answer_missing", False),
            "requires_history": norm.get("requires_history", False),
            "history_dependency_type": norm.get("history_dependency_type", ""),
            "raw_metadata": norm.get("metadata", {}),
        },
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": render_strict_target(norm, getattr(args, "sft_variant", "benchmark_sft"))},
        ],
    }


def build_sft_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    return render_sft_item(normalize_record(rec, args), args)


def _group_key(rec: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else {}
    conversation_id = _conversation_id(rec)
    turn = str(_turn_rank(rec)[0])
    question = to_text(qa.get("question") or rec.get("question"))
    program = to_text(qa.get("program_re") or rec.get("program_re"))
    answer = to_text(qa.get("exe_ans") if qa.get("exe_ans") is not None else qa.get("answer") or rec.get("answer"))
    return (conversation_id, turn, question, program, answer)


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
        "source_dataset": item.get("source_dataset", "ConvFinQA"),
        "record_id": item.get("record_id", ""),
        "metadata": item.get("metadata", {}),
    }

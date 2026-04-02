#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from ..common import combine_instruction_input, to_text


def build_sft_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    question = combine_instruction_input(rec)
    answer = to_text(rec.get("output") or rec.get("answer") or rec.get("response") or rec.get("gold_answer"))
    if not question or not answer:
        return None

    prompt = (
        "你是一名金融问答助手。请给出结构化解释，再给出结论。\n\n"
        f"问题：{question}\n\n"
        "请按以下结构作答：\n问题分析：...\n解释：...\n结论：..."
    )
    response = answer
    if "结论：" not in response:
        response = f"问题分析：这是一个金融问答场景。\n解释：{answer}\n结论：{answer}"

    return {
        "source_dataset": "FinGPT/fingpt-fiqa_qa",
        "task_type": "financial_qa_explained",
        "record_id": to_text(rec.get("id") or rec.get("question_id")),
        "metadata": {},
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response},
        ],
    }


def build_dpo_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    item = build_sft_item(rec, args)
    if item is None:
        return None

    return {
        "system": "",
        "history": [],
        "question": item["conversations"][0]["value"],
        "response_chosen": item["conversations"][1]["value"],
        "response_rejected": "问题分析：这是一个金融问题。\n解释：信息有限，给出简短判断。\n结论：可能与市场波动有关。",
        "source_dataset": item.get("source_dataset", "fiqa_qa"),
        "record_id": item.get("record_id", ""),
        "metadata": item.get("metadata", {}),
    }

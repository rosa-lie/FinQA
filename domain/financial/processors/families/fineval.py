#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from ..common import combine_instruction_input, to_text


def build_sft_item(rec: Dict[str, Any], args: Any) -> Optional[Dict[str, Any]]:
    question = combine_instruction_input(rec)
    answer = to_text(rec.get("output") or rec.get("answer") or rec.get("response"))
    if not question or not answer:
        return None

    prompt = (
        "你是一名中文金融考试辅导助手。请先给出简短推理，再给出正确答案。\n\n"
        f"题目：{question}\n\n"
        "请按以下结构作答：\n题目理解：...\n推理：...\n最终答案：..."
    )
    response = answer
    if "最终答案" not in response:
        response = f"题目理解：这是一道金融考试题。\n推理：{answer}\n最终答案：请以解析中的正确选项为准。"

    return {
        "source_dataset": "FinGPT/fingpt-fineval",
        "task_type": "financial_exam_reasoning_zh",
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
        "response_rejected": "题目理解：这是一道金融考试题。\n推理：根据常识快速判断即可。\n最终答案：A",
        "source_dataset": item.get("source_dataset", "fineval"),
        "record_id": item.get("record_id", ""),
        "metadata": item.get("metadata", {}),
    }

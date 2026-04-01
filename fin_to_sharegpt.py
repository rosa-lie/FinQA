#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert financial reasoning datasets to MedicalGPT SFT (ShareGPT) jsonl.

Supported dataset families:
- convfinqa_turn: conversational financial numerical reasoning
- finqa: table + text + program reasoning
- fineval: Chinese finance exam QA with explanation
- fiqa_qa: financial QA supplement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset


DATASET_FAMILIES = {"auto", "convfinqa_turn", "finqa", "fineval", "fiqa_qa"}


def to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v).strip()
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    return str(value)


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_text_blocks(value: Any, max_items: int, max_chars: int) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = [to_text(value)]
    out = []
    for item in items[:max_items]:
        text = to_text(item)
        if text:
            out.append(truncate_text(text, max_chars))
    return out


def format_table(table: Any, max_rows: int, max_cols: int, max_cell_chars: int) -> str:
    if not isinstance(table, list) or not table:
        return ""
    rows = []
    for row in table[:max_rows]:
        if isinstance(row, list):
            cells = [truncate_text(to_text(cell), max_cell_chars) for cell in row[:max_cols]]
            rows.append(" | ".join(cells))
        else:
            rows.append(truncate_text(to_text(row), max_cell_chars))
    return "\n".join(rows)


def extract_numeric_text(answer: str) -> str:
    return answer.strip()


def infer_dataset_family(rec: Dict[str, Any]) -> str:
    if "cur_dial" in rec or "cur_program" in rec:
        return "convfinqa_turn"
    if "qa" in rec and isinstance(rec.get("qa"), dict):
        return "finqa"
    instruction = to_text(rec.get("instruction")).lower()
    if "选项" in instruction or "单选" in instruction or "多选" in instruction or "判断" in instruction:
        return "fineval"
    if any(k in rec for k in ["gold_answer", "source_answer", "answer", "output", "response"]):
        return "fiqa_qa"
    return "fiqa_qa"


def build_context_sections(rec: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    sections = []
    pre_text = normalize_text_blocks(rec.get("pre_text"), args.max_context_items, args.max_context_chars)
    post_text = normalize_text_blocks(rec.get("post_text"), args.max_context_items, args.max_context_chars)
    if pre_text:
        sections.append("材料（表格前文本）：\n" + "\n".join(f"- {t}" for t in pre_text))
    table_text = format_table(rec.get("table") or rec.get("table_ori"), args.max_table_rows, args.max_table_cols, args.max_cell_chars)
    if table_text:
        sections.append("表格：\n" + table_text)
    if post_text:
        sections.append("材料（表格后文本）：\n" + "\n".join(f"- {t}" for t in post_text))
    return sections


def build_convfinqa_item(rec: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
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
    gold_facts = normalize_text_blocks(gold_ind, args.max_supporting_facts, args.max_context_chars)

    prompt_parts = [
        "你是一名金融数值推理助手。请结合对话历史、文本材料和表格，进行分步推理并给出最终数值答案。",
    ]
    if prior_questions:
        prompt_parts.append("历史对话：\n" + "\n".join(f"- {q}" for q in prior_questions[-args.max_history_turns:]))
    prompt_parts.extend(build_context_sections(rec, args))
    prompt_parts.append(f"当前问题：{current_question}")
    prompt_parts.append("请按以下结构作答：\n问题分析：...\n关键证据：\n- ...\n推理程序：...\n最终答案：...")

    answer_lines = [
        "问题分析：这是一个需要结合历史对话与财务材料进行数值推理的问题。",
        "关键证据：",
    ]
    if gold_facts:
        answer_lines.extend([f"- {fact}" for fact in gold_facts])
    else:
        answer_lines.append("- 需要从给定文本与表格中抽取相关财务指标，并与历史问题保持一致。")
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

def build_finqa_item(rec: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    qa = rec.get("qa") if isinstance(rec.get("qa"), dict) else rec
    question = to_text(qa.get("question") or rec.get("question"))
    if not question:
        return None
    program = to_text(qa.get("program_re") or qa.get("program") or rec.get("program_re") or rec.get("program"))
    final_answer = to_text(qa.get("exe_ans") or rec.get("exe_ans") or qa.get("answer") or rec.get("answer"))
    if not final_answer:
        return None
    gold_inds = qa.get("gold_inds") or rec.get("gold_inds") or []
    supporting = normalize_text_blocks(gold_inds, args.max_supporting_facts, args.max_context_chars)

    prompt_parts = [
        "你是一名金融表文混合推理助手。请结合文本、表格和问题，给出可执行的推理程序与最终答案。",
    ]
    prompt_parts.extend(build_context_sections(rec, args))
    prompt_parts.append(f"问题：{question}")
    prompt_parts.append("请按以下结构作答：\n问题分析：...\n关键证据：\n- ...\n推理程序：...\n最终答案：...")

    answer_lines = [
        "问题分析：需要从财报文本和表格中定位相关指标，并依据程序完成数值计算。",
        "关键证据：",
    ]
    if supporting:
        answer_lines.extend([f"- {fact}" for fact in supporting])
    else:
        answer_lines.append("- 需要使用题目涉及的表格行列和相关文本说明。")
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


def combine_instruction_input(rec: Dict[str, Any]) -> str:
    instruction = to_text(rec.get("instruction"))
    inp = to_text(rec.get("input"))
    question = to_text(rec.get("question") or rec.get("query") or rec.get("prompt"))
    parts = [p for p in [instruction, inp, question] if p]
    return "\n\n".join(parts)


def build_fineval_item(rec: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
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


def build_fiqa_item(rec: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
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


def normalize_record(rec: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    family = args.dataset_family
    if family == "auto":
        family = infer_dataset_family(rec)
    if family == "convfinqa_turn":
        return build_convfinqa_item(rec, args)
    if family == "finqa":
        return build_finqa_item(rec, args)
    if family == "fineval":
        return build_fineval_item(rec, args)
    if family == "fiqa_qa":
        return build_fiqa_item(rec, args)
    raise ValueError(f"Unsupported dataset_family: {family}")


def iter_records(ds: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in ds:
        yield dict(row)


def load_source(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.source_file:
        source_path = Path(args.source_file)
        ext = source_path.suffix.lower()
        if ext not in {".json", ".jsonl"}:
            raise ValueError("--source_file currently supports .json/.jsonl only")
        if ext == ".json":
            with source_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            raise ValueError("JSON source file must contain an object or a list of objects")
        with source_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if not args.dataset_name:
        raise ValueError("Please provide --dataset_name or --source_file")
    return load_dataset(args.dataset_name, args.dataset_config, split=args.split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--dataset_family", type=str, default="auto", choices=sorted(DATASET_FAMILIES))
    parser.add_argument("--max_history_turns", type=int, default=6)
    parser.add_argument("--max_context_items", type=int, default=6)
    parser.add_argument("--max_context_chars", type=int, default=400)
    parser.add_argument("--max_supporting_facts", type=int, default=6)
    parser.add_argument("--max_table_rows", type=int, default=20)
    parser.add_argument("--max_table_cols", type=int, default=12)
    parser.add_argument("--max_cell_chars", type=int, default=80)
    args = parser.parse_args()

    ds = load_source(args)
    rows = []
    skipped = 0
    for rec in iter_records(ds):
        item = normalize_record(rec, args)
        if item is None:
            skipped += 1
            continue
        rows.append(item)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output_file": str(out_path),
        "dataset_family": args.dataset_family,
        "saved_rows": len(rows),
        "skipped_rows": skipped,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

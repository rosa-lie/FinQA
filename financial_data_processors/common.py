#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def build_context_sections(rec: Dict[str, Any], args: Any) -> List[str]:
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


def combine_instruction_input(rec: Dict[str, Any]) -> str:
    instruction = to_text(rec.get("instruction"))
    inp = to_text(rec.get("input"))
    question = to_text(rec.get("question") or rec.get("query") or rec.get("prompt"))
    parts = [p for p in [instruction, inp, question] if p]
    return "\n\n".join(parts)


def load_source(args: Any) -> Iterable[Dict[str, Any]]:
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
    from datasets import load_dataset
    return load_dataset(args.dataset_name, args.dataset_config, split=args.split)


def iter_records(ds: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in ds:
        yield dict(row)


def parse_bool_arg(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    text = to_text(v).lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")

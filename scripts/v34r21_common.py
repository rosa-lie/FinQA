from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TARGET_FAMILIES = [
    "direct_lookup",
    "sum",
    "difference",
    "ratio",
    "percentage_change",
    "share_of_total",
    "growth_rate",
    "multi_step_divide",
    "multi_step_arithmetic",
]


FORBIDDEN_MARKERS = [
    "Reasoning:",
    "Operation Plan:",
    "Formula candidates:",
    "Task Attributes:",
    "Answer:",
    "Normalized Answer:",
]


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        for item in value:
            text = first_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "rows", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON shape for {path}")


def record_id(row: Dict[str, Any]) -> str:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    return first_text(row.get("record_id") or row.get("id") or qa.get("id") or row.get("filename"))


def source_dataset(row: Dict[str, Any]) -> str:
    return first_text(row.get("source_dataset") or row.get("source") or "finqa").lower()


def metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def normalized_question_type(row: Dict[str, Any]) -> str:
    meta = metadata(row)
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    question = first_text(meta.get("question_type") or meta.get("question_raw") or row.get("question") or qa.get("question"))
    question = question.lower().replace("-", "_").replace(" ", "_")
    if not question:
        return "unknown"
    if "direct" in question or bool(meta.get("direct_lookup")):
        return "direct_lookup"
    if "average" in question:
        return "average"
    if "percentage" in question or "percent" in question:
        return "percentage"
    if "change" in question or "increase" in question or "decrease" in question or "growth" in question:
        return "change_or_growth"
    if "ratio" in question or "per_share" in question or "price_per" in question or "represented_how_much_of" in question:
        return "ratio"
    if "total" in question or "sum" in question:
        return "sum_or_total"
    if (
        "difference" in question
        or "variation" in question
        or "by_what_amount" in question
        or "how_much_did" in question
        or "how_much_money" in question
        or "how_bigger" in question
        or "surpass" in question
        or "outperform" in question
    ):
        return "difference"
    if "highest" in question or "lowest" in question or "maximum" in question or "minimum" in question:
        return "direct_lookup"
    return question[:80]


def answer_scale(row: Dict[str, Any]) -> str:
    meta = metadata(row)
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    scale = first_text(meta.get("answer_scale") or qa.get("scale") or row.get("answer_scale"))
    if scale:
        return scale.lower()
    answer = first_text(qa.get("answer") or row.get("answer") or qa.get("exe_ans"))
    if "%" in answer:
        return "percent"
    if "$" in answer:
        return "currency"
    try:
        value = float(answer.replace(",", ""))
    except ValueError:
        return "unknown"
    if -1.0 <= value <= 1.0:
        return "ratio"
    return "plain"


def program_text(row: Dict[str, Any]) -> str:
    meta = metadata(row)
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    program = first_text(meta.get("program_canonical") or row.get("gold_program") or qa.get("program_re") or qa.get("program"))
    return program.lower().replace(" ", "")


def program_ops(row: Dict[str, Any]) -> List[str]:
    meta = metadata(row)
    ops = meta.get("program_ops")
    if isinstance(ops, list):
        return [first_text(op).lower() for op in ops if first_text(op)]
    return [op.lower() for op in re.findall(r"([A-Za-z_]+)\s*\(", program_text(row))]


def is_numeric_literal(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%?", first_text(text).replace(",", "")))


def program_family(row: Dict[str, Any]) -> str:
    meta = metadata(row)
    program = program_text(row)
    ops = program_ops(row)
    qtype = normalized_question_type(row)
    step_count = int(meta.get("program_step_count") or max(len(ops), 1) or 1)
    if bool(meta.get("direct_lookup")) or is_numeric_literal(program):
        return "direct_lookup"
    if "sum" in ops or "add" in ops or program.startswith(("sum(", "add(")):
        return "sum"
    if "subtract" in ops and "divide" not in ops and step_count <= 1:
        return "difference"
    if "divide" in ops and ("subtract" in ops or "change" in qtype or "growth" in qtype):
        return "growth_rate" if "growth" in qtype or "change" in qtype else "percentage_change"
    if "divide" in ops and ("share" in qtype or "percentage" in qtype or answer_scale(row) in {"ratio", "percent"}):
        return "share_of_total" if "total" in qtype or "share" in qtype else "ratio"
    if "divide" in ops and step_count > 1:
        return "multi_step_divide"
    if step_count > 1:
        return "multi_step_arithmetic"
    if "divide" in ops:
        return "ratio"
    return "other"


def strat_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (normalized_question_type(row), answer_scale(row), program_family(row))


def distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    return {
        "question_type": dict(Counter(normalized_question_type(row) for row in rows)),
        "answer_scale": dict(Counter(answer_scale(row) for row in rows)),
        "program_family": dict(Counter(program_family(row) for row in rows)),
        "source_dataset": dict(Counter(source_dataset(row) for row in rows)),
    }


def load_record_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    ids: set[str] = set()
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix == ".json":
        rows = load_json_or_jsonl(path)
    else:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                ids.add(text)
        return ids
    for row in rows:
        rid = record_id(row)
        if rid:
            ids.add(rid)
    return ids


def count_forbidden_markers(text: str) -> Dict[str, int]:
    return {marker: text.count(marker) for marker in FORBIDDEN_MARKERS if marker in text}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0

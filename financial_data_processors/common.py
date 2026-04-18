#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DATASET_FAMILIES = {"auto", "convfinqa_turn", "finqa", "fineval", "fiqa_qa"}

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
CALL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$")
TEXT_ID_RE = re.compile(r"^text_(\d+)$", re.IGNORECASE)
TABLE_ID_RE = re.compile(r"^table_(\d+)$", re.IGNORECASE)


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


def normalize_ws(text: str) -> str:
    return " ".join(to_text(text).replace("\n", " ").split())


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


def _flatten_text_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value).strip()]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_text_values(item))
        return out
    if isinstance(value, dict):
        out: List[str] = []
        for key, item in value.items():
            key_text = str(key).strip().lower()
            item_values = _flatten_text_values(item)
            if key_text and key_text not in {"text", "value", "content", "sentence", "evidence"}:
                item_values = [f"{key}: {text}" for text in item_values]
            out.extend(item_values)
        return out
    return [str(value).strip()]


def summarize_evidence_blocks(value: Any, max_items: int, max_chars: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in _flatten_text_values(value):
        text = normalize_ws(raw)
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(truncate_text(text, max_chars))
        if len(out) >= max_items:
            break
    return out


def summarize_history_questions(questions: List[str], max_turns: int, max_chars: int) -> List[str]:
    out: List[str] = []
    for question in questions[-max_turns:]:
        text = normalize_ws(question)
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



def normalize_audit_text(text: Any) -> str:
    """Normalize text for conservative audit-time substring checks."""
    text = normalize_ws(to_text(text)).lower()
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return normalize_ws(text)


def normalize_question_for_comparison(text: Any) -> str:
    text = normalize_ws(to_text(text)).lower()
    text = re.sub(r"[^a-z0-9%$]+", " ", text)
    return normalize_ws(text)


def clean_question_text(question: Any) -> str:
    text = normalize_ws(question)
    # Deterministic repair for common ConvFinQA concatenations, e.g. "2018what".
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d{4}\b)", " ", text)
    return normalize_ws(text)


def detect_question_text_flags(question: Any) -> List[str]:
    text = normalize_ws(question)
    flags: List[str] = []
    if not text:
        return flags
    if re.search(r"\d{4}[A-Za-z]", text) or re.search(r"[A-Za-z]\d{4}\b", text):
        flags.append("question_text_suspicious")
    token_count = len(text.split())
    if token_count < 4:
        flags.append("question_text_suspicious")
    return sorted(set(flags))


def evidence_item_visible_in_prompt(evidence: Dict[str, Any], prompt: str, program_numbers: Optional[List[str]] = None) -> bool:
    rendered = normalize_ws(evidence.get("rendered_text", ""))
    if not rendered:
        return False
    prompt_norm = normalize_audit_text(prompt)
    rendered_norm = normalize_audit_text(rendered)
    if rendered_norm and rendered_norm in prompt_norm:
        return True
    source_location = to_text(evidence.get("source_location"))
    if source_location.startswith("table["):
        numbers = program_numbers or extract_program_numbers(rendered)
        if numbers:
            return all(_number_token_present(prompt, number) for number in numbers)
        return False
    return False


def _prompt_report_context(prompt: str) -> str:
    text = to_text(prompt)
    report_idx = text.find("\n\nReport context:")
    if report_idx >= 0:
        start = report_idx + len("\n\nReport context:")
        end = len(text)
        for marker in ["\n\nConversation history:", "\n\nConversation history questions:"]:
            idx = text.find(marker, start)
            if idx >= 0:
                end = min(end, idx)
        return text[start:end]
    markers = ["\n\nCurrent question:", "\n\nRespond exactly in this format:"]
    end = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            end = min(end, idx)
    context_start = len(text)
    for marker in ["\n\nText before table:", "\n\nTable:", "\n\nText after table:"]:
        idx = text.find(marker)
        if idx >= 0:
            context_start = min(context_start, idx + 2)
    if context_start == len(text):
        return text[:end]
    return text[context_start:end]


def evidence_visible_in_prompt(norm: Dict[str, Any], prompt: str) -> bool:
    aligned = norm.get("aligned_evidence") or []
    if not aligned:
        return False
    program_numbers = norm.get("program_numbers") or []
    report_context = _prompt_report_context(prompt)
    return all(evidence_item_visible_in_prompt(item, report_context, program_numbers) for item in aligned)


def refresh_quality(norm: Dict[str, Any]) -> None:
    norm["audit_flags"] = sorted(set(norm.get("audit_flags") or []))
    norm["semantic_audit_flags"] = sorted(set(norm.get("semantic_audit_flags") or []))
    quality_tier = classify_quality_tier(norm["audit_flags"], norm["semantic_audit_flags"])
    norm["quality_tier"] = quality_tier
    norm["strict_ok"] = quality_tier != "C"


def add_audit_flags(norm: Dict[str, Any], audit_flags: Optional[List[str]] = None, semantic_flags: Optional[List[str]] = None) -> None:
    if audit_flags:
        norm["audit_flags"] = sorted(set(norm.get("audit_flags") or []) | set(audit_flags))
    if semantic_flags:
        norm["semantic_audit_flags"] = sorted(set(norm.get("semantic_audit_flags") or []) | set(semantic_flags))
    refresh_quality(norm)


def finalize_prompt_audits(norm: Dict[str, Any], prompt: str) -> None:
    audit_flags: List[str] = []
    semantic_flags = detect_question_text_flags(norm.get("question"))
    visible = evidence_visible_in_prompt(norm, prompt)
    norm["evidence_visible_in_prompt"] = visible
    if not visible:
        audit_flags.append("evidence_not_in_rendered_prompt")
    add_audit_flags(norm, audit_flags=audit_flags, semantic_flags=semantic_flags)

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


def build_english_context_sections(rec: Dict[str, Any], args: Any) -> List[str]:
    sections = []
    pre_text = normalize_text_blocks(rec.get("pre_text"), args.max_context_items, args.max_context_chars)
    post_text = normalize_text_blocks(rec.get("post_text"), args.max_context_items, args.max_context_chars)
    if pre_text:
        sections.append("Text before table:\n" + "\n".join(f"- {t}" for t in pre_text))
    table_text = format_table(rec.get("table") or rec.get("table_ori"), args.max_table_rows, args.max_table_cols, args.max_cell_chars)
    if table_text:
        sections.append("Table:\n" + table_text)
    if post_text:
        sections.append("Text after table:\n" + "\n".join(f"- {t}" for t in post_text))
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


def canonicalize_program_re(program_re: Any) -> str:
    program = normalize_ws(to_text(program_re))
    program = re.sub(r"\bconst_(-?\d+(?:\.\d+)?)\b", r"\1", program, flags=re.IGNORECASE)
    program = re.sub(r"\s*,\s*", ", ", program)
    program = re.sub(r"\s*\(\s*", "(", program)
    program = re.sub(r"\s*\)\s*", ")", program)
    return program


def _split_args(text: str) -> List[str]:
    args: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def parse_numeric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = to_text(value)
    if not text:
        return None
    is_paren_negative = bool(re.match(r"^\s*\([^()]+\)\s*$", text))
    cleaned = text.replace("$", "").replace(",", "")
    match = NUM_RE.search(cleaned)
    if not match:
        return None
    try:
        number = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    if is_paren_negative and number > 0:
        number = -number
    return number


def _eval_program_expr(expr: str) -> float:
    expr = expr.strip()
    if re.fullmatch(r"#\d+", expr):
        raise ValueError(f"Unresolved program reference: {expr}")
    number = parse_numeric_value(expr)
    if number is not None and not CALL_RE.match(expr):
        return number

    match = CALL_RE.match(expr)
    if not match:
        raise ValueError(f"Unsupported expression: {expr}")

    fn = match.group(1).lower()
    args = [_eval_program_expr(arg) for arg in _split_args(match.group(2))]
    if fn == "add" and len(args) == 2:
        return args[0] + args[1]
    if fn == "subtract" and len(args) == 2:
        return args[0] - args[1]
    if fn == "multiply" and len(args) == 2:
        return args[0] * args[1]
    if fn == "divide" and len(args) == 2:
        if args[1] == 0:
            raise ZeroDivisionError("division by zero")
        return args[0] / args[1]
    if fn == "exp" and len(args) == 2:
        return args[0] ** args[1]
    if fn in {"max", "maximum", "table_max"} and args:
        return max(args)
    if fn in {"min", "minimum", "table_min"} and args:
        return min(args)
    if fn in {"sum", "table_sum"} and args:
        return sum(args)
    if fn in {"average", "avg", "table_average"} and args:
        return sum(args) / len(args)
    raise ValueError(f"Unsupported operator: {fn}")


def _replace_program_refs(expr: str, step_values: List[float]) -> str:
    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx < 0 or idx >= len(step_values):
            raise ValueError(f"Unknown program reference: #{idx}")
        return _format_float(step_values[idx], decimals=12)

    return re.sub(r"#(\d+)", repl, expr)


def execute_program(program: str) -> Tuple[Optional[float], str | None]:
    if not program:
        return None, "missing_program_re"
    try:
        steps = _split_args(program)
        if not steps:
            return None, "missing_program_re"
        values: List[float] = []
        for step in steps:
            resolved_step = _replace_program_refs(step, values)
            values.append(_eval_program_expr(resolved_step))
        value = values[-1]
    except Exception as exc:
        return None, f"program_parse_or_execute_error:{exc}"
    if not math.isfinite(value):
        return None, "program_non_finite_result"
    return value, None


def numeric_values_equivalent(a: Optional[float], b: Optional[float], allow_percent_scale: bool = False) -> bool:
    if a is None or b is None:
        return False
    pairs = [(a, b)]
    if allow_percent_scale:
        pairs.extend([(a * 100.0, b), (a, b * 100.0)])
    for left, right in pairs:
        tol = max(1e-4, abs(right) * 1e-4)
        if abs(left - right) <= tol:
            return True
        # Percent answers are often rounded to whole percentage points.
        if allow_percent_scale and abs(left - right) <= 1.0 and (abs(left) > 1.0 or abs(right) > 1.0):
            return True
    return False


def format_numeric_answer(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = parse_numeric_value(value)
        if number is None:
            return ""
        if float(number).is_integer():
            return str(int(number))
        return f"{number:.6f}".rstrip("0").rstrip(".")
    text = to_text(value)
    if text:
        return text
    number = parse_numeric_value(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")

def _format_float(value: float, decimals: int = 6) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def choose_answer_display(raw_answer: Any, answer_norm: str, question: str = "") -> str:
    raw_text = to_text(raw_answer)
    norm_text = to_text(answer_norm)
    q = to_text(question).lower()
    if raw_text and any(token in raw_text.lower() for token in ["%", "$", "million", "billion", "thousand"]):
        return raw_text
    norm_num = parse_numeric_value(norm_text)
    if norm_num is not None and any(token in q for token in ["percentage", "percent", "rate"]):
        return _format_float(norm_num * 100.0, 1) + "%"
    return norm_text or raw_text


def infer_answer_unit_scale(question: Any, raw_answer: Any, answer_display: Any) -> Tuple[str, str]:
    text = " ".join([to_text(question), to_text(raw_answer), to_text(answer_display)]).lower()
    answer_unit = "number"
    answer_scale = "absolute"
    if any(token in text for token in ["%", "percent", "percentage", "rate", "growth"]):
        answer_unit = "percent"
        answer_scale = "ratio"
    elif "$" in text:
        answer_unit = "currency"
    if "billion" in text:
        answer_scale = "billion"
    elif "million" in text:
        answer_scale = "million"
    return answer_unit, answer_scale


def extract_program_numbers(program_canonical: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for match in NUM_RE.finditer(to_text(program_canonical)):
        token = match.group(0).replace(",", "")
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _number_token_present(text: str, token: str) -> bool:
    cleaned_text = to_text(text).replace(",", "")
    cleaned_token = to_text(token).replace(",", "")
    if not cleaned_token:
        return False
    if cleaned_token in cleaned_text:
        return True
    try:
        value = float(cleaned_token)
    except ValueError:
        return False
    if float(value).is_integer() and str(int(value)) in cleaned_text:
        return True
    return False


def find_text_sentence_covering_numbers(rec: Dict[str, Any], numbers: List[str], min_numbers: int = 2) -> Tuple[str, str]:
    if not numbers:
        return "", ""
    for loc, source_text in _text_blocks_with_locations(rec):
        text = normalize_ws(source_text)
        if not text:
            continue
        hits = sum(1 for number in numbers if _number_token_present(text, number))
        if hits >= min(min_numbers, len(numbers)):
            return loc, text
    return "", ""


def improve_table_evidence_with_text(rec: Dict[str, Any], aligned: List[Dict[str, Any]], program_numbers: List[str], max_chars: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    flags: List[str] = []
    if not aligned or not any(item.get("evidence_type") == "table" for item in aligned):
        return aligned, flags
    loc, text = find_text_sentence_covering_numbers(rec, program_numbers, min_numbers=2)
    if loc and text:
        return [{
            "evidence_type": "text",
            "raw_id": "text_replacement_for_table",
            "source_location": loc,
            "rendered_text": truncate_text(text, max_chars),
            "match_type": "exact",
        }], flags
    for item in aligned:
        if item.get("evidence_type") == "table":
            item["rendered_text"] = truncate_text(_compact_table_evidence_text(item.get("rendered_text", "")), max_chars)
    return aligned, flags


def _compact_table_evidence_text(text: Any) -> str:
    rendered = normalize_ws(text)
    rendered = re.sub(r"\byear ended june 30 2009 2008\b", "year ended june 30", rendered, flags=re.IGNORECASE)
    rendered = re.sub(r"\s+;\s+", "; ", rendered)
    return rendered



def _question_year_tokens(question: Any) -> List[str]:
    text = to_text(question)
    years = re.findall(r"(?:19|20)\d{2}", text)
    compact_dates = re.findall(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\s*,?\s*((?:19|20)\d{2})", text, flags=re.IGNORECASE)
    return sorted(set(years + compact_dates))


def prune_table_evidence_columns(aligned: List[Dict[str, Any]], question: Any, max_chars: int) -> Tuple[List[Dict[str, Any]], bool]:
    years = _question_year_tokens(question)
    if not years:
        return aligned, False
    changed = False
    year_needles = set(years) | {year[-2:] for year in years}
    compact_year_needles = {re.sub(r"\s+", "", year) for year in year_needles}
    out: List[Dict[str, Any]] = []
    for item in aligned:
        copied = dict(item)
        if copied.get("evidence_type") != "table":
            out.append(copied)
            continue
        text = normalize_ws(copied.get("rendered_text", ""))
        parts = [part.strip() for part in re.split(r"\s*;\s*", text) if part.strip()]
        if len(parts) <= 1:
            out.append(copied)
            continue
        kept = []
        for part in parts:
            compact = re.sub(r"\s+", "", part.lower())
            if any(year in part for year in year_needles) or any(year in compact for year in compact_year_needles):
                kept.append(part)
        if kept and len(kept) < len(parts):
            copied["rendered_text"] = truncate_text("; ".join(kept), max_chars)
            copied["column_pruned"] = True
            changed = True
        out.append(copied)
    return out, changed

def detect_semantic_audit_flags(question: str, aligned_evidence: List[Dict[str, Any]], program_canonical: str) -> List[str]:
    flags: List[str] = []
    q = normalize_ws(question).lower()
    ev = " ".join(normalize_ws(item.get("rendered_text", "")).lower() for item in aligned_evidence)
    sensitivity_terms = ["basis points", "would change by", "sensitivity", "sensitive to"]
    change_terms = ["increase by", "decrease by", "change in", "changed by", "changes by"]
    static_question_terms = ["what is", "what was", "amount", "value", "expense in", "in 2009", "in 2008"]
    asks_change = any(term in q for term in ["change", "increase", "decrease", "percentage", "percent", "basis point"])
    if any(term in ev for term in sensitivity_terms + change_terms) and any(term in q for term in static_question_terms) and not asks_change:
        flags.append("question_semantic_risk")
    return flags


BLOCKING_AUDIT_FLAGS = {
    "missing_program_re",
    "missing_answer_norm",
    "program_answer_mismatch",
    "missing_evidence",
    "non_exact_evidence_alignment",
    "missing_question",
    "evidence_not_in_rendered_prompt",
    "duplicate_current_question_in_history",
    "current_target_leaked_in_history",
}
NONBLOCKING_B_FLAGS = {
    "raw_answer_mismatch_with_answer_norm",
    "question_semantic_risk",
    "weak_table_evidence_rendering",
    "question_text_suspicious",
    "current_answer_repeated_in_history",
}


def classify_quality_tier(audit_flags: List[str], semantic_audit_flags: List[str]) -> str:
    flags = set(audit_flags or []) | set(semantic_audit_flags or [])
    if any(flag in BLOCKING_AUDIT_FLAGS or flag.startswith("program_parse_or_execute_error") for flag in flags):
        return "C"
    if flags & NONBLOCKING_B_FLAGS:
        return "B"
    if flags:
        return "B"
    return "A"


def quality_tier_allowed(norm: Dict[str, Any], args: Any) -> bool:
    raw = to_text(getattr(args, "strict_tiers", "A")) or "A"
    allowed = {item.strip().upper() for item in raw.split(",") if item.strip()}
    return to_text(norm.get("quality_tier", "C")).upper() in allowed


def _iter_gold_items(gold: Any) -> List[Tuple[str, str]]:
    if isinstance(gold, dict):
        return [(to_text(k), to_text(v)) for k, v in gold.items() if to_text(v)]
    if isinstance(gold, list):
        out: List[Tuple[str, str]] = []
        for idx, item in enumerate(gold):
            if isinstance(item, dict):
                out.extend(_iter_gold_items(item))
            else:
                text = to_text(item)
                if text:
                    out.append((f"evidence_{idx}", text))
        return out
    text = to_text(gold)
    return [("evidence", text)] if text else []


def _text_blocks_with_locations(rec: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    pre = rec.get("pre_text") if isinstance(rec.get("pre_text"), list) else []
    post = rec.get("post_text") if isinstance(rec.get("post_text"), list) else []
    for idx, text in enumerate(pre):
        out.append((f"pre_text[{idx}]", to_text(text)))
    for idx, text in enumerate(post):
        out.append((f"post_text[{idx}]", to_text(text)))
    return out


def _render_table_row(table: Any, row_idx: int) -> str:
    if not isinstance(table, list) or row_idx < 0 or row_idx >= len(table):
        return ""
    row = table[row_idx]
    if not isinstance(row, list):
        return normalize_ws(row)
    return " | ".join(to_text(cell) for cell in row)


def align_evidence(rec: Dict[str, Any], gold: Any, max_items: int = 3, max_chars: int = 400) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    aligned: List[Dict[str, Any]] = []
    audit_flags: List[str] = []
    text_blocks = _text_blocks_with_locations(rec)
    table = rec.get("table") or rec.get("table_ori")

    for raw_id, raw_text in _iter_gold_items(gold):
        if len(aligned) >= max_items:
            break
        rendered = normalize_ws(raw_text)
        source_location = ""
        evidence_type = "unknown"
        match_type = "missing"

        text_match = TEXT_ID_RE.match(raw_id)
        table_match = TABLE_ID_RE.match(raw_id)
        if text_match:
            evidence_type = "text"
            idx = int(text_match.group(1))
            if 0 <= idx < len(text_blocks):
                loc, source_text = text_blocks[idx]
                if normalize_ws(source_text).lower() == rendered.lower():
                    source_location = loc
                    match_type = "exact"
            if match_type != "exact":
                for loc, source_text in text_blocks:
                    if normalize_ws(source_text).lower() == rendered.lower():
                        source_location = loc
                        match_type = "exact"
                        break
        elif table_match:
            evidence_type = "table"
            idx = int(table_match.group(1))
            row_text = _render_table_row(table, idx)
            if row_text:
                source_location = f"table[{idx}]"
                match_type = "exact"
                rendered = rendered or normalize_ws(row_text)
        else:
            for loc, source_text in text_blocks:
                if normalize_ws(source_text).lower() == rendered.lower():
                    evidence_type = "text"
                    source_location = loc
                    match_type = "exact"
                    break

        if not rendered:
            continue
        aligned.append({
            "evidence_type": evidence_type,
            "raw_id": raw_id,
            "source_location": source_location,
            "rendered_text": truncate_text(rendered, max_chars),
            "match_type": match_type,
        })

    if not aligned:
        audit_flags.append("missing_evidence")
        return [], "missing", audit_flags
    if any(item.get("match_type") != "exact" for item in aligned):
        audit_flags.append("non_exact_evidence_alignment")
        return aligned, "non_exact", audit_flags
    return aligned, "exact", audit_flags


def choose_answer_norm(raw_answer: Any, exe_ans: Any, program_value: Optional[float]) -> Tuple[str, List[str], bool]:
    audit_flags: List[str] = []
    exe_num = parse_numeric_value(exe_ans)
    raw_num = parse_numeric_value(raw_answer)

    exe_allows_percent_scale = "%" in to_text(exe_ans)
    if program_value is not None and exe_num is not None and numeric_values_equivalent(program_value, exe_num, exe_allows_percent_scale):
        answer_norm = format_numeric_answer(exe_ans)
        answer_matches_program = True
    elif program_value is not None and exe_num is None:
        answer_norm = format_numeric_answer(program_value)
        answer_matches_program = True
    else:
        answer_norm = format_numeric_answer(exe_ans) or format_numeric_answer(raw_answer)
        answer_matches_program = False
        audit_flags.append("program_answer_mismatch")

    answer_norm_num = parse_numeric_value(answer_norm)
    raw_allows_percent_scale = "%" in to_text(raw_answer) or "%" in to_text(answer_norm)
    if raw_num is not None and answer_norm_num is not None and not numeric_values_equivalent(raw_num, answer_norm_num, raw_allows_percent_scale):
        audit_flags.append("raw_answer_mismatch_with_answer_norm")
    if not answer_norm:
        audit_flags.append("missing_answer_norm")
    return answer_norm, audit_flags, answer_matches_program


def build_reasoning_supervision(
    rec: Dict[str, Any],
    *,
    family: str,
    source_dataset: str,
    task_type: str,
    record_id: str,
    question: str,
    program_re: Any,
    raw_answer: Any,
    exe_ans: Any,
    gold_evidence: Any,
    args: Any,
    history_questions: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    audit_flags: List[str] = []
    program_raw = to_text(program_re)
    program_canonical = canonicalize_program_re(program_raw)
    program_value, program_error = execute_program(program_canonical)
    if program_error:
        audit_flags.append(program_error)

    max_evidence = min(3, max(1, int(getattr(args, "max_supporting_facts", 3))))
    max_evidence_chars = int(getattr(args, "max_context_chars", 400))
    aligned_evidence, evidence_match_type, evidence_flags = align_evidence(
        rec,
        gold_evidence,
        max_items=max_evidence,
        max_chars=max_evidence_chars,
    )
    audit_flags.extend(evidence_flags)

    program_numbers = extract_program_numbers(program_canonical)
    aligned_evidence, evidence_improvement_flags = improve_table_evidence_with_text(
        rec, aligned_evidence, program_numbers, max_evidence_chars
    )
    aligned_evidence, table_evidence_column_pruned = prune_table_evidence_columns(
        aligned_evidence, question, max_evidence_chars
    )

    answer_norm, answer_flags, answer_matches_program = choose_answer_norm(raw_answer, exe_ans, program_value)
    answer_display = choose_answer_display(raw_answer, answer_norm, question)
    answer_unit, answer_scale = infer_answer_unit_scale(question, raw_answer, answer_display)
    audit_flags.extend(answer_flags)
    semantic_audit_flags = detect_semantic_audit_flags(question, aligned_evidence, program_canonical)
    semantic_audit_flags.extend(evidence_improvement_flags)
    if not question:
        audit_flags.append("missing_question")
    if not program_raw:
        audit_flags.append("missing_program_re")

    quality_tier = classify_quality_tier(audit_flags, semantic_audit_flags)
    strict_ok = quality_tier != "C"

    return {
        "source_dataset": source_dataset,
        "task_type": task_type,
        "family": family,
        "record_id": record_id,
        "question": question,
        "history_questions": history_questions or [],
        "program_raw": program_raw,
        "program_canonical": program_canonical,
        "program_executable": program_value,
        "answer_raw": to_text(raw_answer),
        "answer_exe": safe_jsonable(exe_ans),
        "answer_norm": answer_norm,
        "answer_display": answer_display,
        "answer_unit": answer_unit,
        "answer_scale": answer_scale,
        "answer_source": "program_executable" if program_value is not None else "raw_answer",
        "answer_matches_program": answer_matches_program,
        "aligned_evidence": aligned_evidence,
        "evidence_match_type": evidence_match_type,
        "audit_flags": sorted(set(audit_flags)),
        "semantic_audit_flags": sorted(set(semantic_audit_flags)),
        "quality_tier": quality_tier,
        "program_numbers": program_numbers,
        "table_evidence_column_pruned": table_evidence_column_pruned,
        "strict_ok": strict_ok,
        "metadata": safe_jsonable(extra_metadata or {}),
    }


def render_strict_target(norm: Dict[str, Any], sft_variant: str = "benchmark_sft") -> str:
    evidence_lines = ["Evidence:"]
    for evidence in norm.get("aligned_evidence") or []:
        text = to_text(evidence.get("rendered_text"))
        if text:
            evidence_lines.append(f"- {text}")
    evidence_text = "\n".join(evidence_lines)
    if sft_variant == "dual_answer_sft":
        return "\n\n".join([
            evidence_text,
            f"Program: {to_text(norm.get('program_canonical'))}",
            f"Answer: {to_text(norm.get('answer_display') or norm.get('answer_norm'))}",
            f"Normalized Answer: {to_text(norm.get('answer_norm'))}",
        ])
    if sft_variant == "program_executor_sft":
        return "\n\n".join([
            evidence_text,
            f"Program: {to_text(norm.get('program_canonical'))}",
        ])
    answer_key = "answer_display" if sft_variant == "assistant_sft" else "answer_norm"
    return "\n\n".join([
        evidence_text,
        f"Program: {to_text(norm.get('program_canonical'))}",
        f"Answer: {to_text(norm.get(answer_key) or norm.get('answer_norm'))}",
    ])


def build_audit_item(norm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_dataset": norm.get("source_dataset", ""),
        "task_type": norm.get("task_type", ""),
        "family": norm.get("family", ""),
        "record_id": norm.get("record_id", ""),
        "question": norm.get("question", ""),
        "audit_flags": norm.get("audit_flags", []),
        "evidence_match_type": norm.get("evidence_match_type", ""),
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
        "quality_tier": norm.get("quality_tier", ""),
        "semantic_audit_flags": norm.get("semantic_audit_flags", []),
        "evidence_visible_in_prompt": norm.get("evidence_visible_in_prompt"),
        "table_evidence_column_pruned": norm.get("table_evidence_column_pruned", False),
        "aligned_evidence": norm.get("aligned_evidence", []),
        "metadata": norm.get("metadata", {}),
    }


def mutate_numeric_text(text: str) -> str:
    match = NUM_RE.search(text)
    if not match:
        return text + " 1"
    token = match.group(0).replace(",", "")
    try:
        value = float(token)
        step = max(abs(value) * 0.12, 1.0)
        mutated = value + step if value >= 0 else value - step
        replacement = str(int(mutated)) if mutated.is_integer() else f"{mutated:.4f}".rstrip("0").rstrip(".")
    except Exception:
        replacement = token + "1"
    return text[:match.start()] + replacement + text[match.end():]


def mutate_program_text(program: str) -> str:
    substitutions = [("divide", "multiply"), ("subtract", "add"), ("multiply", "divide"), ("add", "subtract")]
    for src, dst in substitutions:
        if src in program:
            return program.replace(src, dst, 1)
    return program + " with a nearby unsupported value"


def build_rejected_from_strict_response(chosen: str) -> str:
    out: List[str] = []
    changed_program = False
    changed_answer = False
    for line in chosen.splitlines():
        if line.startswith("Program:"):
            out.append("Program: " + mutate_program_text(line[len("Program:"):].strip()))
            changed_program = True
        elif line.startswith("Answer:"):
            out.append("Answer: " + mutate_numeric_text(line[len("Answer:"):].strip()))
            changed_answer = True
        else:
            out.append(line)
    if not changed_program:
        out.append("Program: use a nearby unsupported calculation")
    if not changed_answer:
        out.append("Answer: " + mutate_numeric_text("0"))
    return "\n".join(out)

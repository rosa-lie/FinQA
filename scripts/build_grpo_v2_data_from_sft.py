#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_financial_benchmarks import execute_prediction_program, parse_number
from financial_data_processors.common import extract_program_numbers, strict_program_invalid_reasons

FORBIDDEN_REFERENCE_ANCHORS = ("Reasoning:", "Answer:", "Normalized Answer:")
SUPPORTED_DATASET_NAMES = {
    "finqa": "finqa",
    "FinQA": "finqa",
    "convfinqa": "convfinqa_turn",
    "ConvFinQA": "convfinqa_turn",
    "convfinqa_turn": "convfinqa_turn",
}
OP_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
FINQA_HARD_QUESTION_TYPES = {"percentage_change", "share_of_total", "ratio", "margin"}
V19_RATIO_QUESTION_TYPES = {"percentage_change", "share_of_total", "ratio", "margin"}
V19_MIX_VERSION = "v19_ratio_dsl_repair"
HISTORY_CUE_RE = re.compile(
    r"\b(that year|this value|previous one|prior|previous|same|it|that|those|then|there|what about|and what)\b",
    re.IGNORECASE,
)


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return str(value).strip()


def read_jsonl(path: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def conversation_text(row: Dict[str, Any], role_names: set[str]) -> str:
    conversations = row.get("conversations") or row.get("messages") or []
    if not isinstance(conversations, list):
        return ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = first_text(message.get("from") or message.get("role")).lower()
        if role in role_names:
            return first_text(message.get("value") or message.get("content"))
    return ""


def user_text(row: Dict[str, Any]) -> str:
    return conversation_text(row, {"human", "user"})


def assistant_text(row: Dict[str, Any]) -> str:
    return conversation_text(row, {"gpt", "assistant", "model"})


def extract_anchor(text: str, anchor: str) -> str:
    start = first_text(text).find(anchor)
    if start < 0:
        return ""
    start += len(anchor)
    tail = text[start:]
    next_positions = [
        pos for marker in ("Evidence:", "Program:", "Reasoning:", "Answer:", "Normalized Answer:")
        if marker != anchor
        for pos in [tail.find(marker)]
        if pos >= 0
    ]
    end = min(next_positions) if next_positions else len(tail)
    return tail[:end].strip()


def extract_current_question(prompt: str) -> str:
    text = first_text(prompt)
    marker = "Current question:"
    start = text.find(marker)
    if start < 0:
        return text
    start += len(marker)
    end = len(text)
    for next_marker in ("\n\nOutput format:", "\n\nReport context:", "\n\nTable:", "\n\nText before table:"):
        pos = text.find(next_marker, start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end].strip()


def normalize_reference_response(response: str) -> Tuple[str, List[str]]:
    text = first_text(response)
    reasons: List[str] = []
    if not text:
        return "", ["missing_reference_response"]
    if any(anchor in text for anchor in FORBIDDEN_REFERENCE_ANCHORS):
        reasons.append("forbidden_schema")
    evidence = extract_anchor(text, "Evidence:")
    program = extract_anchor(text, "Program:")
    if not program:
        reasons.append("missing_program_anchor")
    if not evidence:
        evidence = "- program evidence unavailable in SFT reference."
    normalized = f"Evidence:\n{evidence.strip()}\n\nProgram: {program.strip()}" if program else ""
    return normalized, reasons


def normalize_source_dataset(raw: Any, source_sft_file: str = "") -> str:
    text = first_text(raw)
    if text in SUPPORTED_DATASET_NAMES:
        return SUPPORTED_DATASET_NAMES[text]
    lower = text.lower()
    if lower in SUPPORTED_DATASET_NAMES:
        return SUPPORTED_DATASET_NAMES[lower]
    file_lower = source_sft_file.lower()
    if "convfinqa" in file_lower:
        return "convfinqa_turn"
    if "finqa" in file_lower:
        return "finqa"
    return lower or "unknown"


def program_ops(program: str) -> List[str]:
    return [match.group(1).lower() for match in OP_RE.finditer(first_text(program))]


def program_step_count(program: str) -> int:
    ops = program_ops(program)
    return len(ops)


def is_direct_lookup_program(program: str) -> bool:
    return not program_ops(program) and len(extract_program_numbers(program)) == 1


def infer_question_type(question: str, gold_program: str) -> str:
    q = first_text(question).lower()
    ops = set(program_ops(gold_program))
    if is_direct_lookup_program(gold_program):
        return "direct_lookup"
    if any(token in q for token in ("percentage change", "percent change", "change from", "change in", "increase", "decrease", "compared")) and {"subtract", "divide"} <= ops:
        return "percentage_change"
    if "margin" in q and "divide" in ops:
        return "margin"
    if "ratio" in q and "divide" in ops:
        return "ratio"
    if "average" in q or "avg" in ops or "average" in ops:
        return "average"
    if any(token in q for token in ("combined", "sum", "total of")) or "sum" in ops:
        return "sum"
    if any(token in q for token in ("how much more", "difference", "how much less")) or ("subtract" in ops and "divide" not in ops):
        return "difference"
    if any(token in q for token in ("percent", "percentage", "of total", "as a percent", "share of", "ratio")) and "divide" in ops:
        return "share_of_total"
    return "other"


def infer_answer_scale(question: str, gold_program: str, gold_answer: str) -> str:
    q = first_text(question).lower()
    p = first_text(gold_program).lower()
    ans_num = parse_number(first_text(gold_answer))
    text = " ".join([q, p, first_text(gold_answer).lower()])
    if "$" in text or "dollar" in text or "million" in text or "billion" in text:
        return "currency"
    if any(token in q for token in ("how many", "number of", "count")):
        return "count"
    asks_percent = any(token in q for token in ("percent", "percentage", "rate", "ratio", "share of", "margin"))
    if asks_percent:
        if "multiply(" in p and "100" in p:
            return "percent"
        if ans_num is not None and abs(ans_num) > 2:
            return "percent"
        return "ratio"
    if ans_num is not None:
        return "plain"
    return "unknown"


def infer_answer_unit(question: str, gold_answer: str, scale: str) -> str:
    q = first_text(question).lower()
    if scale in {"ratio", "percent"}:
        return "percent"
    if scale == "currency":
        if "billion" in q:
            return "billion_currency"
        if "million" in q:
            return "million_currency"
        return "currency"
    if scale == "count":
        return "count"
    return "number" if parse_number(gold_answer) is not None else "unknown"


def normalize_answer_scale(raw_scale: Any, question: str, gold_program: str, gold_answer: str) -> str:
    raw = first_text(raw_scale).lower()
    inferred = infer_answer_scale(question, gold_program, gold_answer)
    if raw in {"ratio", "percent", "currency", "count", "plain", "unknown"}:
        return raw
    if raw in {"absolute", "number", "none"}:
        return inferred
    if raw in {"million", "billion", "thousand"}:
        if inferred in {"ratio", "percent", "count"}:
            return inferred
        return "currency" if "$" in question or "dollar" in question.lower() else "plain"
    return inferred or "unknown"


def weak_requires_history(question: str) -> bool:
    q = first_text(question).lower()
    return bool(HISTORY_CUE_RE.search(q)) or any(token in q for token in ("they", "their"))


def extract_gold_program(row: Dict[str, Any], response: str) -> str:
    metadata = row.get("metadata") or {}
    for key in ("program_canonical", "program_raw", "program", "gold_program"):
        value = first_text(metadata.get(key) or row.get(key))
        if value:
            return value
    return extract_anchor(response, "Program:")


def extract_gold_answer(row: Dict[str, Any], executed_value: Optional[float]) -> str:
    metadata = row.get("metadata") or {}
    for key in ("answer_norm", "answer_exe", "gold_answer", "answer", "normalized_answer"):
        value = first_text(metadata.get(key) or row.get(key))
        if value:
            return value
    if executed_value is None:
        return ""
    if float(executed_value).is_integer():
        return str(int(executed_value))
    return f"{executed_value:.12f}".rstrip("0").rstrip(".")


def numeric_close(left: Optional[float], right: Optional[float], abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(left, right, abs_tol=abs_tol, rel_tol=rel_tol)


def convert_sft_row(row: Dict[str, Any], *, source_sft_file: str = "") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    prompt = first_text(row.get("input_prompt_raw")) or user_text(row)
    response = first_text(row.get("reference_response")) or assistant_text(row)
    metadata = dict(row.get("metadata") or {})
    reasons: List[str] = []
    if not prompt:
        reasons.append("missing_input_prompt_raw")
    normalized_reference, reference_reasons = normalize_reference_response(response)
    reasons.extend(reference_reasons)

    gold_program = extract_gold_program(row, response)
    if not gold_program:
        reasons.append("missing_gold_program")
    strict_reasons = strict_program_invalid_reasons(gold_program)
    if strict_reasons:
        reasons.append("strict_dsl_invalid:" + ",".join(strict_reasons))
    executed_value, canonical_program, execute_error = execute_prediction_program(gold_program)
    if executed_value is None:
        reasons.append("gold_program_execute_fail:" + first_text(execute_error))
    gold_answer = extract_gold_answer(row, executed_value)
    gold_num = parse_number(gold_answer)
    if not gold_answer:
        reasons.append("missing_gold_answer")
    elif executed_value is not None and not numeric_close(executed_value, gold_num):
        reasons.append("gold_answer_mismatch")

    if reasons:
        return None, {
            "record_id": first_text(row.get("record_id")),
            "source_dataset": normalize_source_dataset(row.get("source_dataset"), source_sft_file),
            "reasons": reasons,
            "gold_program": gold_program,
            "gold_answer": gold_answer,
            "source_sft_file": source_sft_file,
        }

    source_dataset = normalize_source_dataset(row.get("source_dataset") or metadata.get("source_dataset"), source_sft_file)
    canonical = first_text(canonical_program) or gold_program
    current_question = extract_current_question(prompt)
    question_type = infer_question_type(current_question, canonical)
    answer_scale = normalize_answer_scale(metadata.get("answer_scale"), current_question, canonical, gold_answer)
    answer_unit = first_text(metadata.get("answer_unit")) or infer_answer_unit(prompt, gold_answer, answer_scale)
    requires_history = metadata.get("requires_history")
    if requires_history is None:
        requires_history = bool(source_dataset == "convfinqa_turn" and weak_requires_history(prompt))

    enriched_metadata = dict(metadata)
    enriched_metadata.update(
        {
            "answer_scale": answer_scale or "unknown",
            "answer_unit": answer_unit or "unknown",
            "question_type": question_type,
            "program_ops": program_ops(canonical),
            "program_step_count": program_step_count(canonical),
            "program_numeric_args": extract_program_numbers(canonical),
            "direct_lookup": is_direct_lookup_program(canonical),
            "requires_history": bool(requires_history),
            "history_dependency_type": first_text(metadata.get("history_dependency_type")),
            "source_sft_file": source_sft_file,
        }
    )

    return {
        "input_prompt_raw": prompt,
        "gold_answer": gold_answer,
        "gold_program": canonical,
        "reference_response": normalized_reference,
        "reward_profile": "program_numeric",
        "source_dataset": source_dataset,
        "record_id": first_text(row.get("record_id")),
        "task_type": "program_numeric",
        "metadata": enriched_metadata,
    }, None


def convert_rows(rows: Iterable[Dict[str, Any]], *, source_sft_file: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    converted: List[Dict[str, Any]] = []
    bad_cases: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    seen_prompts: set[str] = set()
    for row in rows:
        item, bad = convert_sft_row(row, source_sft_file=source_sft_file)
        if bad is not None:
            bad_cases.append(bad)
            continue
        assert item is not None
        key = (first_text(item.get("source_dataset")), first_text(item.get("record_id")) or first_text(item.get("input_prompt_raw")))
        if key in seen:
            bad_cases.append({
                "record_id": item.get("record_id"),
                "source_dataset": item.get("source_dataset"),
                "reasons": ["duplicate_record"],
                "source_sft_file": source_sft_file,
            })
            continue
        prompt_key = first_text(item.get("input_prompt_raw"))
        if prompt_key in seen_prompts:
            bad_cases.append({
                "record_id": item.get("record_id"),
                "source_dataset": item.get("source_dataset"),
                "reasons": ["duplicate_prompt"],
                "source_sft_file": source_sft_file,
            })
            continue
        seen.add(key)
        seen_prompts.add(prompt_key)
        converted.append(item)
    return converted, bad_cases


def summarize(rows: List[Dict[str, Any]], bad_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    source = Counter(first_text(row.get("source_dataset")) for row in rows)
    question_type = Counter(first_text((row.get("metadata") or {}).get("question_type")) for row in rows)
    answer_scale = Counter(first_text((row.get("metadata") or {}).get("answer_scale")) for row in rows)
    step_count = Counter(str((row.get("metadata") or {}).get("program_step_count")) for row in rows)
    requires_history = Counter("true" if bool((row.get("metadata") or {}).get("requires_history")) else "false" for row in rows)
    return {
        "rows": len(rows),
        "bad_case_count": len(bad_cases),
        "source_dataset": dict(source.most_common()),
        "question_type": dict(question_type.most_common()),
        "answer_scale": dict(answer_scale.most_common()),
        "program_step_count": dict(step_count.most_common()),
        "requires_history": dict(requires_history.most_common()),
        "bad_reasons": dict(Counter(reason for bad in bad_cases for reason in bad.get("reasons", [])).most_common()),
    }


def is_finqa_hard_row(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if first_text(row.get("source_dataset")) != "finqa":
        return False
    question_type = first_text(metadata.get("question_type"))
    step_count = int(metadata.get("program_step_count") or 0)
    direct_lookup = bool(metadata.get("direct_lookup")) or question_type == "direct_lookup"
    return not direct_lookup and (step_count >= 2 or question_type in FINQA_HARD_QUESTION_TYPES)


def is_convfinqa_history_row(row: Dict[str, Any]) -> bool:
    if first_text(row.get("source_dataset")) != "convfinqa_turn":
        return False
    metadata = row.get("metadata") or {}
    return bool(metadata.get("requires_history")) or weak_requires_history(first_text(row.get("input_prompt_raw")))


def is_v19_ratio_hard_row(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if first_text(row.get("source_dataset")) != "finqa":
        return False
    question_type = first_text(metadata.get("question_type"))
    answer_scale = first_text(metadata.get("answer_scale"))
    direct_lookup = bool(metadata.get("direct_lookup")) or question_type == "direct_lookup"
    return not direct_lookup and (question_type in V19_RATIO_QUESTION_TYPES or answer_scale in {"ratio", "percent"})


def is_v19_history_non_direct_row(row: Dict[str, Any]) -> bool:
    if first_text(row.get("source_dataset")) != "convfinqa_turn":
        return False
    metadata = row.get("metadata") or {}
    question_type = first_text(metadata.get("question_type"))
    direct_lookup = bool(metadata.get("direct_lookup")) or question_type == "direct_lookup"
    return is_convfinqa_history_row(row) and not direct_lookup


def clone_with_v19_bucket(
    row: Dict[str, Any],
    bucket: str,
    *,
    bad_case_error_types: Optional[set[str]] = None,
) -> Dict[str, Any]:
    copied = dict(row)
    copied["metadata"] = dict(row.get("metadata") or {})
    copied["metadata"]["v19_mix_bucket"] = bucket
    copied["metadata"]["v19_mix_version"] = V19_MIX_VERSION
    if bucket == "bad_case_replay" and bad_case_error_types:
        copied["metadata"]["v19_bad_case_error_types"] = sorted(bad_case_error_types)
    return copied


def _sample_bucket(
    rows: List[Dict[str, Any]],
    *,
    count: int,
    rng: random.Random,
    bucket_name: str,
    bad_case_error_types: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected: List[Dict[str, Any]] = []
    while len(selected) < count:
        needed = count - len(selected)
        selected.extend(shuffled[:needed])
        if needed >= len(shuffled):
            rng.shuffle(shuffled)
    return [
        clone_with_v19_bucket(row, bucket_name, bad_case_error_types=bad_case_error_types)
        for row in selected[:count]
    ]


def _bucket_count(target_rows: int, weight: float) -> int:
    return max(0, int(round(target_rows * weight)))


def build_v19_train_mix(
    rows: List[Dict[str, Any]],
    *,
    bad_case_record_ids: Optional[set[str]] = None,
    bad_case_question_types: Optional[set[str]] = None,
    bad_case_error_types: Optional[set[str]] = None,
    seed: int = 42,
    target_rows: int = 0,
    ratio_weight: float = 0.40,
    history_weight: float = 0.35,
    bad_case_weight: float = 0.15,
    easy_weight: float = 0.10,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    rng = random.Random(seed)
    target = target_rows if target_rows > 0 else len(rows)
    bad_ids = bad_case_record_ids or set()
    bad_types = bad_case_question_types or set()
    bad_errors = bad_case_error_types or set()

    ratio_rows = [row for row in rows if is_v19_ratio_hard_row(row)]
    history_rows = [row for row in rows if is_v19_history_non_direct_row(row)]
    if not history_rows:
        history_rows = [row for row in rows if is_convfinqa_history_row(row)]
    bad_rows = [row for row in rows if first_text(row.get("record_id")) in bad_ids]
    if not bad_rows and ("ratio_or_percent" in bad_types or bad_types.intersection(V19_RATIO_QUESTION_TYPES)):
        bad_rows = list(ratio_rows)
    if not bad_rows and bad_errors:
        replay_rows: List[Dict[str, Any]] = []
        if bad_errors.intersection({"missing_ratio_divide", "wrong_number_or_table_cell"}):
            replay_rows.extend(ratio_rows)
        if bad_errors.intersection({"wrong_operation", "calculation_reduced_to_lookup"}):
            replay_rows.extend(
                row
                for row in rows
                if not bool((row.get("metadata") or {}).get("direct_lookup"))
                and first_text((row.get("metadata") or {}).get("question_type")) != "direct_lookup"
            )
        if bad_errors.intersection({"execution_fail"}):
            replay_rows.extend(
                row
                for row in rows
                if int((row.get("metadata") or {}).get("program_step_count") or 0) >= 2
            )
        seen_keys = set()
        bad_rows = []
        for row in replay_rows:
            key = (first_text(row.get("source_dataset")), first_text(row.get("record_id")))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bad_rows.append(row)
    easy_rows = [
        row
        for row in rows
        if row not in ratio_rows and row not in history_rows and row not in bad_rows
    ]
    if not easy_rows:
        easy_rows = list(rows)

    counts = {
        "finqa_ratio_hard": _bucket_count(target, ratio_weight),
        "convfinqa_history_non_direct": _bucket_count(target, history_weight),
        "bad_case_replay": _bucket_count(target, bad_case_weight),
    }
    counts["easy_replay"] = max(0, target - sum(counts.values()))

    mixed: List[Dict[str, Any]] = []
    mixed.extend(_sample_bucket(ratio_rows, count=counts["finqa_ratio_hard"], rng=rng, bucket_name="finqa_ratio_hard"))
    mixed.extend(
        _sample_bucket(
            history_rows,
            count=counts["convfinqa_history_non_direct"],
            rng=rng,
            bucket_name="convfinqa_history_non_direct",
        )
    )
    mixed.extend(
        _sample_bucket(
            bad_rows,
            count=counts["bad_case_replay"],
            rng=rng,
            bucket_name="bad_case_replay",
            bad_case_error_types=bad_errors,
        )
    )
    mixed.extend(_sample_bucket(easy_rows, count=counts["easy_replay"], rng=rng, bucket_name="easy_replay"))

    if len(mixed) < target:
        fallback = ratio_rows + history_rows + bad_rows + easy_rows
        mixed.extend(_sample_bucket(fallback, count=target - len(mixed), rng=rng, bucket_name="fallback_replay"))

    rng.shuffle(mixed)
    return mixed[:target]


def read_bad_case_replay_profile(path_str: str) -> Tuple[set[str], set[str], set[str]]:
    if not path_str:
        return set(), set(), set()
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing V19 bad case file: {path}")
    if path.suffix == ".jsonl":
        ids = set()
        question_types = set()
        error_types = set()
        for row in read_jsonl(path):
            record_id = first_text(row.get("record_id"))
            if record_id:
                ids.add(record_id)
            question_type = first_text(row.get("question_type"))
            if question_type:
                question_types.add(question_type)
            error_type = first_text(row.get("error_type"))
            if error_type:
                error_types.add(error_type)
        return ids, question_types, error_types
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        ids = set()
        question_types = set()
        error_types = set()
        for row in reader:
            record_id = first_text(row.get("record_id"))
            if record_id:
                ids.add(record_id)
            question_type = first_text(row.get("question_type"))
            if question_type:
                question_types.add(question_type)
            error_type = first_text(row.get("error_type"))
            if error_type:
                error_types.add(error_type)
        return ids, question_types, error_types


def build_v18_specialized_splits(
    rows: List[Dict[str, Any]],
    *,
    seed: int = 42,
    max_finqa_hard_rows: int = 0,
    max_convfinqa_history_rows: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    finqa_hard = [row for row in rows if is_finqa_hard_row(row)]
    conv_history = []
    for row in rows:
        if not is_convfinqa_history_row(row):
            continue
        copied = dict(row)
        copied["metadata"] = dict(row.get("metadata") or {})
        copied["metadata"]["requires_history"] = True
        copied["metadata"]["history_split_source"] = (
            "explicit" if bool((row.get("metadata") or {}).get("requires_history")) else "weak_rule"
        )
        conv_history.append(copied)
    rng.shuffle(finqa_hard)
    rng.shuffle(conv_history)
    if max_finqa_hard_rows > 0:
        finqa_hard = finqa_hard[:max_finqa_hard_rows]
    if max_convfinqa_history_rows > 0:
        conv_history = conv_history[:max_convfinqa_history_rows]
    return finqa_hard, conv_history


def split_rows(rows: List[Dict[str, Any]], *, seed: int, valid_ratio: float, smoke_rows: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    valid_count = int(round(len(shuffled) * valid_ratio))
    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]
    smoke = train[: min(smoke_rows, len(train))]
    return train, valid, smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enriched GRPO v2 program_numeric data from strict SFT JSONL files.")
    parser.add_argument("--finqa_sft_file", required=True)
    parser.add_argument("--convfinqa_sft_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--valid_ratio", type=float, default=0.12)
    parser.add_argument("--smoke_rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finqa_hard_output_dir", default="")
    parser.add_argument("--convfinqa_history_output_dir", default="")
    parser.add_argument("--max_finqa_hard_rows", type=int, default=0)
    parser.add_argument("--max_convfinqa_history_rows", type=int, default=0)
    parser.add_argument("--v19_mix_output_dir", default="")
    parser.add_argument("--v19_bad_case_file", default="")
    parser.add_argument("--v19_target_rows", type=int, default=0)
    parser.add_argument("--v19_ratio_weight", type=float, default=0.40)
    parser.add_argument("--v19_history_weight", type=float, default=0.35)
    parser.add_argument("--v19_bad_case_weight", type=float, default=0.15)
    parser.add_argument("--v19_easy_weight", type=float, default=0.10)
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    all_bad: List[Dict[str, Any]] = []
    for path_str in [args.finqa_sft_file, args.convfinqa_sft_file]:
        path = Path(path_str)
        rows = read_jsonl(path, args.max_rows)
        converted, bad = convert_rows(rows, source_sft_file=path.name)
        all_rows.extend(converted)
        all_bad.extend(bad)

    train, valid, smoke = split_rows(all_rows, seed=args.seed, valid_ratio=args.valid_ratio, smoke_rows=args.smoke_rows)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train_grpo_v2_pool.jsonl", train)
    write_jsonl(output_dir / "valid_grpo_v2_pool.jsonl", valid)
    write_jsonl(output_dir / "smoke_grpo_v2_pool.jsonl", smoke)
    write_jsonl(output_dir / "data_enrichment_bad_cases.jsonl", all_bad)
    summary = summarize(all_rows, all_bad)
    summary.update({"train_rows": len(train), "valid_rows": len(valid), "smoke_rows": len(smoke)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_enrichment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    finqa_hard_rows, conv_history_rows = build_v18_specialized_splits(
        all_rows,
        seed=args.seed,
        max_finqa_hard_rows=args.max_finqa_hard_rows,
        max_convfinqa_history_rows=args.max_convfinqa_history_rows,
    )
    for out_dir, split_rows_all, split_name in (
        (args.finqa_hard_output_dir, finqa_hard_rows, "FinQA-Hard"),
        (args.convfinqa_history_output_dir, conv_history_rows, "ConvFinQA-History"),
    ):
        if not out_dir:
            continue
        split_train, split_valid, split_smoke = split_rows(
            split_rows_all,
            seed=args.seed,
            valid_ratio=args.valid_ratio,
            smoke_rows=args.smoke_rows,
        )
        split_output_dir = Path(out_dir)
        write_jsonl(split_output_dir / "train_grpo_v18_pool.jsonl", split_train)
        write_jsonl(split_output_dir / "valid_grpo_v18_pool.jsonl", split_valid)
        write_jsonl(split_output_dir / "smoke_grpo_v18_pool.jsonl", split_smoke)
        split_summary = summarize(split_rows_all, [])
        split_summary.update(
            {
                "split_name": split_name,
                "train_rows": len(split_train),
                "valid_rows": len(split_valid),
                "smoke_rows": len(split_smoke),
            }
        )
        split_output_dir.mkdir(parents=True, exist_ok=True)
        (split_output_dir / "data_enrichment_summary.json").write_text(
            json.dumps(split_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.v19_mix_output_dir:
        bad_case_ids, bad_case_question_types, bad_case_error_types = read_bad_case_replay_profile(args.v19_bad_case_file)
        v19_rows = build_v19_train_mix(
            all_rows,
            bad_case_record_ids=bad_case_ids,
            bad_case_question_types=bad_case_question_types,
            bad_case_error_types=bad_case_error_types,
            seed=args.seed,
            target_rows=args.v19_target_rows,
            ratio_weight=args.v19_ratio_weight,
            history_weight=args.v19_history_weight,
            bad_case_weight=args.v19_bad_case_weight,
            easy_weight=args.v19_easy_weight,
        )
        v19_train, v19_valid, v19_smoke = split_rows(
            v19_rows,
            seed=args.seed,
            valid_ratio=args.valid_ratio,
            smoke_rows=args.smoke_rows,
        )
        v19_output_dir = Path(args.v19_mix_output_dir)
        write_jsonl(v19_output_dir / "train_grpo_v19_mix.jsonl", v19_train)
        write_jsonl(v19_output_dir / "valid_grpo_v19_mix.jsonl", v19_valid)
        write_jsonl(v19_output_dir / "smoke_grpo_v19_mix.jsonl", v19_smoke)
        v19_summary = summarize(v19_rows, [])
        v19_summary.update(
            {
                "split_name": "V19-Ratio-DSL-Repair",
                "mix_version": V19_MIX_VERSION,
                "bad_case_record_ids": len(bad_case_ids),
                "bad_case_question_types": sorted(bad_case_question_types),
                "target_rows": len(v19_rows),
                "train_rows": len(v19_train),
                "valid_rows": len(v19_valid),
                "smoke_rows": len(v19_smoke),
                "mix_bucket": dict(
                    Counter(first_text((row.get("metadata") or {}).get("v19_mix_bucket")) for row in v19_rows).most_common()
                ),
                "weights": {
                    "ratio": args.v19_ratio_weight,
                    "history": args.v19_history_weight,
                    "bad_case": args.v19_bad_case_weight,
                    "easy": args.v19_easy_weight,
                },
            }
        )
        v19_output_dir.mkdir(parents=True, exist_ok=True)
        (v19_output_dir / "data_enrichment_summary.json").write_text(
            json.dumps(v19_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

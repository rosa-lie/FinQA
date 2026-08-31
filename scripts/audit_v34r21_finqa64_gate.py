#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


FORBIDDEN_MARKERS = [
    "Reasoning:",
    "Operation Plan:",
    "Formula candidates:",
    "Task Attributes:",
    "Answer:",
    "Normalized Answer:",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_summary(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    return obj.get("summary_rows") or obj.get("rows") or []


def metric(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None and isinstance(row.get("metrics"), dict):
        value = row["metrics"].get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def first_program(prediction: str) -> str:
    match = re.search(r"Program:\s*(.*)", prediction or "", re.DOTALL)
    if not match:
        return ""
    program = match.group(1).strip().splitlines()[0].strip() if match.group(1).strip() else ""
    if program.lower().startswith("program:"):
        program = program.split(":", 1)[1].strip()
    return program


def dsl_flags(prediction: str) -> Dict[str, int]:
    program = first_program(prediction)
    numeric_infix = bool(re.search(r"-?\d[\d.,\s]*[+\-*/]\s*-?\d", program))
    paren_infix = any(token in program for token in (") +", ") -", ") *", ") /"))
    return {
        "unsupported_infix": int(numeric_infix or paren_infix),
        "symbolic_argument": int(bool(re.search(r"\\([^)A-Za-z]*(?:[A-Za-z_][A-Za-z0-9_]*)[^)]*\\)", program)) and not bool(re.fullmatch(r"[a-z_]+\\([^)]*\\)", program))),
        "assignment_wrapper": int("=" in program),
        "multiple_program": int((prediction or "").count("Program:") > 1),
        "forbidden_marker": int(any(marker in (prediction or "") for marker in FORBIDDEN_MARKERS)),
    }


def by_record(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        if row.get("generation_mode") == "greedy":
            rows[str(row.get("record_id"))] = row
    return rows


def is_correct(row: Dict[str, Any]) -> bool:
    return metric(row, "executed_answer_accuracy") > 0


def compare(base: Dict[str, Dict[str, Any]], current: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    new_correct = []
    new_wrong = []
    for rid, cur in current.items():
        if rid not in base:
            continue
        if is_correct(cur) and not is_correct(base[rid]):
            new_correct.append(rid)
        if is_correct(base[rid]) and not is_correct(cur):
            new_wrong.append(rid)
    return {
        "new_correct_count": len(new_correct),
        "new_wrong_count": len(new_wrong),
        "new_correct": sorted(new_correct),
        "new_wrong": sorted(new_wrong),
    }


def summarize_model(eval_dir: Path, model: str) -> Dict[str, Any]:
    summary_row = next(row for row in load_summary(eval_dir / "benchmark_summary.json") if row.get("model_name") == model and row.get("task_name") == "finqa_test")
    pred_rows = list(by_record(eval_dir / f"{model}_predictions.jsonl").values())
    flags = Counter()
    for row in pred_rows:
        flags.update(dsl_flags(str(row.get("prediction") or "")))
    denom = max(len(pred_rows), 1)
    return {
        "model_name": model,
        "num_examples": int(summary_row.get("num_examples") or len(pred_rows)),
        "greedy_pass1": metric(summary_row, "executed_answer_accuracy"),
        "parse_rate": metric(summary_row, "program_parse_rate"),
        "execution_rate": metric(summary_row, "program_execution_rate"),
        "operation_match": metric(summary_row, "operation_match_rate"),
        "argument_grounding": metric(summary_row, "argument_grounding_rate"),
        "scale_consistency": metric(summary_row, "scale_consistency_rate"),
        "evidence_grounding": metric(summary_row, "evidence_grounding_rate"),
        "strict_contract_rate": 1.0 - (flags["forbidden_marker"] / denom),
        "unsupported_infix_rate": flags["unsupported_infix"] / denom,
        "symbolic_argument_rate": flags["symbolic_argument"] / denom,
        "assignment_wrapper_rate": flags["assignment_wrapper"] / denom,
        "multiple_program_rate": flags["multiple_program"] / denom,
        "forbidden_marker_count": flags["forbidden_marker"],
    }


def classify_case(row: Dict[str, Any]) -> List[str]:
    prediction = str(row.get("prediction") or "")
    program = first_program(prediction).lower()
    flags = []
    if "divide" not in program and any(word in prediction.lower() for word in ("percent", "percentage", "per ", "ratio", "growth")):
        flags.append("missing_divide")
    if re.search(r"divide\\([^,]+,\\s*(?:1|100)\\)", program):
        flags.append("wrong_denominator_candidate")
    if any(marker in prediction for marker in FORBIDDEN_MARKERS):
        flags.append("forbidden_marker")
    dsl = dsl_flags(prediction)
    flags.extend(name for name, value in dsl.items() if value and name != "forbidden_marker")
    return sorted(set(flags))


def badcases(eval_dir: Path, base_model: str, current_model: str) -> List[Dict[str, Any]]:
    base = by_record(eval_dir / f"{base_model}_predictions.jsonl")
    current = by_record(eval_dir / f"{current_model}_predictions.jsonl")
    rows = []
    for rid in sorted(set(base) & set(current)):
        base_ok = is_correct(base[rid])
        cur_ok = is_correct(current[rid])
        if base_ok == cur_ok and not classify_case(current[rid]):
            continue
        if base_ok and not cur_ok:
            direction = f"{base_model}_correct__{current_model}_wrong"
        elif cur_ok and not base_ok:
            direction = f"{base_model}_wrong__{current_model}_correct"
        else:
            direction = "dsl_or_contract_audit"
        rows.append(
            {
                "record_id": rid,
                "direction": direction,
                "base_correct": base_ok,
                "current_correct": cur_ok,
                "audit_flags": classify_case(current[rid]),
                "base_program": first_program(str(base[rid].get("prediction") or "")),
                "current_program": first_program(str(current[rid].get("prediction") or "")),
                "gold_answer": current[rid].get("gold_answer"),
                "executed_answer": current[rid].get("executed_answer"),
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\\n")


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    eval_dir = Path(args.eval_dir)
    models = args.model
    summaries = [summarize_model(eval_dir, model) for model in models]
    predictions = {model: by_record(eval_dir / f"{model}_predictions.jsonl") for model in models}
    comparisons = {}
    for base in args.compare_base:
        for model in models:
            if model != base and base in predictions:
                comparisons[f"{model}_vs_{base}"] = compare(predictions[base], predictions[model])
    badcase_rows = []
    for pair in args.badcase_pair:
        base, current = pair.split("=", 1)
        badcase_rows.extend(badcases(eval_dir, base, current))
    result = {"summary_rows": summaries, "comparisons": comparisons, "badcase_count": len(badcase_rows)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finqa64_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "finqa64_badcases.jsonl", badcase_rows)
    write_jsonl(output_dir / "finqa64_dsl_regressions.jsonl", [row for row in badcase_rows if row["audit_flags"]])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the v34r21 single-seed FinQA-64 gate.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--compare_base", action="append", default=[])
    parser.add_argument("--badcase_pair", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

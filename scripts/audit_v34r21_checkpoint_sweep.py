#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_summary(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("summary_rows") or obj.get("rows") or []
    return []


def metric(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        value = row.get("metrics", {}).get(key) if isinstance(row.get("metrics"), dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def pred_model_files(eval_dir: Path) -> List[Path]:
    return sorted(eval_dir.glob("*_predictions.jsonl"))


def program_from_prediction(prediction: str) -> str:
    match = re.search(r"Program:\s*(.*)", prediction or "", re.DOTALL)
    if not match:
        return ""
    lines = match.group(1).strip().splitlines()
    if not lines:
        return ""
    text = lines[0].strip()
    if text.lower().startswith("program:"):
        text = text.split(":", 1)[1].strip()
    return text


def dsl_flags(prediction: str) -> Dict[str, int]:
    program = program_from_prediction(prediction)
    return {
        "unsupported_infix": int(bool(re.search(r"(?<![A-Za-z_])[\d\).\s]+[/*][\d\(.\s-]+", program))),
        "symbolic_argument": int(bool(re.search(r"\([^\)]*[A-Za-z_][A-Za-z_]+[^\)]*\)", program)) and not bool(re.match(r"^[a-z_]+\(", program))),
        "assignment_wrapper": int("=" in program),
        "multiple_program": int((prediction or "").count("Program:") > 1),
        "forbidden_answer_anchor": int("Answer:" in (prediction or "") or "Normalized Answer:" in (prediction or "")),
        "forbidden_reasoning_anchor": int("Reasoning:" in (prediction or "") or "Formula" in (prediction or "")),
    }


def greedy_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        if row.get("generation_mode") == "greedy":
            rows[str(row.get("record_id"))] = row
    return rows


def compare_predictions(base: Dict[str, Dict[str, Any]], current: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    new_correct = []
    new_wrong = []
    for rid, cur in current.items():
        if rid not in base:
            continue
        base_ok = metric(base[rid], "executed_answer_accuracy") > 0
        cur_ok = metric(cur, "executed_answer_accuracy") > 0
        if cur_ok and not base_ok:
            new_correct.append(rid)
        elif base_ok and not cur_ok:
            new_wrong.append(rid)
    return {"new_correct": new_correct, "new_wrong": new_wrong, "new_correct_count": len(new_correct), "new_wrong_count": len(new_wrong)}


def summarize_eval(eval_dir: Path, model_name: str) -> Dict[str, Any]:
    summary_rows = load_summary(eval_dir / "benchmark_summary.json")
    row = next((item for item in summary_rows if item.get("model_name") == model_name), summary_rows[0] if summary_rows else {})
    pred_path = eval_dir / f"{model_name}_predictions.jsonl"
    rows = [row for row in read_jsonl(pred_path) if row.get("generation_mode") == "greedy"]
    flag_counts = Counter()
    for item in rows:
        flag_counts.update(dsl_flags(str(item.get("prediction") or "")))
    denom = max(len(rows), 1)
    return {
        "model_name": model_name,
        "eval_dir": str(eval_dir),
        "num_examples": int(row.get("num_examples") or len(rows) or 0),
        "greedy_pass1": metric(row, "pass@1_greedy") or metric(row, "executed_answer_accuracy"),
        "parse_rate": metric(row, "program_parse_rate"),
        "execution_rate": metric(row, "program_execution_rate"),
        "strict_contract_rate": metric(row, "strict_program_contract_rate"),
        "operation_match": metric(row, "operation_match_rate"),
        "argument_grounding": metric(row, "argument_grounding_rate"),
        "scale_consistency": metric(row, "scale_consistency_rate"),
        "evidence_grounding": metric(row, "evidence_grounding_rate"),
        "dsl_rates": {key + "_rate": value / denom for key, value in flag_counts.items()},
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    entries = [item.split("=", 2) for item in args.eval_entry]
    rows = []
    predictions = {}
    for label, model_name, eval_dir in entries:
        eval_path = Path(eval_dir)
        rows.append({"label": label, **summarize_eval(eval_path, model_name)})
        predictions[label] = greedy_rows(eval_path / f"{model_name}_predictions.jsonl")
    base_label = args.sft2_label
    ref_label = args.v34r20_label
    comparisons = {}
    for label in predictions:
        if label == base_label:
            continue
        comparisons[f"{label}_vs_{base_label}"] = compare_predictions(predictions.get(base_label, {}), predictions[label])
        if ref_label in predictions and label != ref_label:
            comparisons[f"{label}_vs_{ref_label}"] = compare_predictions(predictions.get(ref_label, {}), predictions[label])
    result = {"summary_rows": rows, "comparisons": comparisons}
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v34r21 checkpoint sweep predictions and summaries.")
    parser.add_argument("--eval_entry", action="append", required=True, help="label=model_name=eval_dir")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--sft2_label", default="sft2")
    parser.add_argument("--v34r20_label", default="v34r20_ckpt30")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

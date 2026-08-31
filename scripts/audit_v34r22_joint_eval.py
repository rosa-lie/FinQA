#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FORBIDDEN_RE = re.compile(
    r"(?im)^\s*(?:Reasoning|Answer|Normalized Answer|Operation Plan|Formula candidates|Formula|Task Attributes)\s*:"
)
PROGRAM_RE = re.compile(r"(?im)^\s*Program\s*:")
INFIX_RE = re.compile(r"^\s*-?\d[\d,]*(?:\.\d+)?%?\s*[+\-*/]\s*-?\d[\d,]*(?:\.\d+)?%?\s*$")
SYMBOLIC_ARG_RE = re.compile(r"\([^\)]*[A-Za-z_][A-Za-z0-9_]*(?!\s*\()[^\)]*\)")
ASSIGNMENT_RE = re.compile(r"(?im)^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_bucket(row: Dict[str, Any]) -> str:
    task = str(row.get("task_name") or "").lower()
    source = str(((row.get("metadata") or {}).get("source_dataset")) or "").lower()
    if "conv" in task or "conv" in source:
        return "convfinqa"
    if "finqa" in task or "finqa" in source:
        return "finqa"
    return "unknown"


def program_text(row: Dict[str, Any]) -> str:
    return str(row.get("executed_program") or "")


def response_text(row: Dict[str, Any]) -> str:
    return str(row.get("prediction") or "")


def requires_history(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if bool(metadata.get("requires_history")):
        return True
    if int(metadata.get("history_turns") or 0) > 0:
        return True
    prompt = str(row.get("prompt") or "")
    return "Previous question:" in prompt or "Conversation history:" in prompt


def dsl_flags(row: Dict[str, Any]) -> Dict[str, int]:
    text = response_text(row)
    program = program_text(row)
    return {
        "forbidden_marker": int(bool(FORBIDDEN_RE.search(text))),
        "unsupported_infix": int(bool(INFIX_RE.fullmatch(program.strip()))),
        "symbolic_argument": int(bool(SYMBOLIC_ARG_RE.search(program))),
        "assignment_wrapper": int(bool(ASSIGNMENT_RE.search(text))),
        "multiple_program": int(len(PROGRAM_RE.findall(text)) > 1),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"num_examples": 0}
    flags = Counter()
    for row in rows:
        flags.update(dsl_flags(row))
    return {
        "num_examples": total,
        "pass@1": sum(float(row.get("answer_correct") or 0.0) for row in rows) / total,
        "parse_rate": sum(float(row.get("program_parse_rate") or 0.0) for row in rows) / total,
        "execution_rate": sum(float(row.get("program_execution_rate") or 0.0) for row in rows) / total,
        "strict_contract_rate": 1.0 - flags["forbidden_marker"] / total - flags["multiple_program"] / total,
        "forbidden_marker_rate": flags["forbidden_marker"] / total,
        "unsupported_infix_rate": flags["unsupported_infix"] / total,
        "symbolic_argument_rate": flags["symbolic_argument"] / total,
        "assignment_wrapper_rate": flags["assignment_wrapper"] / total,
        "multiple_program_rate": flags["multiple_program"] / total,
        "operation_match": avg(rows, "operation_match"),
        "argument_grounding": avg(rows, "argument_grounding"),
        "scale_consistency": avg(rows, "scale_consistency"),
        "evidence_grounding": avg(rows, "evidence_grounding"),
    }


def avg(rows: List[Dict[str, Any]], key: str) -> Any:
    vals = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            value = row.get(f"{key}_rate")
        if value in ("", None):
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return "" if not vals else sum(vals) / len(vals)


def by_record(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("record_id")): row for row in rows if str(row.get("generation_mode")) == "greedy"}


def paired_delta(base: Dict[str, Dict[str, Any]], cand: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    shared = sorted(set(base) & set(cand))
    new_correct = []
    new_wrong = []
    for rid in shared:
        b = float(base[rid].get("answer_correct") or 0.0) > 0
        c = float(cand[rid].get("answer_correct") or 0.0) > 0
        if (not b) and c:
            new_correct.append(rid)
        elif b and (not c):
            new_wrong.append(rid)
    return {
        "shared": len(shared),
        "new_correct_count": len(new_correct),
        "new_wrong_count": len(new_wrong),
        "new_correct": new_correct,
        "new_wrong": new_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline", default="sft2")
    parser.add_argument("--original_v34", default="v34")
    args = parser.parse_args()

    out = Path(args.output_dir)
    model_rows: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(out.glob("*_predictions.jsonl")):
        name = path.name.removesuffix("_predictions.jsonl")
        rows = [row for row in read_jsonl(path) if str(row.get("generation_mode")) == "greedy"]
        model_rows[name] = rows

    metrics: Dict[str, Any] = {}
    for name, rows in model_rows.items():
        split_rows = defaultdict(list)
        for row in rows:
            split_rows[task_bucket(row)].append(row)
        task_metrics = {task: aggregate(items) for task, items in split_rows.items()}
        fin = task_metrics.get("finqa", {}).get("pass@1")
        conv = task_metrics.get("convfinqa", {}).get("pass@1")
        task_metrics["macro"] = {
            "pass@1": (fin + conv) / 2 if isinstance(fin, float) and isinstance(conv, float) else "",
            "num_examples": sum(len(items) for items in split_rows.values()),
        }
        hist = [row for row in rows if task_bucket(row) == "convfinqa" and requires_history(row)]
        non_hist = [row for row in rows if task_bucket(row) == "convfinqa" and not requires_history(row)]
        task_metrics["convfinqa_history"] = aggregate(hist)
        task_metrics["convfinqa_non_history"] = aggregate(non_hist)
        metrics[name] = task_metrics

    deltas: Dict[str, Any] = {}
    base_map = by_record(model_rows.get(args.baseline, []))
    v34_map = by_record(model_rows.get(args.original_v34, []))
    for name, rows in model_rows.items():
        if name != args.baseline and base_map:
            deltas[f"{name}_vs_{args.baseline}"] = paired_delta(base_map, by_record(rows))
        if name != args.original_v34 and v34_map:
            deltas[f"{name}_vs_{args.original_v34}"] = paired_delta(v34_map, by_record(rows))

    summary = {"metrics": metrics, "paired_deltas": deltas}
    (out / "joint_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

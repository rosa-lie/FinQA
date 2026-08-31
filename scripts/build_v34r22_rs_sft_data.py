#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from financial_data_processors.common import canonicalize_program_re, execute_program


FORBIDDEN_RE = re.compile(
    r"(?im)^\s*(?:Reasoning|Answer|Normalized Answer|Operation Plan|Formula candidates|Formula|Task Attributes)\s*:"
)
PROGRAM_RE = re.compile(r"(?im)^\s*Program\s*:")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_record_id(row: Dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or "")


def answer_text(row: Dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    if conversations and len(conversations) >= 2:
        return str(conversations[-1].get("value") or "")
    return str(row.get("reference_response") or row.get("target") or "")


def prompt_text(row: Dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    if conversations:
        for turn in conversations:
            if turn.get("from") == "human":
                return str(turn.get("value") or "")
    return str(row.get("input_prompt_raw") or row.get("prompt") or "")


def program_from_answer(text: str) -> str:
    parts = re.split(r"(?im)^\s*Program\s*:\s*", text or "", maxsplit=1)
    if len(parts) != 2:
        return ""
    return parts[1].strip().splitlines()[0].strip()


def has_multiple_program(text: str) -> bool:
    return len(PROGRAM_RE.findall(text or "")) != 1


def program_family(program: str) -> str:
    program = (program or "").lower()
    if program.startswith("divide(subtract("):
        return "percentage_change"
    if program.startswith("divide("):
        return "ratio_or_share"
    if program.startswith("subtract("):
        return "difference"
    if program.startswith("add(") or program.startswith("sum("):
        return "sum"
    if program.startswith("multiply("):
        return "multiply"
    return "other"


def answer_scale(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("answer_scale") or meta.get("answer_unit") or "unknown")


def requires_history(row: Dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    if bool(meta.get("requires_history")):
        return True
    if int(meta.get("history_turns") or 0) > 0:
        return True
    return "Previous question:" in prompt_text(row) or "Conversation history:" in prompt_text(row)


def normalize_existing_row(row: Dict[str, Any], *, task: str, source: str) -> Dict[str, Any]:
    copied = dict(row)
    copied["source_dataset"] = "ConvFinQA" if task == "convfinqa" else "FinQA"
    copied["v34r22_task"] = task
    copied["v34r22_source"] = source
    copied["v34r22_record_id"] = row_record_id(row)
    return copied


def finqa_grpo_row_to_sft(row: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    record_id = row_record_id(row)
    response = str(row.get("reference_response") or "")
    prompt = str(row.get("input_prompt_raw") or "")
    return {
        "source_dataset": "FinQA",
        "task_type": row.get("task_type") or "financial_table_text_reasoning",
        "record_id": record_id,
        "metadata": row.get("metadata") or {},
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response},
        ],
        "v34r22_task": "finqa",
        "v34r22_source": source,
        "v34r22_record_id": record_id,
    }


def unique_by_record(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for row in rows:
        rid = row_record_id(row)
        if not rid or rid in seen:
            continue
        seen.add(rid)
        unique.append(row)
    return unique


def valid_target(row: Dict[str, Any]) -> Tuple[bool, str]:
    text = answer_text(row)
    if FORBIDDEN_RE.search(text):
        return False, "forbidden_marker"
    if has_multiple_program(text):
        return False, "multiple_program"
    program = program_from_answer(text)
    if not program:
        return False, "missing_program"
    canonical = canonicalize_program_re(program)
    if not canonical:
        return False, "parse_failed"
    value, error = execute_program(canonical)
    if error:
        return False, "execution_failed"
    if value is None:
        return False, "execution_none"
    return True, ""


def load_allow_ids(paths: Sequence[str]) -> set[str]:
    ids = set()
    for path_str in paths:
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                row = json.loads(text)
                ids.add(str(row.get("record_id") or row.get("id") or ""))
            else:
                ids.add(text)
    return {item for item in ids if item}


def sample_rows(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def split(rows: List[Dict[str, Any]], valid_ratio: float, rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_ratio))) if len(shuffled) > 1 else 0
    return shuffled[valid_count:], shuffled[:valid_count]


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_records": len({row_record_id(row) for row in rows}),
        "task_distribution": dict(Counter(str(row.get("v34r22_task") or row.get("source_dataset")) for row in rows)),
        "source_distribution": dict(Counter(str(row.get("v34r22_source") or "unknown") for row in rows)),
        "history_dependent": sum(1 for row in rows if requires_history(row)),
        "program_family_distribution": dict(Counter(program_family(program_from_answer(answer_text(row))) for row in rows)),
        "answer_scale_distribution": dict(Counter(answer_scale(row) for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convfinqa_sft_file", required=True)
    parser.add_argument("--finqa_frontier_file", required=True)
    parser.add_argument("--finqa_retention_file", required=True)
    parser.add_argument("--finqa_clean_sft_file", required=True)
    parser.add_argument("--exclude_allowlist", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total_unique_records", type=int, default=640)
    parser.add_argument("--conv_ratio", type=float, default=0.65)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=34022)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    exclude_ids = load_allow_ids(args.exclude_allowlist)

    conv_rows_raw = unique_by_record(read_jsonl(Path(args.convfinqa_sft_file)))
    conv_rows = []
    bad_counter = Counter()
    for row in conv_rows_raw:
        if row_record_id(row) in exclude_ids:
            continue
        ok, reason = valid_target(row)
        if ok:
            conv_rows.append(normalize_existing_row(row, task="convfinqa", source="convfinqa_train_strict_sft"))
        else:
            bad_counter[f"convfinqa_{reason}"] += 1

    finqa_frontier = [
        finqa_grpo_row_to_sft(row, source="v34r21_frontier")
        for row in read_jsonl(Path(args.finqa_frontier_file))
        if row_record_id(row) not in exclude_ids
    ]
    finqa_retention = [
        finqa_grpo_row_to_sft(row, source="v34r21_retention")
        for row in read_jsonl(Path(args.finqa_retention_file))
        if row_record_id(row) not in exclude_ids
    ]
    finqa_clean_candidates = [
        normalize_existing_row(row, task="finqa", source="sft2_greedy_correct_representative")
        for row in unique_by_record(read_jsonl(Path(args.finqa_clean_sft_file)))
        if row_record_id(row) not in exclude_ids
    ]
    finqa_core = unique_by_record(finqa_frontier + finqa_retention)
    finqa_target = max(1, int(round(args.total_unique_records * (1.0 - args.conv_ratio))))
    finqa_extra_needed = max(0, finqa_target - len(finqa_core))
    finqa_rows = unique_by_record(finqa_core + sample_rows(finqa_clean_candidates, finqa_extra_needed, rng))

    conv_target = max(1, int(round(len(finqa_rows) * args.conv_ratio / max(1.0 - args.conv_ratio, 1e-6))))
    conv_selected = sample_rows(conv_rows, conv_target, rng)

    conv_only_rows = list(conv_selected)
    mixed_rows = unique_by_record(conv_selected + finqa_rows)
    conv_only_train, conv_only_valid = split(conv_only_rows, args.valid_ratio, rng)
    mixed_train, mixed_valid = split(mixed_rows, args.valid_ratio, rng)

    out = Path(args.output_dir)
    write_jsonl(out / "convfinqa_rs_sft.jsonl", conv_selected)
    write_jsonl(out / "finqa_retention_rs_sft.jsonl", finqa_rows)
    write_jsonl(out / "conv_only" / "train.jsonl", conv_only_train)
    write_jsonl(out / "conv_only" / "valid.jsonl", conv_only_valid)
    write_jsonl(out / "retention_aware" / "train.jsonl", mixed_train)
    write_jsonl(out / "retention_aware" / "valid.jsonl", mixed_valid)
    write_jsonl(out / "train_manifest.jsonl", [
        {
            "record_id": row_record_id(row),
            "task": row.get("v34r22_task"),
            "source": row.get("v34r22_source"),
            "requires_history": requires_history(row),
            "program_family": program_family(program_from_answer(answer_text(row))),
            "answer_scale": answer_scale(row),
        }
        for row in mixed_rows
    ])

    train_ids = {row_record_id(row) for row in mixed_train}
    valid_ids = {row_record_id(row) for row in mixed_valid}
    target_checks = Counter()
    for row in mixed_rows + conv_only_rows:
        ok, reason = valid_target(row)
        target_checks["valid_target" if ok else reason] += 1

    summary = {
        "version": "v34r22_convfinqa_retention_rs_sft",
        "seed": args.seed,
        "total_unique_records_requested": args.total_unique_records,
        "conv_ratio_requested": args.conv_ratio,
        "excluded_allowlist_records": len(exclude_ids),
        "convfinqa_unique_records": len({row_record_id(row) for row in conv_selected}),
        "finqa_unique_records": len({row_record_id(row) for row in finqa_rows}),
        "task_ratio_unique": {
            "convfinqa": len(conv_selected) / max(len(mixed_rows), 1),
            "finqa": len(finqa_rows) / max(len(mixed_rows), 1),
        },
        "history_dependent_ratio": sum(1 for row in conv_selected if requires_history(row)) / max(len(conv_selected), 1),
        "target_source": "strict_gold_program_sft_and_v34r21_model_like_frontier_reference_response",
        "current_rollout_winner_count": 0,
        "historical_replay_count": len(finqa_rows),
        "per_record_cap": 1,
        "train_valid_overlap": len(train_ids & valid_ids),
        "train_dev_test_overlap": len({row_record_id(row) for row in mixed_rows} & exclude_ids),
        "forbidden_marker_count": sum(1 for row in mixed_rows if FORBIDDEN_RE.search(answer_text(row))),
        "multiple_program_count": sum(1 for row in mixed_rows if has_multiple_program(answer_text(row))),
        "target_validation_counts": dict(target_checks),
        "bad_source_counts": dict(bad_counter),
        "mixed": summarize_rows(mixed_rows),
        "conv_only": summarize_rows(conv_only_rows),
        "retention_aware_train_rows": len(mixed_train),
        "retention_aware_valid_rows": len(mixed_valid),
        "conv_only_train_rows": len(conv_only_train),
        "conv_only_valid_rows": len(conv_only_valid),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

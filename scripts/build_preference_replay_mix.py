#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REQUIRED_DPO_COLUMNS = ("system", "history", "question", "response_chosen", "response_rejected")


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return json.dumps(value, ensure_ascii=False)


def read_jsonl(path: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def read_preference_root(root: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    paths = sorted((root / "train_dir").glob("*.jsonl")) + sorted((root / "valid_dir").glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No jsonl files found under {root}/train_dir or {root}/valid_dir")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        remaining = 0 if max_rows <= 0 else max_rows - len(rows)
        if max_rows > 0 and remaining <= 0:
            break
        rows.extend(read_jsonl(path, max_rows=remaining))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_pair_schema(pair: Dict[str, Any], source: str) -> None:
    missing = [column for column in REQUIRED_DPO_COLUMNS if column not in pair]
    if missing:
        raise ValueError(f"{source} is missing required DPO columns: {missing}")
    if not first_text(pair.get("question")):
        raise ValueError(f"{source} has empty question")
    if not first_text(pair.get("response_chosen")):
        raise ValueError(f"{source} has empty response_chosen")
    if not first_text(pair.get("response_rejected")):
        raise ValueError(f"{source} has empty response_rejected")


def normalize_pair(pair: Dict[str, Any], mix_source: str) -> Dict[str, Any]:
    row = dict(pair)
    row["system"] = "" if row.get("system") is None else str(row.get("system"))
    row["history"] = row.get("history") if isinstance(row.get("history"), list) else []
    row["question"] = first_text(row.get("question"))
    row["response_chosen"] = first_text(row.get("response_chosen"))
    row["response_rejected"] = first_text(row.get("response_rejected"))
    row["source_dataset"] = first_text(row.get("source_dataset")) or "unknown"
    row["record_id"] = first_text(row.get("record_id"))
    metadata = dict(row.get("metadata") or {})
    metadata["preference_mix_source"] = mix_source
    row["metadata"] = metadata
    validate_pair_schema(row, mix_source)
    return row


def has_program_section(text: str) -> bool:
    return "Program:" in first_text(text)


def has_direct_answer_section(text: str) -> bool:
    text = first_text(text)
    return "\nAnswer:" in text or text.startswith("Answer:")


def main_pair_passes_filters(pair: Dict[str, Any], args: argparse.Namespace) -> bool:
    metadata = pair.get("metadata") or {}
    if args.min_sample_correct_count > 0:
        if int(metadata.get("sample_correct_count") or 0) < args.min_sample_correct_count:
            return False
    if args.require_chosen_program and not has_program_section(pair.get("response_chosen")):
        return False
    if args.require_rejected_program and not has_program_section(pair.get("response_rejected")):
        return False
    if args.drop_chosen_with_answer and has_direct_answer_section(pair.get("response_chosen")):
        return False
    if args.drop_rejected_with_answer and has_direct_answer_section(pair.get("response_rejected")):
        return False
    return True


def record_key(pair: Dict[str, Any]) -> Tuple[str, str]:
    return first_text(pair.get("source_dataset")), first_text(pair.get("record_id"))


def pair_key(pair: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        first_text(pair.get("source_dataset")),
        first_text(pair.get("record_id")),
        first_text(pair.get("question")),
        first_text(pair.get("response_chosen")),
        first_text(pair.get("response_rejected")),
    )


def sample_replay(
    rows: Sequence[Dict[str, Any]],
    *,
    count: int,
    mix_source: str,
    rng: random.Random,
    existing_pair_keys: set[Tuple[str, str, str, str, str]],
    excluded_record_keys: set[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        normalized = normalize_pair(row, mix_source)
        if record_key(normalized) in excluded_record_keys:
            continue
        key = pair_key(normalized)
        if key in existing_pair_keys:
            continue
        candidates.append(normalized)

    rng.shuffle(candidates)
    selected = candidates[:count]
    for row in selected:
        existing_pair_keys.add(pair_key(row))
    return selected


def split_pairs(
    pairs: Sequence[Dict[str, Any]],
    *,
    valid_ratio: float,
    min_valid: int,
    max_valid: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pairs = list(pairs)
    if len(pairs) <= 1 or valid_ratio <= 0.0:
        return pairs, []
    valid_count = int(round(len(pairs) * valid_ratio))
    valid_count = max(min_valid, valid_count)
    valid_count = min(max_valid, valid_count, len(pairs) - 1)
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]
    return train, valid


def replay_count(main_count: int, main_ratio: float, replay_ratio: float, explicit_count: int) -> int:
    if explicit_count >= 0:
        return explicit_count
    if main_ratio <= 0.0:
        raise ValueError("--main_ratio must be positive when replay counts are inferred")
    return int(round(main_count * replay_ratio / main_ratio))


def summarize(rows: Sequence[Dict[str, Any]], train: Sequence[Dict[str, Any]], valid: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mix_counts = Counter(first_text((row.get("metadata") or {}).get("preference_mix_source")) for row in rows)
    source_counts = Counter(first_text(row.get("source_dataset")) or "unknown" for row in rows)
    split_mix_counts = {
        "train": dict(Counter(first_text((row.get("metadata") or {}).get("preference_mix_source")) for row in train)),
        "valid": dict(Counter(first_text((row.get("metadata") or {}).get("preference_mix_source")) for row in valid)),
    }
    return {
        "total_pairs": len(rows),
        "train_pairs": len(train),
        "valid_pairs": len(valid),
        "mix_source_counts": dict(mix_counts),
        "source_dataset_counts": dict(source_counts),
        "split_mix_source_counts": split_mix_counts,
    }


def build_replay_mix(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    raw_main_rows = [normalize_pair(row, "learnable_hard_main") for row in read_preference_root(Path(args.main_data_root), args.max_main_pairs)]
    main_rows = [row for row in raw_main_rows if main_pair_passes_filters(row, args)]
    if not main_rows:
        raise ValueError("No main preference pairs remain after filtering")
    existing_pair_keys = {pair_key(row) for row in main_rows}
    excluded_record_keys = {record_key(row) for row in main_rows if record_key(row)[1]} if args.exclude_main_record_ids else set()

    finqa_count = replay_count(len(main_rows), args.main_ratio, args.finqa_replay_ratio, args.finqa_replay_count)
    conv_count = replay_count(len(main_rows), args.main_ratio, args.convfinqa_replay_ratio, args.convfinqa_replay_count)

    finqa_replay_rows = read_jsonl(Path(args.finqa_replay_file), args.max_replay_pool_rows)
    conv_replay_rows = read_jsonl(Path(args.convfinqa_replay_file), args.max_replay_pool_rows)
    finqa_selected = sample_replay(
        finqa_replay_rows,
        count=finqa_count,
        mix_source="finqa_replay",
        rng=rng,
        existing_pair_keys=existing_pair_keys,
        excluded_record_keys=excluded_record_keys,
    )
    conv_selected = sample_replay(
        conv_replay_rows,
        count=conv_count,
        mix_source="convfinqa_replay",
        rng=rng,
        existing_pair_keys=existing_pair_keys,
        excluded_record_keys=excluded_record_keys,
    )

    all_rows = main_rows + finqa_selected + conv_selected
    train, valid = split_pairs(
        all_rows,
        valid_ratio=args.valid_ratio,
        min_valid=args.min_valid,
        max_valid=args.max_valid,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train_dir" / "train_preference_v7_1.jsonl", train)
    write_jsonl(output_dir / "valid_dir" / "valid_preference_v7_1.jsonl", valid)
    summary = summarize(all_rows, train, valid)
    summary.update(
        {
            "output_dir": str(output_dir),
            "main_data_root": args.main_data_root,
            "finqa_replay_file": args.finqa_replay_file,
            "convfinqa_replay_file": args.convfinqa_replay_file,
            "requested_counts": {
                "main_raw": len(raw_main_rows),
                "main_after_filter": len(main_rows),
                "finqa_replay": finqa_count,
                "convfinqa_replay": conv_count,
            },
            "selected_counts": {
                "main": len(main_rows),
                "finqa_replay": len(finqa_selected),
                "convfinqa_replay": len(conv_selected),
            },
            "ratios": {
                "main_ratio": args.main_ratio,
                "finqa_replay_ratio": args.finqa_replay_ratio,
                "convfinqa_replay_ratio": args.convfinqa_replay_ratio,
            },
            "exclude_main_record_ids": args.exclude_main_record_ids,
            "main_filters": {
                "min_sample_correct_count": args.min_sample_correct_count,
                "require_chosen_program": args.require_chosen_program,
                "require_rejected_program": args.require_rejected_program,
                "drop_chosen_with_answer": args.drop_chosen_with_answer,
                "drop_rejected_with_answer": args.drop_rejected_with_answer,
            },
            "valid_ratio": args.valid_ratio,
            "min_valid": args.min_valid,
            "max_valid": args.max_valid,
            "seed": args.seed,
        }
    )
    (output_dir / "preference_replay_mix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a v7.1 DPO replay mix from learnable-hard pairs and replay pools.")
    parser.add_argument("--main_data_root", default="/root/autodl-tmp/data/financial_reasoning_cot_pot/preference_v7_finqa")
    parser.add_argument("--finqa_replay_file", default="/root/autodl-tmp/data/financial_reasoning_v3/dpo_pairs/finqa_train_dpo.jsonl")
    parser.add_argument("--convfinqa_replay_file", default="/root/autodl-tmp/data/financial_reasoning_v3/dpo_pairs/convfinqa_train_turn_dpo.jsonl")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/data/financial_reasoning_cot_pot/preference_v7_1_replay")
    parser.add_argument("--main_ratio", type=float, default=0.70)
    parser.add_argument("--finqa_replay_ratio", type=float, default=0.15)
    parser.add_argument("--convfinqa_replay_ratio", type=float, default=0.15)
    parser.add_argument("--finqa_replay_count", type=int, default=-1, help="Set >=0 to override inferred FinQA replay count.")
    parser.add_argument("--convfinqa_replay_count", type=int, default=-1, help="Set >=0 to override inferred ConvFinQA replay count.")
    parser.add_argument("--max_main_pairs", type=int, default=0)
    parser.add_argument("--max_replay_pool_rows", type=int, default=0)
    parser.add_argument("--min_sample_correct_count", type=int, default=0)
    parser.add_argument("--require_chosen_program", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_rejected_program", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_chosen_with_answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_rejected_with_answer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--min_valid", type=int, default=16)
    parser.add_argument("--max_valid", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_main_record_ids", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_main_pairs < 0 or args.max_replay_pool_rows < 0:
        raise ValueError("--max_main_pairs and --max_replay_pool_rows must be non-negative")
    if args.min_sample_correct_count < 0:
        raise ValueError("--min_sample_correct_count must be non-negative")
    if not 0.0 <= args.valid_ratio < 1.0:
        raise ValueError("--valid_ratio must be in [0, 1)")
    if args.min_valid < 0 or args.max_valid < 0:
        raise ValueError("--min_valid and --max_valid must be non-negative")
    if args.max_valid and args.min_valid > args.max_valid:
        raise ValueError("--min_valid must be <= --max_valid")
    return args


def main() -> None:
    summary = build_replay_mix(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

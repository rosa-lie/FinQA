#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.build_grpo_v2_data_from_sft import convert_rows, first_text, read_jsonl, summarize


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_finqa_row(row: Dict[str, Any]) -> bool:
    return first_text(row.get("source_dataset")).lower() == "finqa"


def finqa_only(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    kept = [row for row in rows if is_finqa_row(row)]
    return kept, len(rows) - len(kept)




def row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (first_text(row.get("source_dataset")), first_text(row.get("record_id")))


def remove_valid_overlap(train_rows: Sequence[Dict[str, Any]], valid_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    valid_keys = {key for key in (row_key(row) for row in valid_rows) if key[1]}
    if not valid_keys:
        return list(train_rows), 0
    kept = [row for row in train_rows if row_key(row) not in valid_keys]
    return kept, len(train_rows) - len(kept)

def split_train_valid(rows: Sequence[Dict[str, Any]], *, valid_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1)")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    valid_count = int(round(len(shuffled) * valid_ratio))
    if len(shuffled) > 1 and valid_count == 0 and valid_ratio > 0:
        valid_count = 1
    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]
    return train, valid


def build_finqa_only_sft1_grpo_data(
    *,
    sft1_file: Path,
    validation_file: Optional[Path],
    output_dir: Path,
    valid_ratio: float = 0.12,
    smoke_rows: int = 64,
    seed: int = 42,
    max_rows: int = 0,
) -> Dict[str, Any]:
    sft1_raw = read_jsonl(sft1_file, max_rows=max_rows)
    converted_train_source, train_bad = convert_rows(sft1_raw, source_sft_file=sft1_file.name)
    train_source, excluded_train = finqa_only(converted_train_source)

    excluded_valid = 0
    valid_bad: List[Dict[str, Any]] = []
    validation_source = "sft1_split"
    train_overlap_removed = 0
    if validation_file is not None and validation_file.exists():
        valid_raw = read_jsonl(validation_file, max_rows=0)
        converted_valid_source, valid_bad = convert_rows(valid_raw, source_sft_file=validation_file.name)
        valid_rows, excluded_valid = finqa_only(converted_valid_source)
        if valid_rows:
            train_rows, train_overlap_removed = remove_valid_overlap(train_source, valid_rows)
            validation_source = str(validation_file)
        else:
            train_rows, valid_rows = split_train_valid(train_source, valid_ratio=valid_ratio, seed=seed)
    else:
        train_rows, valid_rows = split_train_valid(train_source, valid_ratio=valid_ratio, seed=seed)

    if not train_rows:
        raise ValueError("No FinQA train rows survived conversion/filtering")
    if not valid_rows:
        raise ValueError("No FinQA valid rows survived conversion/filtering")

    smoke = list(train_rows[: min(smoke_rows, len(train_rows))])
    bad_cases = train_bad + valid_bad
    all_kept = list(train_rows) + list(valid_rows)
    summary = summarize(all_kept, bad_cases)
    summary.update(
        {
            "version": "v34r18_finqa_only_sft1_grpo",
            "sft1_file": str(sft1_file),
            "validation_file": str(validation_file) if validation_file is not None else "",
            "validation_source": validation_source,
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "smoke_rows": len(smoke),
            "excluded_non_finqa_rows": excluded_train + excluded_valid,
            "converted_sft1_rows": len(converted_train_source),
            "converted_validation_rows": 0 if validation_file is None or not validation_file.exists() else len(converted_valid_source),
            "record_id_duplicates": len(all_kept) - len({first_text(row.get("record_id")) for row in all_kept}),
            "train_rows_removed_for_valid_overlap": train_overlap_removed,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)
    write_jsonl(output_dir / "smoke.jsonl", smoke)
    write_jsonl(output_dir / "bad_cases.jsonl", bad_cases)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v34r18 FinQA-only GRPO data from processed SFT1 program JSONL.")
    parser.add_argument("--sft1_file", required=True)
    parser.add_argument("--validation_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--valid_ratio", type=float, default=0.12)
    parser.add_argument("--smoke_rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    summary = build_finqa_only_sft1_grpo_data(
        sft1_file=Path(args.sft1_file),
        validation_file=Path(args.validation_file) if args.validation_file else None,
        output_dir=Path(args.output_dir),
        valid_ratio=args.valid_ratio,
        smoke_rows=args.smoke_rows,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

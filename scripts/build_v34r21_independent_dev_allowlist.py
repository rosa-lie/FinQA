#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v34r21_common import distribution, load_json_or_jsonl, load_record_ids, record_id, strat_key, write_jsonl


def select_dev_allowlist(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(strat_key(row), []).append(row)
    for bucket_rows in grouped.values():
        bucket_rows.sort(key=record_id)
        rng.shuffle(bucket_rows)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < sample_size:
        progressed = False
        for key in sorted(grouped):
            if len(selected) >= sample_size:
                break
            while grouped[key]:
                row = grouped[key].pop(0)
                rid = record_id(row)
                if rid and rid not in selected_ids:
                    selected.append(row)
                    selected_ids.add(rid)
                    progressed = True
                    break
        if not progressed:
            break
    selected.sort(key=record_id)
    return selected


def build_allowlist(args: argparse.Namespace) -> Dict[str, Any]:
    excluded: set[str] = set()
    for raw_path in args.exclude_file:
        if raw_path:
            excluded.update(load_record_ids(Path(raw_path)))
    rows = [row for row in load_json_or_jsonl(Path(args.dev_file)) if record_id(row) and record_id(row) not in excluded]
    selected = select_dev_allowlist(rows, int(args.sample_size), int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allow_rows = [{"record_id": record_id(row), "source_dataset": "finqa_dev"} for row in selected]
    write_jsonl(output_dir / "v34r21_independent_finqa_dev_allowlist.jsonl", allow_rows)
    summary = {
        "version": "v34r21_independent_finqa_dev_allowlist",
        "dev_file": args.dev_file,
        "seed": int(args.seed),
        "sample_size": len(selected),
        "eligible_dev_records": len(rows),
        "excluded_record_ids": len(excluded),
        "allowlist_file": str(output_dir / "v34r21_independent_finqa_dev_allowlist.jsonl"),
        "selected_distribution": distribution(selected),
    }
    (output_dir / "v34r21_independent_finqa_dev_allowlist.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent stratified FinQA dev allowlist for v34r21.")
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--exclude_file", action="append", default=[])
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    return args


def main() -> None:
    print(json.dumps(build_allowlist(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

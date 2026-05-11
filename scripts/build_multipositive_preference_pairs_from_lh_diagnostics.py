#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_is_correct(score: Dict[str, Any]) -> bool:
    return float(score.get("executed_answer_accuracy") or 0.0) > 0.0


def score_is_executable(score: Dict[str, Any]) -> bool:
    return float(score.get("program_execution_rate") or 0.0) > 0.0


def prediction_is_usable(text: Any, *, allow_answer: bool = True) -> bool:
    text = first_text(text)
    if not text or "Program:" not in text:
        return False
    if not allow_answer and ("\nAnswer:" in text or text.startswith("Answer:")):
        return False
    return True


def correct_samples(diagnostic: Dict[str, Any], *, max_positive_per_record: int, allow_answer: bool) -> List[Tuple[str, int]]:
    predictions = diagnostic.get("sampled_predictions") or []
    scores = diagnostic.get("sampled_scores") or []
    selected: List[Tuple[str, int]] = []
    seen: set[str] = set()
    for idx, (prediction, score) in enumerate(zip(predictions, scores)):
        prediction = first_text(prediction)
        if not prediction_is_usable(prediction, allow_answer=allow_answer):
            continue
        if not score_is_correct(score):
            continue
        if prediction in seen:
            continue
        seen.add(prediction)
        selected.append((prediction, idx))
        if max_positive_per_record > 0 and len(selected) >= max_positive_per_record:
            break
    return selected


def rejected_candidates(diagnostic: Dict[str, Any], *, allow_answer: bool) -> List[Tuple[str, str, Optional[int]]]:
    candidates: List[Tuple[str, str, Optional[int]]] = []
    greedy_prediction = first_text(diagnostic.get("greedy_prediction"))
    greedy_score = diagnostic.get("greedy_score") or {}
    if prediction_is_usable(greedy_prediction, allow_answer=allow_answer) and not score_is_correct(greedy_score):
        candidates.append((greedy_prediction, "greedy", None))

    predictions = diagnostic.get("sampled_predictions") or []
    scores = diagnostic.get("sampled_scores") or []
    for idx, (prediction, score) in enumerate(zip(predictions, scores)):
        prediction = first_text(prediction)
        if (
            prediction_is_usable(prediction, allow_answer=allow_answer)
            and score_is_executable(score)
            and not score_is_correct(score)
        ):
            candidates.append((prediction, "sampled", idx))
    return candidates


def build_pairs_from_diagnostic(diagnostic: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if first_text(diagnostic.get("bucket")) != "learnable-hard":
        return []
    prompt = first_text(diagnostic.get("input_prompt_raw") or diagnostic.get("question"))
    if not prompt:
        return []
    if int(diagnostic.get("sample_correct_count") or 0) < args.min_sample_correct_count:
        return []

    positives = correct_samples(
        diagnostic,
        max_positive_per_record=args.max_positive_per_record,
        allow_answer=not args.drop_chosen_with_answer,
    )
    negatives = rejected_candidates(diagnostic, allow_answer=not args.drop_rejected_with_answer)
    if not positives or not negatives:
        return []

    pairs: List[Dict[str, Any]] = []
    used_pair_texts: set[Tuple[str, str]] = set()
    for positive_text, positive_idx in positives:
        for negative_text, negative_source, negative_idx in negatives:
            if positive_text == negative_text:
                continue
            pair_text_key = (positive_text, negative_text)
            if pair_text_key in used_pair_texts:
                continue
            used_pair_texts.add(pair_text_key)
            metadata = dict(diagnostic.get("metadata") or {})
            metadata.update(
                {
                    "pair_builder": "multipositive_lh_diagnostics",
                    "chosen_source": "sampled",
                    "rejected_source": negative_source,
                    "chosen_candidate_index": positive_idx,
                    "rejected_candidate_index": negative_idx,
                    "sample_correct_count": int(diagnostic.get("sample_correct_count") or 0),
                    "sample_executable_count": int(diagnostic.get("sample_executable_count") or 0),
                    "gold_answer": first_text(diagnostic.get("gold_answer")),
                    "gold_program": first_text(diagnostic.get("gold_program")),
                }
            )
            pairs.append(
                {
                    "system": "",
                    "history": [],
                    "question": prompt,
                    "response_chosen": positive_text,
                    "response_rejected": negative_text,
                    "source_dataset": first_text(diagnostic.get("source_dataset")) or "finqa",
                    "record_id": first_text(diagnostic.get("record_id")),
                    "metadata": metadata,
                }
            )
            break
            if args.max_pairs_per_record > 0 and len(pairs) >= args.max_pairs_per_record:
                return pairs
        if args.max_pairs_per_record > 0 and len(pairs) >= args.max_pairs_per_record:
            return pairs
    return pairs


def build_pairs(diagnostics: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for diagnostic in diagnostics:
        pairs.extend(build_pairs_from_diagnostic(diagnostic, args))
        if args.max_pairs > 0 and len(pairs) >= args.max_pairs:
            return pairs[: args.max_pairs]
    return pairs


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


def validate_pair_schema(pair: Dict[str, Any]) -> None:
    missing = [column for column in REQUIRED_DPO_COLUMNS if column not in pair]
    if missing:
        raise ValueError(f"Preference pair missing required columns: {missing}")
    for column in ("question", "response_chosen", "response_rejected"):
        if not first_text(pair.get(column)):
            raise ValueError(f"Preference pair has empty {column}")


def summarize(train: Sequence[Dict[str, Any]], valid: Sequence[Dict[str, Any]], all_pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    record_counts = Counter(first_text(pair.get("record_id")) for pair in all_pairs)
    return {
        "total_pairs": len(all_pairs),
        "train_pairs": len(train),
        "valid_pairs": len(valid),
        "unique_records": len(record_counts),
        "pairs_per_record_hist": dict(Counter(str(count) for count in record_counts.values())),
        "source_dataset_counts": dict(Counter(first_text(pair.get("source_dataset")) or "unknown" for pair in all_pairs)),
        "rejected_source_counts": dict(Counter(first_text((pair.get("metadata") or {}).get("rejected_source")) for pair in all_pairs)),
        "sample_correct_count_hist": dict(Counter(str((pair.get("metadata") or {}).get("sample_correct_count")) for pair in all_pairs)),
    }


def write_preference_dataset(pairs: Sequence[Dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    for pair in pairs:
        validate_pair_schema(pair)
    train, valid = split_pairs(
        pairs,
        valid_ratio=args.valid_ratio,
        min_valid=args.min_valid,
        max_valid=args.max_valid,
        seed=args.seed,
    )
    write_jsonl(output_dir / "train_dir" / "train_preference_v8_1.jsonl", train)
    write_jsonl(output_dir / "valid_dir" / "valid_preference_v8_1.jsonl", valid)
    summary = summarize(train, valid, pairs)
    summary.update(
        {
            "output_dir": str(output_dir),
            "diagnostics_file": args.diagnostics_file,
            "max_pairs": args.max_pairs,
            "max_positive_per_record": args.max_positive_per_record,
            "max_pairs_per_record": args.max_pairs_per_record,
            "min_sample_correct_count": args.min_sample_correct_count,
            "drop_chosen_with_answer": args.drop_chosen_with_answer,
            "drop_rejected_with_answer": args.drop_rejected_with_answer,
            "valid_ratio": args.valid_ratio,
            "min_valid": args.min_valid,
            "max_valid": args.max_valid,
            "seed": args.seed,
        }
    )
    (output_dir / "multipositive_preference_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-positive DPO pairs from learnable-hard diagnostics.")
    parser.add_argument("--diagnostics_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--max_positive_per_record", type=int, default=2)
    parser.add_argument("--max_pairs_per_record", type=int, default=2)
    parser.add_argument("--min_sample_correct_count", type=int, default=2)
    parser.add_argument("--drop_chosen_with_answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_rejected_with_answer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--min_valid", type=int, default=16)
    parser.add_argument("--max_valid", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_pairs < 0 or args.max_positive_per_record < 0 or args.max_pairs_per_record < 0:
        raise ValueError("--max_pairs, --max_positive_per_record, and --max_pairs_per_record must be non-negative")
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
    args = parse_args()
    diagnostics = read_jsonl(Path(args.diagnostics_file))
    pairs = build_pairs(diagnostics, args)
    summary = write_preference_dataset(pairs, Path(args.output_dir), args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

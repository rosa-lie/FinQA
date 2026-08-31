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


def prediction_is_usable(text: str) -> bool:
    text = first_text(text)
    return bool(text and "Program:" in text)


def find_chosen_sample(diagnostic: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    predictions = diagnostic.get("sampled_predictions") or []
    scores = diagnostic.get("sampled_scores") or []
    for idx, (prediction, score) in enumerate(zip(predictions, scores)):
        if score_is_correct(score) and prediction_is_usable(prediction):
            return first_text(prediction), idx
    return None, None


def find_rejected_response(diagnostic: Dict[str, Any], chosen: str) -> Tuple[Optional[str], str, Optional[int]]:
    greedy_prediction = first_text(diagnostic.get("greedy_prediction"))
    greedy_score = diagnostic.get("greedy_score") or {}
    if prediction_is_usable(greedy_prediction) and not score_is_correct(greedy_score) and greedy_prediction != chosen:
        return greedy_prediction, "greedy", None

    predictions = diagnostic.get("sampled_predictions") or []
    scores = diagnostic.get("sampled_scores") or []
    for idx, (prediction, score) in enumerate(zip(predictions, scores)):
        prediction = first_text(prediction)
        if (
            prediction_is_usable(prediction)
            and prediction != chosen
            and score_is_executable(score)
            and not score_is_correct(score)
        ):
            return prediction, "sampled", idx
    return None, "", None


def build_pair_from_diagnostic(diagnostic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if first_text(diagnostic.get("bucket")) != "learnable-hard":
        return None
    prompt = first_text(diagnostic.get("input_prompt_raw") or diagnostic.get("question"))
    if not prompt:
        return None

    chosen, chosen_idx = find_chosen_sample(diagnostic)
    if not chosen:
        return None
    rejected, rejected_source, rejected_idx = find_rejected_response(diagnostic, chosen)
    if not rejected:
        return None

    metadata = dict(diagnostic.get("metadata") or {})
    metadata.update(
        {
            "chosen_source": "sampled",
            "rejected_source": rejected_source,
            "chosen_candidate_index": chosen_idx,
            "rejected_candidate_index": rejected_idx,
            "sample_correct_count": int(diagnostic.get("sample_correct_count") or 0),
            "sample_executable_count": int(diagnostic.get("sample_executable_count") or 0),
            "gold_answer": first_text(diagnostic.get("gold_answer")),
            "gold_program": first_text(diagnostic.get("gold_program")),
        }
    )
    return {
        "system": "",
        "history": [],
        "question": prompt,
        "response_chosen": chosen,
        "response_rejected": rejected,
        "source_dataset": first_text(diagnostic.get("source_dataset")) or "finqa",
        "record_id": first_text(diagnostic.get("record_id")),
        "metadata": metadata,
    }


def build_pairs(diagnostics: Sequence[Dict[str, Any]], max_pairs: int = 0) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for diagnostic in diagnostics:
        pair = build_pair_from_diagnostic(diagnostic)
        if pair is None:
            continue
        record_id = first_text(pair.get("record_id"))
        if record_id and record_id in seen_record_ids:
            continue
        if record_id:
            seen_record_ids.add(record_id)
        pairs.append(pair)
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
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


def summarize_pairs(train: Sequence[Dict[str, Any]], valid: Sequence[Dict[str, Any]], all_pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    source_counts = Counter(first_text(pair.get("source_dataset")) or "unknown" for pair in all_pairs)
    rejected_source_counts = Counter(first_text((pair.get("metadata") or {}).get("rejected_source")) for pair in all_pairs)
    sample_correct_hist = Counter(str((pair.get("metadata") or {}).get("sample_correct_count")) for pair in all_pairs)
    return {
        "total_pairs": len(all_pairs),
        "train_pairs": len(train),
        "valid_pairs": len(valid),
        "source_dataset_counts": dict(source_counts),
        "rejected_source_counts": dict(rejected_source_counts),
        "sample_correct_count_hist": dict(sample_correct_hist),
    }


def write_preference_dataset(
    pairs: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    valid_ratio: float,
    min_valid: int,
    max_valid: int,
    seed: int,
) -> Dict[str, Any]:
    train, valid = split_pairs(pairs, valid_ratio=valid_ratio, min_valid=min_valid, max_valid=max_valid, seed=seed)
    write_jsonl(output_dir / "train_dir" / "train_preference_v7.jsonl", train)
    write_jsonl(output_dir / "valid_dir" / "valid_preference_v7.jsonl", valid)
    summary = summarize_pairs(train, valid, pairs)
    summary["output_dir"] = str(output_dir)
    summary["valid_ratio"] = valid_ratio
    summary["min_valid"] = min_valid
    summary["max_valid"] = max_valid
    summary["seed"] = seed
    (output_dir / "preference_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def validate_pair_schema(pair: Dict[str, Any]) -> None:
    missing = [column for column in REQUIRED_DPO_COLUMNS if column not in pair]
    if missing:
        raise ValueError(f"Preference pair missing required columns: {missing}")
    if not first_text(pair["question"]):
        raise ValueError("Preference pair has empty question")
    if not first_text(pair["response_chosen"]):
        raise ValueError("Preference pair has empty response_chosen")
    if not first_text(pair["response_rejected"]):
        raise ValueError("Preference pair has empty response_rejected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO/ORPO preference pairs from learnable-hard diagnostics.")
    parser.add_argument("--diagnostics_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--min_valid", type=int, default=16)
    parser.add_argument("--max_valid", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_pairs < 0:
        raise ValueError("--max_pairs must be non-negative")
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
    pairs = build_pairs(diagnostics, max_pairs=args.max_pairs)
    for pair in pairs:
        validate_pair_schema(pair)
    summary = write_preference_dataset(
        pairs,
        Path(args.output_dir),
        valid_ratio=args.valid_ratio,
        min_valid=args.min_valid,
        max_valid=args.max_valid,
        seed=args.seed,
    )
    summary["diagnostics_file"] = args.diagnostics_file
    summary["max_pairs"] = args.max_pairs
    (Path(args.output_dir) / "preference_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

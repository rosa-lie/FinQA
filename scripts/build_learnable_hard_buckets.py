#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import GenerationConfig
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_financial_benchmarks import (
    BenchmarkExample,
    execute_prediction_program,
    generate_response,
    load_model_and_tokenizer,
    parse_number,
    safe_apply_chat_template,
    score_example,
    unload_model,
)
from training.finqa_program_grpo import STRICT_PROGRAM_SYSTEM_PROMPT, SYSTEM_PROMPT
from scripts.v34r23_prompt_processor import prepare_rows_like_grpo, prompt_target_leakage_reason


REQUIRED_FIELDS = [
    "input_prompt_raw",
    "gold_answer",
    "gold_program",
    "reward_profile",
]


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return json.dumps(value, ensure_ascii=False)


def read_jsonl(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: Dict[str, Any]) -> str:
    record_id = first_text(row.get("record_id"))
    source_dataset = first_text(row.get("source_dataset"))
    if record_id or source_dataset:
        return f"{source_dataset}::{record_id}"
    return first_text(row.get("input_prompt_raw"))


def diagnostic_key(diagnostic: Dict[str, Any]) -> str:
    record_id = first_text(diagnostic.get("record_id"))
    source_dataset = first_text(diagnostic.get("source_dataset"))
    if record_id or source_dataset:
        return f"{source_dataset}::{record_id}"
    return first_text(diagnostic.get("input_prompt_raw"))


def source_filter_values(raw_filter: str) -> set[str]:
    return {item.strip() for item in (raw_filter or "").split(",") if item.strip()}


def filter_rows_by_source(rows: Sequence[Dict[str, Any]], raw_filter: str) -> List[Dict[str, Any]]:
    allowed = source_filter_values(raw_filter)
    if not allowed:
        return list(rows)
    return [row for row in rows if first_text(row.get("source_dataset")) in allowed]


def filter_rows_by_manifest_phase(rows: Sequence[Dict[str, Any]], manifest_file: str, manifest_phase: str) -> List[Dict[str, Any]]:
    if not manifest_file or not manifest_phase or manifest_phase == "all":
        return list(rows)
    allowed_keys = {
        row_key(item)
        for item in read_jsonl(Path(manifest_file))
        if first_text(item.get("manifest_phase")) == manifest_phase
    }
    return [row for row in rows if row_key(row) in allowed_keys]


def read_existing_diagnostics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def score_is_correct(score: Dict[str, Any]) -> bool:
    return float(score.get("executed_answer_accuracy") or 0.0) > 0.0


def score_is_executable(score: Dict[str, Any]) -> bool:
    return float(score.get("program_execution_rate") or 0.0) > 0.0


def score_has_program(score: Dict[str, Any]) -> bool:
    return float(score.get("program_parse_rate") or 0.0) > 0.0


def gold_program_matches_answer(row: Dict[str, Any], abs_tol: float, rel_tol: float) -> Tuple[bool, str, str]:
    gold_answer = first_text(row.get("gold_answer"))
    gold_program = first_text(row.get("gold_program"))
    executed_value, canonical_program, error = execute_prediction_program(gold_program)
    if executed_value is None:
        return False, canonical_program, error or "gold_program_not_executable"
    gold_num = parse_number(gold_answer)
    if gold_num is None:
        return False, canonical_program, "gold_answer_not_numeric"
    if not math.isclose(executed_value, gold_num, abs_tol=abs_tol, rel_tol=rel_tol):
        return False, canonical_program, "gold_program_answer_mismatch"
    return True, canonical_program, ""


def row_to_example(row: Dict[str, Any]) -> BenchmarkExample:
    source_dataset = first_text(row.get("source_dataset")) or "financial_reasoning"
    return BenchmarkExample(
        task_name=source_dataset,
        prompt=first_text(row.get("input_prompt_raw")),
        gold_answer=first_text(row.get("gold_answer")),
        answer_type="numeric",
        record_id=first_text(row.get("record_id")),
        metadata=dict(row.get("metadata") or {}),
        gold_program=first_text(row.get("gold_program")),
    )


def classify_bucket(
    greedy_score: Optional[Dict[str, Any]],
    sampled_scores: Sequence[Dict[str, Any]],
    *,
    noisy: bool,
    invalid_executable_threshold: float = 0.5,
) -> str:
    if noisy:
        return "noisy"
    if greedy_score is None:
        return "noisy"
    if score_is_correct(greedy_score):
        return "easy"

    sample_count = len(sampled_scores)
    if sample_count == 0:
        return "hard"

    correct_count = sum(1 for score in sampled_scores if score_is_correct(score))
    executable_count = sum(1 for score in sampled_scores if score_is_executable(score))
    executable_rate = executable_count / max(sample_count, 1)

    if correct_count > 0:
        return "learnable-hard"
    if executable_rate < invalid_executable_threshold:
        return "invalid-prone"
    return "hard"


def build_mix_rows(
    learnable_rows: Sequence[Dict[str, Any]],
    easy_rows: Sequence[Dict[str, Any]],
    *,
    easy_replay_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if not learnable_rows:
        return []
    if easy_replay_ratio <= 0.0 or not easy_rows:
        return list(learnable_rows)

    replay_count = int(round(len(learnable_rows) * easy_replay_ratio / max(1.0 - easy_replay_ratio, 1e-8)))
    replay_count = min(replay_count, len(easy_rows))
    rng = random.Random(seed)
    selected_easy = rng.sample(list(easy_rows), replay_count)
    mixed = list(learnable_rows) + selected_easy
    rng.shuffle(mixed)
    return mixed


def make_generation_args(args: argparse.Namespace) -> SimpleNamespace:
    system_prompt = args.system_prompt
    if getattr(args, "use_grpo_prompt_processor", False) and not system_prompt:
        system_prompt = STRICT_PROGRAM_SYSTEM_PROMPT
    return SimpleNamespace(
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        repetition_penalty=args.repetition_penalty,
        numeric_abs_tol=args.numeric_abs_tol,
        numeric_rel_tol=args.numeric_rel_tol,
        numeric_output_format=args.numeric_output_format,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )


def score_prediction(example: BenchmarkExample, prediction: str, generation_args: SimpleNamespace) -> Dict[str, Any]:
    return score_example(example, prediction, generation_args)


def generate_and_score(
    model: Any,
    tokenizer: Any,
    example: BenchmarkExample,
    generation_args: SimpleNamespace,
    *,
    temperature: float,
    top_p: Optional[float],
    do_sample: bool,
    seed: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    prediction = generate_response(
        model,
        tokenizer,
        example.prompt,
        generation_args,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        seed=seed,
    )
    return prediction, score_prediction(example, prediction, generation_args)


@torch.inference_mode()
def generate_sample_responses(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_args: SimpleNamespace,
    *,
    num_return_sequences: int,
    temperature: float,
    top_p: Optional[float],
    seed: int,
) -> List[str]:
    messages = []
    if generation_args.system_prompt:
        messages.append({"role": "system", "content": generation_args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    prompt_text = safe_apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generation_config = GenerationConfig(
        max_new_tokens=generation_args.max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=generation_args.repetition_penalty,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if top_p is not None:
        generation_config.top_p = top_p
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )
    return [
        tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def compact_score(score: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer_correct": score.get("answer_correct"),
        "program_parse_rate": score.get("program_parse_rate"),
        "program_execution_rate": score.get("program_execution_rate"),
        "executed_answer_accuracy": score.get("executed_answer_accuracy"),
        "executed_program": score.get("executed_program", ""),
        "executed_program_answer": score.get("executed_program_answer", ""),
        "program_execution_error": score.get("program_execution_error", ""),
    }


def diagnostics_for_row(
    row: Dict[str, Any],
    bucket: str,
    greedy_prediction: str,
    greedy_score: Optional[Dict[str, Any]],
    sampled_predictions: Sequence[str],
    sampled_scores: Sequence[Dict[str, Any]],
    *,
    noisy_reason: str,
    gold_program_canonical: str,
    skipped_sampling: bool,
) -> Dict[str, Any]:
    correct_sample_indices = [idx for idx, score in enumerate(sampled_scores) if score_is_correct(score)]
    sample_programs = [first_text(score.get("executed_program")) for score in sampled_scores if first_text(score.get("executed_program"))]
    return {
        "record_id": first_text(row.get("record_id")),
        "source_dataset": first_text(row.get("source_dataset")),
        "input_prompt_raw": first_text(row.get("input_prompt_raw")),
        "bucket": bucket,
        "noisy_reason": noisy_reason,
        "gold_answer": first_text(row.get("gold_answer")),
        "gold_program": first_text(row.get("gold_program")),
        "gold_program_canonical": gold_program_canonical,
        "greedy_correct": bool(greedy_score and score_is_correct(greedy_score)),
        "greedy_executable": bool(greedy_score and score_is_executable(greedy_score)),
        "greedy_has_program": bool(greedy_score and score_has_program(greedy_score)),
        "greedy_prediction": greedy_prediction,
        "greedy_score": compact_score(greedy_score or {}),
        "skipped_sampling": skipped_sampling,
        "sample_count": len(sampled_scores),
        "sample_correct_count": len(correct_sample_indices),
        "sample_executable_count": sum(1 for score in sampled_scores if score_is_executable(score)),
        "sample_has_program_count": sum(1 for score in sampled_scores if score_has_program(score)),
        "sample_programs_unique": len(set(sample_programs)),
        "correct_sample_indices": correct_sample_indices,
        "sampled_predictions": list(sampled_predictions),
        "sampled_scores": [compact_score(score) for score in sampled_scores],
    }


def summarize_buckets(bucket_rows: Dict[str, List[Dict[str, Any]]], diagnostics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counts = {bucket: len(rows) for bucket, rows in sorted(bucket_rows.items())}
    source_counts = Counter(first_text(item.get("source_dataset")) or "unknown" for item in diagnostics)
    bucket_by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in diagnostics:
        source = first_text(item.get("source_dataset")) or "unknown"
        bucket = first_text(item.get("bucket")) or "unknown"
        bucket_by_source[source][bucket] += 1

    learnable_count = bucket_counts.get("learnable-hard", 0)
    if learnable_count < 200:
        recommended_max_steps = 50
        training_recommendation = "analyze_only_or_short_smoke"
    elif learnable_count <= 1000:
        recommended_max_steps = 100
        training_recommendation = "short_grpo"
    else:
        recommended_max_steps = 200
        training_recommendation = "grpo"

    generation_counts = {
        "greedy": sum(1 for item in diagnostics if item.get("greedy_prediction")),
        "sampled": sum(int(item.get("sample_count") or 0) for item in diagnostics),
    }
    generation_counts["total"] = generation_counts["greedy"] + generation_counts["sampled"]

    return {
        "total": len(diagnostics),
        "bucket_counts": bucket_counts,
        "source_dataset_counts": dict(source_counts),
        "bucket_by_source_dataset": {source: dict(counts) for source, counts in bucket_by_source.items()},
        "generation_count_estimate": generation_counts,
        "recommended_max_steps": recommended_max_steps,
        "training_recommendation": training_recommendation,
    }


def row_is_noisy(row: Dict[str, Any], abs_tol: float, rel_tol: float) -> Tuple[bool, str, str]:
    missing = [field for field in REQUIRED_FIELDS if not first_text(row.get(field))]
    if missing:
        return True, "missing_" + ",".join(missing), ""
    if first_text(row.get("reward_profile")) != "program_numeric":
        return True, "bad_reward_profile", ""
    gold_ok, canonical_program, gold_error = gold_program_matches_answer(row, abs_tol, rel_tol)
    if not gold_ok:
        return True, gold_error, canonical_program
    return False, "", canonical_program


def should_sample_after_greedy(args: argparse.Namespace, greedy_score: Optional[Dict[str, Any]], noisy: bool) -> bool:
    if noisy:
        return False
    if args.num_samples_per_example <= 0:
        return False
    if args.skip_sampling_if_greedy_correct and greedy_score is not None and score_is_correct(greedy_score):
        return False
    return True


def rebuild_bucket_rows_from_diagnostics(
    rows_by_key: Dict[str, Dict[str, Any]],
    diagnostics: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    bucket_rows: Dict[str, List[Dict[str, Any]]] = {
        "easy": [],
        "learnable-hard": [],
        "hard": [],
        "invalid-prone": [],
        "noisy": [],
    }
    for diagnostic in diagnostics:
        key = diagnostic_key(diagnostic)
        row = rows_by_key.get(key)
        bucket = first_text(diagnostic.get("bucket"))
        if row is not None and bucket in bucket_rows:
            bucket_rows[bucket].append(row)
    return bucket_rows


def write_outputs(
    output_dir: Path,
    input_file: Path,
    args: argparse.Namespace,
    bucket_rows: Dict[str, List[Dict[str, Any]]],
    diagnostics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    mix_rows = build_mix_rows(
        bucket_rows["learnable-hard"],
        bucket_rows["easy"],
        easy_replay_ratio=args.easy_replay_ratio,
        seed=args.sample_seed,
    )

    write_jsonl(output_dir / "train_learnable_hard.jsonl", bucket_rows["learnable-hard"])
    write_jsonl(output_dir / "train_easy_replay.jsonl", bucket_rows["easy"])
    write_jsonl(output_dir / "train_hard.jsonl", bucket_rows["hard"])
    write_jsonl(output_dir / "train_invalid_prone.jsonl", bucket_rows["invalid-prone"])
    write_jsonl(output_dir / "train_noisy.jsonl", bucket_rows["noisy"])
    write_jsonl(output_dir / "train_grpo_lh_mix.jsonl", mix_rows)
    write_jsonl(output_dir / "learnable_hard_diagnostics.jsonl", diagnostics)

    summary = summarize_buckets(bucket_rows, diagnostics)
    summary["input_file"] = str(input_file)
    summary["output_dir"] = str(output_dir)
    summary["max_samples"] = args.max_samples
    summary["source_dataset_filter"] = args.source_dataset_filter
    summary["skip_sampling_if_greedy_correct"] = args.skip_sampling_if_greedy_correct
    summary["resume_from_diagnostics"] = args.resume_from_diagnostics
    summary["num_samples_per_example"] = args.num_samples_per_example
    summary["sample_temperature"] = args.sample_temperature
    summary["sample_top_p"] = args.sample_top_p
    summary["sample_seed"] = args.sample_seed
    summary["easy_replay_ratio"] = args.easy_replay_ratio
    summary["mix_count"] = len(mix_rows)
    summary["use_grpo_prompt_processor"] = bool(getattr(args, "use_grpo_prompt_processor", False))
    (output_dir / "bucket_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def process_rows(args: argparse.Namespace) -> Dict[str, Any]:
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    rows = filter_rows_by_source(read_jsonl(input_file), args.source_dataset_filter)
    rows = filter_rows_by_manifest_phase(
        rows,
        getattr(args, "manifest_file", ""),
        getattr(args, "manifest_phase", ""),
    )
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if getattr(args, "use_grpo_prompt_processor", False):
        processed_rows = prepare_rows_like_grpo(rows, is_main_process=True)
        rewritten_rows = []
        for raw_row, processed_row in zip(rows, processed_rows):
            row = dict(raw_row)
            row["input_prompt_raw"] = first_text(processed_row.get("input_prompt_raw"))
            row["reward_profile"] = first_text(processed_row.get("reward_profile"))
            row["source_dataset"] = first_text(processed_row.get("source_dataset"))
            row["gold_answer"] = first_text(processed_row.get("gold_answer"))
            row["gold_program"] = first_text(processed_row.get("gold_program"))
            meta = dict(raw_row.get("metadata") or {})
            meta.update(processed_row.get("metadata") or {})
            row["metadata"] = meta
            rewritten_rows.append(row)
        rows = [row for row in rewritten_rows if not prompt_target_leakage_reason(row)]
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "learnable_hard_diagnostics.jsonl"
    rows_by_key = {row_key(row): row for row in rows}

    diagnostics: List[Dict[str, Any]] = []
    processed_keys: set[str] = set()
    if args.resume_from_diagnostics:
        diagnostics = [
            item for item in read_existing_diagnostics(diagnostics_path)
            if diagnostic_key(item) in rows_by_key
        ]
        processed_keys = {diagnostic_key(item) for item in diagnostics}
    bucket_rows = rebuild_bucket_rows_from_diagnostics(rows_by_key, diagnostics)
    if len(processed_keys) >= len(rows):
        return write_outputs(output_dir, input_file, args, bucket_rows, diagnostics)

    generation_args = make_generation_args(args)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.tokenizer_path,
        args.adapter_path,
        generation_args,
    )

    try:
        for example_index, row in enumerate(tqdm(rows, desc="Bucketing examples")):
            key = row_key(row)
            if key in processed_keys:
                continue
            noisy, noisy_reason, gold_program_canonical = row_is_noisy(row, args.numeric_abs_tol, args.numeric_rel_tol)
            greedy_prediction = ""
            greedy_score: Optional[Dict[str, Any]] = None
            sampled_predictions: List[str] = []
            sampled_scores: List[Dict[str, Any]] = []
            skipped_sampling = False

            if not noisy:
                example = row_to_example(row)
                greedy_prediction, greedy_score = generate_and_score(
                    model,
                    tokenizer,
                    example,
                    generation_args,
                    temperature=0.0,
                    top_p=None,
                    do_sample=False,
                    seed=None,
                )
                if should_sample_after_greedy(args, greedy_score, noisy):
                    if args.batch_sample_generation and args.num_samples_per_example > 1:
                        sample_seed = args.sample_seed + example_index * args.num_samples_per_example
                        batch_predictions = generate_sample_responses(
                            model,
                            tokenizer,
                            example.prompt,
                            generation_args,
                            num_return_sequences=args.num_samples_per_example,
                            temperature=args.sample_temperature,
                            top_p=args.sample_top_p,
                            seed=sample_seed,
                        )
                        for prediction in batch_predictions:
                            score = score_prediction(example, prediction, generation_args)
                            sampled_predictions.append(prediction)
                            sampled_scores.append(score)
                    else:
                        for candidate_index in range(args.num_samples_per_example):
                            sample_seed = args.sample_seed + example_index * args.num_samples_per_example + candidate_index
                            prediction, score = generate_and_score(
                                model,
                                tokenizer,
                                example,
                                generation_args,
                                temperature=args.sample_temperature,
                                top_p=args.sample_top_p,
                                do_sample=True,
                                seed=sample_seed,
                            )
                            sampled_predictions.append(prediction)
                            sampled_scores.append(score)
                else:
                    skipped_sampling = True

            bucket = classify_bucket(
                greedy_score,
                sampled_scores,
                noisy=noisy,
                invalid_executable_threshold=args.invalid_executable_threshold,
            )
            bucket_rows[bucket].append(row)
            processed_keys.add(key)
            diagnostics.append(
                diagnostics_for_row(
                    row,
                    bucket,
                    greedy_prediction,
                    greedy_score,
                    sampled_predictions,
                    sampled_scores,
                    noisy_reason=noisy_reason,
                    gold_program_canonical=gold_program_canonical,
                    skipped_sampling=skipped_sampling,
                )
            )
            if args.flush_every > 0 and len(diagnostics) % args.flush_every == 0:
                write_outputs(output_dir, input_file, args, bucket_rows, diagnostics)
    finally:
        unload_model(model, tokenizer)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return write_outputs(output_dir, input_file, args, bucket_rows, diagnostics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build learnable-hard buckets for Program GRPO.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--manifest_file", default="")
    parser.add_argument("--manifest_phase", choices=["", "pilot", "extension", "all"], default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--source_dataset_filter", type=str, default="")
    parser.add_argument("--skip_sampling_if_greedy_correct", action="store_true")
    parser.add_argument("--flush_every", type=int, default=0)
    parser.add_argument("--resume_from_diagnostics", action="store_true")
    parser.add_argument("--num_samples_per_example", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=0.7)
    parser.add_argument("--sample_top_p", type=float, default=0.95)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--batch_sample_generation", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--numeric_abs_tol", type=float, default=1e-4)
    parser.add_argument("--numeric_rel_tol", type=float, default=1e-4)
    parser.add_argument("--numeric_output_format", type=str, default="program_executor")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--easy_replay_ratio", type=float, default=0.2)
    parser.add_argument("--invalid_executable_threshold", type=float, default=0.5)
    parser.add_argument("--use_grpo_prompt_processor", action="store_true")
    args = parser.parse_args()
    if args.num_samples_per_example < 0:
        raise ValueError("--num_samples_per_example must be non-negative")
    if args.num_samples_per_example > 0 and args.sample_temperature <= 0:
        raise ValueError("--sample_temperature must be positive when sampling")
    if not 0.0 < args.sample_top_p <= 1.0:
        raise ValueError("--sample_top_p must be in (0, 1]")
    if not 0.0 <= args.easy_replay_ratio < 1.0:
        raise ValueError("--easy_replay_ratio must be in [0, 1)")
    if not 0.0 <= args.invalid_executable_threshold <= 1.0:
        raise ValueError("--invalid_executable_threshold must be in [0, 1]")
    if args.flush_every < 0:
        raise ValueError("--flush_every must be non-negative")
    return args


def main() -> None:
    summary = process_rows(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

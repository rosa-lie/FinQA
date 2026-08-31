#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v34r23_frontier_grpo_data import (
    clean_rollout_prediction,
    prompt_target_leakage_reason,
    score_correct,
    score_executable,
    scored_predictions,
    valid_train_row,
)
from scripts.v34r21_common import first_text, read_jsonl, record_id, source_dataset
from scripts.v34r23_prompt_processor import prepare_rows_like_grpo


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0}
    ordered = sorted(float(v) for v in values)
    def q(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(math.floor((len(ordered) - 1) * p))))
        return ordered[idx]
    return {"mean": sum(ordered) / len(ordered), "median": median(ordered), "p10": q(0.10), "p25": q(0.25), "p75": q(0.75), "min": ordered[0]}


def prediction_program(prediction: str) -> str:
    text = first_text(prediction)
    if "Program:" not in text:
        return ""
    return text.split("Program:", 1)[1].strip()


def normalize_program_exact(program: str) -> str:
    text = first_text(program).lower()
    text = re.sub(r"\s+", "", text)
    return text.replace("％", "%")


def canonical_skeleton(program: str) -> str:
    text = normalize_program_exact(program)
    return re.sub(r"-?\d+(?:\.\d+)?%?", "NUM", text)


def program_ops(program: str) -> List[str]:
    return re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", first_text(program))


def is_literal_program(program: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%?", normalize_program_exact(program)))


def program_family_from_program(program: str) -> str:
    if is_literal_program(program):
        return "literal_direct_lookup"
    ops = program_ops(program)
    if not ops:
        return "other"
    if len(ops) > 1:
        if "divide" in ops and "subtract" in ops:
            return "percentage_change_skeleton"
        return "multi_step_nested"
    return {"add": "add", "sum": "add", "subtract": "subtract", "multiply": "multiply", "divide": "divide"}.get(ops[0], "other")


def reward_from_score(score: Dict[str, Any]) -> float:
    if score_correct(score):
        return 1.0
    if score_executable(score):
        return -0.1
    return -1.0


def score_program(score: Dict[str, Any], prediction: str) -> str:
    return first_text(score.get("executed_program") or score.get("program") or prediction_program(prediction))


def phase_for_diag(diag: Dict[str, Any], path_hint: str = "") -> str:
    phase = first_text(diag.get("v34r23_manifest_phase"))
    if phase:
        return phase
    lowered = path_hint.lower()
    if "extension" in lowered:
        return "extension"
    if "pilot" in lowered or "current_policy_frontier_acquisition" in lowered:
        return "pilot"
    return "unknown"


def load_diagnostics(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(Path(path)):
            out = dict(row)
            out["audit_phase"] = phase_for_diag(out, path)
            rows.append(out)
    return rows


def load_source_rows(path: str, use_grpo_prompt_processor: bool = True) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows = read_jsonl(Path(path))
    if use_grpo_prompt_processor:
        processed = prepare_rows_like_grpo(rows, is_main_process=True)
        rewritten = []
        for raw, proc in zip(rows, processed):
            row = dict(raw)
            row["input_prompt_raw"] = first_text(proc.get("input_prompt_raw"))
            row["reward_profile"] = first_text(proc.get("reward_profile"))
            row["source_dataset"] = first_text(proc.get("source_dataset"))
            row["gold_answer"] = first_text(proc.get("gold_answer"))
            row["gold_program"] = first_text(proc.get("gold_program"))
            meta = dict(raw.get("metadata") or {})
            meta.update(proc.get("metadata") or {})
            row["metadata"] = meta
            rewritten.append(row)
        rows = rewritten
    return {(source_dataset(row), record_id(row)): row for row in rows if source_dataset(row) and record_id(row)}


def diagnostic_key(diag: Dict[str, Any]) -> Tuple[str, str]:
    return (source_dataset(diag), first_text(diag.get("record_id")))


def analyze_record_candidates(diag: Dict[str, Any]) -> Dict[str, Any]:
    triples = scored_predictions(diag)
    completions = [prediction for _idx, _score, prediction in triples]
    programs = [score_program(score, prediction) for _idx, score, prediction in triples if score_program(score, prediction)]
    executable_programs = [score_program(score, prediction) for _idx, score, prediction in triples if score_executable(score) and score_program(score, prediction)]
    skeletons = [canonical_skeleton(program) for program in programs]
    rewards = [reward_from_score(score) for _idx, score, _prediction in triples]
    correct = [score for _idx, score, _prediction in triples if score_correct(score)]
    wrong_exec = [score for _idx, score, _prediction in triples if score_executable(score) and not score_correct(score)]
    invalid = [score for _idx, score, _prediction in triples if not score_executable(score)]
    reward_mean = sum(rewards) / max(len(rewards), 1)
    reward_std = math.sqrt(sum((reward - reward_mean) ** 2 for reward in rewards) / max(len(rewards), 1)) if rewards else 0.0
    program_counts = Counter(programs)
    max_program_share = max(program_counts.values()) / max(len(programs), 1) if programs else 1.0
    return {
        "record_id": first_text(diag.get("record_id")),
        "phase": first_text(diag.get("audit_phase")),
        "candidate_count": len(triples),
        "exact_completion_unique_ratio": len(set(completions)) / max(len(completions), 1),
        "within_record_unique_program_ratio": len(set(programs)) / max(len(triples), 1),
        "executable_program_unique_ratio": len(set(executable_programs)) / max(len(executable_programs), 1),
        "canonical_skeleton_unique_ratio": len(set(skeletons)) / max(len(triples), 1),
        "reward_unique_count": len(set(rewards)),
        "reward_std": reward_std,
        "correct_candidate_count": len(correct),
        "wrong_executable_candidate_count": len(wrong_exec),
        "invalid_candidate_count": len(invalid),
        "mixed_reward_group": len(set(rewards)) >= 2,
        "answer_variance_group": bool(correct) and len(correct) < len(triples),
        "executable_contrast_group": bool(correct) and bool(wrong_exec),
        "effective_candidate_diversity": len(set(executable_programs)) >= 2,
        "canonical_candidate_diversity": len(set(skeletons)) >= 2,
        "duplicate_collapsed": max_program_share >= 0.75,
        "max_program_share": max_program_share,
    }


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def share_top(counter: Counter[str], k: int) -> float:
    total = sum(counter.values())
    return sum(count for _item, count in counter.most_common(k)) / max(total, 1)


def summarize_diversity(frontier_rows: Sequence[Dict[str, Any]], diagnostics_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    winner_programs = []
    skeletons = []
    families = []
    candidate_metrics = []
    for row in frontier_rows:
        meta = row.get("metadata") or {}
        winner = prediction_program(first_text(meta.get("v34r23_winner_prediction") or row.get("reference_response")))
        winner_programs.append(normalize_program_exact(winner))
        skeletons.append(canonical_skeleton(winner))
        families.append(program_family_from_program(winner))
        diag = diagnostics_by_id.get(record_id(row))
        if diag:
            candidate_metrics.append(analyze_record_candidates(diag))
    skeleton_counts = Counter(skeletons)
    family_counts = Counter(families)
    candidate_fields = ["within_record_unique_program_ratio", "exact_completion_unique_ratio", "executable_program_unique_ratio", "canonical_skeleton_unique_ratio", "reward_std", "correct_candidate_count", "wrong_executable_candidate_count", "invalid_candidate_count"]
    candidate_summary = {field: quantiles([metric[field] for metric in candidate_metrics]) for field in candidate_fields}
    bool_summary = {
        "mixed_reward_group_ratio": sum(1 for metric in candidate_metrics if metric["mixed_reward_group"]) / max(len(candidate_metrics), 1),
        "answer_variance_group_ratio": sum(1 for metric in candidate_metrics if metric["answer_variance_group"]) / max(len(candidate_metrics), 1),
        "executable_contrast_group_ratio": sum(1 for metric in candidate_metrics if metric["executable_contrast_group"]) / max(len(candidate_metrics), 1),
        "effective_candidate_diversity_ratio": sum(1 for metric in candidate_metrics if metric["effective_candidate_diversity"]) / max(len(candidate_metrics), 1),
        "canonical_candidate_diversity_ratio": sum(1 for metric in candidate_metrics if metric["canonical_candidate_diversity"]) / max(len(candidate_metrics), 1),
        "duplicate_collapse_ratio": sum(1 for metric in candidate_metrics if metric["duplicate_collapsed"]) / max(len(candidate_metrics), 1),
    }
    literal_share = family_counts.get("literal_direct_lookup", 0) / max(len(frontier_rows), 1)
    single_op = sum(family_counts.get(name, 0) for name in ["add", "subtract", "multiply", "divide"]) / max(len(frontier_rows), 1)
    multi_step = (family_counts.get("multi_step_nested", 0) + family_counts.get("percentage_change_skeleton", 0)) / max(len(frontier_rows), 1)
    return {
        "records": len(frontier_rows),
        "winner_exact_unique_count": len(set(winner_programs)),
        "winner_exact_unique_ratio": len(set(winner_programs)) / max(len(winner_programs), 1),
        "unique_canonical_skeleton_count": len(skeleton_counts),
        "skeleton_entropy": entropy(skeleton_counts),
        "top1_skeleton_share": share_top(skeleton_counts, 1),
        "top3_skeleton_share": share_top(skeleton_counts, 3),
        "top5_skeleton_share": share_top(skeleton_counts, 5),
        "direct_lookup_literal_share": literal_share,
        "single_operation_share": single_op,
        "multi_step_share": multi_step,
        "skeleton_distribution": dict(skeleton_counts.most_common()),
        "program_family_distribution": dict(family_counts),
        "candidate_metrics": candidate_summary,
        "grpo_signal_metrics": bool_summary,
        "record_candidate_metrics": candidate_metrics,
    }


def attrition_for_phase(phase: str, diagnostics: Sequence[Dict[str, Any]], source_rows: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    phase_diags = [diag for diag in diagnostics if first_text(diag.get("audit_phase")) == phase]
    steps = ["diagnostic", "learnable_hard_coarse", "greedy_wrong", "sampled_correct_exists", "wrong_executable_exists", "winner_current_rollout", "no_reference_leakage", "no_winner_leakage", "no_hard_negative_leakage", "no_assignment", "no_forbidden_marker", "no_multiple_program", "executor_clean", "non_noisy", "strict_frontier"]
    reason_examples: Dict[str, List[str]] = defaultdict(list)

    def passes(diag: Dict[str, Any], step: str) -> bool:
        key = diagnostic_key(diag)
        row = source_rows.get(key, {})
        triples = scored_predictions(diag)
        correct_clean = [(idx, score, prediction) for idx, score, prediction in triples if score_correct(score) and clean_rollout_prediction(prediction)]
        wrong_exec = [(idx, score, prediction) for idx, score, prediction in triples if score_executable(score) and not score_correct(score)]
        if step == "learnable_hard_coarse":
            return first_text(diag.get("bucket")) == "learnable-hard"
        if step == "greedy_wrong":
            return not bool(diag.get("greedy_correct"))
        if step == "sampled_correct_exists":
            return bool(correct_clean)
        if step == "wrong_executable_exists":
            return bool(wrong_exec)
        if step == "winner_current_rollout":
            return bool(correct_clean)
        if step in {"no_reference_leakage", "no_winner_leakage", "no_hard_negative_leakage"}:
            winner = correct_clean[0][2] if correct_clean else ""
            negative = wrong_exec[0][2] if wrong_exec else ""
            prompt = first_text(row.get("input_prompt_raw"))
            if step == "no_reference_leakage":
                return not bool(winner and winner in prompt)
            if step == "no_winner_leakage":
                return not bool(winner and winner in prompt)
            return not bool(negative and negative in prompt)
        if step == "no_assignment":
            winner = correct_clean[0][2] if correct_clean else ""
            return "=" not in prediction_program(winner)
        if step == "no_forbidden_marker":
            winner = correct_clean[0][2] if correct_clean else ""
            return not any(marker in winner for marker in ["Reasoning:", "Answer:", "Normalized Answer:", "Operation Plan", "Formula candidates"])
        if step == "no_multiple_program":
            winner = correct_clean[0][2] if correct_clean else ""
            return winner.count("Program:") == 1
        if step in {"executor_clean", "non_noisy"}:
            ok, _reason = valid_train_row(row)
            return ok
        return True

    current = list(phase_diags)
    funnel = []
    for step in steps:
        input_count = len(current)
        if step == "diagnostic":
            passed, failed = current, []
        else:
            passed, failed = [], []
            for diag in current:
                if passes(diag, step):
                    passed.append(diag)
                else:
                    failed.append(diag)
                    if len(reason_examples[step]) < 5:
                        reason_examples[step].append(first_text(diag.get("record_id")))
        funnel.append({"step": step, "input": input_count, "filtered": len(failed), "remaining": len(passed), "examples": reason_examples.get(step, [])})
        current = passed
    return {"phase": phase, "diagnostic_records": len(phase_diags), "funnel": funnel, "strict_frontier_records": len(current)}


def prompt_processor_consistency(args: argparse.Namespace, source_rows: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    summaries = {}
    for name, path in [("pilot", args.pilot_summary), ("extension", args.extension_summary)]:
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text())
            summaries[name] = {"path": path, "use_grpo_prompt_processor": bool(data.get("use_grpo_prompt_processor")), "total": data.get("total")}
    mismatch = summaries.get("pilot", {}).get("use_grpo_prompt_processor") != summaries.get("extension", {}).get("use_grpo_prompt_processor")
    examples = list(source_rows.values())[:3]
    return {"summaries": summaries, "pilot_extension_prompt_processor_mismatch": bool(mismatch), "processed_prompt_hash_examples": [hashlib.sha256(first_text(row.get("input_prompt_raw")).encode()).hexdigest() for row in examples]}


def gate_decision(summary: Dict[str, Any]) -> Dict[str, Any]:
    contract = summary["contract_metrics"]
    div = summary["diversity"]
    signal = div["grpo_signal_metrics"]
    candidate = div["candidate_metrics"]
    checks = {
        "leakage_zero": contract["leakage_count"] == 0,
        "frontier_total_ge_64": div["records"] >= 64,
        "train_frontier_ge_56": contract["train_frontier"] >= 56,
        "valid_frontier_ge_8": contract["valid_frontier"] >= 8,
        "executable_contrast_ge_0_80": signal["executable_contrast_group_ratio"] >= 0.80,
        "mixed_reward_ge_0_60": signal["mixed_reward_group_ratio"] >= 0.60,
        "duplicate_collapse_le_0_40": signal["duplicate_collapse_ratio"] <= 0.40,
        "within_exec_unique_mean_ge_0_25": candidate["executable_program_unique_ratio"]["mean"] >= 0.25,
        "sampled_executable_ge_0_70": contract["sampled_executable_rate"] >= 0.70,
        "direct_lookup_le_0_40": div["direct_lookup_literal_share"] <= 0.40,
        "top1_skeleton_le_0_40": div["top1_skeleton_share"] <= 0.40,
        "history_and_nonhistory_present": contract["history_frontier"] > 0 and contract["nonhistory_frontier"] > 0,
        "program_family_ge_3": len([key for key, value in div["program_family_distribution"].items() if value > 0]) >= 3,
        "prompt_processor_consistent": not summary["prompt_processor_consistency"]["pilot_extension_prompt_processor_mismatch"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if summary["prompt_processor_consistency"]["pilot_extension_prompt_processor_mismatch"]:
        code = "pilot_extension_prompt_processor_mismatch_reacquisition_required"
        allow = False
    elif failed:
        code = "clean_frontier_candidate_diversity_insufficient"
        allow = False
    else:
        code = "clean_frontier_quality_gate_redefined_online_probe_allowed"
        allow = True
    return {"checks": checks, "failed_checks": failed, "online_probe_allowed_next_round": allow, "conclusion_code": code}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier_dir", required=True)
    parser.add_argument("--source_train_file", required=True)
    parser.add_argument("--diagnostics_file", action="append", required=True)
    parser.add_argument("--pilot_summary", default="")
    parser.add_argument("--extension_summary", default="")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    frontier_dir = Path(args.frontier_dir)
    frontier_rows = read_jsonl(frontier_dir / "frontier.jsonl")
    train_rows = read_jsonl(frontier_dir / "train.jsonl")
    valid_rows = read_jsonl(frontier_dir / "valid.jsonl")
    frontier_summary = json.loads((frontier_dir / "summary.json").read_text())
    diagnostics = load_diagnostics(args.diagnostics_file)
    diagnostics_by_id = {first_text(diag.get("record_id")): diag for diag in diagnostics}
    source_rows = load_source_rows(args.source_train_file, use_grpo_prompt_processor=True)
    leakage_count = sum(1 for row in frontier_rows + train_rows + valid_rows if prompt_target_leakage_reason(row))
    frontier_ids = {record_id(row) for row in frontier_rows}
    train_ids = {record_id(row) for row in train_rows}
    valid_ids = {record_id(row) for row in valid_rows}
    summary = {
        "version": "v34r23_frontier_program_diversity_audit",
        "frontier_dir": args.frontier_dir,
        "diagnostics_files": args.diagnostics_file,
        "metric_definition_audit": {
            "scripts/probe_v34r23_online_reward_variance.py:133": "unique_program_ratio = unique parsed Program strings across generated completions divided by non-empty program count; computed over probe output rows/completions; no numeric skeleton normalization; Evidence excluded; winner-only no.",
            "training/finqa_program_grpo.py:2176": "program/unique_program_ratio = unique non-empty Program strings in current reward batch divided by non-empty Program count; no numeric skeleton normalization; Evidence excluded; winner-only no.",
            "scripts/build_v34r21_sft2_stratified_frontier_grpo.py:164-187": "unique_candidate_program_ratio = per-record unique candidate executed programs divided by candidate program count, then averaged; no cross-record winner ratio.",
            "current_failed_gate": "winner_unique_program_ratio was computed ad hoc as cross-record unique winner Program strings / frontier records; this was not the original unique_program_ratio semantics and should be descriptive only.",
        },
        "contract_metrics": {
            "frontier_records": len(frontier_rows),
            "train_frontier": len(train_ids & frontier_ids),
            "valid_frontier": len(valid_ids & frontier_ids),
            "train_valid_overlap": len(train_ids & valid_ids),
            "leakage_count": leakage_count,
            "sampled_executable_rate": frontier_summary.get("sampled_executable_rate", 0.0),
            "history_frontier": sum(1 for row in frontier_rows if (row.get("metadata") or {}).get("v34r23_requires_history")),
            "nonhistory_frontier": sum(1 for row in frontier_rows if not (row.get("metadata") or {}).get("v34r23_requires_history")),
            "pilot_frontier_contribution": frontier_summary.get("pilot_frontier_contribution"),
            "extension_frontier_contribution": frontier_summary.get("extension_frontier_contribution"),
        },
    }
    summary["diversity"] = summarize_diversity(frontier_rows, diagnostics_by_id)
    summary["attrition"] = {"pilot": attrition_for_phase("pilot", diagnostics, source_rows), "extension": attrition_for_phase("extension", diagnostics, source_rows)}
    summary["prompt_processor_consistency"] = prompt_processor_consistency(args, source_rows)
    summary["revised_gate_decision"] = gate_decision(summary)
    write_json(Path(args.output_file), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

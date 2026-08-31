from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_financial_benchmarks import safe_apply_chat_template
from training.finqa_program_grpo import ScriptArguments, first_text, prepare_dataset


def training_bucket(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return first_text(meta.get("v34r23_bucket") or row.get("training_bucket") or row.get("bucket") or "unknown")


def is_history_dependent(row: Dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    return bool(meta.get("v34r23_requires_history") or meta.get("requires_history"))


def safe_metadata_for_prepare(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("metadata") or {}
    return {
        "requires_history": bool(meta.get("requires_history") or meta.get("v34r23_requires_history")),
        "v34r23_requires_history": bool(meta.get("v34r23_requires_history") or meta.get("requires_history")),
        "v34r23_bucket": first_text(meta.get("v34r23_bucket") or row.get("training_bucket") or row.get("bucket")),
        "question_type": first_text(meta.get("question_type")),
        "operation_type": first_text(meta.get("operation_type")),
        "answer_scale": first_text(meta.get("answer_scale")),
        "difficulty_bucket": first_text(meta.get("difficulty_bucket")),
        "error_bucket": first_text(meta.get("error_bucket")),
    }


def dataset_row_for_prepare(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "record_id": first_text(row.get("record_id")),
        "input_prompt_raw": first_text(row.get("input_prompt_raw")),
        "gold_answer": first_text(row.get("gold_answer")),
        "gold_program": first_text(row.get("gold_program")),
        "reward_profile": first_text(row.get("reward_profile")),
        "source_dataset": first_text(row.get("source_dataset")),
        "metadata": safe_metadata_for_prepare(row),
        "reference_response": first_text(row.get("reference_response")),
    }


def default_script_args(reward_mode: str = "frontier_execution_calibration") -> ScriptArguments:
    return ScriptArguments(reward_mode=reward_mode, reward_profile_expected="program_numeric")


def prepare_rows_like_grpo(rows: Sequence[Dict[str, Any]], script_args: ScriptArguments | None = None, is_main_process: bool = True) -> List[Dict[str, Any]]:
    args = script_args or default_script_args()
    dataset_rows = [dataset_row_for_prepare(row) for row in rows]
    processed = prepare_dataset(Dataset.from_list(dataset_rows), args, is_main_process)
    return [dict(row) for row in processed]


def build_model_prompt_text(tokenizer: Any, row: Dict[str, Any], script_args: ScriptArguments | None = None) -> str:
    processed = prepare_rows_like_grpo([row], script_args=script_args, is_main_process=False)[0]
    return safe_apply_chat_template(tokenizer, processed["prompt"])


def prompt_target_leakage_flags(raw_row: Dict[str, Any], prompt_text: str) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    for field in ["reference_response", "response", "target", "sampled_correct_winner", "completion"]:
        value = first_text(raw_row.get(field))
        flags[field] = bool(value and value in prompt_text)
    meta = raw_row.get("metadata") or {}
    for field in ["v34r23_winner_prediction", "v34r23_hard_negative_prediction"]:
        value = first_text(meta.get(field))
        flags[field] = bool(value and value in prompt_text)
    return flags


def prompt_target_leakage_reason(row: Dict[str, Any], prompt_field: str = "input_prompt_raw") -> str:
    prompt = first_text(row.get(prompt_field))
    meta = row.get("metadata") or {}
    checks = [
        ("reference_response", first_text(row.get("reference_response"))),
        ("winner_prediction", first_text(meta.get("v34r23_winner_prediction"))),
        ("hard_negative_prediction", first_text(meta.get("v34r23_hard_negative_prediction"))),
    ]
    leaked = [name for name, value in checks if value and value in prompt]
    return "target_leakage_" + "+".join(leaked) if leaked else ""


def row_content_hash(rows: Sequence[Dict[str, Any]]) -> str:
    import hashlib

    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

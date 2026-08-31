from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_preference_pairs_from_lh_diagnostics import (
    build_pair_from_diagnostic,
    split_pairs,
    write_preference_dataset,
)


def diagnostic(
    *,
    bucket: str = "learnable-hard",
    greedy_prediction: str = "Evidence:\n- wrong\n\nProgram: subtract(2, 1)",
    greedy_correct: bool = False,
    sampled_predictions=None,
    sampled_scores=None,
):
    if sampled_predictions is None:
        sampled_predictions = [
            "Evidence:\n- correct\n\nProgram: add(1, 1)",
            "Evidence:\n- wrong sampled\n\nProgram: subtract(3, 1)",
        ]
    if sampled_scores is None:
        sampled_scores = [
            {"executed_answer_accuracy": 1.0, "program_execution_rate": 1.0},
            {"executed_answer_accuracy": 0.0, "program_execution_rate": 1.0},
        ]
    return {
        "record_id": "rec-1",
        "source_dataset": "finqa",
        "bucket": bucket,
        "input_prompt_raw": "Question text",
        "gold_answer": "2",
        "gold_program": "add(1, 1)",
        "greedy_prediction": greedy_prediction,
        "greedy_score": {
            "executed_answer_accuracy": 1.0 if greedy_correct else 0.0,
            "program_execution_rate": 1.0,
        },
        "sample_correct_count": sum(1 for score in sampled_scores if score.get("executed_answer_accuracy")),
        "sample_executable_count": sum(1 for score in sampled_scores if score.get("program_execution_rate")),
        "correct_sample_indices": [idx for idx, score in enumerate(sampled_scores) if score.get("executed_answer_accuracy")],
        "sampled_predictions": sampled_predictions,
        "sampled_scores": sampled_scores,
    }


def test_build_pair_uses_correct_sample_as_chosen_and_greedy_wrong_as_rejected():
    pair = build_pair_from_diagnostic(diagnostic())
    assert pair is not None
    assert pair["system"] == ""
    assert pair["history"] == []
    assert pair["question"] == "Question text"
    assert pair["response_chosen"].startswith("Evidence:\n- correct")
    assert pair["response_rejected"].startswith("Evidence:\n- wrong")
    assert pair["metadata"]["chosen_candidate_index"] == 0
    assert pair["metadata"]["rejected_source"] == "greedy"


def test_build_pair_falls_back_to_sampled_executable_wrong_when_greedy_missing():
    item = diagnostic(greedy_prediction="")
    pair = build_pair_from_diagnostic(item)
    assert pair is not None
    assert pair["response_rejected"].startswith("Evidence:\n- wrong sampled")
    assert pair["metadata"]["rejected_source"] == "sampled"
    assert pair["metadata"]["rejected_candidate_index"] == 1


def test_build_pair_skips_non_learnable_hard_or_no_positive():
    assert build_pair_from_diagnostic(diagnostic(bucket="easy")) is None
    item = diagnostic(
        sampled_predictions=["Evidence:\n- wrong\n\nProgram: 1"],
        sampled_scores=[{"executed_answer_accuracy": 0.0, "program_execution_rate": 1.0}],
    )
    assert build_pair_from_diagnostic(item) is None


def test_split_pairs_keeps_record_ids_disjoint():
    pairs = []
    for idx in range(20):
        item = diagnostic()
        item["record_id"] = f"rec-{idx}"
        pair = build_pair_from_diagnostic(item)
        assert pair is not None
        pairs.append(pair)
    train, valid = split_pairs(pairs, valid_ratio=0.1, min_valid=3, max_valid=5, seed=42)
    assert len(valid) == 3
    assert len(train) == 17
    assert {row["record_id"] for row in train}.isdisjoint({row["record_id"] for row in valid})


def test_write_preference_dataset_writes_train_valid_and_summary(tmp_path: Path):
    pairs = []
    for idx in range(20):
        item = diagnostic()
        item["record_id"] = f"rec-{idx}"
        pair = build_pair_from_diagnostic(item)
        assert pair is not None
        pairs.append(pair)
    summary = write_preference_dataset(pairs, tmp_path, valid_ratio=0.1, min_valid=3, max_valid=5, seed=42)
    train_file = tmp_path / "train_dir" / "train_preference_v7.jsonl"
    valid_file = tmp_path / "valid_dir" / "valid_preference_v7.jsonl"
    summary_file = tmp_path / "preference_pair_summary.json"
    assert train_file.exists()
    assert valid_file.exists()
    assert summary_file.exists()
    assert summary["train_pairs"] == 17
    assert summary["valid_pairs"] == 3
    first_train = json.loads(train_file.read_text(encoding="utf-8").splitlines()[0])
    assert {"system", "history", "question", "response_chosen", "response_rejected"}.issubset(first_train)


if __name__ == "__main__":
    import tempfile

    test_build_pair_uses_correct_sample_as_chosen_and_greedy_wrong_as_rejected()
    test_build_pair_falls_back_to_sampled_executable_wrong_when_greedy_missing()
    test_build_pair_skips_non_learnable_hard_or_no_positive()
    test_split_pairs_keeps_record_ids_disjoint()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_preference_dataset_writes_train_valid_and_summary(Path(tmp))

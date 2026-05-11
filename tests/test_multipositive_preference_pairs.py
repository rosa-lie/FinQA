from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_multipositive_preference_pairs_from_lh_diagnostics import (
    build_pairs_from_diagnostic,
    split_pairs,
    write_preference_dataset,
)


class Args:
    pass


def args():
    item = Args()
    item.diagnostics_file = "diagnostics.jsonl"
    item.max_pairs = 0
    item.max_positive_per_record = 2
    item.max_pairs_per_record = 2
    item.min_sample_correct_count = 2
    item.drop_chosen_with_answer = True
    item.drop_rejected_with_answer = False
    item.valid_ratio = 0.1
    item.min_valid = 2
    item.max_valid = 4
    item.seed = 42
    return item


def diagnostic(*, sample_correct_count: int = 2):
    return {
        "record_id": "rec-1",
        "source_dataset": "finqa",
        "bucket": "learnable-hard",
        "input_prompt_raw": "Question text",
        "gold_answer": "2",
        "gold_program": "add(1, 1)",
        "greedy_prediction": "Evidence:\n- wrong greedy\n\nProgram: subtract(3, 1)",
        "greedy_score": {"executed_answer_accuracy": 0.0, "program_execution_rate": 1.0},
        "sample_correct_count": sample_correct_count,
        "sample_executable_count": 4,
        "sampled_predictions": [
            "Evidence:\n- correct one\n\nProgram: add(1, 1)",
            "Evidence:\n- correct two\n\nProgram: sum(1, 1)",
            "Evidence:\n- wrong sample\n\nProgram: multiply(2, 2)",
            "Evidence:\n- bad chosen\n\nProgram: add(1, 1)\nAnswer: 2",
        ],
        "sampled_scores": [
            {"executed_answer_accuracy": 1.0, "program_execution_rate": 1.0},
            {"executed_answer_accuracy": 1.0, "program_execution_rate": 1.0},
            {"executed_answer_accuracy": 0.0, "program_execution_rate": 1.0},
            {"executed_answer_accuracy": 1.0, "program_execution_rate": 1.0},
        ],
    }


def test_builds_multiple_positive_pairs_against_greedy_negative():
    pairs = build_pairs_from_diagnostic(diagnostic(), args())
    assert len(pairs) == 2
    assert pairs[0]["response_chosen"].startswith("Evidence:\n- correct one")
    assert pairs[1]["response_chosen"].startswith("Evidence:\n- correct two")
    assert all(pair["response_rejected"].startswith("Evidence:\n- wrong greedy") for pair in pairs)
    assert [pair["metadata"]["chosen_candidate_index"] for pair in pairs] == [0, 1]
    assert all(pair["metadata"]["rejected_source"] == "greedy" for pair in pairs)


def test_skips_low_confidence_or_answer_leaking_chosen():
    low_conf_args = args()
    assert build_pairs_from_diagnostic(diagnostic(sample_correct_count=1), low_conf_args) == []

    leak_args = args()
    leak_args.max_positive_per_record = 4
    leak_args.max_pairs_per_record = 4
    pairs = build_pairs_from_diagnostic(diagnostic(), leak_args)
    assert len(pairs) == 2
    assert not any("\nAnswer:" in pair["response_chosen"] for pair in pairs)


def test_write_dataset_outputs_train_valid_and_summary(tmp_path: Path):
    rows = []
    for idx in range(10):
        item = diagnostic()
        item["record_id"] = f"rec-{idx}"
        rows.extend(build_pairs_from_diagnostic(item, args()))
    summary = write_preference_dataset(rows, tmp_path, args())
    assert summary["total_pairs"] == 20
    assert summary["unique_records"] == 10
    assert summary["valid_pairs"] == 2
    train_file = tmp_path / "train_dir" / "train_preference_v8_1.jsonl"
    valid_file = tmp_path / "valid_dir" / "valid_preference_v8_1.jsonl"
    assert train_file.exists()
    assert valid_file.exists()
    first_train = json.loads(train_file.read_text(encoding="utf-8").splitlines()[0])
    assert {"system", "history", "question", "response_chosen", "response_rejected"}.issubset(first_train)


def test_split_pairs_is_deterministic():
    rows = []
    for idx in range(10):
        item = diagnostic()
        item["record_id"] = f"rec-{idx}"
        rows.extend(build_pairs_from_diagnostic(item, args()))
    train_a, valid_a = split_pairs(rows, valid_ratio=0.1, min_valid=2, max_valid=4, seed=42)
    train_b, valid_b = split_pairs(rows, valid_ratio=0.1, min_valid=2, max_valid=4, seed=42)
    assert len(train_a) == 18
    assert len(valid_a) == 2
    assert [row["record_id"] for row in valid_a] == [row["record_id"] for row in valid_b]


if __name__ == "__main__":
    import tempfile

    test_builds_multiple_positive_pairs_against_greedy_negative()
    test_skips_low_confidence_or_answer_leaking_chosen()
    test_split_pairs_is_deterministic()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_dataset_outputs_train_valid_and_summary(Path(tmp))

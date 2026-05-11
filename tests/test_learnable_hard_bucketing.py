from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from argparse import Namespace

from scripts.build_learnable_hard_buckets import (
    build_mix_rows,
    classify_bucket,
    filter_rows_by_source,
    should_sample_after_greedy,
    summarize_buckets,
)


def score(correct: bool = False, executable: bool = True, has_program: bool = True):
    return {
        "executed_answer_accuracy": 1.0 if correct else 0.0,
        "program_execution_rate": 1.0 if executable else 0.0,
        "program_parse_rate": 1.0 if has_program else 0.0,
    }


def test_greedy_correct_is_easy():
    bucket = classify_bucket(score(correct=True), [score(correct=False)], noisy=False)
    assert bucket == "easy"


def test_greedy_wrong_sample_correct_is_learnable_hard():
    bucket = classify_bucket(
        score(correct=False),
        [score(correct=False), score(correct=True), score(correct=False)],
        noisy=False,
    )
    assert bucket == "learnable-hard"


def test_all_wrong_executable_samples_are_hard():
    bucket = classify_bucket(
        score(correct=False),
        [score(correct=False), score(correct=False), score(correct=False)],
        noisy=False,
    )
    assert bucket == "hard"


def test_mostly_unexecutable_samples_are_invalid_prone():
    bucket = classify_bucket(
        score(correct=False),
        [
            score(correct=False, executable=False, has_program=False),
            score(correct=False, executable=False, has_program=True),
            score(correct=False, executable=True, has_program=True),
        ],
        noisy=False,
    )
    assert bucket == "invalid-prone"


def test_noisy_overrides_scores():
    bucket = classify_bucket(score(correct=True), [score(correct=True)], noisy=True)
    assert bucket == "noisy"


def test_mix_rows_uses_roughly_twenty_percent_easy_replay():
    learnable = [{"record_id": f"lh-{idx}"} for idx in range(8)]
    easy = [{"record_id": f"easy-{idx}"} for idx in range(10)]
    mixed = build_mix_rows(learnable, easy, easy_replay_ratio=0.2, seed=42)
    easy_count = sum(1 for row in mixed if row["record_id"].startswith("easy-"))
    learnable_count = sum(1 for row in mixed if row["record_id"].startswith("lh-"))
    assert learnable_count == 8
    assert easy_count == 2
    assert len(mixed) == 10


def test_source_dataset_filter_keeps_only_requested_sources():
    rows = [
        {"record_id": "1", "source_dataset": "finqa"},
        {"record_id": "2", "source_dataset": "convfinqa_turn"},
        {"record_id": "3", "source_dataset": "finqa"},
    ]
    filtered = filter_rows_by_source(rows, "finqa")
    assert [row["record_id"] for row in filtered] == ["1", "3"]


def test_source_dataset_filter_accepts_comma_separated_sources():
    rows = [
        {"record_id": "1", "source_dataset": "finqa"},
        {"record_id": "2", "source_dataset": "convfinqa_turn"},
        {"record_id": "3", "source_dataset": "other"},
    ]
    filtered = filter_rows_by_source(rows, "finqa,convfinqa_turn")
    assert [row["record_id"] for row in filtered] == ["1", "2"]


def test_skip_sampling_if_greedy_correct_saves_easy_samples():
    args = Namespace(num_samples_per_example=4, skip_sampling_if_greedy_correct=True)
    assert should_sample_after_greedy(args, score(correct=True), noisy=False) is False
    assert should_sample_after_greedy(args, score(correct=False), noisy=False) is True


def test_summary_reports_generation_counts():
    diagnostics = [
        {
            "record_id": "1",
            "source_dataset": "finqa",
            "bucket": "easy",
            "greedy_prediction": "Evidence:\n- a\n\nProgram: 1",
            "sample_count": 0,
        },
        {
            "record_id": "2",
            "source_dataset": "finqa",
            "bucket": "learnable-hard",
            "greedy_prediction": "Evidence:\n- a\n\nProgram: 2",
            "sample_count": 4,
        },
    ]
    summary = summarize_buckets(
        {
            "easy": [{"record_id": "1"}],
            "learnable-hard": [{"record_id": "2"}],
            "hard": [],
            "invalid-prone": [],
            "noisy": [],
        },
        diagnostics,
    )
    assert summary["generation_count_estimate"] == {"greedy": 2, "sampled": 4, "total": 6}

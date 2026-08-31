import json
from pathlib import Path

import pytest

from scripts.run_v34r23_controlled_smoke import (
    FIXED_SMOKE_CONFIG,
    assert_controlled_smoke_config,
    reward_group_summary,
    select_frontier_smoke_rows,
    smoke_train_row_count,
)


def row(record_id, bucket="frontier", history=False):
    return {
        "record_id": record_id,
        "source_dataset": "convfinqa_turn",
        "input_prompt_raw": f"Question {record_id}",
        "gold_answer": "1",
        "gold_program": "1",
        "reward_profile": "program_numeric",
        "metadata": {"v34r23_bucket": bucket, "v34r23_requires_history": history},
    }


def test_select_frontier_smoke_rows_excludes_retention_and_records_metadata():
    frontier_rows = [row(f"f{index}", "frontier", bool(index % 2)) for index in range(40)]
    rows = [row("r1", "retention_variance"), *frontier_rows]
    selected = select_frontier_smoke_rows(rows, max_steps=2, seed=34023)
    assert len(selected) == smoke_train_row_count(2)
    assert all(item["metadata"]["v34r23_bucket"] == "frontier" for item in selected)


def test_controlled_smoke_config_rejects_steps_above_five_and_mutated_sampling():
    good = dict(FIXED_SMOKE_CONFIG)
    assert_controlled_smoke_config(good)

    bad_steps = dict(good, max_steps=6)
    with pytest.raises(ValueError, match="max_steps"):
        assert_controlled_smoke_config(bad_steps)

    bad_temp = dict(good, temperature=0.9)
    with pytest.raises(ValueError, match="temperature"):
        assert_controlled_smoke_config(bad_temp)


def test_reward_group_summary_includes_reward_std_and_strict_markers():
    completions = [
        "Evidence:\n- one\n\nProgram: 1",
        "Evidence:\n- two\n\nProgram: 2\nProgram: 3",
        "Reasoning: x\nEvidence:\n- one\n\nProgram: 1",
    ]
    diagnostics = [
        {"exact_match": 1.0, "executable": 1.0, "invalid": 0.0, "wrong_executable": 0.0, "program": "1"},
        {"exact_match": 0.0, "executable": 1.0, "invalid": 0.0, "wrong_executable": 1.0, "program": "2"},
        {"exact_match": 1.0, "executable": 1.0, "invalid": 0.0, "wrong_executable": 0.0, "program": "1"},
    ]
    summary = reward_group_summary([1.0, -0.1, 0.25], completions, diagnostics)
    assert summary["reward_std"] > 0
    assert summary["mixed_reward"] is True
    assert summary["sampled_correct_rate"] == 2 / 3
    assert summary["wrong_executable_rate"] == 1 / 3
    assert summary["reasoning_marker_rate"] == 1 / 3
    assert summary["multiple_program_rate"] == 1 / 3

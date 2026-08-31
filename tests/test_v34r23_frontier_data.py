from argparse import Namespace
import json
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v34r23_frontier_grpo_data import build_frontier_data, classify_diagnostic


def score(correct=False, executable=True, program="1"):
    return {
        "executed_answer_accuracy": 1.0 if correct else 0.0,
        "program_execution_rate": 1.0 if executable else 0.0,
        "program_parse_rate": 1.0 if executable else 0.0,
        "executed_program": program,
    }


def diagnostic(record_id, greedy_correct, sampled):
    return {
        "record_id": record_id,
        "source_dataset": "finqa",
        "input_prompt_raw": f"prompt {record_id}",
        "gold_answer": "1",
        "gold_program": "1",
        "bucket": "learnable-hard",
        "greedy_correct": greedy_correct,
        "greedy_prediction": "Evidence:\n- x\n\nProgram: 2",
        "sampled_predictions": [f"Evidence:\n- x\n\nProgram: {idx}" for idx, _ in enumerate(sampled)],
        "sampled_scores": sampled,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_classify_requires_correct_and_wrong_executable_candidate():
    assert classify_diagnostic(diagnostic("f", False, [score(True, True, "1"), score(False, True, "2")])) == "frontier"
    assert classify_diagnostic(diagnostic("w", False, [score(False, True, "2")])) == "all_wrong"
    assert classify_diagnostic(diagnostic("c", True, [score(True, True, "1"), score(False, True, "2")])) == "retention_variance"
    assert classify_diagnostic(diagnostic("a", True, [score(True, True, "1")])) == "all_correct"


def test_build_frontier_data_uses_only_frontier_and_variance_retention(tmp_path):
    diagnostics = tmp_path / "diag.jsonl"
    source = tmp_path / "source.jsonl"
    rows = [
        {"record_id": rid, "source_dataset": "finqa", "input_prompt_raw": f"prompt {rid}", "gold_answer": "1", "gold_program": "1", "reward_profile": "program_numeric"}
        for rid in ["f1", "f2", "r1", "easy"]
    ]
    write_jsonl(source, rows)
    write_jsonl(
        diagnostics,
        [
            diagnostic("f1", False, [score(True, True, "1"), score(False, True, "2")]),
            diagnostic("f2", False, [score(True, True, "3"), score(False, True, "4")]),
            diagnostic("r1", True, [score(True, True, "1"), score(False, True, "2")]),
            diagnostic("easy", True, [score(True, True, "1"), score(True, True, "1")]),
        ],
    )

    summary = build_frontier_data(
        Namespace(
            task="finqa",
            diagnostics_file=str(diagnostics),
            source_train_file=str(source),
            output_dir=str(tmp_path / "out"),
            exclude_record_ids=[],
            frontier_ratio=0.8,
            retention_ratio=0.2,
            valid_ratio=0.0,
            per_record_cap=1,
            seed=11,
            min_frontier_records=1,
        )
    )

    train_rows = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert summary["frontier_unique_records"] == 2
    assert summary["retention_unique_records"] == 1
    assert {row["record_id"] for row in train_rows} == {"f1", "f2", "r1"}
    assert all(row["metadata"]["v34r23_winner_source"] == "current_rollout" for row in train_rows if row["record_id"].startswith("f"))
    f1 = next(row for row in train_rows if row["record_id"] == "f1")
    assert f1["reference_response"] == "Evidence:\n- x\n\nProgram: 0"
    assert f1["metadata"]["v34r23_winner_sample_index"] == 0
    assert f1["metadata"]["v34r23_hard_negative_sample_index"] == 1


def test_build_frontier_data_filters_assignment_style_winner(tmp_path):
    diagnostics = tmp_path / "diag.jsonl"
    source = tmp_path / "source.jsonl"
    write_jsonl(
        source,
        [
            {
                "record_id": "bad_assign",
                "source_dataset": "finqa",
                "input_prompt_raw": "prompt",
                "gold_answer": "1",
                "gold_program": "1",
                "reward_profile": "program_numeric",
            }
        ],
    )
    row = diagnostic("bad_assign", False, [score(True, True, "x = 1"), score(False, True, "2")])
    row["sampled_predictions"][0] = "Evidence:\n- x\n\nProgram: x = 1"
    write_jsonl(diagnostics, [row])

    summary = build_frontier_data(
        Namespace(
            task="finqa",
            diagnostics_file=str(diagnostics),
            source_train_file=str(source),
            output_dir=str(tmp_path / "out"),
            exclude_record_ids=[],
            frontier_ratio=0.8,
            retention_ratio=0.2,
            valid_ratio=0.0,
            per_record_cap=1,
            seed=11,
            min_frontier_records=1,
        )
    )

    assert summary["frontier_unique_records"] == 0
    assert summary["bucket_counts"]["noisy"] == 1


def test_build_frontier_data_filters_prompt_target_leakage(tmp_path):
    diagnostics = tmp_path / "diag.jsonl"
    source = tmp_path / "source.jsonl"
    leaked_winner = "Evidence:\n- x\n\nProgram: 0"
    write_jsonl(
        source,
        [
            {
                "record_id": "leaked",
                "source_dataset": "finqa",
                "input_prompt_raw": "prompt with current target\n" + leaked_winner,
                "gold_answer": "1",
                "gold_program": "1",
                "reward_profile": "program_numeric",
            },
            {
                "record_id": "clean",
                "source_dataset": "finqa",
                "input_prompt_raw": "clean prompt",
                "gold_answer": "1",
                "gold_program": "1",
                "reward_profile": "program_numeric",
            },
        ],
    )
    write_jsonl(
        diagnostics,
        [
            diagnostic("leaked", False, [score(True, True, "1"), score(False, True, "2")]),
            diagnostic("clean", False, [score(True, True, "1"), score(False, True, "2")]),
        ],
    )
    summary = build_frontier_data(
        Namespace(
            task="finqa",
            diagnostics_file=str(diagnostics),
            source_train_file=str(source),
            output_dir=str(tmp_path / "out"),
            exclude_record_ids=[],
            frontier_ratio=0.8,
            retention_ratio=0.2,
            valid_ratio=0.0,
            per_record_cap=1,
            seed=11,
            min_frontier_records=1,
        )
    )
    train_rows = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert {row["record_id"] for row in train_rows} == {"clean"}
    assert summary["frontier_unique_records"] == 1
    assert summary["counters"]["invalid_source_target_leakage_reference_response+winner_prediction"] == 1

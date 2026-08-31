import json
from pathlib import Path

from scripts.build_v34r20_sft2_frontier_finqa_grpo import (
    build_v34r20_sft2_frontier_finqa_grpo_data,
)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_row(record_id, *, question_type="multi_step_divide", program="divide(10, 2)", answer="5"):
    return {
        "input_prompt_raw": f"Question for {record_id}",
        "gold_answer": answer,
        "gold_program": program,
        "reference_response": f"Evidence:\n- {record_id}\n\nProgram: {program}",
        "reward_profile": "program_numeric",
        "source_dataset": "finqa",
        "record_id": record_id,
        "metadata": {
            "question_type": question_type,
            "program_ops": ["divide"] if "divide" in program else ["sum"],
            "direct_lookup": program.replace(".", "", 1).isdigit(),
        },
    }


def diagnostic(record_id, *, greedy_correct=False, sample_correct=True, wrong_executable=True):
    sampled_scores = []
    if sample_correct:
        sampled_scores.append({"executed_answer_accuracy": 1.0, "program_execution_rate": 1.0})
    if wrong_executable:
        sampled_scores.append({"executed_answer_accuracy": 0.0, "program_execution_rate": 1.0})
    return {
        "record_id": record_id,
        "source_dataset": "finqa",
        "greedy_correct": greedy_correct,
        "greedy_score": {
            "executed_answer_accuracy": 1.0 if greedy_correct else 0.0,
            "program_execution_rate": 1.0,
        },
        "sampled_scores": sampled_scores,
        "sample_correct_count": sum(1 for score in sampled_scores if score["executed_answer_accuracy"] > 0),
        "sample_executable_count": sum(1 for score in sampled_scores if score["program_execution_rate"] > 0),
    }


def test_builds_frontier_and_retention_with_allowlist_exclusion(tmp_path):
    train_file = tmp_path / "train.jsonl"
    diagnostics_file = tmp_path / "learnable_hard_diagnostics.jsonl"
    allowlist = tmp_path / "allowlist.jsonl"
    output = tmp_path / "out"

    rows = [
        train_row("frontier-1", question_type="growth-rate"),
        train_row("frontier-2", question_type="share-of-total"),
        train_row("frontier-test", question_type="multi-step divide"),
        train_row("no-wrong-exec", question_type="sum", program="sum(1, 4)"),
        train_row("easy-direct", question_type="direct lookup", program="5", answer="5"),
        train_row("easy-sum", question_type="sum", program="sum(2, 3)"),
        train_row("easy-growth", question_type="growth-rate"),
        train_row("easy-share", question_type="share-of-total"),
        {"record_id": "conv-1", "source_dataset": "convfinqa_turn"},
    ]
    write_jsonl(train_file, rows)
    write_jsonl(
        diagnostics_file,
        [
            diagnostic("frontier-1"),
            diagnostic("frontier-2"),
            diagnostic("frontier-test"),
            diagnostic("no-wrong-exec", sample_correct=True, wrong_executable=False),
            diagnostic("easy-direct", greedy_correct=True, sample_correct=False, wrong_executable=False),
            diagnostic("easy-sum", greedy_correct=True, sample_correct=False, wrong_executable=False),
            diagnostic("easy-growth", greedy_correct=True, sample_correct=False, wrong_executable=False),
            diagnostic("easy-share", greedy_correct=True, sample_correct=False, wrong_executable=False),
            {**diagnostic("conv-1"), "source_dataset": "convfinqa_turn"},
        ],
    )
    write_jsonl(allowlist, [{"record_id": "frontier-test"}])

    summary = build_v34r20_sft2_frontier_finqa_grpo_data(
        diagnostics_file=diagnostics_file,
        source_train_file=train_file,
        allowlist_file=allowlist,
        output_dir=output,
        seed=42,
        retention_ratio=0.2,
        retention_min=8,
        retention_max=64,
        valid_ratio=0.25,
    )

    train_rows = read_jsonl(output / "train.jsonl")
    valid_rows = read_jsonl(output / "valid.jsonl")
    frontier_rows = read_jsonl(output / "frontier.jsonl")
    retention_rows = read_jsonl(output / "retention.jsonl")
    summary_json = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert [row["record_id"] for row in frontier_rows] == ["frontier-1", "frontier-2"]
    assert {row["record_id"] for row in retention_rows} == {"easy-direct", "easy-sum", "easy-growth", "easy-share"}
    assert all(row["source_dataset"] == "finqa" for row in train_rows + valid_rows)
    assert all({"input_prompt_raw", "gold_answer", "gold_program", "reward_profile", "source_dataset", "record_id"} <= row.keys() for row in train_rows + valid_rows)
    assert not ({row["record_id"] for row in train_rows + valid_rows} & {"frontier-test"})
    assert summary["frontier_rows"] == 2
    assert summary["retention_rows"] == 4
    assert summary["allowlist_excluded_frontier_candidates"] == 1
    assert summary["frontier_diagnostics"]["greedy_wrong_sample_correct_wrong_executable"] == 2
    assert summary_json["recommend_pass16"] is True


def test_retention_count_uses_ratio_cap_and_seeded_priority(tmp_path):
    train_file = tmp_path / "train.jsonl"
    diagnostics_file = tmp_path / "learnable_hard_diagnostics.jsonl"
    output = tmp_path / "out"

    frontier = [train_row(f"frontier-{idx}") for idx in range(100)]
    easy = [
        train_row("easy-direct", question_type="direct lookup", program="7", answer="7"),
        train_row("easy-sum", question_type="sum", program="sum(3, 4)", answer="7"),
        train_row("easy-growth", question_type="growth-rate"),
        train_row("easy-share", question_type="share-of-total"),
        train_row("easy-divide", question_type="multi-step divide"),
    ] + [train_row(f"easy-extra-{idx}", question_type="other") for idx in range(40)]
    write_jsonl(train_file, frontier + easy)
    write_jsonl(
        diagnostics_file,
        [diagnostic(row["record_id"]) for row in frontier]
        + [diagnostic(row["record_id"], greedy_correct=True, sample_correct=False, wrong_executable=False) for row in easy],
    )

    summary = build_v34r20_sft2_frontier_finqa_grpo_data(
        diagnostics_file=diagnostics_file,
        source_train_file=train_file,
        allowlist_file=None,
        output_dir=output,
        seed=42,
        retention_ratio=0.2,
        retention_min=8,
        retention_max=16,
        valid_ratio=0.0,
    )

    retention_ids = [row["record_id"] for row in read_jsonl(output / "retention.jsonl")]
    assert summary["retention_target_rows"] == 16
    assert len(retention_ids) == 16
    assert {"easy-direct", "easy-sum", "easy-growth", "easy-share", "easy-divide"} <= set(retention_ids)
    assert len(read_jsonl(output / "train.jsonl")) == 116
    assert len(read_jsonl(output / "valid.jsonl")) == 0

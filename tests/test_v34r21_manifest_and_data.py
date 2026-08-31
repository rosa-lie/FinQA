import json
from pathlib import Path

from scripts.build_v34r21_finqa_stratified_manifest import build_manifest
from scripts.build_v34r21_independent_dev_allowlist import build_allowlist
from scripts.build_v34r21_sft2_stratified_frontier_grpo import build_data


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row(record_id, family="ratio", scale="ratio", greedy=False):
    program = {
        "direct_lookup": "5",
        "sum": "add(2, 3)",
        "difference": "subtract(7, 2)",
        "growth_rate": "divide(subtract(7, 2), 2)",
        "multi_step_divide": "divide(add(2, 3), 5)",
    }.get(family, "divide(10, 2)")
    return {
        "record_id": record_id,
        "source_dataset": "finqa",
        "input_prompt_raw": f"Question {record_id}",
        "gold_answer": "5",
        "gold_program": program,
        "reward_profile": "program_numeric",
        "reference_response": f"Evidence:\n- x\n\nProgram: {program}",
        "metadata": {
            "question_type": family,
            "answer_scale": scale,
            "program_canonical": program,
            "program_ops": ["divide"] if "divide" in program else ["add"],
            "program_step_count": 2 if "add(" in program and "divide(" in program else 1,
            "direct_lookup": family == "direct_lookup",
            "program_executable": 5,
            "answer_exe": 5,
        },
    }


def diag(record_id, greedy_correct=False, sample_correct=True, wrong_exec=True):
    scores = []
    responses = []
    programs = []
    if sample_correct:
        scores.append({"executed_answer_accuracy": 1.0, "program_execution_rate": 1.0})
        responses.append("Evidence:\n- ok\n\nProgram: divide(10, 2)")
        programs.append("divide(10, 2)")
    if wrong_exec:
        scores.append({"executed_answer_accuracy": 0.0, "program_execution_rate": 1.0})
        responses.append("Evidence:\n- bad\n\nProgram: subtract(10, 2)")
        programs.append("subtract(10, 2)")
    return {
        "record_id": record_id,
        "source_dataset": "finqa",
        "greedy_correct": greedy_correct,
        "greedy_score": {"executed_answer_accuracy": 1.0 if greedy_correct else 0.0, "program_execution_rate": 1.0},
        "greedy_response": "Evidence:\n- g\n\nProgram: divide(1, 2)",
        "sampled_scores": scores,
        "sampled_responses": responses,
        "sampled_programs": programs,
    }


def test_manifest_is_seeded_and_excludes_history(tmp_path):
    source = tmp_path / "train.jsonl"
    exclude = tmp_path / "exclude.jsonl"
    rows = [row(f"r{i}", family=("direct_lookup" if i == 0 else "sum" if i < 4 else "ratio")) for i in range(12)]
    write_jsonl(source, rows)
    write_jsonl(exclude, [{"record_id": "r1"}])

    summary = build_manifest(
        Args(
            source_train_file=str(source),
            output_dir=str(tmp_path / "manifest"),
            sample_size=6,
            seed=42,
            exclude_allowlist=str(exclude),
            exclude_dev_file="",
            exclude_test_file="",
        )
    )

    manifest = read_jsonl(tmp_path / "manifest" / "acquisition_manifest.jsonl")
    assert summary["sample_size"] == 6
    assert "r1" not in {item["record_id"] for item in manifest}
    assert summary["selected_unique_records"] == 6
    assert (tmp_path / "manifest" / "acquisition_input.jsonl").exists()


def test_frontier_builder_keeps_same_question_winner_negative_and_retention(tmp_path):
    source = tmp_path / "train.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    exclude = tmp_path / "exclude.jsonl"
    rows = [
        row("frontier-1", "ratio"),
        row("frontier-excluded", "ratio"),
        row("all-wrong", "ratio"),
        row("easy-1", "direct_lookup"),
        row("easy-2", "sum"),
    ]
    write_jsonl(source, rows)
    write_jsonl(
        diagnostics,
        [
            diag("frontier-1"),
            diag("frontier-excluded"),
            diag("all-wrong", sample_correct=False, wrong_exec=True),
            diag("easy-1", greedy_correct=True, sample_correct=False, wrong_exec=False),
            diag("easy-2", greedy_correct=True, sample_correct=False, wrong_exec=False),
        ],
    )
    write_jsonl(manifest, [{"record_id": item["record_id"]} for item in rows])
    write_jsonl(exclude, [{"record_id": "frontier-excluded"}])

    summary = build_data(
        Args(
            diagnostics_file=str(diagnostics),
            source_train_file=str(source),
            manifest_file=str(manifest),
            output_dir=str(tmp_path / "out"),
            exclude_record_ids=[str(exclude)],
            seed=42,
            retention_ratio=0.5,
            retention_min=1,
            retention_max=2,
            valid_ratio=0.0,
            min_frontier_records=1,
            numeric_abs_tol=1e-4,
            numeric_rel_tol=1e-4,
        )
    )

    frontier = read_jsonl(tmp_path / "out" / "frontier.jsonl")
    retention = read_jsonl(tmp_path / "out" / "retention.jsonl")
    assert [item["record_id"] for item in frontier] == ["frontier-1"]
    assert len(retention) == 1
    assert summary["frontier_unique_records"] == 1
    assert summary["test_dev_overlap"] == 1
    assert summary["winner_source_distribution"]["current_sft2_rollout"] == 1
    assert summary["hard_negative_type_distribution"]["wrong_executable"] == 1
    assert summary["recommend_stop_no_learnable_frontier"] is False


def test_dev_allowlist_excludes_train_and_history(tmp_path):
    dev = tmp_path / "dev.json"
    exclude = tmp_path / "exclude.jsonl"
    dev.write_text(
        json.dumps(
            [
                {"id": "d1", "qa": {"id": "d1", "program": "divide(10, 2)", "answer": "5"}},
                {"id": "d2", "qa": {"id": "d2", "program": "add(2, 3)", "answer": "5"}},
                {"id": "d3", "qa": {"id": "d3", "program": "subtract(7, 2)", "answer": "5"}},
            ]
        ),
        encoding="utf-8",
    )
    write_jsonl(exclude, [{"record_id": "d2"}])

    summary = build_allowlist(
        Args(dev_file=str(dev), output_dir=str(tmp_path / "devout"), sample_size=2, seed=42, exclude_file=[str(exclude)])
    )
    allow = read_jsonl(tmp_path / "devout" / "v34r21_independent_finqa_dev_allowlist.jsonl")
    assert summary["sample_size"] == 2
    assert "d2" not in {item["record_id"] for item in allow}

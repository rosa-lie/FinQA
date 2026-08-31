import json
from pathlib import Path

import pytest

from scripts.build_v34r18_finqa_only_sft1_grpo import build_finqa_only_sft1_grpo_data


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sft_row(record_id: str, source_dataset: str, question: str, program: str, answer: str):
    return {
        "record_id": record_id,
        "source_dataset": source_dataset,
        "task_type": "program_numeric",
        "conversations": [
            {"from": "human", "value": f"Current question:\n{question}\n\nOutput format:\nEvidence:\n- ...\n\nProgram: ..."},
            {"from": "gpt", "value": f"Evidence:\n- evidence for {record_id}.\n\nProgram: {program}"},
        ],
        "metadata": {
            "program_canonical": program,
            "answer_norm": answer,
            "answer_scale": "ratio" if "divide" in program else "plain",
            "answer_unit": "percent" if "divide" in program else "number",
        },
    }


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_builds_only_finqa_grpo_rows_from_sft1_and_filtered_validation(tmp_path):
    sft1 = tmp_path / "train_sft1_program_strict.jsonl"
    validation = tmp_path / "valid_program_balanced.jsonl"
    output = tmp_path / "out"

    write_jsonl(
        sft1,
        [
            sft_row("finqa-train-1", "FinQA", "what is 121.4 divided by 4187.8?", "divide(121.4, 4187.8)", "0.028997086232866898"),
            sft_row("finqa-train-2", "FinQA", "what is the amount?", "93", "93"),
            sft_row("conv-train-1", "ConvFinQA", "what about in 2008?", "181001", "181001"),
        ],
    )
    write_jsonl(
        validation,
        [
            sft_row("finqa-valid-1", "FinQA", "what is 10 minus 4?", "subtract(10, 4)", "6"),
            sft_row("conv-valid-1", "ConvFinQA", "what about in 2009?", "206588", "206588"),
        ],
    )

    summary = build_finqa_only_sft1_grpo_data(
        sft1_file=sft1,
        validation_file=validation,
        output_dir=output,
        valid_ratio=0.5,
        smoke_rows=1,
        seed=7,
    )

    train_rows = read_jsonl(output / "train.jsonl")
    valid_rows = read_jsonl(output / "valid.jsonl")
    smoke_rows = read_jsonl(output / "smoke.jsonl")
    summary_json = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert {row["source_dataset"] for row in train_rows + valid_rows + smoke_rows} == {"finqa"}
    assert all(row["reward_profile"] == "program_numeric" for row in train_rows + valid_rows)
    assert all("input_prompt_raw" in row and "gold_program" in row and "gold_answer" in row for row in train_rows + valid_rows)
    assert [row["record_id"] for row in valid_rows] == ["finqa-valid-1"]
    assert len(train_rows) == 2
    assert len(smoke_rows) == 1
    assert summary["source_dataset"] == {"finqa": 3}
    assert summary_json["source_dataset"] == {"finqa": 3}
    assert summary_json["excluded_non_finqa_rows"] == 2
    assert not ({row["record_id"] for row in train_rows} & {row["record_id"] for row in valid_rows})


def test_validation_record_ids_are_removed_from_train_split(tmp_path):
    sft1 = tmp_path / "train_sft1_program_strict.jsonl"
    validation = tmp_path / "valid_program_balanced.jsonl"
    output = tmp_path / "out"
    write_jsonl(
        sft1,
        [
            sft_row("shared-finqa", "FinQA", "what is 10 minus 4?", "subtract(10, 4)", "6"),
            sft_row("train-only", "FinQA", "what is 2 plus 3?", "add(2, 3)", "5"),
        ],
    )
    write_jsonl(
        validation,
        [sft_row("shared-finqa", "FinQA", "what is 10 minus 4?", "subtract(10, 4)", "6")],
    )

    summary = build_finqa_only_sft1_grpo_data(
        sft1_file=sft1,
        validation_file=validation,
        output_dir=output,
        valid_ratio=0.5,
        smoke_rows=4,
        seed=3,
    )

    assert [row["record_id"] for row in read_jsonl(output / "valid.jsonl")] == ["shared-finqa"]
    assert [row["record_id"] for row in read_jsonl(output / "train.jsonl")] == ["train-only"]
    assert summary["train_rows_removed_for_valid_overlap"] == 1


def test_falls_back_to_sft1_split_when_no_validation_file(tmp_path):
    sft1 = tmp_path / "train_sft1_program_strict.jsonl"
    output = tmp_path / "out"
    write_jsonl(
        sft1,
        [
            sft_row("finqa-1", "FinQA", "what is 10 minus 4?", "subtract(10, 4)", "6"),
            sft_row("finqa-2", "FinQA", "what is 2 plus 3?", "add(2, 3)", "5"),
            sft_row("finqa-3", "FinQA", "what is 9?", "9", "9"),
        ],
    )

    summary = build_finqa_only_sft1_grpo_data(
        sft1_file=sft1,
        validation_file=None,
        output_dir=output,
        valid_ratio=1 / 3,
        smoke_rows=2,
        seed=13,
    )

    assert summary["validation_source"] == "sft1_split"
    assert len(read_jsonl(output / "valid.jsonl")) == 1
    assert len(read_jsonl(output / "train.jsonl")) == 2
    assert len(read_jsonl(output / "smoke.jsonl")) == 2

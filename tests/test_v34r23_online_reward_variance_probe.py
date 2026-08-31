import json
from pathlib import Path

from scripts.probe_v34r23_online_reward_variance import (
    acquisition_summary_for_rows,
    audit_prompt_leakage,
    dataset_row_for_prepare,
    classify_reward_group,
    config_diff,
    select_rows,
    summarize_probe_rows,
    training_bucket,
)


def test_target_fields_do_not_count_when_absent_from_prompt():
    row = {
        "reference_response": "Evidence:\n- secret\n\nProgram: 42",
        "gold_program": "42",
        "gold_answer": "42",
        "metadata": {"v34r23_winner_prediction": "Evidence:\n- secret\n\nProgram: 42"},
    }
    prompt = "Current question: what is the value?\nTable:\nvalue | 42\nProgram: ..."
    leaks = audit_prompt_leakage(row, prompt)
    assert not leaks["reference_response"]
    assert not leaks["v34r23_winner_prediction"]


def test_bucket_selection_preserves_frontier_and_retention_metadata():
    rows = [
        {"record_id": "f", "metadata": {"v34r23_bucket": "frontier", "v34r23_requires_history": True}},
        {"record_id": "r", "metadata": {"v34r23_bucket": "retention_variance"}},
    ]
    assert training_bucket(rows[0]) == "frontier"
    assert [row["record_id"] for row in select_rows(rows, "frontier", 10, 1)] == ["f"]
    assert [row["record_id"] for row in select_rows(rows, "retention", 10, 1)] == ["r"]
    assert [row["record_id"] for row in select_rows(rows, "history_frontier", 10, 1)] == ["f"]


def test_reward_group_classification_and_std():
    mixed = classify_reward_group([1.0, -0.1, 1.0], [True, False, True])
    assert mixed["reward_std"] > 0
    assert mixed["mixed_reward"]
    all_correct = classify_reward_group([1.0, 1.0], [True, True])
    assert all_correct["zero_std"]
    assert all_correct["all_correct"]
    all_wrong = classify_reward_group([-0.1, -0.1], [False, False])
    assert all_wrong["zero_std"]
    assert all_wrong["all_wrong"]


def test_probe_summary_counts_zero_std_and_unique_programs():
    rows = [
        {"training_bucket": "frontier", "history_dependent": True, "zero_std": False, "all_correct": False, "all_wrong": False, "mixed_reward": True, "completions": ["a", "b"], "scores": [{"answer_correct": True, "executable": True, "program": "1"}, {"answer_correct": False, "executable": True, "program": "2"}]},
        {"training_bucket": "frontier", "history_dependent": False, "zero_std": True, "all_correct": True, "all_wrong": False, "mixed_reward": False, "completions": ["c"], "scores": [{"answer_correct": True, "executable": True, "program": "1"}]},
    ]
    summary = summarize_probe_rows(rows)
    assert summary["records"] == 2
    assert summary["mixed_reward_ratio"] == 0.5
    assert summary["zero_std_ratio"] == 0.5
    assert summary["sampled_correct_rate"] == 2 / 3


def test_acquisition_and_probe_config_diff_marks_mismatches():
    diff = config_diff({"temperature": 0.72, "reward": "executor"}, {"temperature": 0.72, "reward": "frontier"})
    assert diff["temperature"]["match"]
    assert not diff["reward"]["match"]


def test_acquisition_summary_estimates_all_correct_and_all_wrong():
    rows = [
        {"metadata": {"v34r23_sampled_score_count": 8, "v34r23_sampled_correct_count": 8, "v34r23_wrong_executable_count": 0}},
        {"metadata": {"v34r23_sampled_score_count": 8, "v34r23_sampled_correct_count": 0, "v34r23_wrong_executable_count": 8}},
        {"metadata": {"v34r23_sampled_score_count": 8, "v34r23_sampled_correct_count": 4, "v34r23_wrong_executable_count": 4}},
    ]
    summary = acquisition_summary_for_rows(rows)
    assert summary["all_correct_ratio"] == 1 / 3
    assert summary["all_wrong_ratio"] == 1 / 3
    assert summary["zero_std_ratio_estimate"] == 2 / 3


def test_dataset_prepare_view_drops_mixed_raw_metadata_values():
    row = {
        "record_id": "r1",
        "input_prompt_raw": "Question only",
        "gold_answer": "402",
        "gold_program": "402",
        "reward_profile": "program_numeric",
        "source_dataset": "convfinqa_turn",
        "reference_response": "Evidence:\n- leaked\n\nProgram: 402",
        "metadata": {
            "program_executable": 402.0,
            "program_raw": "402",
            "raw_metadata": {"program": "402", "exe_ans": 402.0},
            "v34r23_bucket": "frontier",
            "v34r23_requires_history": True,
            "answer_scale": None,
        },
    }
    prepared = dataset_row_for_prepare(row)
    assert prepared["metadata"]["v34r23_bucket"] == "frontier"
    assert prepared["metadata"]["requires_history"] is True
    assert "program_executable" not in prepared["metadata"]
    assert "raw_metadata" not in prepared["metadata"]
    assert prepared["reference_response"].startswith("Evidence:")


def test_v34r23_frontier_prepare_dataset_uses_program_only_contract():
    from datasets import Dataset
    from training.finqa_program_grpo import ScriptArguments, prepare_dataset

    raw_prompt = (
        "Question: What is the value?\n\n"
        "Output format:\n"
        "Evidence:\n"
        "- ...\n\n"
        "Program: ...\n\n"
        "Report context:\nvalue | 42"
    )
    row = {
        "record_id": "conv_1",
        "input_prompt_raw": raw_prompt,
        "gold_answer": "42",
        "gold_program": "42",
        "reward_profile": "program_numeric",
        "source_dataset": "convfinqa",
        "reference_response": "Evidence:\n- value is 42\n\nProgram: 42",
        "metadata": {"v34r23_bucket": "frontier"},
    }

    processed = prepare_dataset(
        Dataset.from_list([row]),
        ScriptArguments(reward_mode="frontier_execution_calibration", reward_profile_expected="program_numeric", preprocessing_num_workers=1),
        is_main_process=False,
    )[0]
    prompt_text = "\n".join(message["content"] for message in processed["prompt"])

    assert "Reasoning:" not in prompt_text
    assert "Answer:" not in prompt_text
    assert "Normalized Answer:" not in prompt_text
    assert "Output only Evidence and Program" in prompt_text
    assert "Evidence:" in prompt_text
    assert "Program:" in prompt_text


def test_v34r23_shared_prompt_processor_does_not_insert_reasoning():
    from scripts.v34r23_prompt_processor import prepare_rows_like_grpo

    row = {
        "record_id": "conv_2",
        "input_prompt_raw": (
            "Question: What is the value?\n\n"
            "Output format:\n"
            "Evidence:\n"
            "- ...\n\n"
            "Program: ...\n\n"
            "Conversation history:\nPrevious answer: 13\n\n"
            "Report context:\nvalue | 42"
        ),
        "gold_answer": "42",
        "gold_program": "42",
        "reward_profile": "program_numeric",
        "source_dataset": "convfinqa",
        "reference_response": "Evidence:\n- value is 42\n\nProgram: 42",
        "metadata": {
            "v34r23_bucket": "frontier",
            "v34r23_winner_prediction": "Evidence:\n- secret winner\n\nProgram: 42",
            "v34r23_hard_negative_prediction": "Evidence:\n- secret negative\n\nProgram: 41",
        },
    }

    processed = prepare_rows_like_grpo([row], is_main_process=False)[0]
    prompt_text = "\n".join(message["content"] for message in processed["prompt"])

    assert "Reasoning:" not in prompt_text
    assert "Evidence:" in prompt_text
    assert "Program:" in prompt_text
    assert "secret winner" not in prompt_text
    assert "secret negative" not in prompt_text


def test_non_frontier_reasoning_prompt_modes_remain_supported():
    from training.finqa_program_grpo import apply_cot_pot_output_format

    raw_prompt = "Question: x\n\nOutput format:\nEvidence:\n- ...\n\nProgram: ...\n\nReport context:\nx"

    default_prompt = apply_cot_pot_output_format(raw_prompt)
    strict_prompt = apply_cot_pot_output_format(raw_prompt, strict_program_only=True)

    assert "Reasoning:" in default_prompt
    assert "Reasoning:" not in strict_prompt

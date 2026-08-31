import math

from scripts.audit_v34r23_frontier_program_diversity import (
    analyze_record_candidates,
    canonical_skeleton,
    gate_decision,
    is_literal_program,
    program_family_from_program,
    summarize_diversity,
)


def score(correct=False, executable=True, program="1"):
    return {
        "executed_answer_accuracy": 1.0 if correct else 0.0,
        "program_execution_rate": 1.0 if executable else 0.0,
        "executed_program": program,
    }


def diagnostic(record_id="r1"):
    return {
        "record_id": record_id,
        "source_dataset": "convfinqa_turn",
        "audit_phase": "pilot",
        "bucket": "learnable-hard",
        "greedy_correct": False,
        "sampled_predictions": [
            "Evidence:\n- a\n\nProgram: divide(43, 126)",
            "Evidence:\n- b\n\nProgram: subtract(50, 7)",
            "Evidence:\n- c\n\nProgram: N/A",
        ],
        "sampled_scores": [
            score(True, True, "divide(43, 126)"),
            score(False, True, "subtract(50, 7)"),
            score(False, False, ""),
        ],
    }


def test_canonical_skeleton_normalizes_numbers_only():
    assert canonical_skeleton("divide(43, 126)") == "divide(NUM,NUM)"
    assert canonical_skeleton("multiply(divide(subtract(43, 45), 45), 100)") == "multiply(divide(subtract(NUM,NUM),NUM),NUM)"


def test_literal_and_family_classification():
    assert is_literal_program("43")
    assert program_family_from_program("43") == "literal_direct_lookup"
    assert program_family_from_program("subtract(50, 7)") == "subtract"
    assert program_family_from_program("divide(subtract(50, 43), 43)") == "percentage_change_skeleton"


def test_analyze_record_candidates_grpo_signal_metrics():
    metrics = analyze_record_candidates(diagnostic())
    assert metrics["mixed_reward_group"]
    assert metrics["answer_variance_group"]
    assert metrics["executable_contrast_group"]
    assert metrics["effective_candidate_diversity"]
    assert metrics["reward_unique_count"] == 3
    assert metrics["wrong_executable_candidate_count"] == 1
    assert metrics["invalid_candidate_count"] == 1


def test_summarize_diversity_uses_winner_and_candidate_metrics():
    frontier_rows = [
        {"record_id": "r1", "reference_response": "Evidence:\n- a\n\nProgram: divide(43, 126)", "metadata": {"v34r23_winner_prediction": "Evidence:\n- a\n\nProgram: divide(43, 126)"}},
        {"record_id": "r2", "reference_response": "Evidence:\n- b\n\nProgram: divide(44, 127)", "metadata": {"v34r23_winner_prediction": "Evidence:\n- b\n\nProgram: divide(44, 127)"}},
    ]
    diags = {"r1": diagnostic("r1"), "r2": diagnostic("r2")}
    summary = summarize_diversity(frontier_rows, diags)
    assert summary["winner_exact_unique_ratio"] == 1.0
    assert summary["unique_canonical_skeleton_count"] == 1
    assert summary["top1_skeleton_share"] == 1.0
    assert summary["grpo_signal_metrics"]["mixed_reward_group_ratio"] == 1.0


def test_gate_decision_blocks_prompt_processor_mismatch_first():
    summary = {
        "contract_metrics": {"leakage_count": 0, "train_frontier": 56, "valid_frontier": 8, "sampled_executable_rate": 0.8, "history_frontier": 1, "nonhistory_frontier": 1},
        "diversity": {
            "records": 64,
            "direct_lookup_literal_share": 0.1,
            "top1_skeleton_share": 0.1,
            "program_family_distribution": {"subtract": 20, "divide": 20, "literal_direct_lookup": 24},
            "grpo_signal_metrics": {"executable_contrast_group_ratio": 0.9, "mixed_reward_group_ratio": 0.8, "duplicate_collapse_ratio": 0.2},
            "candidate_metrics": {"executable_program_unique_ratio": {"mean": 0.5}},
        },
        "prompt_processor_consistency": {"pilot_extension_prompt_processor_mismatch": True},
    }
    decision = gate_decision(summary)
    assert decision["conclusion_code"] == "pilot_extension_prompt_processor_mismatch_reacquisition_required"
    assert not decision["online_probe_allowed_next_round"]

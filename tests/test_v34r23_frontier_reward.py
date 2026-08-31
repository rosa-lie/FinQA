import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import finqa_program_grpo as grpo


def make_args():
    return grpo.ScriptArguments(
        reward_mode="frontier_execution_calibration",
        schema_hard_gate=True,
        process_reward_enabled=True,
        risk_aware_reward=True,
        dense_reward_shaping=True,
        wrong_executable_reward_cap=0.02,
        program_max_lines=1,
    )


def evaluate(completion, gold_answer="3", gold_program="add(1, 2)"):
    return grpo.evaluate_program_completion(
        make_args(),
        completion,
        gold_answer,
        gold_program,
        prompt_text="values 1 and 2",
        metadata={"question_type": "sum"},
    )


def test_frontier_reward_uses_five_conservative_outcome_bands():
    original = grpo.execute_prediction_program
    values = {
        "add(1,2)": (3.0, "add(1,2)", ""),
        "add(1,4)": (5.0, "add(1,4)", ""),
    }
    grpo.execute_prediction_program = lambda program: values.get(program.replace(" ", ""), (None, "", "invalid"))
    try:
        correct = evaluate("Evidence:\n- values 1 and 2\n\nProgram: add(1, 2)")
        correct_bad_contract = evaluate("Evidence:\n- values 1 and 2\n\nProgram: add(1, 2)\n\nAnswer: 3")
        wrong = evaluate("Evidence:\n- values 1 and 4\n\nProgram: add(1, 4)")
        wrong_bad_contract = evaluate("Evidence:\n- values 1 and 4\n\nProgram: add(1, 4)\n\nAnswer: 5")
        invalid = evaluate("Evidence:\n- values 1 and 2\n\nProgram: N/A")

        assert correct["core_score"] == 1.0
        assert correct_bad_contract["core_score"] == 0.25
        assert wrong["core_score"] == -0.1
        assert wrong_bad_contract["core_score"] == -0.3
        assert invalid["core_score"] == grpo.HARD_INVALID_REWARD
        assert correct["process_score"] == 0.0
        assert correct["semantic_process_score"] == 0.0
    finally:
        grpo.execute_prediction_program = original

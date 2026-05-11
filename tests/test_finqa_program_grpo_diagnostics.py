import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import finqa_program_grpo as grpo


def test_make_reward_funcs_records_program_diagnostics():
    original_executor = grpo.execute_prediction_program
    grpo.execute_prediction_program = lambda program: (3.0, None, None)
    try:
        script_args = grpo.ScriptArguments()
        diagnostics = grpo.ProgramDiagnostics()
        reward_funcs = grpo.make_reward_funcs(script_args, diagnostics=diagnostics)

        completions = ["Evidence:\n- values 1 and 2\n\nProgram: add(1, 2)"]
        kwargs = {
            "gold_answer": ["3"],
            "gold_program": ["add(1, 2)"],
            "input_prompt_raw": ["The report contains values 1 and 2."],
        }

        for reward_func in reward_funcs:
            rewards = reward_func(completions, **kwargs)
            assert len(rewards) == 1

        metrics = diagnostics.pop_means()

        assert metrics["program/core_score"] >= 1.0
        assert metrics["program/executable_rate"] == 1.0
        assert metrics["program/exact_match_rate"] == 1.0
        assert metrics["program/invalid_rate"] == 0.0
        assert metrics["program/wrong_executable_rate"] == 0.0
        assert metrics["program/forbidden_anchor_rate"] == 0.0
        assert metrics["program/multiple_program_rate"] == 0.0
        assert metrics["program/post_program_text_rate"] == 0.0
        assert metrics["program/has_program_rate"] == 1.0
        assert metrics["program/unique_program_ratio"] == 1.0
        assert metrics["program/completion_words"] > 0
    finally:
        grpo.execute_prediction_program = original_executor


if __name__ == "__main__":
    test_make_reward_funcs_records_program_diagnostics()

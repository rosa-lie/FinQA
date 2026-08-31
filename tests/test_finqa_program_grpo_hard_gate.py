import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import finqa_program_grpo as grpo


def run_rewards(completions, gold_answer=None, gold_program=None):
    rewards = []
    for reward_func in grpo.make_reward_funcs(grpo.ScriptArguments()):
        rewards_by_func = reward_func(
            completions,
            gold_answer=gold_answer or ["3"] * len(completions),
            gold_program=gold_program or ["add(1, 2)"] * len(completions),
            input_prompt_raw=["values 1 and 2"] * len(completions),
        )
        rewards.append(rewards_by_func)
    return [sum(values) for values in zip(*rewards)]


def test_invalid_program_has_hard_negative_reward():
    completion = "Evidence:\n- values 1 and 2\n\nProgram: N/A"

    assert run_rewards([completion]) == [-1.0]


def test_wrong_executable_program_is_capped():
    original_executor = grpo.execute_prediction_program
    grpo.execute_prediction_program = lambda program: (5.0, None, None)
    try:
        completion = "Evidence:\n- values 1 and 2\n\nProgram: add(1, 2)"

        reward = run_rewards([completion])[0]

        assert 0.0 <= reward <= 0.2
    finally:
        grpo.execute_prediction_program = original_executor


def test_correct_executable_program_beats_wrong_and_invalid():
    original_executor = grpo.execute_prediction_program
    values = iter([(3.0, None, None), (5.0, None, None)])
    grpo.execute_prediction_program = lambda program: next(values)
    try:
        correct = "Evidence:\n- values 1 and 2\n\nProgram: add(1, 2)"
        wrong = "Evidence:\n- values 1 and 2\n\nProgram: add(1, 2)"
        invalid = "Evidence:\n- values 1 and 2\n\nProgram: N/A"

        correct_reward, wrong_reward, invalid_reward = run_rewards([correct, wrong, invalid])

        assert correct_reward >= 1.0
        assert correct_reward > wrong_reward > invalid_reward
    finally:
        grpo.execute_prediction_program = original_executor


def test_hacking_diagnostics_are_recorded():
    diagnostics = grpo.ProgramDiagnostics()
    reward_funcs = grpo.make_reward_funcs(grpo.ScriptArguments(), diagnostics=diagnostics)
    completion = (
        "Evidence:\n- values 1 and 2\n\n"
        "Program: add(1, 2)\n"
        "Program: subtract(2, 1)\n"
        "Answer: 3"
    )
    original_executor = grpo.execute_prediction_program
    grpo.execute_prediction_program = lambda program: (3.0, None, None)
    try:
        for reward_func in reward_funcs:
            reward_func(
                [completion],
                gold_answer=["3"],
                gold_program=["add(1, 2)"],
                input_prompt_raw=["values 1 and 2"],
            )

        metrics = diagnostics.pop_means()

        assert metrics["program/forbidden_anchor_rate"] == 1.0
        assert metrics["program/multiple_program_rate"] == 1.0
        assert metrics["program/post_program_text_rate"] == 1.0
    finally:
        grpo.execute_prediction_program = original_executor


if __name__ == "__main__":
    test_invalid_program_has_hard_negative_reward()
    test_wrong_executable_program_is_capped()
    test_correct_executable_program_beats_wrong_and_invalid()
    test_hacking_diagnostics_are_recorded()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import finqa_program_grpo as grpo


def make_args():
    return grpo.ScriptArguments(
        schema_hard_gate=True,
        process_reward_enabled=True,
        wrong_executable_reward_cap=0.03,
        correct_format_penalty_floor=0.0,
        program_max_lines=1,
    )


def patch_executor():
    original = grpo.execute_prediction_program

    def fake_executor(program):
        normalized = grpo.first_text(program).strip().lower().replace(" ", "")
        values = {
            "93": 93.0,
            "sum(93)": 93.0,
            "add(86000,93000)": 179000.0,
        }
        if normalized.startswith("get("):
            return (None, None, "program_strict_validation_error:unsupported_operator")
        if "=" in normalized:
            return (None, None, "program_strict_validation_error:assignment_or_placeholder")
        return (values.get(normalized), normalized, "" if normalized in values else "unknown")

    grpo.execute_prediction_program = fake_executor
    return original


def evaluate(completion, gold_answer="93", gold_program="93"):
    return grpo.evaluate_program_completion(
        make_args(),
        completion,
        gold_answer,
        gold_program,
        prompt_text="Current question:\nwhat was the equipment rents payable in 2008?",
        metadata={"question_type": "lookup"},
    )


def test_direct_lookup_accepts_only_literal_program_for_execution_contract():
    original = patch_executor()
    try:
        literal = "Evidence:\n- Equipment rents payable in 2008 was 93.\n\nProgram: 93"
        wrapped = "Evidence:\n- Equipment rents payable in 2008 was 93.\n\nProgram: sum(93)"

        literal_result = evaluate(literal)
        wrapped_result = evaluate(wrapped)

        assert literal_result["exact_match"] == 1.0
        assert literal_result["direct_lookup_literal_ok"] == 1.0
        assert wrapped_result["invalid"] == 1.0
        assert wrapped_result["direct_lookup_wrapper_error"] == 1.0
        assert wrapped_result["core_score"] == grpo.HARD_INVALID_REWARD
    finally:
        grpo.execute_prediction_program = original


def test_unsupported_get_and_assignment_remain_invalid_even_with_correct_number():
    original = patch_executor()
    try:
        get_program = "Evidence:\n- Equipment rents payable in 2008 was 93.\n\nProgram: get(93)"
        assignment = "Evidence:\n- Equipment rents payable in 2008 was 93.\n\nProgram: answer = 93"

        get_result = evaluate(get_program)
        assignment_result = evaluate(assignment)

        assert get_result["invalid"] == 1.0
        assert get_result["invalid_operator_get"] == 1.0
        assert get_result["core_score"] == grpo.HARD_INVALID_REWARD
        assert assignment_result["invalid"] == 1.0
        assert assignment_result["assignment_or_placeholder"] == 1.0
        assert assignment_result["core_score"] == grpo.HARD_INVALID_REWARD
    finally:
        grpo.execute_prediction_program = original


def test_direct_lookup_rejects_extra_operation_even_if_it_is_executable():
    original = patch_executor()
    try:
        wrong = "Evidence:\n- The table shows 86000 and 93000.\n\nProgram: add(86000, 93000)"
        result = evaluate(wrong, gold_answer="93000", gold_program="93000")

        assert result["invalid"] == 1.0
        assert result["direct_lookup_wrapper_error"] == 1.0
        assert result["core_score"] == grpo.HARD_INVALID_REWARD
    finally:
        grpo.execute_prediction_program = original



def make_v46_args():
    return grpo.ScriptArguments(
        schema_hard_gate=True,
        process_reward_enabled=True,
        wrong_executable_reward_cap=0.03,
        correct_format_penalty_floor=0.0,
        program_max_lines=1,
        relaxed_executor_canonicalization=True,
        recoverable_syntax_soft_gate=True,
        direct_lookup_wrapper_soft_penalty=True,
        dense_reward_shaping=True,
    )


def evaluate_v46(completion, gold_answer="0.028997086232866898", gold_program="divide(121.4, 4187.8)"):
    return grpo.evaluate_program_completion(
        make_v46_args(),
        completion,
        gold_answer,
        gold_program,
        prompt_text="Current question:\nwhat portion of total purchase price is related to stock awards?",
        metadata={"question_type": "ratio", "operation_type": "divide"},
    )


def test_v46_recoverable_infix_gets_soft_reward_instead_of_hard_invalid():
    recoverable = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: 121.4 / 4187.8\n"
        "Explanation: this is the ratio."
    )
    invalid_text = "Evidence:\n- stock awards were 121.4\n\nProgram: divide the awards by purchase price"

    recoverable_result = evaluate_v46(recoverable)
    invalid_result = evaluate_v46(invalid_text)

    assert recoverable_result["recoverable_infix_rate"] == 1.0
    assert recoverable_result["invalid"] == 0.0
    assert recoverable_result["exact_match"] == 1.0
    assert recoverable_result["core_score"] > invalid_result["core_score"]
    assert recoverable_result["core_score"] < grpo.CORRECT_EXECUTABLE_BASE_REWARD


def test_v46_wrong_denominator_executable_stays_capped_below_correct_recoverable():
    correct_recoverable = "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\nProgram: 121.4 / 4187.8"
    wrong_denominator = "Evidence:\n- stock awards were 121.4\n- wrong denominator was 100.0\n\nProgram: divide(121.4, 100.0)"

    correct_result = evaluate_v46(correct_recoverable)
    wrong_result = evaluate_v46(wrong_denominator)

    assert wrong_result["executable"] == 1.0
    assert wrong_result["wrong_executable"] == 1.0
    assert wrong_result["core_score"] <= make_v46_args().wrong_executable_reward_cap
    assert correct_result["core_score"] > wrong_result["core_score"]


def test_v46_direct_lookup_wrapper_can_be_soft_penalized_when_answer_is_correct():
    result = grpo.evaluate_program_completion(
        make_v46_args(),
        "Evidence:\n- equipment rents payable in 2008 was 93.\n\nProgram: sum(93)",
        "93",
        "93",
        prompt_text="Current question:\nwhat was the equipment rents payable in 2008?",
        metadata={"question_type": "lookup"},
    )

    assert result["direct_lookup_wrapper_error"] == 1.0
    assert result["invalid"] == 0.0
    assert result["exact_match"] == 1.0
    assert result["core_score"] > grpo.HARD_INVALID_REWARD


def make_v47_args():
    return grpo.ScriptArguments(
        reward_mode="program_first_answer_aux",
        schema_hard_gate=True,
        process_reward_enabled=True,
        wrong_executable_reward_cap=0.03,
        correct_format_penalty_floor=0.0,
        program_max_lines=1,
        relaxed_executor_canonicalization=True,
        recoverable_syntax_soft_gate=True,
        dense_reward_shaping=True,
    )


def evaluate_v47(completion, gold_answer="0.028997086232866898", gold_program="divide(121.4, 4187.8)"):
    return grpo.evaluate_program_completion(
        make_v47_args(),
        completion,
        gold_answer,
        gold_program,
        prompt_text="Current question:\nwhat portion of total purchase price is related to stock awards?",
        metadata={"question_type": "ratio", "operation_type": "divide"},
    )


def test_v47_program_first_answer_aux_rewards_consistent_program_and_answer_highest():
    consistent = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: divide(121.4, 4187.8)\n\n"
        "Answer: 0.028997086232866898"
    )
    program_correct_answer_wrong = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: divide(121.4, 4187.8)\n\n"
        "Answer: 1.214"
    )
    program_wrong_answer_correct = (
        "Evidence:\n- stock awards were 121.4\n- wrong denominator was 100.0\n\n"
        "Program: divide(121.4, 100.0)\n\n"
        "Answer: 0.028997086232866898"
    )

    consistent_result = evaluate_v47(consistent)
    answer_wrong_result = evaluate_v47(program_correct_answer_wrong)
    program_wrong_result = evaluate_v47(program_wrong_answer_correct)

    assert consistent_result["invalid"] == 0.0
    assert consistent_result["exact_match"] == 1.0
    assert consistent_result["answer_first_exact_match"] == 1.0
    assert consistent_result["program_answer_consistency_reward"] == 1.0
    assert answer_wrong_result["exact_match"] == 1.0
    assert answer_wrong_result["answer_first_exact_match"] == 0.0
    assert answer_wrong_result["program_answer_consistency_reward"] == 0.0
    assert program_wrong_result["wrong_executable"] == 1.0
    assert program_wrong_result["answer_first_exact_match"] == 1.0
    assert program_wrong_result["core_score"] <= make_v47_args().wrong_executable_reward_cap
    assert consistent_result["core_score"] > answer_wrong_result["core_score"] > program_wrong_result["core_score"]


def test_v47_program_first_answer_aux_allows_answer_anchor_but_missing_program_is_invalid():
    with_answer = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: 121.4 / 4187.8\n\n"
        "Answer: 0.028997086232866898"
    )
    missing_program = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Answer: 0.028997086232866898"
    )

    with_answer_result = evaluate_v47(with_answer)
    missing_program_result = evaluate_v47(missing_program)

    assert with_answer_result["invalid"] == 0.0
    assert with_answer_result["recoverable_infix_rate"] == 1.0
    assert with_answer_result["answer_anchor_coverage"] == 1.0
    assert missing_program_result["invalid"] == 1.0
    assert missing_program_result["core_score"] == grpo.HARD_INVALID_REWARD



def make_v48_args():
    return grpo.ScriptArguments(
        reward_mode="program_gated_answer_aux",
        schema_hard_gate=True,
        process_reward_enabled=True,
        wrong_executable_reward_cap=0.02,
        correct_format_penalty_floor=0.92,
        program_max_lines=1,
        relaxed_executor_canonicalization=True,
        recoverable_syntax_soft_gate=True,
        dense_reward_shaping=True,
    )


def evaluate_v48(completion, gold_answer="0.028997086232866898", gold_program="divide(121.4, 4187.8)"):
    return grpo.evaluate_program_completion(
        make_v48_args(),
        completion,
        gold_answer,
        gold_program,
        prompt_text="Current question:\nwhat portion of total purchase price is related to stock awards?",
        metadata={"question_type": "ratio", "operation_type": "divide"},
    )


def test_v48_program_gated_answer_rewards_only_consistent_correct_program():
    consistent = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: divide(121.4, 4187.8)\n\n"
        "Answer: 0.028997086232866898"
    )
    program_correct_answer_wrong = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: divide(121.4, 4187.8)\n\n"
        "Answer: 1.214"
    )
    program_correct_missing_answer = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: divide(121.4, 4187.8)"
    )
    program_wrong_answer_correct = (
        "Evidence:\n- stock awards were 121.4\n- wrong denominator was 100.0\n\n"
        "Program: divide(121.4, 100.0)\n\n"
        "Answer: 0.028997086232866898"
    )

    consistent_result = evaluate_v48(consistent)
    answer_wrong_result = evaluate_v48(program_correct_answer_wrong)
    missing_answer_result = evaluate_v48(program_correct_missing_answer)
    program_wrong_result = evaluate_v48(program_wrong_answer_correct)

    assert consistent_result["invalid"] == 0.0
    assert consistent_result["exact_match"] == 1.0
    assert consistent_result["answer_first_exact_match"] == 1.0
    assert consistent_result["program_answer_consistency_reward"] == 1.0
    assert consistent_result["answer_correctness_reward"] == 1.0
    assert consistent_result["program_correct_answer_consistent_rate"] == 1.0
    assert consistent_result["answer_shortcut_rate"] == 0.0

    assert answer_wrong_result["exact_match"] == 1.0
    assert answer_wrong_result["answer_first_exact_match"] == 0.0
    assert answer_wrong_result["program_answer_consistency_reward"] == 0.0
    assert answer_wrong_result["program_correct_answer_inconsistent_rate"] == 1.0
    assert answer_wrong_result["core_score"] <= 0.25

    assert missing_answer_result["exact_match"] == 1.0
    assert missing_answer_result["answer_anchor_coverage"] == 0.0
    assert missing_answer_result["program_correct_answer_missing_rate"] == 1.0
    assert missing_answer_result["core_score"] > answer_wrong_result["core_score"]

    assert program_wrong_result["wrong_executable"] == 1.0
    assert program_wrong_result["answer_first_exact_match"] == 1.0
    assert program_wrong_result["answer_correctness_reward"] == 0.0
    assert program_wrong_result["answer_shortcut_rate"] == 1.0
    assert program_wrong_result["answer_correct_program_wrong_rate"] == 1.0
    assert program_wrong_result["core_score"] <= make_v48_args().wrong_executable_reward_cap
    assert consistent_result["core_score"] > missing_answer_result["core_score"] > answer_wrong_result["core_score"] > program_wrong_result["core_score"]


def test_v48_correct_answer_cannot_compensate_symbolic_or_get_program():
    symbolic = (
        "Evidence:\n- stock awards were 121.4\n- total purchase price was 4187.8\n\n"
        "Program: divide(value_of_metavante_stock_awards, total_purchase_price)\n\n"
        "Answer: 0.028997086232866898"
    )
    get_program = "Evidence:\n- equipment rents payable in 2008 was 93.\n\nProgram: get(93)\n\nAnswer: 93"

    symbolic_result = evaluate_v48(symbolic)
    get_result = grpo.evaluate_program_completion(
        make_v48_args(),
        get_program,
        "93",
        "93",
        prompt_text="Current question:\nwhat was the equipment rents payable in 2008?",
        metadata={"question_type": "lookup"},
    )

    assert symbolic_result["invalid"] == 1.0
    assert symbolic_result["symbolic_variable_rate"] == 1.0
    assert symbolic_result["answer_first_exact_match"] == 1.0
    assert symbolic_result["answer_shortcut_rate"] == 1.0
    assert symbolic_result["answer_correct_program_wrong_rate"] == 1.0
    assert symbolic_result["answer_correctness_reward"] == 0.0
    assert symbolic_result["core_score"] == grpo.HARD_INVALID_REWARD

    assert get_result["invalid"] == 1.0
    assert get_result["invalid_operator_get"] == 1.0
    assert get_result["answer_first_exact_match"] == 1.0
    assert get_result["answer_shortcut_rate"] == 1.0
    assert get_result["answer_correct_program_wrong_rate"] == 1.0
    assert get_result["answer_correctness_reward"] == 0.0
    assert get_result["core_score"] == grpo.HARD_INVALID_REWARD


def test_v48_direct_lookup_literal_with_consistent_answer_gets_high_reward():
    result = grpo.evaluate_program_completion(
        make_v48_args(),
        "Evidence:\n- equipment rents payable in 2008 was 93.\n\nProgram: 93\n\nAnswer: 93",
        "93",
        "93",
        prompt_text="Current question:\nwhat was the equipment rents payable in 2008?",
        metadata={"question_type": "lookup"},
    )

    assert result["invalid"] == 0.0
    assert result["direct_lookup_literal_ok"] == 1.0
    assert result["exact_match"] == 1.0
    assert result["answer_first_exact_match"] == 1.0
    assert result["program_answer_consistency_reward"] == 1.0
    assert result["program_correct_answer_consistent_rate"] == 1.0
    assert result["answer_shortcut_rate"] == 0.0
    assert result["core_score"] >= make_v48_args().correct_format_penalty_floor

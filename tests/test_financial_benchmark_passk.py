from evaluation.evaluate_financial_benchmarks import (
    aggregate_scores,
    build_example_summary_score,
    compute_pass_metrics,
    parse_number,
    parse_pass_k_values,
    score_example,
    BenchmarkExample,
)


def score(answer_correct, task_name="finqa_test"):
    return {
        "task_name": task_name,
        "record_id": "row",
        "gold_answer": "1",
        "prediction": "1" if answer_correct else "0",
        "answer_correct": float(answer_correct),
        "program_correct": None,
        "answer_coverage": 1.0,
        "normalized_answer_coverage": 1.0,
        "final_answer_coverage": 1.0,
        "program_section_coverage": 0.0,
        "structured_response_coverage": 1.0,
        "prediction_chars": 1,
        "numeric_parse_rate": 1.0,
        "program_parse_rate": 1.0,
        "program_execution_rate": 1.0,
        "executed_answer_accuracy": float(answer_correct),
        "model_normalized_answer_accuracy": float(answer_correct),
        "program_answer_consistency": 1.0,
        "program_string_accuracy": None,
    }


def test_parse_pass_k_values_sorts_and_dedupes():
    assert parse_pass_k_values("8,1,4,4") == [1, 4, 8]


def test_compute_pass_metrics_uses_available_sampled_candidates():
    metrics = compute_pass_metrics(score(0), [score(0), score(0), score(1)], [1, 4, 8])

    assert metrics["pass@1_greedy"] == 0.0
    assert metrics["pass@1_sampled"] == 0.0
    assert metrics["pass@4"] == 1.0
    assert metrics["pass@8"] == 1.0
    assert "pass@1" not in metrics


def test_aggregate_scores_keeps_greedy_accuracy_and_passk_fields():
    rows = aggregate_scores(
        "sft2_v2",
        [
            build_example_summary_score(score(1), [score(0), score(0)], [1, 4, 8]),
            build_example_summary_score(score(0), [score(0), score(1)], [1, 4, 8]),
            build_example_summary_score(score(0), [], [1, 4, 8]),
        ],
    )
    finqa_row = next(row for row in rows if row["task_name"] == "finqa_test")

    assert finqa_row["answer_accuracy"] == 0.333333
    assert finqa_row["primary_metric"] == 0.333333
    assert finqa_row["pass@1_greedy"] == 0.333333
    assert finqa_row["pass@1_sampled"] == 0.0
    assert finqa_row["pass@4"] == 0.5
    assert finqa_row["pass@8"] == 0.5
    assert "program_accuracy" in finqa_row


def test_parse_number_prefers_normalized_answer():
    assert parse_number("Answer: 10.745%\nNormalized Answer: 0.10745") == 0.10745


def test_score_example_uses_normalized_answer_and_program():
    example = BenchmarkExample(
        task_name="finqa_test",
        prompt="Question",
        gold_answer="0.10745",
        answer_type="numeric",
        record_id="row",
        metadata={},
        gold_program="divide(662, 6161)",
    )
    args = type("Args", (), {"numeric_abs_tol": 1e-4, "numeric_rel_tol": 1e-4})()
    score_row = score_example(
        example,
        "Evidence:\n- x\n\nProgram: divide(662, 6161)\nAnswer: 10.745%\nNormalized Answer: 0.10745",
        args,
    )

    assert score_row["answer_correct"] == 1.0
    assert score_row["executed_answer_accuracy"] == 1.0
    assert score_row["model_normalized_answer_accuracy"] == 1.0
    assert score_row["program_correct"] == 1.0
    assert score_row["answer_coverage"] == 1.0
    assert score_row["normalized_answer_coverage"] == 1.0


def test_score_example_uses_executed_program_when_model_answer_is_rounded_wrong():
    example = BenchmarkExample(
        task_name="finqa_test",
        prompt="Question",
        gold_answer="0.14464",
        answer_type="numeric",
        record_id="row",
        metadata={},
        gold_program="divide(8.1, 56.0)",
    )
    args = type("Args", (), {"numeric_abs_tol": 1e-4, "numeric_rel_tol": 1e-4})()
    score_row = score_example(
        example,
        "Evidence:\n- x\n\nProgram: divide(8.1, 56.0)\nAnswer: 14.4%\nNormalized Answer: 0.1443",
        args,
    )

    assert score_row["answer_correct"] == 1.0
    assert score_row["executed_answer_accuracy"] == 1.0
    assert score_row["model_normalized_answer_accuracy"] == 0.0
    assert score_row["program_answer_consistency"] == 0.0


def test_score_example_marks_unexecutable_program_wrong_even_with_model_answer():
    example = BenchmarkExample(
        task_name="finqa_test",
        prompt="Question",
        gold_answer="0.14464",
        answer_type="numeric",
        record_id="row",
        metadata={},
        gold_program="divide(8.1, 56.0)",
    )
    args = type("Args", (), {"numeric_abs_tol": 1e-4, "numeric_rel_tol": 1e-4})()
    score_row = score_example(
        example,
        "Evidence:\n- x\n\nProgram: use nearby numbers\nAnswer: 14.464%\nNormalized Answer: 0.14464",
        args,
    )

    assert score_row["answer_correct"] == 0.0
    assert score_row["executed_answer_accuracy"] == 0.0
    assert score_row["model_normalized_answer_accuracy"] == 1.0
    assert score_row["program_execution_rate"] == 0.0

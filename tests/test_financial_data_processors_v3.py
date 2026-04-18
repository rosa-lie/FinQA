from types import SimpleNamespace

from financial_data_processors.common import (
    build_reasoning_supervision,
    canonicalize_program_re,
    render_strict_target,
)
from financial_data_processors.families import finqa


def args():
    return SimpleNamespace(
        max_context_items=6,
        max_context_chars=400,
        max_supporting_facts=3,
        max_table_rows=20,
        max_table_cols=12,
        max_cell_chars=80,
        strict_tiers="A",
        sft_variant="dual_answer_sft",
    )


def test_canonicalize_program_re_replaces_const_tokens():
    assert canonicalize_program_re("divide(subtract(110.14, const_100), const_100)") == "divide(subtract(110.14, 100), 100)"
    assert canonicalize_program_re("divide(455, const_7)") == "divide(455, 7)"


def test_dual_answer_target_contains_display_and_normalized_answer():
    rec = {
        "pre_text": ["cash provided by operating activities increased from 6161 to 6823."],
        "table": [["year", "2013", "2012"], ["cash", "6823", "6161"]],
        "post_text": [],
    }
    norm = build_reasoning_supervision(
        rec,
        family="finqa",
        source_dataset="FinQA",
        task_type="financial_table_text_reasoning",
        record_id="row",
        question="what was the percentage change during this time?",
        program_re="divide(subtract(6823, 6161), 6161)",
        raw_answer="10.7%",
        exe_ans=0.10745,
        gold_evidence={"text_0": "cash provided by operating activities increased from 6161 to 6823."},
        args=args(),
    )
    target = render_strict_target(norm, "dual_answer_sft")

    assert "Answer: 10.7%" in target
    assert "Normalized Answer: 0.10745" in target
    assert norm["answer_unit"] == "percent"
    assert norm["answer_scale"] == "ratio"


def test_program_executor_target_excludes_model_answer_fields():
    rec = {
        "pre_text": ["leased facilities were 8.1 and total facilities were 56.0."],
        "table": [["metric", "value"], ["leased", "8.1"], ["total", "56.0"]],
        "post_text": [],
    }
    norm = build_reasoning_supervision(
        rec,
        family="finqa",
        source_dataset="FinQA",
        task_type="financial_table_text_reasoning",
        record_id="row",
        question="what percentage of total facilities are leased?",
        program_re="divide(8.1, 56.0)",
        raw_answer="14.5%",
        exe_ans=0.14464,
        gold_evidence={"text_0": "leased facilities were 8.1 and total facilities were 56.0."},
        args=args(),
    )
    target = render_strict_target(norm, "program_executor_sft")

    assert "Evidence:" in target
    assert "Program: divide(8.1, 56.0)" in target
    assert "Answer:" not in target
    assert "Normalized Answer:" not in target
    assert norm["answer_norm"] == "0.14464"
    assert norm["program_executable"] is not None


def test_dual_answer_prompt_puts_question_before_context():
    rec = {
        "pre_text": ["long report context"],
        "table": [["metric", "value"], ["revenue", "94"]],
        "post_text": [],
    }
    prompt = finqa.build_prompt(rec, "what is the net change?", args())

    assert prompt.index("Current question:") < prompt.index("Report context:")
    assert "Normalized Answer: ..." in prompt
    assert "For percentage questions, Normalized Answer must be a decimal ratio." in prompt


def test_program_executor_prompt_puts_program_instruction_before_context():
    rec = {
        "pre_text": ["long report context"],
        "table": [["metric", "value"], ["revenue", "94"]],
        "post_text": [],
    }
    program_args = args()
    program_args.sft_variant = "program_executor_sft"
    prompt = finqa.build_prompt(rec, "what is the net change?", program_args)

    assert prompt.index("Current question:") < prompt.index("Report context:")
    assert "Program: ..." in prompt
    assert "The final numeric answer will be computed by executing Program." in prompt
    assert "Normalized Answer: ..." not in prompt

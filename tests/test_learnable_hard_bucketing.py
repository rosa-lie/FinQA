from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from argparse import Namespace

from scripts.build_learnable_hard_buckets import (
    build_mix_rows,
    classify_bucket,
    filter_rows_by_source,
    should_sample_after_greedy,
    summarize_buckets,
)


def score(correct: bool = False, executable: bool = True, has_program: bool = True):
    return {
        "executed_answer_accuracy": 1.0 if correct else 0.0,
        "program_execution_rate": 1.0 if executable else 0.0,
        "program_parse_rate": 1.0 if has_program else 0.0,
    }


def test_greedy_correct_is_easy():
    bucket = classify_bucket(score(correct=True), [score(correct=False)], noisy=False)
    assert bucket == "easy"


def test_greedy_wrong_sample_correct_is_learnable_hard():
    bucket = classify_bucket(
        score(correct=False),
        [score(correct=False), score(correct=True), score(correct=False)],
        noisy=False,
    )
    assert bucket == "learnable-hard"


def test_all_wrong_executable_samples_are_hard():
    bucket = classify_bucket(
        score(correct=False),
        [score(correct=False), score(correct=False), score(correct=False)],
        noisy=False,
    )
    assert bucket == "hard"


def test_mostly_unexecutable_samples_are_invalid_prone():
    bucket = classify_bucket(
        score(correct=False),
        [
            score(correct=False, executable=False, has_program=False),
            score(correct=False, executable=False, has_program=True),
            score(correct=False, executable=True, has_program=True),
        ],
        noisy=False,
    )
    assert bucket == "invalid-prone"


def test_noisy_overrides_scores():
    bucket = classify_bucket(score(correct=True), [score(correct=True)], noisy=True)
    assert bucket == "noisy"


def test_mix_rows_uses_roughly_twenty_percent_easy_replay():
    learnable = [{"record_id": f"lh-{idx}"} for idx in range(8)]
    easy = [{"record_id": f"easy-{idx}"} for idx in range(10)]
    mixed = build_mix_rows(learnable, easy, easy_replay_ratio=0.2, seed=42)
    easy_count = sum(1 for row in mixed if row["record_id"].startswith("easy-"))
    learnable_count = sum(1 for row in mixed if row["record_id"].startswith("lh-"))
    assert learnable_count == 8
    assert easy_count == 2
    assert len(mixed) == 10


def test_source_dataset_filter_keeps_only_requested_sources():
    rows = [
        {"record_id": "1", "source_dataset": "finqa"},
        {"record_id": "2", "source_dataset": "convfinqa_turn"},
        {"record_id": "3", "source_dataset": "finqa"},
    ]
    filtered = filter_rows_by_source(rows, "finqa")
    assert [row["record_id"] for row in filtered] == ["1", "3"]


def test_source_dataset_filter_accepts_comma_separated_sources():
    rows = [
        {"record_id": "1", "source_dataset": "finqa"},
        {"record_id": "2", "source_dataset": "convfinqa_turn"},
        {"record_id": "3", "source_dataset": "other"},
    ]
    filtered = filter_rows_by_source(rows, "finqa,convfinqa_turn")
    assert [row["record_id"] for row in filtered] == ["1", "2"]


def test_filter_rows_by_manifest_phase_keeps_only_requested_records(tmp_path):
    import scripts.build_learnable_hard_buckets as buckets

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"record_id": "1", "source_dataset": "finqa", "manifest_phase": "pilot"}\n'
        '{"record_id": "2", "source_dataset": "finqa", "manifest_phase": "extension"}\n',
        encoding="utf-8",
    )
    rows = [
        {"record_id": "1", "source_dataset": "finqa"},
        {"record_id": "2", "source_dataset": "finqa"},
        {"record_id": "3", "source_dataset": "finqa"},
    ]

    filtered = buckets.filter_rows_by_manifest_phase(rows, str(manifest), "pilot")

    assert [row["record_id"] for row in filtered] == ["1"]


def test_skip_sampling_if_greedy_correct_saves_easy_samples():
    args = Namespace(num_samples_per_example=4, skip_sampling_if_greedy_correct=True)
    assert should_sample_after_greedy(args, score(correct=True), noisy=False) is False
    assert should_sample_after_greedy(args, score(correct=False), noisy=False) is True


def test_summary_reports_generation_counts():
    diagnostics = [
        {
            "record_id": "1",
            "source_dataset": "finqa",
            "bucket": "easy",
            "greedy_prediction": "Evidence:\n- a\n\nProgram: 1",
            "sample_count": 0,
        },
        {
            "record_id": "2",
            "source_dataset": "finqa",
            "bucket": "learnable-hard",
            "greedy_prediction": "Evidence:\n- a\n\nProgram: 2",
            "sample_count": 4,
        },
    ]
    summary = summarize_buckets(
        {
            "easy": [{"record_id": "1"}],
            "learnable-hard": [{"record_id": "2"}],
            "hard": [],
            "invalid-prone": [],
            "noisy": [],
        },
        diagnostics,
    )
    assert summary["generation_count_estimate"] == {"greedy": 2, "sampled": 4, "total": 6}


def test_process_rows_passes_adapter_path_to_loader(tmp_path, monkeypatch):
    import scripts.build_learnable_hard_buckets as buckets

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        '{"record_id": "r1", "source_dataset": "finqa", "input_prompt_raw": "prompt", '
        '"gold_answer": "1", "gold_program": "1", "reward_profile": "program_numeric"}\n',
        encoding="utf-8",
    )
    captured = {}

    def fake_load_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_args):
        captured["model_path"] = model_path
        captured["tokenizer_path"] = tokenizer_path
        captured["adapter_path"] = adapter_path
        return object(), object()

    monkeypatch.setattr(buckets, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(buckets, "unload_model", lambda model, tokenizer: None)
    monkeypatch.setattr(buckets, "row_is_noisy", lambda row, abs_tol, rel_tol: (True, "forced_noisy", ""))
    monkeypatch.setattr(buckets.gc, "collect", lambda: None)
    monkeypatch.setattr(buckets.torch.cuda, "is_available", lambda: False)

    args = Namespace(
        input_file=str(input_file),
        output_dir=str(tmp_path / "out"),
        model_path="base-model",
        adapter_path="adapter-ckpt20",
        tokenizer_path="tokenizer",
        max_samples=0,
        source_dataset_filter="",
        skip_sampling_if_greedy_correct=False,
        flush_every=0,
        resume_from_diagnostics=False,
        num_samples_per_example=8,
        sample_temperature=0.72,
        sample_top_p=0.95,
        sample_seed=42,
        batch_sample_generation=False,
        max_new_tokens=16,
        repetition_penalty=1.0,
        system_prompt="",
        numeric_abs_tol=1e-4,
        numeric_rel_tol=1e-4,
        numeric_output_format="program_executor",
        load_in_4bit=False,
        load_in_8bit=False,
        easy_replay_ratio=0.2,
        invalid_executable_threshold=0.5,
    )

    buckets.process_rows(args)

    assert captured["adapter_path"] == "adapter-ckpt20"

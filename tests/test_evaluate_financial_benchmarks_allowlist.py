import json
from pathlib import Path

from evaluation.evaluate_financial_benchmarks import (
    BenchmarkExample,
    filter_examples_by_record_id_allowlist,
    load_record_id_allowlist,
)


def example(record_id: str) -> BenchmarkExample:
    return BenchmarkExample(
        task_name="finqa_test",
        prompt=f"prompt {record_id}",
        gold_answer="1",
        answer_type="numeric",
        record_id=record_id,
        metadata={},
        gold_program="1",
    )


def test_record_id_allowlist_filters_examples_without_reordering(tmp_path):
    allowlist = tmp_path / "allowlist.jsonl"
    allowlist.write_text(
        "\n".join(
            [
                json.dumps({"record_id": "keep-b"}),
                json.dumps({"record_id": "keep-a"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    allowed = load_record_id_allowlist(str(allowlist))
    filtered = filter_examples_by_record_id_allowlist(
        [example("keep-a"), example("drop"), example("keep-b")],
        allowed,
    )

    assert [item.record_id for item in filtered] == ["keep-a", "keep-b"]


def test_empty_record_id_allowlist_path_keeps_all_examples():
    filtered = filter_examples_by_record_id_allowlist([example("a"), example("b")], None)

    assert [item.record_id for item in filtered] == ["a", "b"]

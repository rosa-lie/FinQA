from argparse import Namespace
import json
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v34r23_current_policy_manifest import build_manifest, normalize_for_acquisition


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_normalize_convfinqa_sharegpt_row_to_program_numeric():
    row = {
        "record_id": "c1",
        "source_dataset": "convfinqa_turn",
        "conversations": [
            {"from": "human", "value": "prompt"},
            {"from": "gpt", "value": "Evidence:\n- x\n\nProgram: add(1, 2)"},
        ],
        "metadata": {
            "answer_norm": "3",
            "program_canonical": "add(1, 2)",
            "answer_scale": "absolute",
            "requires_history": True,
            "history_dependency_type": "previous_turn_reuse",
        },
    }

    out = normalize_for_acquisition(row, "convfinqa")

    assert out["source_dataset"] == "convfinqa_turn"
    assert out["input_prompt_raw"] == "prompt"
    assert out["gold_answer"] == "3"
    assert out["gold_program"] == "add(1, 2)"
    assert out["reward_profile"] == "program_numeric"
    assert out["metadata"]["v34r23_requires_history"] is True
    assert out["metadata"]["v34r23_history_dependency_type"] == "previous_turn_reuse"


def test_build_manifest_excludes_reserved_and_marks_pilot(tmp_path):
    source = tmp_path / "source.jsonl"
    reserved = tmp_path / "reserved.jsonl"
    rows = [
        {"record_id": f"r{idx}", "source_dataset": "finqa", "input_prompt_raw": f"p{idx}", "gold_answer": "1", "gold_program": "1", "reward_profile": "program_numeric"}
        for idx in range(5)
    ]
    write_jsonl(source, rows)
    write_jsonl(reserved, [{"record_id": "r0"}])

    summary = build_manifest(
        Namespace(
            task="finqa",
            source_file=str(source),
            output_dir=str(tmp_path / "out"),
            sample_size=4,
            pilot_size=2,
            seed=7,
            exclude_allowlist=[str(reserved)],
        )
    )

    manifest_rows = [json.loads(line) for line in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()]
    input_rows = [json.loads(line) for line in (tmp_path / "out" / "acquisition_input.jsonl").read_text().splitlines()]

    assert summary["selected_unique_records"] == 4
    assert "r0" not in {row["record_id"] for row in manifest_rows}
    assert [row["manifest_phase"] for row in manifest_rows].count("pilot") == 2
    assert [row["manifest_phase"] for row in manifest_rows].count("extension") == 2
    assert all(row["reward_profile"] == "program_numeric" for row in input_rows)

import json
from pathlib import Path

import pytest

from scripts.run_v34r23_grpo40_checkpoint_sweep import (
    FIXED_GRPO40_CONFIG,
    SMOKE_ADAPTER,
    assert_grpo40_config,
    build_frontier_only_data,
    checkpoint_integrity,
    select_frontier_train_rows,
)


def row(record_id, bucket="frontier", history=False):
    return {
        "record_id": record_id,
        "source_dataset": "convfinqa_turn",
        "input_prompt_raw": f"Question {record_id}",
        "gold_answer": "1",
        "gold_program": "1",
        "reward_profile": "program_numeric",
        "metadata": {"v34r23_bucket": bucket, "v34r23_requires_history": history},
    }


def test_grpo40_config_locks_steps_sampling_and_checkpointing():
    good = dict(FIXED_GRPO40_CONFIG)
    assert_grpo40_config(good, "/tmp/rs_sft_checkpoint_20")

    with pytest.raises(ValueError, match="max_steps"):
        assert_grpo40_config(dict(good, max_steps=41), "/tmp/rs_sft_checkpoint_20")
    with pytest.raises(ValueError, match="save_steps"):
        assert_grpo40_config(dict(good, save_steps=20), "/tmp/rs_sft_checkpoint_20")
    with pytest.raises(ValueError, match="temperature"):
        assert_grpo40_config(dict(good, temperature=0.9), "/tmp/rs_sft_checkpoint_20")
    with pytest.raises(ValueError, match="controlled smoke adapter"):
        assert_grpo40_config(good, SMOKE_ADAPTER)


def test_select_frontier_train_rows_excludes_retention():
    rows = [row("r1", "retention_variance"), row("f1", "frontier", True), row("f2", "frontier", False)]
    selected = select_frontier_train_rows(rows)
    assert [item["record_id"] for item in selected] == ["f1", "f2"]
    assert all(item["metadata"]["v34r23_bucket"] == "frontier" for item in selected)


def test_build_frontier_only_data_writes_manifest_and_keeps_valid_out_of_train(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_rows = [row("f1", "frontier", True), row("ret1", "retention_variance"), row("f2", "frontier", False)]
    valid_rows = [row("vf1", "frontier"), row("vret", "retention_variance")]
    for name, rows in (("train.jsonl", train_rows), ("valid.jsonl", valid_rows)):
        with (data_dir / name).open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item) + "\n")

    result = build_frontier_only_data(data_dir, tmp_path / "out")
    written = [json.loads(line) for line in Path(result["train_file"]).read_text().splitlines()]
    valid = [json.loads(line) for line in Path(result["valid_file"]).read_text().splitlines()]
    manifest = result["manifest"]

    assert [item["record_id"] for item in written] == ["f1", "f2"]
    assert [item["record_id"] for item in valid] == ["vf1"]
    assert manifest["frontier_train_rows"] == 2
    assert manifest["frontier_valid_rows"] == 1
    assert manifest["history_rows"] == 1
    assert manifest["nonhistory_rows"] == 1


def test_checkpoint_integrity_detects_adapter_only_checkpoint(tmp_path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_bytes(b"adapter")

    info = checkpoint_integrity(tmp_path)["checkpoint-10"]
    assert info["exists"] is True
    assert info["adapter_config"] is True
    assert info["adapter_weights"] is True
    assert info["has_full_model_safetensors"] is False
    assert info["size_bytes"] > 0

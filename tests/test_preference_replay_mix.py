from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_preference_replay_mix import build_replay_mix, replay_count, split_pairs


def pair(idx: int, *, source_dataset: str = "finqa", record_prefix: str = "rec"):
    return {
        "system": "",
        "history": [],
        "question": f"question {source_dataset} {idx}",
        "response_chosen": f"Evidence:\n- chosen {idx}\n\nProgram: add({idx}, 1)",
        "response_rejected": f"Evidence:\n- rejected {idx}\n\nProgram: subtract({idx}, 1)",
        "source_dataset": source_dataset,
        "record_id": f"{record_prefix}-{idx}",
        "metadata": {"existing": True, "sample_correct_count": 2},
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Args:
    pass


def make_args(tmp_path: Path):
    main_root = tmp_path / "main"
    write_jsonl(main_root / "train_dir" / "train_preference_v7.jsonl", [pair(i) for i in range(7)])
    write_jsonl(main_root / "valid_dir" / "valid_preference_v7.jsonl", [pair(i) for i in range(7, 10)])
    finqa_replay = tmp_path / "finqa_replay.jsonl"
    conv_replay = tmp_path / "conv_replay.jsonl"
    write_jsonl(finqa_replay, [pair(0), *[pair(i, record_prefix="finqa-replay") for i in range(20)]])
    write_jsonl(conv_replay, [pair(i, source_dataset="convfinqa_turn", record_prefix="conv-replay") for i in range(20)])

    args = Args()
    args.main_data_root = str(main_root)
    args.finqa_replay_file = str(finqa_replay)
    args.convfinqa_replay_file = str(conv_replay)
    args.output_dir = str(tmp_path / "out")
    args.main_ratio = 0.70
    args.finqa_replay_ratio = 0.15
    args.convfinqa_replay_ratio = 0.15
    args.finqa_replay_count = 2
    args.convfinqa_replay_count = 2
    args.max_main_pairs = 0
    args.max_replay_pool_rows = 0
    args.min_sample_correct_count = 0
    args.require_chosen_program = True
    args.require_rejected_program = True
    args.drop_chosen_with_answer = True
    args.drop_rejected_with_answer = False
    args.valid_ratio = 0.2
    args.min_valid = 2
    args.max_valid = 3
    args.seed = 42
    args.exclude_main_record_ids = True
    return args


def test_replay_count_infers_ratio_against_main_count():
    assert replay_count(236, 0.70, 0.15, -1) == 51
    assert replay_count(236, 0.70, 0.15, 12) == 12


def test_split_pairs_is_deterministic_and_respects_bounds():
    rows = [pair(i) for i in range(20)]
    train_a, valid_a = split_pairs(rows, valid_ratio=0.1, min_valid=3, max_valid=5, seed=42)
    train_b, valid_b = split_pairs(rows, valid_ratio=0.1, min_valid=3, max_valid=5, seed=42)
    assert [row["record_id"] for row in valid_a] == [row["record_id"] for row in valid_b]
    assert len(train_a) == 17
    assert len(valid_a) == 3


def test_build_replay_mix_keeps_main_and_adds_replay_with_metadata(tmp_path: Path):
    args = make_args(tmp_path)
    summary = build_replay_mix(args)
    assert summary["total_pairs"] == 14
    assert summary["selected_counts"] == {"main": 10, "finqa_replay": 2, "convfinqa_replay": 2}
    assert summary["mix_source_counts"] == {
        "learnable_hard_main": 10,
        "finqa_replay": 2,
        "convfinqa_replay": 2,
    }
    train_file = Path(args.output_dir) / "train_dir" / "train_preference_v7_1.jsonl"
    valid_file = Path(args.output_dir) / "valid_dir" / "valid_preference_v7_1.jsonl"
    assert train_file.exists()
    assert valid_file.exists()
    rows = read_jsonl(train_file) + read_jsonl(valid_file)
    assert len(rows) == 14
    assert all((row.get("metadata") or {}).get("preference_mix_source") for row in rows)
    assert not any(row["record_id"] == "rec-0" and row["metadata"]["preference_mix_source"] == "finqa_replay" for row in rows)


def test_build_replay_mix_can_filter_low_confidence_main_pairs(tmp_path: Path):
    args = make_args(tmp_path)
    main_root = Path(args.main_data_root)
    rows = [pair(0), pair(1), pair(2)]
    rows[0]["metadata"]["sample_correct_count"] = 1
    rows[1]["response_chosen"] += "\nAnswer: 3"
    write_jsonl(main_root / "train_dir" / "train_preference_v7.jsonl", rows)
    write_jsonl(main_root / "valid_dir" / "valid_preference_v7.jsonl", [])
    args.finqa_replay_count = 0
    args.convfinqa_replay_count = 0
    args.min_sample_correct_count = 2

    summary = build_replay_mix(args)

    assert summary["requested_counts"]["main_raw"] == 3
    assert summary["selected_counts"]["main"] == 1
    output_rows = read_jsonl(Path(args.output_dir) / "train_dir" / "train_preference_v7_1.jsonl")
    assert [row["record_id"] for row in output_rows] == ["rec-2"]


if __name__ == "__main__":
    import tempfile

    test_replay_count_infers_ratio_against_main_count()
    test_split_pairs_is_deterministic_and_respects_bounds()
    with tempfile.TemporaryDirectory() as tmp:
        test_build_replay_mix_keeps_main_and_adds_replay_with_metadata(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_build_replay_mix_can_filter_low_confidence_main_pairs(Path(tmp))

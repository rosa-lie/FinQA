#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal runnable pipeline:
1) download FinGPT data
2) convert to MedicalGPT SFT / DPO format
3) run SFT
4) merge LoRA
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from datasets import load_dataset


def run(cmd: str) -> None:
    print(f"\n[RUN] {cmd}")
    subprocess.run(shlex.split(cmd), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--fin_dataset", default="FinGPT/fingpt-sentiment-train")
    p.add_argument("--fin_split", default="train")
    p.add_argument("--out_dir", default="data/fingpt_min")
    p.add_argument("--sft_out", default="outputs/fingpt_sft_lora")
    p.add_argument("--merged_out", default="outputs/fingpt_sft_merged")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    sft_dir = out_dir / "sft"
    dpo_dir = out_dir / "dpo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / "fingpt_raw.jsonl"
    print(f"[1/4] Downloading dataset: {args.fin_dataset} ({args.fin_split})")
    ds = load_dataset(args.fin_dataset, split=args.fin_split)
    with raw_file.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    print(f"Saved {len(ds)} rows -> {raw_file}")

    print("[2/4] Converting to MedicalGPT formats")
    run(
        f"python fin_to_sharegpt.py --source_file {raw_file} --output_file {sft_dir / 'fingpt_sft_sharegpt.jsonl'}"
    )
    run(
        f"python fin_to_dpo_pairs.py --source_file {raw_file} --output_file {dpo_dir / 'fingpt_dpo_pairs.jsonl'}"
    )

    print("[3/4] Training SFT (LoRA)")
    run(
        "python supervised_finetuning.py "
        f"--model_name_or_path {args.base_model} "
        f"--tokenizer_name_or_path {args.base_model} "
        f"--train_file_dir {sft_dir} "
        "--validation_split_percentage 1 "
        "--do_train "
        "--num_train_epochs 1 "
        "--per_device_train_batch_size 2 "
        "--gradient_accumulation_steps 2 "
        "--learning_rate 2e-4 "
        "--max_steps 50 "
        "--logging_steps 5 "
        "--save_steps 50 "
        "--model_max_length 512 "
        f"--output_dir {args.sft_out}"
    )

    print("[4/4] Merging LoRA")
    run(
        "python merge_peft_adapter.py "
        f"--base_model {args.base_model} "
        f"--tokenizer_path {args.base_model} "
        f"--lora_model {args.sft_out} "
        f"--output_dir {args.merged_out}"
    )

    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

"""Finance dataset cleaning and conversion pipeline for MedicalGPT formats.

Pipeline goals:
1) Read raw finance instruction data from HF/local files.
2) Clean with auditable filtering rules (quality/noise/safety/task alignment).
3) Build SFT/RM/RL datasets and an evaluation set split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


QUESTION_KEYS = ["instruction", "question", "query", "prompt", "input"]
ANSWER_KEYS = ["output", "answer", "response", "completion", "target"]
REFUSAL_PATTERNS = [r"作为AI", r"我不能", r"无法提供", r"抱歉", r"不知道"]
UNSUPPORTED_CLAIMS = [r"稳赚", r"保证收益", r"100%", r"无风险"]


@dataclass
class CleanConfig:
    min_q_len: int = 8
    min_a_len: int = 32
    max_q_len: int = 1200
    max_a_len: int = 4000
    min_quality: float = 0.58
    dedup_on: str = "qa"  # qa|q


def pick_first(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in d and d[k] is not None:
            v = str(d[k]).strip()
            if v:
                return v
    return ""


def load_raw(source: str, split: str = "train") -> Dataset:
    path = Path(source)
    if path.exists():
        ext = path.suffix.lower()
        if ext in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        if ext == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local extension: {ext}")
    return load_dataset(source, split=split)


def normalize_record(rec: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    q = pick_first(rec, QUESTION_KEYS)
    a = pick_first(rec, ANSWER_KEYS)
    ins = str(rec.get("instruction", "")).strip()
    inp = str(rec.get("input", "")).strip()

    if not q and ins:
        q = ins if not inp else f"{ins}\n\n补充信息：{inp}"

    meta = {
        "source": rec.get("source", "unknown"),
        "task_type": rec.get("task_type", "unknown"),
    }
    return q, a, meta


def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t


def quality_score(q: str, a: str) -> float:
    score = 1.0
    if len(q) < 20:
        score -= 0.15
    if len(a) < 80:
        score -= 0.25
    if len(a) > 2500:
        score -= 0.10

    refusal_hits = sum(bool(re.search(p, a, flags=re.I)) for p in REFUSAL_PATTERNS)
    score -= 0.08 * refusal_hits

    claim_hits = sum(bool(re.search(p, a, flags=re.I)) for p in UNSUPPORTED_CLAIMS)
    score -= 0.15 * claim_hits

    # repetition
    toks = a.split()
    if toks:
        unique_ratio = len(set(toks)) / len(toks)
        if unique_ratio < 0.25:
            score -= 0.15

    return max(0.0, min(1.0, score))


def infer_difficulty(q: str, a: str) -> str:
    if len(q) > 140 or len(a) > 800:
        return "hard"
    if len(q) > 60 or len(a) > 260:
        return "medium"
    return "easy"


def hash_key(q: str, a: str, mode: str = "qa") -> str:
    content = q if mode == "q" else f"{q}||{a}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def clean_dataset(ds: Dataset, cfg: CleanConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "raw": 0,
        "empty": 0,
        "length": 0,
        "dup": 0,
        "low_quality": 0,
        "kept": 0,
    }
    rows = []
    seen = set()

    for rec in ds:
        stats["raw"] += 1
        q, a, meta = normalize_record(rec)
        q, a = normalize_text(q), normalize_text(a)
        if not q or not a:
            stats["empty"] += 1
            continue
        if not (cfg.min_q_len <= len(q) <= cfg.max_q_len and cfg.min_a_len <= len(a) <= cfg.max_a_len):
            stats["length"] += 1
            continue

        key = hash_key(q, a, cfg.dedup_on)
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)

        qscore = quality_score(q, a)
        if qscore < cfg.min_quality:
            stats["low_quality"] += 1
            continue

        rows.append(
            {
                "question": q,
                "answer": a,
                "quality_score": qscore,
                "difficulty": infer_difficulty(q, a),
                "source": meta["source"],
                "task_type": meta["task_type"],
            }
        )

    stats["kept"] = len(rows)
    return pd.DataFrame(rows), stats


def augment_report_style(df: pd.DataFrame) -> pd.DataFrame:
    template = (
        "请以行业研究报告格式回答，必须包含："
        "\n1) 核心结论\n2) 关键驱动\n3) 估值视角\n4) 风险提示\n5) 跟踪指标"
    )
    out = []
    for _, r in df.iterrows():
        base = r.to_dict()
        base.update({"synthetic": False, "generator": "raw"})
        out.append(base)

        aug = r.to_dict()
        aug["question"] = f"{template}\n\n问题：{r['question']}"
        aug["synthetic"] = True
        aug["generator"] = "report_template_v2"
        aug["quality_score"] = min(1.0, float(r["quality_score"]) + 0.03)
        out.append(aug)

    return pd.DataFrame(out)


def build_preference(df: pd.DataFrame) -> pd.DataFrame:
    risk_hint = "风险提示：以上内容不构成投资建议，应结合最新公告与流动性变化。"
    rows = []
    for _, r in df.iterrows():
        chosen = r["answer"].strip()
        if risk_hint not in chosen:
            chosen = f"{chosen}\n\n{risk_hint}"

        rejected = re.sub(r"风险|不确定|波动|假设", "", r["answer"])
        rejected = rejected[: max(30, int(len(rejected) * 0.7))]
        if len(rejected) < 30:
            rejected = "结论看多，预计上涨，细节略。"

        rows.append(
            {
                "question": r["question"],
                "response_chosen": chosen,
                "response_rejected": rejected,
            }
        )
    return pd.DataFrame(rows)


def split3(df: pd.DataFrame, test_size: float = 0.02, val_size: float = 0.03):
    if len(df) < 100:
        return df.copy(), df.iloc[:0].copy(), df.iloc[:0].copy()
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train, test_size=val_size, random_state=42)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_medicalgpt(aug_df: pd.DataFrame, pref_df: pd.DataFrame, out_dir: Path):
    sft_train, sft_val, sft_test = split3(aug_df)
    rm_train, rm_val, rm_test = split3(pref_df)

    write_jsonl(
        (
            {"conversations": [{"from": "human", "value": r.question}, {"from": "gpt", "value": r.answer}]}
            for r in sft_train.itertuples()
        ),
        out_dir / "finance_sft_train.jsonl",
    )
    write_jsonl(
        (
            {"conversations": [{"from": "human", "value": r.question}, {"from": "gpt", "value": r.answer}]}
            for r in sft_val.itertuples()
        ),
        out_dir / "finance_sft_val.jsonl",
    )

    write_jsonl(
        (
            {
                "question": r.question,
                "response_chosen": r.response_chosen,
                "response_rejected": r.response_rejected,
            }
            for r in rm_train.itertuples()
        ),
        out_dir / "finance_rm_train.jsonl",
    )
    write_jsonl(
        (
            {
                "question": r.question,
                "response_chosen": r.response_chosen,
                "response_rejected": r.response_rejected,
            }
            for r in rm_val.itertuples()
        ),
        out_dir / "finance_rm_val.jsonl",
    )

    write_jsonl(
        (
            {"instruction": r.question, "input": "", "output": r.answer}
            for r in sft_train.itertuples()
        ),
        out_dir / "finance_rl_train.jsonl",
    )

    # eval set (for manual/automatic benchmark)
    eval_rows = (
        {
            "question": r.question,
            "gold_answer": r.answer,
            "difficulty": r.difficulty,
            "quality_score": float(r.quality_score),
        }
        for r in sft_test.itertuples()
    )
    write_jsonl(eval_rows, out_dir / "finance_eval.jsonl")

    return {
        "sft_train": len(sft_train),
        "sft_val": len(sft_val),
        "sft_test": len(sft_test),
        "rm_train": len(rm_train),
        "rm_val": len(rm_val),
        "rm_test": len(rm_test),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="BAAI/IndustryInstruction_Finance-Economics", help="HF dataset name or local file path")
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", default="data/finance")
    p.add_argument("--min_quality", type=float, default=0.58)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CleanConfig(min_quality=args.min_quality)

    ds = load_raw(args.source, args.split)
    clean_df, stats = clean_dataset(ds, cfg)

    aug_df = augment_report_style(clean_df)
    pref_df = build_preference(aug_df)
    counts = export_medicalgpt(aug_df, pref_df, Path(args.output_dir))

    report = {
        "config": asdict(cfg),
        "clean_stats": stats,
        "output_counts": counts,
        "difficulty_dist": clean_df["difficulty"].value_counts(normalize=True).round(4).to_dict() if len(clean_df) else {},
    }
    report_path = Path(args.output_dir) / "cleaning_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark base, SFT1, and SFT2 models on financial reasoning tasks."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from financial_data_processors.common import iter_records
from financial_data_processors.families import FAMILY_MODULES


OPTION_RE = re.compile(r"\b([A-F])\b", re.IGNORECASE)
STRICT_OPTION_PATTERNS = [
    re.compile(r"^[\s\(\[（【]*([A-F])[\s\)\]）】\.。,，:：-]*$", re.IGNORECASE),
    re.compile(r"(?:最终答案|答案|正确答案|选项|answer|option)\s*(?:是|为|[:：])\s*([A-F])\b", re.IGNORECASE),
]
FINAL_ANSWER_RE = re.compile(r"(?:最终答案|答案|answer)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
PROGRAM_RE = re.compile(r"(?:推理程序|program)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
NUMERIC_STRUCTURED_ANCHORS = ["问题分析：", "关键证据：", "推理程序：", "最终答案："]
MCQ_STRUCTURED_ANCHORS = ["题目理解：", "推理：", "最终答案："]


@dataclass
class BenchmarkExample:
    task_name: str
    prompt: str
    gold_answer: str
    answer_type: str
    record_id: str
    metadata: Dict[str, Any]
    gold_program: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate financial reasoning benchmarks.")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--model_entry",
        action="append",
        default=[],
        help="Model spec in name=path format. Repeat to compare multiple models.",
    )
    parser.add_argument(
        "--adapter_entry",
        action="append",
        default=[],
        help="Optional PEFT adapter spec in name=path format. The name must match --model_entry.",
    )
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--convfinqa_test_file", type=str, default="")
    parser.add_argument("--convfinqa_max_samples", type=int, default=0)
    parser.add_argument("--finqa_test_file", type=str, default="")
    parser.add_argument("--finqa_max_samples", type=int, default=0)
    parser.add_argument("--fineval_dataset_name", type=str, default="FinGPT/fingpt-fineval")
    parser.add_argument("--fineval_split", type=str, default="test")
    parser.add_argument("--fineval_local_file", type=str, default="")
    parser.add_argument(
        "--run_fineval",
        action="store_true",
        help="Enable Fineval benchmark loading. Disabled by default to avoid unintended Hub requests.",
    )
    parser.add_argument("--fineval_max_samples", type=int, default=0)
    parser.add_argument(
        "--cflue_task_file",
        action="append",
        default=[],
        help="CFLUE task spec in task_name=/path/to/file.{json,jsonl} format. Repeat for multiple tasks.",
    )
    parser.add_argument("--cflue_max_samples_per_task", type=int, default=0)
    parser.add_argument("--numeric_abs_tol", type=float, default=1e-4)
    parser.add_argument("--numeric_rel_tol", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def parse_name_path_entries(entries: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid name=path entry: {entry}")
        name, path = entry.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid name=path entry: {entry}")
        parsed[name] = path
    return parsed


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_text_blocks(value: Any, max_items: int, max_chars: int) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    output = []
    for item in items[:max_items]:
        text = to_text(item)
        if text:
            output.append(truncate_text(text, max_chars))
    return output


def format_table(table: Any, max_rows: int = 30, max_cols: int = 12, max_cell_chars: int = 100) -> str:
    if not isinstance(table, list) or not table:
        return ""
    lines = []
    for row in table[:max_rows]:
        if isinstance(row, list):
            cells = [truncate_text(to_text(cell), max_cell_chars) for cell in row[:max_cols]]
            lines.append(" | ".join(cells))
        else:
            lines.append(truncate_text(to_text(row), max_cell_chars))
    return "\n".join(lines)


def build_context_sections(record: Dict[str, Any], args: SimpleNamespace) -> List[str]:
    sections = []
    pre_text = normalize_text_blocks(record.get("pre_text"), args.max_context_items, args.max_context_chars)
    post_text = normalize_text_blocks(record.get("post_text"), args.max_context_items, args.max_context_chars)
    if pre_text:
        sections.append("材料（表格前文本）：\n" + "\n".join(f"- {item}" for item in pre_text))
    table_text = format_table(record.get("table") or record.get("table_ori"))
    if table_text:
        sections.append("表格：\n" + table_text)
    if post_text:
        sections.append("材料（表格后文本）：\n" + "\n".join(f"- {item}" for item in post_text))
    return sections


PROCESSOR_ARGS = SimpleNamespace(
    max_history_turns=6,
    max_context_items=6,
    max_context_chars=400,
    max_supporting_facts=6,
    max_table_rows=20,
    max_table_cols=12,
    max_cell_chars=80,
)


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        for key in ["data", "records", "examples", "items"]:
            if isinstance(data.get(key), list):
                return [dict(row) for row in data[key]]
    raise ValueError(f"Unsupported file format for {path}")


def limit_records(records: List[Any], max_samples: int) -> List[Any]:
    if max_samples and max_samples > 0:
        return records[:max_samples]
    return records


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return (text or "").strip()


def build_numeric_example_from_processor(record: Dict[str, Any], family: str, task_name: str) -> Optional[BenchmarkExample]:
    module = FAMILY_MODULES[family]
    item = module.build_sft_item(record, PROCESSOR_ARGS)
    if item is None:
        return None
    conversations = item.get("conversations") or []
    if len(conversations) < 2:
        return None
    prompt = to_text(conversations[0].get("value"))
    chosen = to_text(conversations[1].get("value"))
    gold_answer = extract_final_answer(chosen)
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("source_dataset", item.get("source_dataset", family))
    gold_program = to_text(metadata.get("program"))
    if not prompt or not gold_answer:
        return None
    return BenchmarkExample(
        task_name=task_name,
        prompt=prompt,
        gold_answer=gold_answer,
        answer_type="numeric",
        record_id=to_text(item.get("record_id") or record.get("id") or record.get("filename")),
        metadata=metadata,
        gold_program=gold_program,
    )


def load_numeric_examples(test_file: str, family: str, task_name: str, max_samples: int) -> List[BenchmarkExample]:
    if not test_file:
        return []
    records = [dict(row) for row in iter_records(load_json_records(Path(test_file)))]
    if family == "convfinqa_turn":
        records, dedupe_stats = FAMILY_MODULES[family].dedupe_final_turn(records)
        dropped = dedupe_stats.get("dedup_dropped_rows", 0)
        if dropped:
            print(f"[prep] {family}: dedup dropped {dropped} rows before evaluation")
    examples: List[BenchmarkExample] = []
    build_skipped = 0
    for record in records:
        example = build_numeric_example_from_processor(record, family, task_name)
        if example is None:
            build_skipped += 1
            continue
        examples.append(example)
    if build_skipped:
        print(f"[prep] {family}: skipped {build_skipped} rows during benchmark example build")
    return limit_records(examples, max_samples)


def load_finqa_examples(test_file: str, max_samples: int) -> List[BenchmarkExample]:
    return load_numeric_examples(test_file, family="finqa", task_name="finqa_test", max_samples=max_samples)

def combine_instruction_input(record: Dict[str, Any]) -> str:
    parts = [
        to_text(record.get("instruction")),
        to_text(record.get("input")),
        to_text(record.get("question") or record.get("query") or record.get("prompt")),
    ]
    return "\n\n".join([part for part in parts if part])


def render_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{key}. {to_text(value)}" for key, value in options.items())
    if isinstance(options, list):
        rendered = []
        for idx, option in enumerate(options):
            if isinstance(option, dict):
                key = to_text(option.get("label") or option.get("key") or option.get("id") or chr(ord("A") + idx))
                text = to_text(option.get("text") or option.get("value") or option.get("content") or option)
                rendered.append(f"{key}. {text}")
            else:
                rendered.append(f"{chr(ord('A') + idx)}. {to_text(option)}")
        return "\n".join(rendered)
    return to_text(options)


def extract_option(text: str) -> str:
    final_section = extract_section(text, FINAL_ANSWER_RE)
    candidates = [final_section] if final_section else []
    if text:
        candidates.append(text)
    for candidate in candidates:
        normalized = candidate.strip().upper()
        if not normalized:
            continue
        for pattern in STRICT_OPTION_PATTERNS:
            match = pattern.search(normalized)
            if match:
                return match.group(1).upper()
        unique_options = list(dict.fromkeys(OPTION_RE.findall(normalized)))
        if len(unique_options) == 1:
            return unique_options[0].upper()
    return ""


def infer_gold_option(record: Dict[str, Any]) -> str:
    candidates = [
        record.get("gold_option"),
        record.get("correct_option"),
        record.get("label"),
        record.get("answer"),
        record.get("output"),
        record.get("response"),
        record.get("gold_answer"),
    ]
    for candidate in candidates:
        text = to_text(candidate)
        option = extract_option(text)
        if option:
            return option
    return ""


def build_mcq_example(record: Dict[str, Any], task_name: str, source_name: str) -> BenchmarkExample:
    question = combine_instruction_input(record)
    option_text = render_options(record.get("options") or record.get("choices") or record.get("candidates"))
    if option_text:
        question = f"{question}\n\n选项：\n{option_text}"
    prompt = (
        "你是一名金融问答助手。请先做简短推理，再给出最终答案选项。\n\n"
        f"题目：{question}\n\n"
        "请按以下结构作答：\n题目理解：...\n推理：...\n最终答案：..."
    )
    gold_option = infer_gold_option(record)
    gold_answer = gold_option or to_text(
        record.get("answer") or record.get("output") or record.get("response") or record.get("gold_answer")
    )
    return BenchmarkExample(
        task_name=task_name,
        prompt=prompt,
        gold_answer=gold_answer,
        answer_type="mcq",
        record_id=to_text(record.get("id") or record.get("record_id") or record.get("question_id")),
        metadata={"source_dataset": source_name},
    )


def load_convfinqa_examples(test_file: str, max_samples: int) -> List[BenchmarkExample]:
    return load_numeric_examples(test_file, family="convfinqa_turn", task_name="convfinqa_test", max_samples=max_samples)

def load_fineval_examples(dataset_name: str, split: str, local_file: str, max_samples: int) -> List[BenchmarkExample]:
    if local_file:
        records = load_json_records(Path(local_file))
    else:
        records = [dict(row) for row in load_dataset(dataset_name, split=split)]
    records = limit_records(records, max_samples)
    return [build_mcq_example(record, "fineval_test", "FinGPT/fingpt-fineval") for record in records]


def load_cflue_examples(task_files: Dict[str, str], max_samples_per_task: int) -> List[BenchmarkExample]:
    examples: List[BenchmarkExample] = []
    for task_name, file_path in task_files.items():
        path = Path(file_path)
        if not path.exists():
            print(f"[skip] Missing CFLUE task file: {path}")
            continue
        records = limit_records(load_json_records(path), max_samples_per_task)
        examples.extend(build_mcq_example(record, f"cflue_{task_name}", f"CFLUE/{task_name}") for record in records)
    return examples


def safe_apply_chat_template(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[-1]["content"]


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    adapter_path: str,
    args: argparse.Namespace,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="left")
    load_dtype = torch.float16
    config_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.load_in_8bit:
        config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_dtype,
        )
    else:
        config_kwargs["torch_dtype"] = load_dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **config_kwargs)
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    except OSError:
        pass
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=load_dtype, device_map="auto")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.inference_mode()
def generate_response(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    prompt_text = safe_apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0.0,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def extract_section(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text or "")
    if not match:
        return ""
    return match.group(1).strip().split("\n")[0].strip()


def normalize_program(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    final_section = extract_section(text, FINAL_ANSWER_RE) or text
    matches = NUMBER_RE.findall(final_section.replace("，", ","))
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1]
    try:
        number = float(value)
    except ValueError:
        return None
    if is_percent:
        return number / 100.0
    return number


def normalize_answer_text(text: str) -> str:
    text = extract_section(text, FINAL_ANSWER_RE) or text
    return re.sub(r"\s+", "", text).strip().lower()


def score_example(example: BenchmarkExample, prediction: str, args: argparse.Namespace) -> Dict[str, Any]:
    final_answer = extract_section(prediction, FINAL_ANSWER_RE)
    program_section = extract_section(prediction, PROGRAM_RE)
    if example.answer_type == "numeric":
        structured_anchors = NUMERIC_STRUCTURED_ANCHORS
    elif example.answer_type == "mcq":
        structured_anchors = MCQ_STRUCTURED_ANCHORS
    else:
        structured_anchors = ["最终答案："]

    result = {
        "task_name": example.task_name,
        "record_id": example.record_id,
        "gold_answer": example.gold_answer,
        "prediction": prediction,
        "answer_correct": 0.0,
        "program_correct": None,
        "final_answer_coverage": float(bool(final_answer)),
        "program_section_coverage": float(bool(program_section)),
        "structured_response_coverage": float(all(anchor in (prediction or "") for anchor in structured_anchors)),
        "prediction_chars": len((prediction or "").strip()),
        "numeric_parse_rate": None,
    }
    if example.answer_type == "numeric":
        pred_num = parse_number(prediction)
        gold_num = parse_number(example.gold_answer)
        result["numeric_parse_rate"] = float(pred_num is not None)
        if pred_num is not None and gold_num is not None:
            result["answer_correct"] = float(
                math.isclose(pred_num, gold_num, abs_tol=args.numeric_abs_tol, rel_tol=args.numeric_rel_tol)
            )
        else:
            result["answer_correct"] = float(normalize_answer_text(prediction) == normalize_answer_text(example.gold_answer))
        if example.gold_program:
            result["program_correct"] = float(normalize_program(program_section) == normalize_program(example.gold_program))
    elif example.answer_type == "mcq":
        pred_option = extract_option(prediction)
        gold_option = extract_option(example.gold_answer)
        if pred_option and gold_option:
            result["answer_correct"] = float(pred_option == gold_option)
        else:
            result["answer_correct"] = float(normalize_answer_text(prediction) == normalize_answer_text(example.gold_answer))
    else:
        result["answer_correct"] = float(normalize_answer_text(prediction) == normalize_answer_text(example.gold_answer))
    return result


def benchmark_bucket(task_name: str) -> str:
    if task_name.startswith("convfinqa"):
        return "convfinqa"
    if task_name.startswith("finqa"):
        return "finqa"
    if task_name.startswith("fineval"):
        return "fineval"
    if task_name.startswith("cflue_"):
        return "cflue"
    return "other"


def _mean(values: List[float]) -> Any:
    if not values:
        return ""
    return round(sum(values) / len(values), 6)


def _build_summary_row(model_name: str, task_name: str, scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    answer_scores = [item["answer_correct"] for item in scores]
    program_scores = [item["program_correct"] for item in scores if item["program_correct"] is not None]
    final_answer_scores = [item["final_answer_coverage"] for item in scores]
    program_section_scores = [item["program_section_coverage"] for item in scores]
    structured_scores = [item["structured_response_coverage"] for item in scores]
    prediction_chars = [item["prediction_chars"] for item in scores]
    numeric_parse_scores = [item["numeric_parse_rate"] for item in scores if item["numeric_parse_rate"] is not None]
    answer_acc = sum(answer_scores) / len(answer_scores)
    return {
        "model_name": model_name,
        "task_name": task_name,
        "num_examples": len(scores),
        "answer_accuracy": round(answer_acc, 6),
        "primary_metric": round(answer_acc, 6),
        "program_accuracy": _mean(program_scores),
        "final_answer_coverage": _mean(final_answer_scores),
        "program_section_coverage": _mean(program_section_scores),
        "structured_response_coverage": _mean(structured_scores),
        "numeric_parse_rate": _mean(numeric_parse_scores),
        "avg_prediction_chars": _mean(prediction_chars),
    }


def aggregate_scores(model_name: str, scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for score in scores:
        grouped.setdefault(score["task_name"], []).append(score)

    rows: List[Dict[str, Any]] = []
    bucket_scores: Dict[str, List[Dict[str, Any]]] = {}
    all_task_rows: List[Dict[str, Any]] = []
    for task_name, task_scores in sorted(grouped.items()):
        row = _build_summary_row(model_name, task_name, task_scores)
        rows.append(row)
        all_task_rows.append(row)
        bucket_scores.setdefault(benchmark_bucket(task_name), []).extend(task_scores)

    for bucket_name, grouped_scores in sorted(bucket_scores.items()):
        rows.append(_build_summary_row(model_name, f"bucket_{bucket_name}", grouped_scores))

    if scores:
        rows.append(_build_summary_row(model_name, "macro_average", scores))
    return rows


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unload_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = parse_name_path_entries(args.model_entry)
    if not model_paths:
        raise ValueError("Please provide at least one --model_entry name=path")
    adapter_paths = parse_name_path_entries(args.adapter_entry)
    tokenizer_path = args.tokenizer_path or next(iter(model_paths.values()))

    benchmark_examples: List[BenchmarkExample] = []
    benchmark_examples.extend(load_convfinqa_examples(args.convfinqa_test_file, args.convfinqa_max_samples))
    benchmark_examples.extend(load_finqa_examples(args.finqa_test_file, args.finqa_max_samples))
    run_fineval = bool(args.run_fineval or args.fineval_local_file)
    if run_fineval:
        benchmark_examples.extend(
            load_fineval_examples(
                args.fineval_dataset_name,
                args.fineval_split,
                args.fineval_local_file,
                args.fineval_max_samples,
            )
        )
    else:
        print("[skip] Fineval disabled (use --run_fineval or set --fineval_local_file to enable).")
    cflue_task_files = parse_name_path_entries(args.cflue_task_file)
    benchmark_examples.extend(load_cflue_examples(cflue_task_files, args.cflue_max_samples_per_task))

    if not benchmark_examples:
        raise ValueError("No benchmark examples were loaded. Please check your dataset paths and splits.")

    manifest = {
        "models": model_paths,
        "adapters": adapter_paths,
        "tokenizer_path": tokenizer_path,
        "convfinqa_test_file": args.convfinqa_test_file,
        "finqa_test_file": args.finqa_test_file,
        "fineval_dataset_name": args.fineval_dataset_name,
        "fineval_split": args.fineval_split,
        "run_fineval": run_fineval,
        "fineval_local_file": args.fineval_local_file,
        "cflue_task_files": cflue_task_files,
        "num_examples": len(benchmark_examples),
    }
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_summary_rows: List[Dict[str, Any]] = []
    for model_name, model_path in model_paths.items():
        adapter_path = adapter_paths.get(model_name, "")
        print(f"[eval] model={model_name} path={model_path} adapter={adapter_path or 'none'}")
        model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, adapter_path, args)
        raw_predictions: List[Dict[str, Any]] = []
        scores: List[Dict[str, Any]] = []
        for example in tqdm(benchmark_examples, desc=f"Evaluating {model_name}"):
            prediction = generate_response(model, tokenizer, example.prompt, args)
            score = score_example(example, prediction, args)
            scores.append(score)
            raw_predictions.append(
                {
                    "model_name": model_name,
                    "task_name": example.task_name,
                    "record_id": example.record_id,
                    "prompt": example.prompt,
                    "gold_answer": example.gold_answer,
                    "gold_program": example.gold_program,
                    "prediction": prediction,
                    "answer_correct": score["answer_correct"],
                    "program_correct": score["program_correct"],
                    "metadata": example.metadata,
                }
            )

        save_jsonl(output_dir / f"{model_name}_predictions.jsonl", raw_predictions)
        summary_rows = aggregate_scores(model_name, scores)
        save_jsonl(output_dir / f"{model_name}_summary.jsonl", summary_rows)
        all_summary_rows.extend(summary_rows)
        unload_model(model, tokenizer)

    save_csv(output_dir / "benchmark_summary.csv", all_summary_rows)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(all_summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] Saved benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()

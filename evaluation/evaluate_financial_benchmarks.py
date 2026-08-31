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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from financial_data_processors.common import (
    canonicalize_program_re,
    execute_program,
    format_numeric_answer,
    iter_records,
)
from financial_data_processors.families import FAMILY_MODULES


OPTION_RE = re.compile(r"\b([A-F])\b", re.IGNORECASE)
STRICT_OPTION_PATTERNS = [
    re.compile(r"^[\s\(\[（【]*([A-F])[\s\)\]）】\.。,，:：-]*$", re.IGNORECASE),
    re.compile(r"(?:最终答案|答案|正确答案|选项|answer|option)\s*(?:是|为|[:：])\s*([A-F])\b", re.IGNORECASE),
]
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"(?:最终答案|答案|answer)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?m)^Answer\s*[:：]\s*(.+)$", re.IGNORECASE)
NORMALIZED_ANSWER_RE = re.compile(r"(?m)^Normalized Answer\s*[:：]\s*(.+)$", re.IGNORECASE)
PROGRAM_RE = re.compile(r"(?:推理程序|program)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
REASONING_RE = re.compile(r"(?m)^Reasoning\s*[:：]\s*(.+)$", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
PROGRAM_STOP_RE = re.compile(
    r"(?im)^\s*(?:Answer|Normalized Answer|Final Answer|Explanation|The final numeric answer)\s*[:：]?.*$"
)
PROGRAM_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)$", re.DOTALL)
NUMERIC_OUTPUT_FORMATS = [
    "program_executor",
    "cot_program",
    "reasoning_program_executor",
    "answer_first_reasoning_program",
    "program_first_answer",
]
NUMERIC_STRUCTURED_ANCHORS = {
    "program_executor": ["Evidence:", "Program:"],
    "cot_program": ["Reasoning:", "Evidence:", "Program:"],
    "reasoning_program_executor": ["Evidence:", "Program:"],
    "answer_first_reasoning_program": ["Reasoning:", "Evidence:", "Program:", "Answer:"],
    "program_first_answer": ["Evidence:", "Program:", "Answer:"],
}
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
    parser.add_argument(
        "--pass_k",
        type=str,
        default="1,4,8",
        help="Comma-separated sampled pass@k values to report, e.g. 1,4,8.",
    )
    parser.add_argument(
        "--num_samples_per_example",
        type=int,
        default=8,
        help="Number of sampled candidates per example for pass@k.",
    )
    parser.add_argument("--sample_temperature", type=float, default=0.7)
    parser.add_argument("--sample_top_p", type=float, default=0.95)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--only_model",
        action="append",
        default=[],
        help="Run only the named model(s). Repeat for multiple names. Existing model entries are filtered.",
    )
    parser.add_argument(
        "--skip_existing_predictions",
        action="store_true",
        help="If a model predictions JSONL already exists in output_dir, reuse it and recompute summaries.",
    )
    parser.add_argument(
        "--score_only_predictions",
        action="store_true",
        help="Do not load models. Re-score existing *_predictions.jsonl files in output_dir and rebuild summaries.",
    )
    parser.add_argument(
        "--allow_scoring_config_mismatch",
        action="store_true",
        help="Allow re-scoring existing predictions even when their stored scoring_config differs from current CLI args.",
    )
    parser.add_argument(
        "--stratified_max_per_bucket",
        type=int,
        default=0,
        help="Limit numeric examples to at most N per source/question/history bucket. 0 disables stratified selection.",
    )
    parser.add_argument(
        "--relaxed_executor_canonicalization",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y", "on"},
        default=False,
        help="Allow recoverable numeric infix and direct numeric programs to execute for v46-style evaluation.",
    )
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--convfinqa_test_file", type=str, default="")
    parser.add_argument("--convfinqa_max_samples", type=int, default=0)
    parser.add_argument("--finqa_test_file", type=str, default="")
    parser.add_argument("--finqa_max_samples", type=int, default=0)
    parser.add_argument(
        "--record_id_allowlist_jsonl",
        type=str,
        default="",
        help="Optional JSONL/TXT file containing record_id values to keep after benchmark examples are built.",
    )
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
    parser.add_argument(
        "--processor_sft_variant",
        type=str,
        default="dual_answer_sft",
        choices=["dual_answer_sft", "program_executor_sft"],
        help="Prompt/target variant used by the data processors when building benchmark examples.",
    )
    parser.add_argument(
        "--numeric_output_format",
        type=str,
        default="program_executor",
        choices=NUMERIC_OUTPUT_FORMATS,
        help=(
            "Eval-only output format for numeric benchmarks. The default preserves the existing "
            "Evidence/Program executor behavior."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def parse_pass_k_values(value: str) -> List[int]:
    values: List[int] = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            k = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid pass@k value: {item}") from exc
        if k <= 0:
            raise ValueError(f"pass@k values must be positive integers: {item}")
        if k not in values:
            values.append(k)
    if not values:
        raise ValueError("Please provide at least one pass@k value.")
    return sorted(values)


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


def filter_model_entries(
    model_paths: Dict[str, str],
    adapter_paths: Dict[str, str],
    only_models: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    requested = [name.strip() for name in only_models if name.strip()]
    if not requested:
        return model_paths, adapter_paths
    missing = [name for name in requested if name not in model_paths]
    if missing:
        raise ValueError(f"Requested --only_model entries are not defined by --model_entry: {missing}")
    filtered_models = {name: model_paths[name] for name in requested}
    filtered_adapters = {name: adapter_paths[name] for name in requested if name in adapter_paths}
    return filtered_models, filtered_adapters



def load_record_id_allowlist(path: str) -> Optional[set[str]]:
    if not path:
        return None
    allowlist_path = Path(path)
    allowed: set[str] = set()
    with allowlist_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            record_id = ""
            if text.startswith("{"):
                row = json.loads(text)
                value = row.get("record_id") or row.get("id")
                record_id = to_text(value)
            else:
                record_id = text
            if record_id:
                allowed.add(record_id)
    if not allowed:
        raise ValueError(f"No record_id values loaded from allowlist: {path}")
    return allowed


def filter_examples_by_record_id_allowlist(
    examples: Sequence[BenchmarkExample],
    allowed_record_ids: Optional[set[str]],
) -> List[BenchmarkExample]:
    if allowed_record_ids is None:
        return list(examples)
    return [example for example in examples if example.record_id in allowed_record_ids]

def scoring_config_from_args(args: argparse.Namespace, pass_k_values: Sequence[int]) -> Dict[str, Any]:
    return {
        "numeric_output_format": get_numeric_output_format(args),
        "processor_sft_variant": to_text(getattr(args, "processor_sft_variant", "")),
        "numeric_abs_tol": float(getattr(args, "numeric_abs_tol", 1e-4)),
        "numeric_rel_tol": float(getattr(args, "numeric_rel_tol", 1e-4)),
        "pass_k": [int(value) for value in pass_k_values],
        "num_samples_per_example": int(getattr(args, "num_samples_per_example", 0) or 0),
    }


def validate_prediction_scoring_config(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    pass_k_values: Sequence[int],
) -> None:
    if getattr(args, "allow_scoring_config_mismatch", False):
        return
    expected = scoring_config_from_args(args, pass_k_values)
    for row in rows:
        stored = row.get("scoring_config")
        if stored is None:
            continue
        if stored != expected:
            raise ValueError(
                "scoring_config mismatch for existing predictions; "
                f"expected={expected}, found={stored}. "
                "Use --allow_scoring_config_mismatch to intentionally re-score with different settings."
            )


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
    sft_variant="dual_answer_sft",
    convfinqa_mode="turn_level",
    convfinqa_keep_final_only="false",
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




def extract_answer_body(text: str) -> str:
    tag_match = ANSWER_TAG_RE.search(text or "")
    if tag_match:
        return tag_match.group(1).strip()
    return text or ""


def extract_final_answer(text: str) -> str:
    text = extract_answer_body(text or "")
    normalized = extract_normalized_answer(text)
    if normalized:
        return normalized
    answer = extract_display_answer(text)
    if answer:
        return answer
    match = FINAL_ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return (text or "").strip()


def extract_display_answer(text: str) -> str:
    match = ANSWER_LINE_RE.search(extract_answer_body(text or ""))
    if not match:
        return ""
    return match.group(1).strip().split("\n")[0].strip()


def extract_normalized_answer(text: str) -> str:
    match = NORMALIZED_ANSWER_RE.search(extract_answer_body(text or ""))
    if not match:
        return ""
    return match.group(1).strip().split("\n")[0].strip()


def get_numeric_output_format(args: Any) -> str:
    value = getattr(args, "numeric_output_format", "program_executor")
    if value not in NUMERIC_OUTPUT_FORMATS:
        return "program_executor"
    return value


def numeric_eval_instruction(output_format: str) -> str:
    if output_format == "program_first_answer":
        return (
            "Output format:\n"
            "Reasoning: ... (optional, keep it brief)\n\n"
            "Evidence:\n"
            "- ...\n\n"
            "Program: ...\n\n"
            "Answer: ...\n\n"
            "Program rule:\n"
            "- Program is primary and must be one executable numeric DSL expression when possible.\n"
            "- Use numeric literals or DSL expressions: add, subtract, multiply, divide, max, min, sum, average.\n\n"
            "Answer rule:\n"
            "- Put the final normalized numeric answer after Answer:.\n"
            "- The final numeric answer must match executing Program.\n"
            "- Use raw ratios such as 0.02899 unless the question explicitly asks for a percent."
        )
    if output_format == "answer_first_reasoning_program":
        return (
            "Output format:\n"
            "Reasoning: ... (brief financial reasoning is allowed)\n\n"
            "Evidence:\n"
            "- ...\n\n"
            "Program: ...\n\n"
            "Answer: ...\n\n"
            "Answer rule:\n"
            "- Put the final normalized numeric answer after Answer:.\n"
            "- Use raw ratios such as 0.02899 unless the question explicitly asks for a percent.\n\n"
            "Program rule:\n"
            "- Program is auxiliary but should be an executable numeric DSL expression when possible, using add, subtract, multiply, divide, max, min, sum, average."
        )
    if output_format in {"cot_program", "reasoning_program_executor"}:
        return (
            "Output format:\n"
            "Reasoning: ... (optional, keep it brief)\n\n"
            "Evidence:\n"
            "- ...\n\n"
            "Program: ...\n\n"
            "The final numeric answer will be computed by executing Program.\n"
            "Do not calculate or round the final answer yourself.\n\n"
            "Program rule:\n"
            "- Use only executable numeric DSL expressions such as add, subtract, multiply, divide, max, min, sum, average."
        )
    return ""


def apply_numeric_eval_output_format(prompt: str, output_format: str) -> str:
    if output_format == "program_executor":
        return prompt
    instruction = numeric_eval_instruction(output_format)
    if not instruction:
        return prompt
    return re.sub(
        r"Output format:\n.*?(?=\n\nReport context:|\n\nConversation history:|\n\nConversation history questions:|\Z)",
        instruction,
        prompt,
        count=1,
        flags=re.DOTALL,
    )


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
    prompt = apply_numeric_eval_output_format(prompt, get_numeric_output_format(PROCESSOR_ARGS))
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("source_dataset", item.get("source_dataset", family))
    gold_answer = to_text(metadata.get("answer_norm")) or extract_final_answer(chosen)
    gold_program = to_text(metadata.get("program_canonical") or metadata.get("program"))
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
        records, multiturn_stats = FAMILY_MODULES[family].prepare_multiturn_records(records, PROCESSOR_ARGS)
        print(
            f"[prep] {family}: prepared {multiturn_stats.get('conversation_count', 0)} conversations "
            f"with {multiturn_stats.get('history_full_reasoning_turns', 0)} full-history turns"
        )
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


def infer_eval_question_type(example: BenchmarkExample) -> str:
    metadata = example.metadata or {}
    question_type = to_text(metadata.get("question_type"))
    if question_type:
        return question_type
    prompt = to_text(example.prompt).lower()
    ops = set(program_ops_for_metric(example.gold_program))
    if not ops:
        return "direct_lookup"
    if any(token in prompt for token in ("percent", "percentage", "ratio", "margin", "rate", "portion", "share of")):
        return "ratio_or_percent"
    if any(token in prompt for token in ("difference", "change", "increase", "decrease", "more", "less")) or "subtract" in ops:
        return "change_or_difference"
    if ops.intersection({"sum", "average", "max", "min", "table_sum", "table_average", "table_max", "table_min"}):
        return "aggregate"
    return "calculation"


def stratified_bucket_key(example: BenchmarkExample) -> Tuple[str, str, str]:
    metadata = example.metadata or {}
    source = benchmark_bucket(example.task_name)
    question_type = infer_eval_question_type(example)
    requires_history = "history" if bool(metadata.get("requires_history")) else "no_history"
    return source, question_type, requires_history


def select_stratified_examples(examples: Sequence[BenchmarkExample], max_per_bucket: int) -> List[BenchmarkExample]:
    if max_per_bucket <= 0:
        return list(examples)
    bucket_counts: Dict[Tuple[str, str, str], int] = {}
    selected: List[BenchmarkExample] = []
    for example in examples:
        key = stratified_bucket_key(example)
        count = bucket_counts.get(key, 0)
        if count >= max_per_bucket:
            continue
        selected.append(example)
        bucket_counts[key] = count + 1
    return selected


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
def generate_response(
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    do_sample: Optional[bool] = None,
    seed: Optional[int] = None,
) -> str:
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    prompt_text = safe_apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    generation_temperature = args.temperature if temperature is None else temperature
    generation_do_sample = generation_temperature > 0.0 if do_sample is None else do_sample
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=generation_do_sample,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if generation_do_sample:
        generation_config.temperature = generation_temperature
        if top_p is not None:
            generation_config.top_p = top_p
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )
    generated_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def extract_section(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text or "")
    if not match:
        return ""
    return match.group(1).strip().split("\n")[0].strip()


def clean_prediction_program_text(text: str) -> str:
    """Keep evaluation strict on DSL semantics while tolerating common wrappers."""
    program = (text or "").strip()
    if not program:
        return ""

    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)```", program, flags=re.DOTALL)
    if fenced:
        program = fenced.group(1).strip()

    kept_lines = []
    for raw_line in program.splitlines():
        line = raw_line.strip()
        if not line:
            if kept_lines:
                break
            continue
        if PROGRAM_STOP_RE.match(line):
            break
        if kept_lines and line.lower().startswith(("this program", "executing the program")):
            break
        if line.startswith("```"):
            continue
        kept_lines.append(line)
    program = "\n".join(kept_lines).strip()
    if not program:
        return ""

    if program.lower().startswith("program:"):
        program = program.split(":", 1)[1].strip()

    assignment = PROGRAM_ASSIGNMENT_RE.match(program)
    if assignment:
        program = assignment.group(1).strip()

    return program.strip()


def extract_program_section(text: str) -> str:
    match = PROGRAM_RE.search(text or "")
    if not match:
        return ""
    return clean_prediction_program_text(match.group(1))


def extract_evidence_section(text: str) -> str:
    text = text or ""
    if "Evidence:" not in text:
        return ""
    start = text.find("Evidence:") + len("Evidence:")
    end = text.find("Program:", start)
    return text[start:end if end >= 0 else len(text)].strip()


def evidence_bullet_count(evidence: str) -> int:
    lines = [line.strip() for line in (evidence or "").splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("-") or re.match(r"^\d+[\.)]\s+", line)]
    return len(bullet_lines) if bullet_lines else (1 if (evidence or "").strip() else 0)


def has_post_program_text(text: str) -> bool:
    text = text or ""
    if "Program:" not in text:
        return False
    tail = text[text.rfind("Program:") + len("Program:"):].strip()
    if not tail:
        return False
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return len(lines) > 1


def strict_program_contract_rate(prediction: str, evidence: str, program_section: str) -> float:
    if evidence_bullet_count(evidence) > 2:
        return 0.0
    if len((evidence or "").strip()) > 180:
        return 0.0
    if len([line for line in (program_section or "").splitlines() if line.strip()]) != 1:
        return 0.0
    if has_post_program_text(prediction):
        return 0.0
    return 1.0


def normalize_program(text: str) -> str:
    return re.sub(r"\s+", "", clean_prediction_program_text(text).lower())


def program_ops_for_metric(program: str) -> List[str]:
    return [op.lower() for op in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", to_text(program))]


def numeric_args_for_metric(text: str) -> List[str]:
    return [match.rstrip("%").replace(",", "") for match in NUMBER_RE.findall(to_text(text))]


def multiset_f1_for_metric(pred_items: List[str], gold_items: List[str]) -> float:
    if not pred_items or not gold_items:
        return 0.0
    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for item in pred_items:
        pred_counts[item] = pred_counts.get(item, 0) + 1
    for item in gold_items:
        gold_counts[item] = gold_counts.get(item, 0) + 1
    overlap = sum(min(pred_counts.get(item, 0), gold_counts.get(item, 0)) for item in set(pred_counts) | set(gold_counts))
    precision = overlap / max(len(pred_items), 1)
    recall = overlap / max(len(gold_items), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def scale_consistency_metric(executed_value: Any, gold_answer: str, args: argparse.Namespace) -> float:
    pred_num = parse_number(str(executed_value))
    gold_num = parse_number(gold_answer)
    if pred_num is None or gold_num is None or gold_num == 0:
        return 1.0
    if math.isclose(pred_num, gold_num, abs_tol=args.numeric_abs_tol, rel_tol=args.numeric_rel_tol):
        return 1.0
    if math.isclose(pred_num, gold_num * 100.0, rel_tol=1e-3, abs_tol=1e-3):
        return 0.0
    if math.isclose(pred_num * 100.0, gold_num, rel_tol=1e-3, abs_tol=1e-3):
        return 0.0
    return 1.0


def evidence_grounding_metric(evidence: str, program_section: str) -> float:
    pred_nums = sorted(set(numeric_args_for_metric(program_section)))
    if not pred_nums:
        return 1.0
    evidence_nums = set(numeric_args_for_metric(evidence))
    if not evidence_nums:
        return 0.0
    return sum(1 for num in pred_nums if num in evidence_nums) / max(len(pred_nums), 1)


def history_grounding_metric(example: BenchmarkExample, evidence: str) -> Any:
    requires_history = bool((example.metadata or {}).get("requires_history"))
    if not (example.task_name.startswith("convfinqa") and requires_history):
        return ""
    evidence_text = to_text(evidence).lower()
    prompt = example.prompt.lower()
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", prompt))
    if years and any(year in evidence_text for year in years):
        return 1.0
    cues = ("previous", "prior", "same", "that year", "this value", "previous one", "earlier", "follow-up")
    return 1.0 if any(cue in evidence_text for cue in cues) else 0.0


RELAXED_INFIX_RE = re.compile(
    r"^\s*(" + NUMBER_RE.pattern + r")\s*([+\-*/])\s*(" + NUMBER_RE.pattern + r")\s*$"
)
RELAXED_OPERATOR_MAP = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
}


def _normalize_program_number(token: str) -> str:
    return token.rstrip("%").replace(",", "") + ("%" if token.endswith("%") else "")


def relaxed_canonicalize_program(program_text: str) -> str:
    raw = to_text(program_text)
    if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=", raw):
        return canonicalize_program_re(raw)
    program = clean_prediction_program_text(raw)
    if not program:
        return ""
    if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=", program):
        return canonicalize_program_re(program)
    if re.fullmatch(NUMBER_RE.pattern, program.replace("，", ",")):
        return _normalize_program_number(program.replace("，", ","))
    match = RELAXED_INFIX_RE.fullmatch(program.replace("，", ","))
    if match:
        left, operator, right = match.groups()
        op_name = RELAXED_OPERATOR_MAP[operator]
        return f"{op_name}({_normalize_program_number(left)}, {_normalize_program_number(right)})"
    return canonicalize_program_re(program)


def execute_prediction_program(
    program_text: str,
    relaxed_canonicalization: bool = False,
) -> Tuple[Optional[float], str, str]:
    program_canonical = (
        relaxed_canonicalize_program(program_text)
        if relaxed_canonicalization
        else canonicalize_program_re(clean_prediction_program_text(program_text))
    )
    if not program_canonical:
        return None, "", "missing_program"
    value, error = execute_program(program_canonical)
    return value, program_canonical, error or ""


def parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    final_section = extract_normalized_answer(text) or extract_display_answer(text) or extract_section(extract_answer_body(text), FINAL_ANSWER_RE) or extract_answer_body(text) or text
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
    text = extract_normalized_answer(text) or extract_display_answer(text) or extract_section(extract_answer_body(text), FINAL_ANSWER_RE) or extract_answer_body(text) or text
    return re.sub(r"\s+", "", text).strip().lower()


def score_example(example: BenchmarkExample, prediction: str, args: argparse.Namespace) -> Dict[str, Any]:
    numeric_output_format = get_numeric_output_format(args)
    normalized_answer = extract_normalized_answer(prediction)
    display_answer = extract_display_answer(prediction)
    answer = display_answer or extract_section(extract_answer_body(prediction), FINAL_ANSWER_RE) or extract_answer_body(prediction).strip().split("\n")[0].strip()
    explicit_model_answer = normalized_answer or display_answer
    reasoning = extract_section(prediction, REASONING_RE)
    evidence = extract_evidence_section(prediction)
    program_section = extract_program_section(prediction)
    executed_value, executed_program_canonical, program_error = execute_prediction_program(
        program_section,
        relaxed_canonicalization=bool(getattr(args, "relaxed_executor_canonicalization", False)),
    )
    gold_num = parse_number(example.gold_answer)
    model_pred_num = parse_number(prediction)
    model_answer_correct = None
    if model_pred_num is not None and gold_num is not None:
        model_answer_correct = float(
            math.isclose(model_pred_num, gold_num, abs_tol=args.numeric_abs_tol, rel_tol=args.numeric_rel_tol)
        )
    executed_answer_correct = None
    if executed_value is not None and gold_num is not None:
        executed_answer_correct = float(
            math.isclose(executed_value, gold_num, abs_tol=args.numeric_abs_tol, rel_tol=args.numeric_rel_tol)
        )
    program_answer_consistency = None
    if executed_value is not None and model_pred_num is not None:
        program_answer_consistency = float(
            math.isclose(executed_value, model_pred_num, abs_tol=args.numeric_abs_tol, rel_tol=args.numeric_rel_tol)
        )
    if example.answer_type == "numeric":
        structured_anchors = NUMERIC_STRUCTURED_ANCHORS[numeric_output_format]
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
        "answer_coverage": float(bool(answer)),
        "normalized_answer_coverage": float(bool(normalized_answer)),
        "final_answer_coverage": float(bool(answer)),
        "reasoning_coverage": float(bool(reasoning)),
        "evidence_bullet_count": evidence_bullet_count(evidence),
        "evidence_over_two_bullets_rate": float(evidence_bullet_count(evidence) > 2),
        "evidence_chars": len((evidence or "").strip()),
        "post_program_text_rate": float(has_post_program_text(prediction)),
        "strict_program_contract_rate": strict_program_contract_rate(prediction, evidence, program_section),
        "program_section_coverage": float(bool(program_section)),
        "structured_response_coverage": float(all(anchor in (prediction or "") for anchor in structured_anchors)),
        "prediction_chars": len((prediction or "").strip()),
        "numeric_parse_rate": None,
        "program_parse_rate": None,
        "program_execution_rate": None,
        "executed_answer_accuracy": None,
        "model_normalized_answer_accuracy": None,
        "program_answer_consistency": None,
        "program_string_accuracy": None,
        "executed_program": executed_program_canonical,
        "executed_program_answer": format_numeric_answer(executed_value) if executed_value is not None else "",
        "program_execution_error": program_error,
        "operation_match_rate": "",
        "argument_grounding_rate": "",
        "scale_consistency_rate": "",
        "evidence_grounding_rate": "",
        "history_grounding_rate": history_grounding_metric(example, evidence),
    }
    if example.answer_type == "numeric":
        operation_match = ""
        argument_grounding = ""
        if example.gold_program:
            operation_match = multiset_f1_for_metric(
                program_ops_for_metric(program_section),
                program_ops_for_metric(example.gold_program),
            )
            argument_grounding = multiset_f1_for_metric(
                numeric_args_for_metric(program_section),
                numeric_args_for_metric(example.gold_program),
            )
        result["numeric_parse_rate"] = float(model_pred_num is not None)
        result["program_parse_rate"] = float(bool(program_section))
        result["program_execution_rate"] = float(executed_value is not None)
        result["executed_answer_accuracy"] = float(executed_answer_correct or 0.0)
        result["model_normalized_answer_accuracy"] = float(model_answer_correct or 0.0) if model_answer_correct is not None else 0.0
        result["program_answer_consistency"] = program_answer_consistency
        result["operation_match_rate"] = operation_match
        result["argument_grounding_rate"] = argument_grounding
        result["scale_consistency_rate"] = scale_consistency_metric(executed_value, example.gold_answer, args) if executed_value is not None else 0.0
        result["evidence_grounding_rate"] = evidence_grounding_metric(evidence, program_section)
        program_primary_format = numeric_output_format in {
            "program_executor",
            "cot_program",
            "reasoning_program_executor",
            "program_first_answer",
        }
        if gold_num is None:
            result["answer_correct"] = float(normalize_answer_text(prediction) == normalize_answer_text(example.gold_answer))
        elif program_primary_format:
            result["answer_correct"] = result["executed_answer_accuracy"]
        elif explicit_model_answer and model_answer_correct is not None:
            result["answer_correct"] = result["model_normalized_answer_accuracy"]
        else:
            result["answer_correct"] = result["executed_answer_accuracy"]
        if example.gold_program:
            result["program_correct"] = float(normalize_program(program_section) == normalize_program(example.gold_program))
            result["program_string_accuracy"] = result["program_correct"]
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


def _numeric_score_values(scores: List[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for score in scores:
        value = score.get(key, "")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def compute_pass_metrics(
    greedy_score: Dict[str, Any],
    sampled_scores: Sequence[Dict[str, Any]],
    pass_k_values: Sequence[int],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "pass@1_greedy": float(greedy_score["answer_correct"]),
        "pass@1_sampled": "",
    }
    sampled_correct = [float(item["answer_correct"]) for item in sampled_scores]
    if sampled_correct:
        metrics["pass@1_sampled"] = float(sampled_correct[0] > 0.0)
    for k in pass_k_values:
        if k == 1:
            continue
        effective_k = min(k, len(sampled_correct))
        metrics[f"pass@{k}"] = float(any(score > 0.0 for score in sampled_correct[:effective_k])) if effective_k else ""
    return metrics


def build_example_summary_score(
    greedy_score: Dict[str, Any],
    sampled_scores: Sequence[Dict[str, Any]],
    pass_k_values: Sequence[int],
) -> Dict[str, Any]:
    summary_score = dict(greedy_score)
    summary_score.update(compute_pass_metrics(greedy_score, sampled_scores, pass_k_values))
    return summary_score


def _build_summary_row(model_name: str, task_name: str, scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    answer_scores = [item["answer_correct"] for item in scores]
    program_scores = [item["program_correct"] for item in scores if item["program_correct"] is not None]
    program_parse_scores = [item["program_parse_rate"] for item in scores if item.get("program_parse_rate") is not None]
    program_execution_scores = [item["program_execution_rate"] for item in scores if item.get("program_execution_rate") is not None]
    executed_answer_scores = [item["executed_answer_accuracy"] for item in scores if item.get("executed_answer_accuracy") is not None]
    model_normalized_scores = [item["model_normalized_answer_accuracy"] for item in scores if item.get("model_normalized_answer_accuracy") is not None]
    program_answer_consistency_scores = [item["program_answer_consistency"] for item in scores if item.get("program_answer_consistency") is not None]
    program_string_scores = [item["program_string_accuracy"] for item in scores if item.get("program_string_accuracy") is not None]
    answer_coverage_scores = [item.get("answer_coverage", item.get("final_answer_coverage", 0.0)) for item in scores]
    normalized_answer_scores = [item.get("normalized_answer_coverage", 0.0) for item in scores]
    final_answer_scores = [item.get("final_answer_coverage", item.get("answer_coverage", 0.0)) for item in scores]
    reasoning_scores = [item.get("reasoning_coverage", 0.0) for item in scores]
    evidence_bullet_counts = [item.get("evidence_bullet_count", 0.0) for item in scores]
    evidence_over_two_scores = [item.get("evidence_over_two_bullets_rate", 0.0) for item in scores]
    evidence_chars = [item.get("evidence_chars", 0.0) for item in scores]
    post_program_text_scores = [item.get("post_program_text_rate", 0.0) for item in scores]
    strict_contract_scores = [item.get("strict_program_contract_rate", 0.0) for item in scores]
    program_section_scores = [item["program_section_coverage"] for item in scores]
    structured_scores = [item["structured_response_coverage"] for item in scores]
    prediction_chars = [item["prediction_chars"] for item in scores]
    numeric_parse_scores = [item["numeric_parse_rate"] for item in scores if item["numeric_parse_rate"] is not None]
    operation_match_scores = [item.get("operation_match_rate", "") for item in scores if item.get("operation_match_rate", "") != ""]
    argument_grounding_scores = [item.get("argument_grounding_rate", "") for item in scores if item.get("argument_grounding_rate", "") != ""]
    scale_consistency_scores = [item.get("scale_consistency_rate", "") for item in scores if item.get("scale_consistency_rate", "") != ""]
    evidence_grounding_scores = [item.get("evidence_grounding_rate", "") for item in scores if item.get("evidence_grounding_rate", "") != ""]
    history_grounding_scores = [item.get("history_grounding_rate", "") for item in scores if item.get("history_grounding_rate", "") != ""]
    answer_acc = sum(answer_scores) / len(answer_scores)
    row = {
        "model_name": model_name,
        "task_name": task_name,
        "num_examples": len(scores),
        "answer_accuracy": round(answer_acc, 6),
        "primary_metric": round(answer_acc, 6),
        "pass@1_greedy": _mean([item["pass@1_greedy"] for item in scores if item.get("pass@1_greedy") != ""]),
        "pass@1_sampled": _mean([item["pass@1_sampled"] for item in scores if item.get("pass@1_sampled") != ""]),
        "program_accuracy": _mean(program_scores),
        "program_parse_rate": _mean(program_parse_scores),
        "program_execution_rate": _mean(program_execution_scores),
        "executed_answer_accuracy": _mean(executed_answer_scores),
        "model_normalized_answer_accuracy": _mean(model_normalized_scores),
        "program_answer_consistency": _mean(program_answer_consistency_scores),
        "program_string_accuracy": _mean(program_string_scores),
        "answer_coverage": _mean(answer_coverage_scores),
        "normalized_answer_coverage": _mean(normalized_answer_scores),
        "final_answer_coverage": _mean(final_answer_scores),
        "reasoning_coverage": _mean(reasoning_scores),
        "avg_evidence_bullet_count": _mean(evidence_bullet_counts),
        "evidence_over_two_bullets_rate": _mean(evidence_over_two_scores),
        "avg_evidence_chars": _mean(evidence_chars),
        "post_program_text_rate": _mean(post_program_text_scores),
        "strict_program_contract_rate": _mean(strict_contract_scores),
        "program_section_coverage": _mean(program_section_scores),
        "structured_response_coverage": _mean(structured_scores),
        "numeric_parse_rate": _mean(numeric_parse_scores),
        "operation_match_rate": _mean(operation_match_scores),
        "argument_grounding_rate": _mean(argument_grounding_scores),
        "scale_consistency_rate": _mean(scale_consistency_scores),
        "evidence_grounding_rate": _mean(evidence_grounding_scores),
        "history_grounding_rate": _mean(history_grounding_scores),
        "generation_quality_score": _mean(
            [
                item
                for item in (
                    _mean(program_execution_scores),
                    _mean(operation_match_scores),
                    _mean(argument_grounding_scores),
                    _mean(scale_consistency_scores),
                    _mean(evidence_grounding_scores),
                )
                if item != ""
            ]
        ),
        "selection_quality_score": "",
        "avg_prediction_chars": _mean(prediction_chars),
    }
    pass_keys = sorted(
        {key for score in scores for key in score if key.startswith("pass@") and key not in {"pass@1_greedy", "pass@1_sampled"}},
        key=lambda item: int(item.split("@", 1)[1]),
    )
    for key in pass_keys:
        row[key] = _mean([item[key] for item in scores if item.get(key) != ""])
    selection_keys = ["pass@1_greedy"] + pass_keys
    row["selection_quality_score"] = _mean(
        [item for key in selection_keys for item in [_mean(_numeric_score_values(scores, key))] if item != ""]
    )
    return row


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
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prediction_row(
    model_name: str,
    example: BenchmarkExample,
    prediction: str,
    score: Dict[str, Any],
    generation_mode: str,
    candidate_index: int,
    *,
    scoring_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "model_name": model_name,
        "task_name": example.task_name,
        "record_id": example.record_id,
        "generation_mode": generation_mode,
        "candidate_index": candidate_index,
        "prompt": example.prompt,
        "gold_answer": example.gold_answer,
        "gold_program": example.gold_program,
        "prediction": prediction,
        "answer_correct": score["answer_correct"],
        "program_correct": score["program_correct"],
        "reasoning_coverage": score.get("reasoning_coverage"),
        "structured_response_coverage": score.get("structured_response_coverage"),
        "program_parse_rate": score.get("program_parse_rate"),
        "program_execution_rate": score.get("program_execution_rate"),
        "executed_answer_accuracy": score.get("executed_answer_accuracy"),
        "executed_program": score.get("executed_program", ""),
        "executed_program_answer": score.get("executed_program_answer", ""),
        "program_execution_error": score.get("program_execution_error", ""),
        "model_normalized_answer_accuracy": score.get("model_normalized_answer_accuracy"),
        "program_answer_consistency": score.get("program_answer_consistency"),
        "program_string_accuracy": score.get("program_string_accuracy"),
        "metadata": example.metadata,
    }
    if scoring_config is not None:
        row["scoring_config"] = dict(scoring_config)
    return row


def example_from_prediction_row(row: Dict[str, Any]) -> BenchmarkExample:
    return BenchmarkExample(
        task_name=to_text(row.get("task_name")),
        prompt=to_text(row.get("prompt")),
        gold_answer=to_text(row.get("gold_answer")),
        answer_type="mcq" if to_text(row.get("task_name")).startswith(("fineval", "cflue_")) else "numeric",
        record_id=to_text(row.get("record_id")),
        metadata=dict(row.get("metadata") or {}),
        gold_program=to_text(row.get("gold_program")),
    )


def aggregate_prediction_rows(
    model_name: str,
    raw_predictions: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    pass_k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in raw_predictions:
        if to_text(row.get("model_name")) != model_name:
            continue
        key = (to_text(row.get("task_name")), to_text(row.get("record_id")))
        grouped.setdefault(key, []).append(dict(row))
    validate_prediction_scoring_config(
        [row for rows in grouped.values() for row in rows],
        args,
        pass_k_values,
    )

    scores: List[Dict[str, Any]] = []
    for _, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: (to_text(item.get("generation_mode")) != "greedy", int(item.get("candidate_index") or 0)))
        greedy_row = next((row for row in rows if row.get("generation_mode") == "greedy"), rows[0])
        example = example_from_prediction_row(greedy_row)
        greedy_score = score_example(example, to_text(greedy_row.get("prediction")), args)
        sampled_scores = [
            score_example(example_from_prediction_row(row), to_text(row.get("prediction")), args)
            for row in rows
            if row.get("generation_mode") == "sampled"
        ]
        scores.append(build_example_summary_score(greedy_score, sampled_scores, pass_k_values))
    return aggregate_scores(model_name, scores)


def load_existing_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_generation(
    model,
    tokenizer,
    model_name: str,
    example: BenchmarkExample,
    args: argparse.Namespace,
    *,
    generation_mode: str,
    candidate_index: int,
    temperature: float,
    top_p: Optional[float],
    do_sample: bool,
    seed: Optional[int],
    scoring_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prediction = generate_response(
        model,
        tokenizer,
        example.prompt,
        args,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        seed=seed,
    )
    score = score_example(example, prediction, args)
    row = build_prediction_row(
        model_name,
        example,
        prediction,
        score,
        generation_mode,
        candidate_index,
        scoring_config=scoring_config,
    )
    return score, row


def unload_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    PROCESSOR_ARGS.sft_variant = args.processor_sft_variant
    PROCESSOR_ARGS.numeric_output_format = args.numeric_output_format
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_k_values = parse_pass_k_values(args.pass_k)
    if args.num_samples_per_example < 0:
        raise ValueError("--num_samples_per_example must be non-negative.")
    if args.num_samples_per_example > 0 and args.sample_temperature <= 0.0:
        raise ValueError("--sample_temperature must be positive when sampling candidates.")
    if not 0.0 < args.sample_top_p <= 1.0:
        raise ValueError("--sample_top_p must be in the interval (0, 1].")
    if args.num_samples_per_example < max(pass_k_values):
        print(
            f"[warn] num_samples_per_example={args.num_samples_per_example} is smaller than max pass@k={max(pass_k_values)}; "
            "metrics will use available candidates only."
        )
    scoring_config = scoring_config_from_args(args, pass_k_values)

    model_paths = parse_name_path_entries(args.model_entry)
    if not model_paths:
        raise ValueError("Please provide at least one --model_entry name=path")
    adapter_paths = parse_name_path_entries(args.adapter_entry)
    model_paths, adapter_paths = filter_model_entries(model_paths, adapter_paths, args.only_model)
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
    allowed_record_ids = load_record_id_allowlist(args.record_id_allowlist_jsonl)
    before_allowlist_count = len(benchmark_examples)
    benchmark_examples = filter_examples_by_record_id_allowlist(benchmark_examples, allowed_record_ids)
    if allowed_record_ids is not None:
        print(
            f"[allowlist] kept {len(benchmark_examples)}/{before_allowlist_count} examples from {args.record_id_allowlist_jsonl}"
        )
    benchmark_examples = select_stratified_examples(benchmark_examples, args.stratified_max_per_bucket)

    if not benchmark_examples:
        raise ValueError("No benchmark examples were loaded. Please check your dataset paths and splits.")

    manifest = {
        "models": model_paths,
        "adapters": adapter_paths,
        "tokenizer_path": tokenizer_path,
        "convfinqa_test_file": args.convfinqa_test_file,
        "finqa_test_file": args.finqa_test_file,
        "record_id_allowlist_jsonl": args.record_id_allowlist_jsonl,
        "fineval_dataset_name": args.fineval_dataset_name,
        "fineval_split": args.fineval_split,
        "run_fineval": run_fineval,
        "fineval_local_file": args.fineval_local_file,
        "cflue_task_files": cflue_task_files,
        "num_examples": len(benchmark_examples),
        "pass_k": pass_k_values,
        "num_samples_per_example": args.num_samples_per_example,
        "sample_temperature": args.sample_temperature,
        "sample_top_p": args.sample_top_p,
        "sample_seed": args.sample_seed,
        "processor_sft_variant": args.processor_sft_variant,
        "numeric_output_format": args.numeric_output_format,
        "scoring_config": scoring_config,
        "primary_metric": "executed_answer_accuracy",
    }
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_summary_rows: List[Dict[str, Any]] = []
    if args.score_only_predictions:
        for model_name in model_paths:
            predictions_path = output_dir / f"{model_name}_predictions.jsonl"
            raw_predictions = load_existing_predictions(predictions_path)
            summary_rows = aggregate_prediction_rows(model_name, raw_predictions, args, pass_k_values)
            save_jsonl(output_dir / f"{model_name}_summary.jsonl", summary_rows)
            all_summary_rows.extend(summary_rows)
        save_csv(output_dir / "benchmark_summary.csv", all_summary_rows)
        (output_dir / "benchmark_summary.json").write_text(
            json.dumps(all_summary_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[done] Re-scored benchmark outputs in: {output_dir}")
        return

    for model_name, model_path in model_paths.items():
        adapter_path = adapter_paths.get(model_name, "")
        predictions_path = output_dir / f"{model_name}_predictions.jsonl"
        if args.skip_existing_predictions and predictions_path.exists():
            print(f"[reuse] model={model_name} predictions={predictions_path}")
            raw_predictions = load_existing_predictions(predictions_path)
            summary_rows = aggregate_prediction_rows(model_name, raw_predictions, args, pass_k_values)
            save_jsonl(output_dir / f"{model_name}_summary.jsonl", summary_rows)
            all_summary_rows.extend(summary_rows)
            continue
        print(f"[eval] model={model_name} path={model_path} adapter={adapter_path or 'none'}")
        model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, adapter_path, args)
        raw_predictions: List[Dict[str, Any]] = []
        scores: List[Dict[str, Any]] = []
        for example_index, example in enumerate(tqdm(benchmark_examples, desc=f"Evaluating {model_name}")):
            greedy_score, greedy_row = score_generation(
                model,
                tokenizer,
                model_name,
                example,
                args,
                generation_mode="greedy",
                candidate_index=0,
                temperature=0.0,
                top_p=None,
                do_sample=False,
                seed=None,
                scoring_config=scoring_config,
            )
            raw_predictions.append(greedy_row)

            sampled_scores: List[Dict[str, Any]] = []
            for candidate_index in range(args.num_samples_per_example):
                sample_seed = args.sample_seed + example_index * args.num_samples_per_example + candidate_index
                sampled_score, sampled_row = score_generation(
                    model,
                    tokenizer,
                    model_name,
                    example,
                    args,
                    generation_mode="sampled",
                    candidate_index=candidate_index,
                    temperature=args.sample_temperature,
                    top_p=args.sample_top_p,
                    do_sample=True,
                    seed=sample_seed,
                    scoring_config=scoring_config,
                )
                sampled_scores.append(sampled_score)
                raw_predictions.append(sampled_row)

            scores.append(build_example_summary_score(greedy_score, sampled_scores, pass_k_values))

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

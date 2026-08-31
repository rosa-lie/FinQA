from __future__ import annotations

import os
raw_omp_num_threads = os.environ.get("OMP_NUM_THREADS")
if raw_omp_num_threads is None or not raw_omp_num_threads.strip().isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import math

import torch
from torch.utils.data import SequentialSampler
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

from evaluation.evaluate_financial_benchmarks import execute_prediction_program

PROGRAM_NUMERIC_REQUIRED = ["Evidence:", "Program:"]
PROGRAM_NUMERIC_FORBIDDEN = ["Answer:", "Normalized Answer:", "Program: N/A"]
PROGRAM_FIRST_ANSWER_FORBIDDEN = ["Program: N/A"]
HARD_INVALID_REWARD = -1.0
WRONG_EXECUTABLE_REWARD_CAP = 0.2
CORRECT_EXECUTABLE_BASE_REWARD = 1.0
PROGRAM_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
    "average",
    "sum",
    "max",
    "min",
]
AGGREGATE_OPS = {"average", "sum", "max", "min", "table_average", "table_sum", "table_max", "table_min"}
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?%?")
ANCHORS = ["Evidence:", "Program:", "Reasoning:", "Answer:", "Normalized Answer:"]
SYSTEM_PROMPT = (
    "You are a financial numerical reasoning assistant. Follow the requested schema exactly. "
    "Output brief Reasoning if useful, then Evidence and Program. Reasoning is optional and must be concise. "
    "Evidence must be at most 2 short bullets and only cite numbers used in Program. "
    "Program must be exactly one executable DSL line. Stop immediately after Program."
)
STRICT_PROGRAM_SYSTEM_PROMPT = (
    "You are a financial numerical reasoning assistant. Follow the requested schema exactly. "
    "Output only Evidence and Program. Do not output Reasoning, Answer, Normalized Answer, or any extra section. "
    "Evidence must be at most 2 short bullets and only cite numbers used in Program. "
    "Program must be exactly one executable DSL line. Stop immediately after Program."
)
ANSWER_FIRST_SYSTEM_PROMPT = (
    "You are a financial numerical reasoning assistant. Follow the requested schema exactly. "
    "Output brief Reasoning if useful, then Evidence, Program, and Answer. "
    "Evidence must be short and grounded in the report. "
    "Program should be one executable DSL line when possible. "
    "Answer must contain the final normalized numeric answer."
)
PROGRAM_FIRST_ANSWER_SYSTEM_PROMPT = (
    "You are a financial numerical reasoning assistant. Follow the requested schema exactly. "
    "Output brief Reasoning if useful, then Evidence, Program, and Answer. "
    "Program is primary and must be one executable DSL line. "
    "Answer must contain the final normalized numeric answer and match executing Program."
)
V26_COT_POT_OUTPUT_FORMAT = (
    "Output format:\n"
    "Reasoning: ... (optional, keep it brief)\n\n"
    "Evidence:\n"
    "- ...\n\n"
    "Program: ...\n\n"
    "The final numeric answer will be computed by executing Program.\n"
    "Do not calculate or round the final answer yourself.\n\n"
    "Program rule:\n"
    "- The Program line must be executable by the DSL executor.\n"
    "- Use only numeric literals or DSL expressions: add, subtract, multiply, divide, max, min, sum, average.\n"
    "- Do not use unsupported functions, variable assignment, natural language, or SQL-like expressions.\n"
    "- For direct lookup answers, output only the numeric literal. Correct: Program: 93. Wrong: Program: get(93) or Program: answer = 93."
)
STRICT_PROGRAM_OUTPUT_FORMAT = (
    "Output format:\n"
    "Evidence:\n"
    "- ...\n\n"
    "Program: ...\n\n"
    "The final numeric answer will be computed by executing Program.\n"
    "Do not calculate or round the final answer yourself.\n\n"
    "Contract rule:\n"
    "- Output only Evidence and Program.\n"
    "- Do not output Reasoning, Answer, Normalized Answer, Operation Plan, or Formula candidates.\n\n"
    "Program rule:\n"
    "- The Program line must be executable by the DSL executor.\n"
    "- Use only numeric literals or DSL expressions: add, subtract, multiply, divide, max, min, sum, average.\n"
    "- Do not use unsupported functions, variable assignment, natural language, or SQL-like expressions.\n"
    "- For direct lookup answers, output only the numeric literal. Correct: Program: 93. Wrong: Program: get(93) or Program: answer = 93."
)
ANSWER_FIRST_COT_POT_OUTPUT_FORMAT = (
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
    "- Program is auxiliary but should be an executable numeric DSL expression when possible, using add, subtract, multiply, divide, max, min, sum, average.\n"
    "- Do not use variable assignment, natural language instructions, markdown code blocks, or multiple Program sections."
)
PROGRAM_FIRST_ANSWER_COT_POT_OUTPUT_FORMAT = (
    "Output format:\n"
    "Reasoning: ... (optional, keep it brief)\n\n"
    "Evidence:\n"
    "- ...\n\n"
    "Program: ...\n\n"
    "Answer: ...\n\n"
    "Program rule:\n"
    "- Program is primary and must be one executable numeric DSL expression when possible.\n"
    "- Use numeric literals or DSL expressions: add, subtract, multiply, divide, max, min, sum, average.\n"
    "- Do not use unsupported functions, variable assignment, natural language, or SQL-like expressions.\n\n"
    "Answer rule:\n"
    "- Put the final normalized numeric answer after Answer:.\n"
    "- The final numeric answer must match executing Program.\n"
    "- Use raw ratios such as 0.02899 unless the question explicitly asks for a percent."
)


@dataclass
class ScriptArguments:
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer path. Defaults to model_name_or_path."}
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional existing PEFT adapter path to load trainably on top of model_name_or_path."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "Training json/jsonl file path."})
    valid_file: Optional[str] = field(default=None, metadata={"help": "Validation json/jsonl file path."})
    preprocessing_num_workers: int = field(default=4)
    reward_profile_expected: str = field(default="program_numeric")
    strict_schema_required: bool = field(default=True)
    reward_mode: str = field(
        default="execution",
        metadata={"help": "Reward mode: execution uses DSL executor correctness; frontier_execution_calibration uses conservative outcome-only bands for short frontier GRPO; answer_only is a loose ablation; answer_first requires an explicit Answer anchor and uses Program as auxiliary signal; program_first_answer_aux keeps Program primary and uses explicit Answer for auxiliary reward/consistency; program_gated_answer_aux only rewards Answer after the Program is executable, correct, and consistent."},
    )
    outcome_first_reward: bool = field(
        default=False,
        metadata={"help": "Prioritize executable answer correctness over schema hard-gates; used by v16 outcome-first GRPO."},
    )
    schema_hard_gate: bool = field(
        default=False,
        metadata={"help": "Hard-invalid completions that violate strict Evidence/Program schema or DSL constraints."},
    )
    relaxed_executor_canonicalization: bool = field(
        default=False,
        metadata={"help": "Allow recoverable numeric infix/direct numeric programs through the shared executor."},
    )
    recoverable_syntax_soft_gate: bool = field(
        default=False,
        metadata={"help": "Treat recoverable DSL syntax issues as soft penalties instead of hard schema failures."},
    )
    direct_lookup_wrapper_soft_penalty: bool = field(
        default=False,
        metadata={"help": "Soft-penalize direct lookup wrappers when execution is numerically correct instead of hard invalidating them."},
    )
    dense_reward_shaping: bool = field(
        default=False,
        metadata={"help": "Use denser v46 reward shaping for executable but non-strict programs."},
    )
    evidence_max_bullets: int = field(default=0, metadata={"help": "Hard-gate evidence bullet count above this value. 0 disables."})
    evidence_max_chars: int = field(default=0, metadata={"help": "Hard-gate evidence text longer than this many chars. 0 disables."})
    program_max_lines: int = field(default=0, metadata={"help": "Hard-gate Program content with more non-empty lines. 0 disables."})
    program_max_chars: int = field(default=0, metadata={"help": "Hard-gate Program content longer than this many chars. 0 disables."})
    answer_abs_tol: float = field(default=1e-4)
    answer_rel_tol: float = field(default=1e-4)
    format_reward_weight: float = field(default=0.03)
    program_executable_reward_weight: float = field(default=0.15)
    program_execution_closeness_reward_weight: float = field(default=0.30)
    program_structure_reward_weight: float = field(default=0.20)
    program_argument_coverage_reward_weight: float = field(default=0.12)
    program_step_count_reward_weight: float = field(default=0.08)
    program_exact_match_bonus_weight: float = field(default=0.05)
    evidence_reward_weight: float = field(default=0.04)
    history_grounding_reward_weight: float = field(
        default=0.0,
        metadata={"help": "Penalty weight for ConvFinQA history-dependent turns when evidence is not history-grounded."},
    )
    brevity_reward_weight: float = field(default=0.02)
    correct_format_penalty_floor: float = field(
        default=0.85,
        metadata={"help": "Minimum reward retained for exact executable answers after schema/format penalties."},
    )
    wrong_executable_reward_cap: float = field(
        default=WRONG_EXECUTABLE_REWARD_CAP,
        metadata={"help": "Maximum reward for executable programs whose answer is numerically wrong."},
    )
    invalid_program_reward: float = field(
        default=HARD_INVALID_REWARD,
        metadata={"help": "Reward assigned to missing, invalid, or unexecutable programs."},
    )
    schema_soft_penalty_weight: float = field(default=0.05)
    schema_soft_penalty_max: float = field(default=0.15)
    contract_soft_penalty_weight: float = field(default=0.05)
    contract_soft_penalty_max: float = field(default=0.15)
    length_penalty_chars: int = field(
        default=0,
        metadata={"help": "Apply a soft penalty when completion characters exceed this value. 0 disables it."},
    )
    length_penalty_weight: float = field(default=0.0)
    length_penalty_max: float = field(default=0.15)
    reasoning_max_chars: int = field(
        default=0,
        metadata={"help": "Apply a soft penalty when optional Reasoning exceeds this many chars. 0 disables it."},
    )
    reasoning_max_lines: int = field(
        default=0,
        metadata={"help": "Apply a soft penalty when optional Reasoning exceeds this many non-empty lines. 0 disables it."},
    )
    reasoning_penalty_weight: float = field(default=0.0)
    reasoning_penalty_max: float = field(default=0.12)
    risk_aware_reward: bool = field(default=False)
    risk_group_size: int = field(
        default=0,
        metadata={"help": "Number of completions per prompt for group risk shaping. 0 disables group shaping."},
    )
    risk_q25_floor: float = field(default=0.0)
    risk_invalid_threshold: float = field(default=0.25)
    risk_penalty_weight: float = field(default=0.10)
    risk_wrong_executable_threshold: float = field(default=1.0)
    risk_process_q25_floor: float = field(default=0.0)
    risk_semantic_penalty_weight: float = field(default=0.0)
    risk_exact_bonus_weight: float = field(default=0.0)
    risk_skip_all_wrong_groups: bool = field(
        default=False,
        metadata={"help": "Set all rewards in all-wrong groups equal so GRPO gets zero advantage from noisy groups."},
    )
    risk_single_correct_advantage_clip: float = field(
        default=0.0,
        metadata={"help": "If >0, cap a single correct completion to best wrong reward plus this margin."},
    )
    bundle_diagnostics_enabled: bool = field(
        default=False,
        metadata={"help": "Record RiskPO-style cross-prompt bundle score diagnostics without changing rewards."},
    )
    bundle_size: int = field(
        default=4,
        metadata={"help": "Number of distinct prompts per diagnostic bundle."},
    )
    bundle_num_generations: int = field(
        default=0,
        metadata={"help": "Number of completions per prompt used to reconstruct diagnostic bundles. 0 disables."},
    )
    bundle_bucket_by_source: bool = field(
        default=True,
        metadata={"help": "Bucket bundle diagnostics by source/history/question metadata before bundling."},
    )
    bundle_grpo_enabled: bool = field(
        default=False,
        metadata={"help": "Replace prompt-local GRPO advantages with RiskPO-style bundle MVaR advantages."},
    )
    bundle_grpo_quantile_down: float = field(default=0.2)
    bundle_grpo_quantile_up: float = field(default=0.8)
    bundle_grpo_cvar_weight: float = field(default=1.0)
    bundle_grpo_normalize_by_bundle_std: bool = field(default=True)
    bundle_grpo_fallback_to_prompt_advantage: bool = field(default=True)
    bundle_ordered_sampling: bool = field(
        default=False,
        metadata={"help": "Use sequential training sampling so bucket-ordered Bundle-MVaR JSONL rows stay adjacent in batches."},
    )
    process_reward_enabled: bool = field(default=False)
    process_reward_weight: float = field(default=0.12)
    evidence_program_alignment_weight: float = field(default=0.04)
    program_arg_precision_weight: float = field(default=0.04)
    operation_prior_weight: float = field(default=0.03)
    dsl_validity_weight: float = field(default=0.01)
    ratio_divide_requirement_weight: float = field(
        default=0.0,
        metadata={"help": "Reward weight for ratio/percent questions whose predicted program includes required divide ops."},
    )
    denominator_grounding_weight: float = field(
        default=0.0,
        metadata={"help": "Reward weight for matching ratio/percent divide denominators against the gold program."},
    )
    table_argument_grounding_weight: float = field(
        default=0.0,
        metadata={"help": "Reward weight for using the same numeric table/cell arguments as the gold program."},
    )
    lookup_shortcut_penalty_weight: float = field(
        default=0.0,
        metadata={"help": "Reward weight for avoiding direct lookup shortcuts on calculation questions."},
    )
    log_prompt_completions: bool = field(default=False)
    tensorboard_logging_dir: Optional[str] = field(default=None)


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(first_text(v) for v in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return first_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return first_text(completion["content"])
        if "text" in completion:
            return first_text(completion["text"])
    if isinstance(completion, list) and completion:
        return completion_text(completion[0])
    return first_text(completion)


def extract_anchor(text: str, anchor: str) -> str:
    text = first_text(text)
    if not text or anchor not in text:
        return ""
    start = text.index(anchor) + len(anchor)
    end = len(text)
    for other in ANCHORS:
        if other == anchor:
            continue
        pos = text.find(other, start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end].strip()


def anchor_count(text: str, anchor: str) -> int:
    return first_text(text).count(anchor)


def forbidden_anchors_for_reward_mode(reward_mode: str) -> List[str]:
    if reward_mode in {"answer_first", "program_first_answer_aux", "program_gated_answer_aux"}:
        return PROGRAM_FIRST_ANSWER_FORBIDDEN
    return PROGRAM_NUMERIC_FORBIDDEN


def has_post_program_text(text: str) -> bool:
    text = first_text(text)
    marker = "Program:"
    if marker not in text:
        return False
    program_tail = text[text.rfind(marker) + len(marker):].strip()
    if not program_tail:
        return False
    lines = [line.strip() for line in program_tail.splitlines() if line.strip()]
    if len(lines) <= 1:
        return False
    return True


def non_empty_lines(text: str) -> List[str]:
    return [line.strip() for line in first_text(text).splitlines() if line.strip()]


def evidence_bullet_count(evidence: str) -> int:
    lines = non_empty_lines(evidence)
    bullet_lines = [
        line for line in lines
        if line.startswith("-") or re.match(r"^\d+[\.)]\s+", line)
    ]
    return len(bullet_lines) if bullet_lines else (1 if first_text(evidence).strip() else 0)


def prompt_contract_reasons(
    evidence: str,
    pred_program: str,
    script_args: ScriptArguments,
) -> List[str]:
    reasons: List[str] = []
    if script_args.evidence_max_bullets > 0 and evidence_bullet_count(evidence) > script_args.evidence_max_bullets:
        reasons.append("too_many_evidence_bullets")
    if script_args.evidence_max_chars > 0 and len(first_text(evidence).strip()) > script_args.evidence_max_chars:
        reasons.append("evidence_too_long")
    if script_args.program_max_lines > 0 and len(non_empty_lines(pred_program)) > script_args.program_max_lines:
        reasons.append("program_too_many_lines")
    if script_args.program_max_chars > 0 and len(first_text(pred_program).strip()) > script_args.program_max_chars:
        reasons.append("program_too_long")
    return reasons


def normalize_number(text: str) -> Optional[float]:
    text = first_text(text)
    if not text:
        return None
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    raw = matches[-1]
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1]
    try:
        value = float(raw)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def numeric_equal(pred: str, gold: str, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> bool:
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    if pred_num is None or gold_num is None:
        return first_text(pred).strip().lower() == first_text(gold).strip().lower()
    return abs(pred_num - gold_num) <= max(abs_tol, abs(gold_num) * rel_tol)


def numeric_closeness(pred: str, gold: str, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> float:
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    if pred_num is None or gold_num is None:
        return 1.0 if first_text(pred).strip().lower() == first_text(gold).strip().lower() else 0.0
    tolerance = max(abs_tol, abs(gold_num) * rel_tol)
    abs_error = abs(pred_num - gold_num)
    if abs_error <= tolerance:
        return 1.0
    scaled_error = abs_error / max(abs(gold_num), 1.0)
    return max(0.0, 1.0 - min(scaled_error, 2.0) / 2.0)


def program_ops(program: str) -> List[str]:
    program = first_text(program).lower()
    return [op for op in PROGRAM_OPS if re.search(rf"\b{re.escape(op)}\b", program)]


def parse_program_steps(program: str) -> List[tuple[str, List[str]]]:
    program = first_text(program)
    matches = re.findall(r"([A-Za-z_]+)\(([^()]*)\)", program)
    steps = []
    for op, raw_args in matches:
        args = [arg.strip().lower() for arg in raw_args.split(",") if arg.strip()]
        steps.append((op.lower(), args))
    return steps


def program_numeric_args(program: str) -> List[str]:
    text = re.sub(r"#\d+", "", first_text(program).replace(",", ""))
    return [match.rstrip("%") for match in NUMBER_RE.findall(text)]


def split_top_level_args(raw_args: str) -> List[str]:
    args: List[str] = []
    current: List[str] = []
    depth = 0
    for char in first_text(raw_args):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def find_call_args(program: str, op_name: str) -> List[List[str]]:
    text = first_text(program)
    calls: List[List[str]] = []
    pattern = re.compile(rf"\b{re.escape(op_name)}\s*\(", re.IGNORECASE)
    for match in pattern.finditer(text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth > 0:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        if depth == 0:
            calls.append(split_top_level_args(text[start:index - 1]))
    return calls


def divide_denominator_args(program: str) -> List[str]:
    denominators: List[str] = []
    for args in find_call_args(program, "divide"):
        if len(args) >= 2:
            denominators.extend(program_numeric_args(args[1]) or [args[1].strip().lower()])
    return denominators


def unsupported_program_ops(program: str) -> List[str]:
    ops = [op.lower() for op in re.findall(r"([A-Za-z_]+)\s*\(", first_text(program))]
    return [op for op in ops if op not in PROGRAM_OPS]


def program_has_symbolic_args(program: str) -> bool:
    allowed_refs = re.compile(r"^#\d+$")
    for _, args in parse_program_steps(program):
        for arg in args:
            if allowed_refs.match(arg):
                continue
            if re.search(r"[A-Za-z_]", arg):
                return True
    return False


def program_has_mixed_infix(program: str) -> bool:
    program = first_text(program).strip()
    if not program:
        return False
    if "/" in program or "*" in program:
        return True
    return False


def program_has_assignment_or_placeholder(program: str) -> bool:
    program = first_text(program)
    if "=" in program:
        return True
    if re.search(r"(?<![A-Za-z0-9])_(?![A-Za-z0-9])", program):
        return True
    return False


def program_has_recoverable_infix(program: str) -> bool:
    text = first_text(program).strip()
    if not text or not re.search(r"[+\-*/]", text):
        return False
    if re.search(r"[A-Za-z_]\s*\(", text):
        return False
    return len(program_numeric_args(text)) >= 2


def is_numeric_literal_program(program: str) -> bool:
    text = first_text(program).strip().replace(",", "")
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%?", text))


def direct_lookup_gold_program(gold_program: str) -> bool:
    return is_numeric_literal_program(gold_program) and not program_ops(gold_program)


def direct_lookup_literal_ok(pred_program: str, gold_program: str) -> bool:
    return direct_lookup_gold_program(gold_program) and is_numeric_literal_program(pred_program)


def direct_lookup_wrapper_error(pred_program: str, gold_program: str) -> bool:
    if not direct_lookup_gold_program(gold_program):
        return False
    if not first_text(pred_program).strip():
        return False
    return not direct_lookup_literal_ok(pred_program, gold_program)


def invalid_operator_get(program: str) -> bool:
    return bool(re.search(r"\bget\s*\(", first_text(program), re.IGNORECASE))


def answer_only_exact_match_score(text: str, gold_answer: str, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> float:
    answer_text = extract_anchor(text, "Answer:") or extract_anchor(text, "Normalized Answer:") or first_text(text)
    for number in program_numeric_args(answer_text):
        if numeric_equal(number, gold_answer, abs_tol, rel_tol):
            return 1.0
    return 0.0


def answer_first_exact_match_score(text: str, gold_answer: str, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> float:
    answer_text = extract_anchor(text, "Normalized Answer:") or extract_anchor(text, "Answer:")
    if not answer_text:
        return 0.0
    for number in program_numeric_args(answer_text):
        if numeric_equal(number, gold_answer, abs_tol, rel_tol):
            return 1.0
    return 0.0


def calculation_shortcut_score(pred_program: str, gold_program: str) -> float:
    if direct_lookup_gold_program(gold_program):
        return 0.0
    if is_numeric_literal_program(pred_program) and program_ops(gold_program):
        return 1.0
    return 0.0


def reasoning_stats(text: str) -> Dict[str, float]:
    reasoning = extract_anchor(text, "Reasoning:")
    lines = non_empty_lines(reasoning)
    return {
        "reasoning_chars": float(len(first_text(reasoning).strip())),
        "reasoning_lines": float(len(lines)),
        "reasoning_number_count": float(len(program_numeric_args(reasoning))),
    }


def reasoning_soft_penalty(text: str, script_args: ScriptArguments) -> tuple[float, float]:
    stats = reasoning_stats(text)
    penalty = 0.0
    too_long = 0.0
    if script_args.reasoning_max_chars > 0 and stats["reasoning_chars"] > script_args.reasoning_max_chars:
        over = (stats["reasoning_chars"] - script_args.reasoning_max_chars) / max(script_args.reasoning_max_chars, 1)
        penalty += script_args.reasoning_penalty_weight * over
        too_long = 1.0
    if script_args.reasoning_max_lines > 0 and stats["reasoning_lines"] > script_args.reasoning_max_lines:
        over_lines = stats["reasoning_lines"] - script_args.reasoning_max_lines
        penalty += script_args.reasoning_penalty_weight * over_lines
        too_long = 1.0
    return min(script_args.reasoning_penalty_max, penalty), too_long


def strict_dsl_invalid_reasons(program: str) -> List[str]:
    program = first_text(program).strip()
    reasons = []
    if not program:
        return reasons
    unsupported = unsupported_program_ops(program)
    if unsupported:
        reasons.append("unsupported_op")
    if program_has_symbolic_args(program):
        reasons.append("symbolic_arg")
    if program_has_mixed_infix(program):
        reasons.append("mixed_infix")
    if program_has_assignment_or_placeholder(program):
        reasons.append("assignment_or_placeholder")
    if invalid_operator_get(program):
        reasons.append("invalid_operator_get")
    if not parse_program_steps(program) and len(program_numeric_args(program)) != 1:
        reasons.append("non_dsl_text")
    return reasons


def dsl_validity_score(program: str) -> float:
    return 0.0 if strict_dsl_invalid_reasons(program) else 1.0


def execute_prediction_program_for_reward(program: str, script_args: ScriptArguments) -> tuple[Any, str, str]:
    try:
        return execute_prediction_program(
            program,
            relaxed_canonicalization=script_args.relaxed_executor_canonicalization,
        )
    except TypeError:
        return execute_prediction_program(program)


def schema_hard_gate_reasons(
    text: str,
    pred_program: str,
    *,
    forbidden_anchor: bool,
    multiple_program: bool,
    post_program_text: bool,
    dsl_reasons: List[str],
    gold_program: str = "",
    recoverable_syntax_soft_gate: bool = False,
    direct_lookup_wrapper_soft_penalty: bool = False,
) -> List[str]:
    reasons: List[str] = []
    recoverable_infix = program_has_recoverable_infix(pred_program)
    if "Evidence:" not in first_text(text):
        reasons.append("missing_evidence")
    if "Program:" not in first_text(text):
        reasons.append("missing_program")
    if forbidden_anchor:
        reasons.append("forbidden_anchor")
    if multiple_program:
        reasons.append("multiple_program")
    if post_program_text and not (recoverable_syntax_soft_gate and recoverable_infix):
        reasons.append("post_program_text")
    if not first_text(pred_program) or first_text(pred_program).strip().upper() == "N/A":
        reasons.append("empty_program")
    if direct_lookup_wrapper_error(pred_program, gold_program) and not direct_lookup_wrapper_soft_penalty:
        reasons.append("direct_lookup_wrapper_error")
    soft_dsl_reasons = {"mixed_infix", "non_dsl_text"} if recoverable_syntax_soft_gate and recoverable_infix else set()
    reasons.extend(reason for reason in dsl_reasons if reason not in soft_dsl_reasons)
    return sorted(set(reasons))


def multiset_f1(pred_items: List[str], gold_items: List[str]) -> float:
    if not pred_items or not gold_items:
        return 0.0
    pred_counter = Counter(pred_items)
    gold_counter = Counter(gold_items)
    overlap = sum(min(pred_counter[item], gold_counter[item]) for item in pred_counter.keys() | gold_counter.keys())
    precision = overlap / max(len(pred_items), 1)
    recall = overlap / max(len(gold_items), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def program_similarity(pred_program: str, gold_program: str) -> float:
    pred_steps = parse_program_steps(pred_program)
    gold_steps = parse_program_steps(gold_program)
    if not pred_steps or not gold_steps:
        return 0.0

    pred_ops = [op for op, _ in pred_steps]
    gold_ops = [op for op, _ in gold_steps]
    op_score = multiset_f1(pred_ops, gold_ops)

    pred_args = [arg for _, args in pred_steps for arg in args]
    gold_args = [arg for _, args in gold_steps for arg in args]
    arg_score = multiset_f1(pred_args, gold_args)

    prefix_match = 0
    for pred_step, gold_step in zip(pred_steps, gold_steps):
        if pred_step[0] != gold_step[0]:
            break
        prefix_match += 1
    prefix_score = prefix_match / max(len(gold_steps), 1)

    step_count_gap = abs(len(pred_steps) - len(gold_steps)) / max(len(gold_steps), 1)
    length_score = max(0.0, 1.0 - min(step_count_gap, 1.0))
    return 0.45 * op_score + 0.25 * arg_score + 0.20 * prefix_score + 0.10 * length_score


def program_step_count_score(pred_program: str, gold_program: str) -> float:
    pred_steps = parse_program_steps(pred_program)
    gold_steps = parse_program_steps(gold_program)
    if not pred_steps or not gold_steps:
        return 0.0
    gap = abs(len(pred_steps) - len(gold_steps))
    return max(0.0, 1.0 - gap / max(len(gold_steps), 1))


def program_argument_coverage(pred_program: str, gold_program: str) -> float:
    pred_steps = parse_program_steps(pred_program)
    gold_steps = parse_program_steps(gold_program)
    pred_args = [arg for _, args in pred_steps for arg in args]
    gold_args = [arg for _, args in gold_steps for arg in args]
    if pred_args or gold_args:
        return multiset_f1(pred_args, gold_args)
    pred_nums = program_numeric_args(pred_program)
    gold_nums = program_numeric_args(gold_program)
    if not pred_nums and not gold_nums:
        return 1.0
    return multiset_f1(pred_nums, gold_nums)


def evidence_program_alignment(evidence: str, pred_program: str) -> float:
    pred_nums = sorted(set(program_numeric_args(pred_program)))
    if not pred_nums:
        return 1.0
    evidence_nums = set(program_numeric_args(evidence))
    if not evidence_nums:
        return 0.0
    covered = sum(1 for num in pred_nums if num in evidence_nums)
    return covered / max(len(pred_nums), 1)


def program_arg_source_score(pred_program: str, prompt_text: str, evidence: str = "") -> float:
    pred_nums = program_numeric_args(pred_program)
    if not pred_nums:
        return 1.0
    source_nums = set(program_numeric_args(f"{prompt_text}\n{evidence}"))
    if not source_nums:
        return 0.0
    covered = sum(1 for num in pred_nums if num in source_nums)
    return covered / max(len(pred_nums), 1)


def program_arg_precision(pred_program: str, gold_program: str) -> float:
    pred_nums = program_numeric_args(pred_program)
    if not pred_nums:
        return 1.0
    gold_nums = Counter(program_numeric_args(gold_program))
    if not gold_nums:
        return 0.0
    overlap = 0
    for num, count in Counter(pred_nums).items():
        overlap += min(count, gold_nums.get(num, 0))
    return overlap / max(len(pred_nums), 1)


def extra_number_penalty(pred_program: str, gold_program: str) -> float:
    pred_nums = Counter(program_numeric_args(pred_program))
    if not pred_nums:
        return 0.0
    gold_nums = Counter(program_numeric_args(gold_program))
    extra = 0
    for num, count in pred_nums.items():
        extra += max(0, count - gold_nums.get(num, 0))
    return min(1.0, extra / max(sum(pred_nums.values()), 1))


def scale_consistency_score(executed_value: Any, gold_answer: str, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> float:
    pred_num = normalize_number(str(executed_value))
    gold_num = normalize_number(gold_answer)
    if pred_num is None or gold_num is None or gold_num == 0:
        return 1.0
    if numeric_equal(str(pred_num), str(gold_num), abs_tol, rel_tol):
        return 1.0
    if math.isclose(pred_num, gold_num * 100.0, rel_tol=1e-3, abs_tol=1e-3):
        return 0.0
    if math.isclose(pred_num * 100.0, gold_num, rel_tol=1e-3, abs_tol=1e-3):
        return 0.0
    return 1.0


RATIO_DIVIDE_QUESTION_TYPES = {"percentage_change", "share_of_total", "ratio", "margin"}


def _metadata_text(metadata: Optional[Dict[str, Any]], key: str) -> str:
    if not isinstance(metadata, dict):
        return ""
    return first_text(metadata.get(key)).strip().lower()


def ratio_divide_required(prompt_text: str, gold_program: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    gold_ops = set(program_ops(gold_program))
    if "divide" not in gold_ops:
        return False
    question_type = _metadata_text(metadata, "question_type")
    answer_scale = _metadata_text(metadata, "answer_scale")
    if question_type in RATIO_DIVIDE_QUESTION_TYPES or answer_scale in {"ratio", "percent"}:
        return True
    prompt = first_text(prompt_text).lower()
    return any(
        phrase in prompt
        for phrase in (
            "percentage change",
            "percent change",
            "change in percentage",
            "growth rate",
            "increase rate",
            "decrease rate",
            "ratio",
            "margin",
            "rate",
            "percentage",
            "percent",
            "share of",
            "of total",
        )
    )


def ratio_divide_requirement_score(
    prompt_text: str,
    pred_program: str,
    gold_program: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    if not ratio_divide_required(prompt_text, gold_program, metadata):
        return 1.0
    pred_ops = set(program_ops(pred_program))
    return 1.0 if "divide" in pred_ops else 0.0


def denominator_grounding_score(
    prompt_text: str,
    pred_program: str,
    gold_program: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    if not ratio_divide_required(prompt_text, gold_program, metadata):
        return 1.0
    gold_denominators = divide_denominator_args(gold_program)
    if not gold_denominators:
        return 1.0
    pred_denominators = divide_denominator_args(pred_program)
    if not pred_denominators:
        return 0.0
    return multiset_f1(pred_denominators, gold_denominators)


def table_argument_grounding_score(pred_program: str, gold_program: str) -> float:
    pred_nums = program_numeric_args(pred_program)
    gold_nums = program_numeric_args(gold_program)
    if not pred_nums and not gold_nums:
        return 1.0
    if not pred_nums or not gold_nums:
        return 0.0
    return multiset_f1(pred_nums, gold_nums)


def lookup_shortcut_score(pred_program: str, gold_program: str) -> float:
    gold_ops = set(program_ops(gold_program))
    pred_ops = set(program_ops(pred_program))
    if direct_lookup_gold_program(gold_program):
        if pred_ops:
            return 0.0
        return program_argument_coverage(pred_program, gold_program)
    if gold_ops and not pred_ops:
        return 0.0
    return 1.0


def operation_prior_score(
    prompt_text: str,
    pred_program: str,
    gold_program: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    prompt = first_text(prompt_text).lower()
    pred_ops = set(program_ops(pred_program))
    gold_ops = set(program_ops(gold_program))

    if ratio_divide_required(prompt_text, gold_program, metadata) and "divide" not in pred_ops:
        return 0.0

    asks_change = any(
        phrase in prompt
        for phrase in (
            "percentage change",
            "percent change",
            "change in percentage",
            "growth rate",
            "increase rate",
            "decrease rate",
        )
    )
    if asks_change:
        return 1.0 if {"subtract", "divide"}.issubset(pred_ops) else 0.0

    asks_ratio = any(phrase in prompt for phrase in ("ratio", "margin", "rate", "percentage", "percent"))
    if asks_ratio and "divide" in gold_ops:
        return 1.0 if "divide" in pred_ops else 0.0

    direct_lookup_gold = direct_lookup_gold_program(gold_program)
    if direct_lookup_gold:
        return 1.0 if not pred_ops and program_argument_coverage(pred_program, gold_program) >= 1.0 else 0.0

    if gold_ops:
        return multiset_f1(list(pred_ops), list(gold_ops))
    return 1.0


def weighted_process_score(script_args: ScriptArguments, result: Dict[str, Any]) -> float:
    weights = {
        "evidence_program_alignment": script_args.evidence_program_alignment_weight,
        "program_arg_precision": script_args.program_arg_precision_weight,
        "operation_prior_score": script_args.operation_prior_weight,
        "dsl_validity_score": script_args.dsl_validity_weight,
        "ratio_divide_requirement_score": script_args.ratio_divide_requirement_weight,
        "denominator_grounding_score": script_args.denominator_grounding_weight,
        "table_argument_grounding_score": script_args.table_argument_grounding_weight,
        "lookup_shortcut_score": script_args.lookup_shortcut_penalty_weight,
    }
    weight_sum = sum(max(0.0, weight) for weight in weights.values())
    if weight_sum <= 0:
        return 0.0
    return sum(max(0.0, weights[name]) * float(result[name]) for name in weights) / weight_sum


def semantic_process_score(result: Dict[str, Any]) -> float:
    base = float(result.get("process_score", 0.0))
    source = float(result.get("program_arg_source_score", 0.0))
    scale = float(result.get("scale_consistency_score", 1.0))
    extra = float(result.get("extra_number_penalty", 0.0))
    strict = float(result.get("dsl_strict_validity_score", result.get("dsl_validity_score", 0.0)))
    ratio_divide = float(result.get("ratio_divide_requirement_score", 1.0))
    denominator = float(result.get("denominator_grounding_score", 1.0))
    table_args = float(result.get("table_argument_grounding_score", 1.0))
    lookup = float(result.get("lookup_shortcut_score", 1.0))
    shortcut = 1.0 - float(result.get("calculation_shortcut", 0.0))
    semantic_gate = min(ratio_divide, denominator, lookup, shortcut)
    return max(
        0.0,
        min(
            1.0,
            0.24 * base
            + 0.14 * source
            + 0.14 * scale
            + 0.16 * (1.0 - extra)
            + 0.12 * ratio_divide
            + 0.10 * denominator
            + 0.06 * table_args
            + 0.04 * lookup,
        )
        * strict
        * semantic_gate,
    )


def evidence_overlap_score(evidence: str, prompt_text: str) -> float:
    ev_nums = set(NUMBER_RE.findall(first_text(evidence).replace(",", "")))
    prompt_nums = set(NUMBER_RE.findall(first_text(prompt_text).replace(",", "")))
    if not ev_nums or not prompt_nums:
        return 0.0
    overlap = len(ev_nums.intersection(prompt_nums))
    return min(overlap, 4) / 4.0


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = first_text(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def metadata_requires_history(metadata: Any, explicit: Any = None) -> bool:
    if explicit is not None:
        return truthy(explicit)
    if isinstance(metadata, dict):
        return truthy(metadata.get("requires_history"))
    return False


def history_grounding_score(evidence: str, prompt_text: str, requires_history: bool) -> float:
    if not requires_history:
        return 0.0
    evidence_text = first_text(evidence).lower()
    prompt = first_text(prompt_text).lower()
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", prompt))
    if years and any(year in evidence_text for year in years):
        return 1.0
    history_terms = (
        "previous",
        "prior",
        "same",
        "that year",
        "this value",
        "previous one",
        "earlier",
        "follow-up",
    )
    return 1.0 if any(term in evidence_text for term in history_terms) else 0.0


def length_shape_score(text: str, target_low: int = 80, target_high: int = 160, hard_cap: int = 220) -> float:
    word_count = len(first_text(text).split())
    if word_count == 0 or word_count > hard_cap:
        return 0.0
    if target_low <= word_count <= target_high:
        return 1.0
    if word_count < target_low:
        return max(0.0, word_count / max(target_low, 1))
    tail = hard_cap - target_high
    return max(0.0, 1.0 - ((word_count - target_high) / max(tail, 1)))


def soft_length_penalty(text: str, target_chars: int, weight: float, max_penalty: float) -> float:
    if target_chars <= 0 or weight <= 0 or max_penalty <= 0:
        return 0.0
    char_count = len(first_text(text))
    if char_count <= target_chars:
        return 0.0
    over_ratio = (char_count - target_chars) / max(target_chars, 1)
    return min(max_penalty, weight * over_ratio)


def percentile(values: List[float], q: float) -> Optional[float]:
    clean = sorted(value for value in values if value == value)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    q = min(max(q, 0.0), 1.0)
    pos = (len(clean) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return clean[lower]
    weight = pos - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def prompt_text_from_any(prompt_value: Any) -> str:
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, list):
        parts = []
        for item in prompt_value:
            if isinstance(item, dict):
                parts.append(first_text(item.get("content")))
            else:
                parts.append(first_text(item))
        return "\n".join(p for p in parts if p)
    if isinstance(prompt_value, dict):
        return first_text(prompt_value.get("content") or prompt_value.get("text") or prompt_value.get("value"))
    return first_text(prompt_value)


def apply_cot_pot_output_format(
    prompt: str,
    *,
    answer_first: bool = False,
    program_first_answer: bool = False,
    strict_program_only: bool = False,
) -> str:
    text = first_text(prompt)
    if not text:
        return text
    if "Output format:" not in text:
        return text
    if strict_program_only:
        output_format = STRICT_PROGRAM_OUTPUT_FORMAT
    elif program_first_answer:
        output_format = PROGRAM_FIRST_ANSWER_COT_POT_OUTPUT_FORMAT
    elif answer_first:
        output_format = ANSWER_FIRST_COT_POT_OUTPUT_FORMAT
    else:
        output_format = V26_COT_POT_OUTPUT_FORMAT
    return re.sub(
        r"Output format:\n.*?(?=\n\nReport context:|\n\nConversation history:|\n\nConversation history questions:|\Z)",
        output_format,
        text,
        count=1,
        flags=re.DOTALL,
    )


def ensure_local_path(path_str: Optional[str], field_name: str) -> Path:
    if not path_str:
        raise ValueError(f"Missing required path for {field_name}")
    path = Path(path_str)
    if path.exists():
        return path
    raise FileNotFoundError(f"{field_name} does not exist: {path}")


def get_checkpoint(training_args: GRPOConfig):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def _normalize_report_to(report_to: Any) -> List[str]:
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [item.strip() for item in report_to.split(",") if item.strip()]
    if isinstance(report_to, (list, tuple, set)):
        return [str(item).strip() for item in report_to if str(item).strip()]
    return [str(report_to).strip()]


def configure_tensorboard_reporting(
    training_args: GRPOConfig,
    script_args: ScriptArguments,
    is_main_process: bool,
    run_label: str,
) -> Optional[Path]:
    logging_dir = (
        Path(script_args.tensorboard_logging_dir)
        if script_args.tensorboard_logging_dir
        else (Path(training_args.logging_dir) if training_args.logging_dir else None)
    )
    if logging_dir is None:
        return None

    logging_dir.mkdir(parents=True, exist_ok=True)
    training_args.logging_dir = str(logging_dir)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    report_to = _normalize_report_to(training_args.report_to)
    if script_args.tensorboard_logging_dir:
        report_to = [item for item in report_to if item.lower() != "none"]
        if not any(item.lower() == "tensorboard" for item in report_to):
            report_to.append("tensorboard")
        training_args.report_to = report_to
    if is_main_process:
        logger.info(
            f"{run_label} TensorBoard reporting: report_to={training_args.report_to}, "
            f"logging_dir={training_args.logging_dir}"
        )
    return logging_dir


def apply_grpo_training_arg_defaults(training_args: GRPOConfig) -> None:
    if getattr(training_args, "multi_objective_aggregation", None) is None:
        training_args.multi_objective_aggregation = "normalize_then_sum"
    if getattr(training_args, "scale_rewards", None) is None:
        training_args.scale_rewards = "none"
    if getattr(training_args, "loss_type", None) is None:
        training_args.loss_type = "dapo"


def validate_grpo_reward_contract(script_args: ScriptArguments, training_args: GRPOConfig) -> None:
    num_generations = int(getattr(training_args, "num_generations", 0) or 0)
    if script_args.bundle_diagnostics_enabled or script_args.bundle_grpo_enabled:
        if script_args.bundle_size < 2:
            raise ValueError(f"bundle_size must be >= 2 when bundle diagnostics or bundle GRPO are enabled; got {script_args.bundle_size}")
        if script_args.bundle_num_generations <= 0:
            raise ValueError("bundle_num_generations must be > 0 when bundle diagnostics or bundle GRPO are enabled")
        if script_args.bundle_num_generations != num_generations:
            raise ValueError(
                "bundle_num_generations must equal num_generations when bundle diagnostics or bundle GRPO are enabled; "
                f"got bundle_num_generations={script_args.bundle_num_generations}, num_generations={num_generations}"
            )
    if script_args.bundle_grpo_enabled:
        q_down = script_args.bundle_grpo_quantile_down
        q_up = script_args.bundle_grpo_quantile_up
        if not (0.0 <= q_down < q_up <= 1.0):
            raise ValueError(
                "bundle_grpo_quantile_down must be >= 0 and < bundle_grpo_quantile_up <= 1; "
                f"got bundle_grpo_quantile_down={q_down}, bundle_grpo_quantile_up={q_up}"
            )
        if script_args.bundle_grpo_cvar_weight < 0.0:
            raise ValueError("bundle_grpo_cvar_weight must be non-negative")
    if not script_args.risk_aware_reward or script_args.risk_group_size <= 1:
        return
    if script_args.risk_group_size != num_generations:
        raise ValueError(
            "risk_group_size must equal num_generations when risk_aware_reward is enabled; "
            f"got risk_group_size={script_args.risk_group_size}, num_generations={num_generations}"
        )


def find_all_linear_names(peft_model, int4: bool = False, int8: bool = False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb

        cls = bnb.nn.Linear4bit if int4 else bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if "lm_head" in name or "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def audit_program_numeric_rows(rows: List[Dict[str, Any]], script_args: ScriptArguments) -> Dict[str, Any]:
    execute_fail = 0
    answer_mismatch = 0
    forbidden_schema = 0
    missing_program = 0
    missing_answer = 0
    bad_reward_profile = 0
    duplicate_prompt = 0
    seen_prompts = set()
    source_counts: Dict[str, int] = {}
    gold_program_step_hist: Dict[str, int] = {}

    for row in rows:
        prompt = first_text(row.get("input_prompt_raw"))
        if prompt in seen_prompts:
            duplicate_prompt += 1
        seen_prompts.add(prompt)

        reward_profile = first_text(row.get("reward_profile"))
        if reward_profile != script_args.reward_profile_expected:
            bad_reward_profile += 1
        source = first_text(row.get("source_dataset")) or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

        gold_answer = first_text(row.get("gold_answer"))
        gold_program = first_text(row.get("gold_program"))
        gold_step_count = len(parse_program_steps(gold_program))
        gold_program_step_hist[str(gold_step_count)] = gold_program_step_hist.get(str(gold_step_count), 0) + 1
        if not gold_answer:
            missing_answer += 1
        if not gold_program:
            missing_program += 1

        executed_value, _, _ = execute_prediction_program(gold_program)
        if executed_value is None:
            execute_fail += 1
        elif not numeric_equal(str(executed_value), gold_answer, script_args.answer_abs_tol, script_args.answer_rel_tol):
            answer_mismatch += 1

        if script_args.strict_schema_required:
            ref = first_text(row.get("reference_response"))
            if any(anchor in ref for anchor in forbidden_anchors_for_reward_mode(script_args.reward_mode)):
                forbidden_schema += 1

    return {
        "rows": len(rows),
        "reward_profile_expected": script_args.reward_profile_expected,
        "source_counts": source_counts,
        "gold_program_step_hist": gold_program_step_hist,
        "program_execute_fail_rows": execute_fail,
        "program_answer_mismatch_rows": answer_mismatch,
        "forbidden_reference_schema_rows": forbidden_schema,
        "missing_program_rows": missing_program,
        "missing_answer_rows": missing_answer,
        "bad_reward_profile_rows": bad_reward_profile,
        "duplicate_prompt_rows": duplicate_prompt,
    }


class ProgramDiagnostics:
    def __init__(self):
        self.pending: Dict[str, List[float]] = {}

    def add(self, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        self.pending.setdefault(name, []).append(float(value))

    def pop_means(self) -> Dict[str, float]:
        metrics = {}
        for name, values in self.pending.items():
            valid = [value for value in values if value == value]
            if valid:
                metrics[name] = sum(valid) / len(valid)
        self.pending.clear()
        return metrics


class ProgramDiagnosticGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        program_diagnostics: Optional[ProgramDiagnostics] = None,
        script_args: Optional[ScriptArguments] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.program_diagnostics = program_diagnostics
        self.script_args = script_args
        self._bundle_grpo_rewards_per_func: Optional[torch.Tensor] = None
        self._bundle_grpo_contexts: List[Dict[str, Any]] = []

    def _get_train_sampler(self, train_dataset=None):
        if self.script_args is not None and self.script_args.bundle_ordered_sampling:
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if self.script_args is not None and self.script_args.bundle_grpo_enabled:
            local_contexts = []
            for example in inputs:
                local_contexts.append(
                    {
                        "source_dataset": example.get("source_dataset", ""),
                        "metadata": example.get("metadata", {}) or {},
                        "requires_history": example.get("requires_history", None),
                    }
                )
            self._bundle_grpo_contexts = gather_object(local_contexts)
            self._bundle_grpo_rewards_per_func = rewards_per_func.detach()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor | Any]:
        output = super()._generate_and_score_completions(inputs)
        if (
            self.script_args is None
            or not self.script_args.bundle_grpo_enabled
            or self._bundle_grpo_rewards_per_func is None
            or not self._bundle_grpo_contexts
        ):
            return output
        rewards = (
            self._bundle_grpo_rewards_per_func
            * self.reward_weights.to(self._bundle_grpo_rewards_per_func.device).unsqueeze(0)
        ).nansum(dim=1)
        base_global_advantages = torch.tensor(self._logs["advantages"], device=rewards.device, dtype=rewards.dtype)
        if base_global_advantages.numel() != rewards.numel():
            return output
        bundle_advantages, bundle_metrics = compute_bundle_mvar_advantages(
            rewards,
            base_global_advantages,
            self._bundle_grpo_contexts,
            self.script_args,
        )
        process_slice = slice(
            self.accelerator.process_index * len(output["advantages"]),
            (self.accelerator.process_index + 1) * len(output["advantages"]),
        )
        output["advantages"] = bundle_advantages[process_slice].to(output["advantages"].device)
        mode = "train" if self.model.training else "eval"
        for name, value in bundle_metrics.items():
            self._metrics[mode][name].append(float(value))
        if self._logs["advantages"]:
            for index, value in enumerate(bundle_advantages.detach().cpu().tolist()):
                if index < len(self._logs["advantages"]):
                    self._logs["advantages"][index] = value
        return output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        if self.program_diagnostics is not None:
            device = self.accelerator.device
            for name, local_mean in sorted(self.program_diagnostics.pop_means().items()):
                gathered = self.accelerator.gather(torch.tensor(local_mean, device=device))
                self._metrics[mode][name].append(gathered.nanmean().item())
        super().log(logs, start_time)


def evaluate_program_completion(
    script_args: ScriptArguments,
    completion: Any,
    gold_ans: str,
    gold_prog: str,
    prompt_text: str = "",
    source_dataset: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    requires_history: Any = None,
) -> Dict[str, Any]:
    text = completion_text(completion)
    pred_program = extract_anchor(text, "Program:")
    evidence = extract_anchor(text, "Evidence:")
    forbidden_anchors = forbidden_anchors_for_reward_mode(script_args.reward_mode)
    forbidden_anchor = any(anchor in text for anchor in forbidden_anchors)
    multiple_program = anchor_count(text, "Program:") > 1
    post_program_text = has_post_program_text(text)
    post_program_text_for_gate = post_program_text
    if script_args.reward_mode in {"program_first_answer_aux", "program_gated_answer_aux"} and (extract_anchor(text, "Answer:") or extract_anchor(text, "Normalized Answer:")):
        post_program_text_for_gate = False
    dsl_score = dsl_validity_score(pred_program)
    dsl_reasons = strict_dsl_invalid_reasons(pred_program)
    reasoning_info = reasoning_stats(text)
    reasoning_penalty, reasoning_too_long = reasoning_soft_penalty(text, script_args)
    schema_gate_reasons = schema_hard_gate_reasons(
        text,
        pred_program,
        forbidden_anchor=forbidden_anchor,
        multiple_program=multiple_program,
        post_program_text=post_program_text_for_gate,
        dsl_reasons=dsl_reasons,
        gold_program=gold_prog,
        recoverable_syntax_soft_gate=script_args.recoverable_syntax_soft_gate,
        direct_lookup_wrapper_soft_penalty=script_args.direct_lookup_wrapper_soft_penalty,
    )
    contract_reasons = prompt_contract_reasons(evidence, pred_program, script_args)
    if script_args.recoverable_syntax_soft_gate and program_has_recoverable_infix(pred_program):
        contract_reasons = [reason for reason in contract_reasons if reason != "program_too_many_lines"]
    hard_gate_reasons = sorted(set(schema_gate_reasons + contract_reasons))
    result: Dict[str, Any] = {
        "text": text,
        "program": pred_program,
        "core_score": script_args.invalid_program_reward if script_args.outcome_first_reward else HARD_INVALID_REWARD,
        "executable": 0.0,
        "exact_match": 0.0,
        "invalid": 1.0,
        "wrong_executable": 0.0,
        "structure_score": 0.0,
        "argument_coverage": 0.0,
        "step_count_score": 0.0,
        "completion_words": float(len(text.split())),
        "completion_chars": float(len(text)),
        "evidence_bullet_count": float(evidence_bullet_count(evidence)),
        "evidence_chars": float(len(first_text(evidence).strip())),
        "program_lines": float(len(non_empty_lines(pred_program))),
        "program_chars": float(len(first_text(pred_program).strip())),
        "length_penalty": 0.0,
        "reasoning_penalty": reasoning_penalty,
        "reasoning_too_long": reasoning_too_long,
        "reasoning_chars": reasoning_info["reasoning_chars"],
        "reasoning_lines": reasoning_info["reasoning_lines"],
        "reasoning_number_count": reasoning_info["reasoning_number_count"],
        "forbidden_anchor": 1.0 if forbidden_anchor else 0.0,
        "multiple_program": 1.0 if multiple_program else 0.0,
        "post_program_text": 1.0 if post_program_text else 0.0,
        "schema_hard_gate_violation": 1.0 if schema_gate_reasons else 0.0,
        "prompt_contract_violation": 1.0 if contract_reasons else 0.0,
        "answer_only_exact_match": answer_only_exact_match_score(text, gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol),
        "answer_first_exact_match": answer_first_exact_match_score(text, gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol),
        "answer_anchor_coverage": 1.0 if (extract_anchor(text, "Answer:") or extract_anchor(text, "Normalized Answer:")) else 0.0,
        "direct_lookup_literal_ok": 1.0 if direct_lookup_literal_ok(pred_program, gold_prog) else 0.0,
        "direct_lookup_wrapper_error": 1.0 if direct_lookup_wrapper_error(pred_program, gold_prog) else 0.0,
        "invalid_operator_get": 1.0 if invalid_operator_get(pred_program) else 0.0,
        "assignment_or_placeholder": 1.0 if program_has_assignment_or_placeholder(pred_program) else 0.0,
        "evidence_program_alignment": 0.0,
        "program_arg_precision": 0.0,
        "operation_prior_score": 0.0,
        "ratio_divide_requirement_score": 1.0,
        "denominator_grounding_score": 1.0,
        "table_argument_grounding_score": 1.0,
        "lookup_shortcut_score": 1.0,
        "dsl_validity_score": dsl_score,
        "dsl_strict_validity_score": dsl_score,
        "symbolic_variable_rate": 1.0 if "symbolic_arg" in dsl_reasons else 0.0,
        "mixed_infix_rate": 1.0 if "mixed_infix" in dsl_reasons else 0.0,
        "recoverable_infix_rate": 1.0 if program_has_recoverable_infix(pred_program) else 0.0,
        "direct_lookup_gold": 1.0 if direct_lookup_gold_program(gold_prog) else 0.0,
        "calculation_shortcut": calculation_shortcut_score(pred_program, gold_prog),
        "program_arg_source_score": 0.0,
        "extra_number_penalty": 0.0,
        "scale_consistency_score": 1.0,
        "semantic_process_score": 0.0,
        "process_score": 0.0,
        "process_adjustment": 0.0,
        "answer_correctness_reward": 0.0,
        "program_execution_reward": 0.0,
        "program_answer_consistency_reward": 0.0,
        "operation_match_reward": 0.0,
        "argument_grounding_reward": 0.0,
        "denominator_grounding_reward": 0.0,
        "table_argument_grounding_reward": 0.0,
        "lookup_shortcut_reward": 0.0,
        "scale_consistency_reward": 0.0,
        "evidence_grounding_reward": 0.0,
        "history_grounding_reward": 0.0,
        "structure_reward": 0.0,
        "answer_shortcut_rate": 0.0,
        "answer_correct_program_wrong_rate": 0.0,
        "program_correct_answer_missing_rate": 0.0,
        "program_correct_answer_inconsistent_rate": 0.0,
        "program_correct_answer_consistent_rate": 0.0,
    }
    if script_args.reward_mode == "program_gated_answer_aux" and result["answer_first_exact_match"] > 0.0:
        result["answer_shortcut_rate"] = 1.0
        result["answer_correct_program_wrong_rate"] = 1.0
    if script_args.reward_mode == "answer_only":
        score = result["answer_only_exact_match"]
        result.update(
            {
                "core_score": score,
                "exact_match": score,
                "invalid": 0.0 if score > 0.0 else result["invalid"],
                "answer_correctness_reward": score,
            }
        )
        return result
    if script_args.reward_mode == "answer_first":
        answer_score = result["answer_first_exact_match"]
        result["answer_correctness_reward"] = answer_score
        if answer_score <= 0.0:
            result.update({"core_score": 0.0, "exact_match": 0.0, "invalid": 0.0})
            return result

        executed_value, _, _ = execute_prediction_program_for_reward(pred_program, script_args) if pred_program else (None, "", "missing_program")
        program_correct = 0.0
        program_consistent = 0.0
        if executed_value is not None:
            program_correct = 1.0 if numeric_equal(str(executed_value), gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol) else 0.0
            answer_text = extract_anchor(text, "Normalized Answer:") or extract_anchor(text, "Answer:")
            answer_num = normalize_number(answer_text)
            if answer_num is not None:
                program_consistent = 1.0 if numeric_equal(str(executed_value), str(answer_num), script_args.answer_abs_tol, script_args.answer_rel_tol) else 0.0
            result["executable"] = 1.0
            result["program_execution_reward"] = program_correct
            result["scale_consistency_score"] = scale_consistency_score(executed_value, gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol)
        result["exact_match"] = answer_score
        result["invalid"] = 0.0

        schema_penalty = min(script_args.schema_soft_penalty_max, script_args.schema_soft_penalty_weight * len(schema_gate_reasons))
        contract_penalty = min(script_args.contract_soft_penalty_max, script_args.contract_soft_penalty_weight * len(contract_reasons))
        dsl_penalty = 0.0 if dsl_score > 0.0 else 0.12
        length_penalty = soft_length_penalty(
            text,
            script_args.length_penalty_chars,
            script_args.length_penalty_weight,
            script_args.length_penalty_max,
        )
        core_score = 1.0 + 0.15 * program_correct + 0.05 * program_consistent
        core_score = max(0.55, core_score - schema_penalty - contract_penalty - dsl_penalty - length_penalty - reasoning_penalty)
        result.update(
            {
                "core_score": core_score,
                "length_penalty": length_penalty,
                "reasoning_penalty": reasoning_penalty,
                "wrong_executable": 0.0 if program_correct > 0.0 or executed_value is None else 1.0,
            }
        )
        return result
    if (
        script_args.schema_hard_gate
        and hard_gate_reasons
        and not script_args.outcome_first_reward
        and script_args.reward_mode != "frontier_execution_calibration"
    ):
        return result
    if not pred_program or pred_program.strip().upper() == "N/A":
        return result

    executed_value, executed_program, _ = execute_prediction_program_for_reward(pred_program, script_args)
    if executed_value is None:
        return result
    if dsl_score <= 0.0 and not (script_args.recoverable_syntax_soft_gate and executed_program):
        return result

    effective_program = executed_program or pred_program
    closeness = numeric_closeness(str(executed_value), gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol)
    structure = program_similarity(effective_program, gold_prog)
    argument_coverage = program_argument_coverage(effective_program, gold_prog)
    step_count = program_step_count_score(effective_program, gold_prog)
    exact_match = numeric_equal(str(executed_value), gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol)
    evidence_alignment = evidence_program_alignment(evidence, effective_program)
    arg_precision = program_arg_precision(effective_program, gold_prog)
    op_prior = operation_prior_score(prompt_text, effective_program, gold_prog, metadata)
    ratio_divide_score = ratio_divide_requirement_score(prompt_text, effective_program, gold_prog, metadata)
    denominator_score = denominator_grounding_score(prompt_text, effective_program, gold_prog, metadata)
    table_arg_score = table_argument_grounding_score(effective_program, gold_prog)
    lookup_score = lookup_shortcut_score(effective_program, gold_prog)
    arg_source = program_arg_source_score(effective_program, prompt_text, evidence)
    extra_penalty = extra_number_penalty(effective_program, gold_prog)
    scale_score = scale_consistency_score(executed_value, gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol)
    conv_history_enabled = (
        first_text(source_dataset) == "convfinqa_turn"
        and metadata_requires_history(metadata, requires_history)
    )
    history_score = history_grounding_score(evidence, prompt_text, conv_history_enabled)
    explicit_answer_text = extract_anchor(text, "Normalized Answer:") or extract_anchor(text, "Answer:")
    explicit_answer_num = normalize_number(explicit_answer_text)
    answer_aux_score = result["answer_first_exact_match"] if script_args.reward_mode in {"program_first_answer_aux", "program_gated_answer_aux"} else 0.0
    program_answer_consistency = 0.0
    if script_args.reward_mode in {"program_first_answer_aux", "program_gated_answer_aux"} and explicit_answer_num is not None:
        program_answer_consistency = 1.0 if numeric_equal(str(executed_value), str(explicit_answer_num), script_args.answer_abs_tol, script_args.answer_rel_tol) else 0.0

    if script_args.reward_mode == "frontier_execution_calibration":
        contract_violation = bool(hard_gate_reasons or contract_reasons or dsl_score <= 0.0)
        if exact_match:
            core_score = 0.25 if contract_violation else 1.0
        else:
            core_score = -0.3 if contract_violation else -0.1
        result.update(
            {
                "core_score": core_score,
                "executable": 1.0,
                "exact_match": 1.0 if exact_match else 0.0,
                "invalid": 0.0,
                "wrong_executable": 0.0 if exact_match else 1.0,
                "program_execution_reward": 1.0 if exact_match else 0.0,
                "answer_correctness_reward": 1.0 if exact_match else 0.0,
                "process_score": 0.0,
                "semantic_process_score": 0.0,
                "process_adjustment": 0.0,
                "operation_match_reward": 0.0,
                "argument_grounding_reward": 0.0,
                "denominator_grounding_reward": 0.0,
                "table_argument_grounding_reward": 0.0,
                "lookup_shortcut_reward": 0.0,
                "scale_consistency_reward": 0.0,
                "evidence_grounding_reward": 0.0,
                "history_grounding_reward": 0.0,
            }
        )
        return result

    if script_args.outcome_first_reward or script_args.dense_reward_shaping:
        schema_soft_penalty = min(
            script_args.schema_soft_penalty_max,
            script_args.schema_soft_penalty_weight * len(schema_gate_reasons),
        )
        contract_soft_penalty = min(
            script_args.contract_soft_penalty_max,
            script_args.contract_soft_penalty_weight * len(contract_reasons),
        )
        format_penalty = schema_soft_penalty + contract_soft_penalty
        if script_args.dense_reward_shaping:
            format_penalty += 0.04 if post_program_text_for_gate else 0.0
            format_penalty += 0.06 if program_has_recoverable_infix(pred_program) else 0.0
            format_penalty += 0.06 if direct_lookup_wrapper_error(pred_program, gold_prog) else 0.0
            format_penalty += 0.04 * (1.0 - min(scale_score, 1.0))
    else:
        format_penalty = 0.0
        format_penalty += 0.05 if forbidden_anchor else 0.0
        format_penalty += 0.05 if multiple_program else 0.0
        format_penalty += 0.05 if post_program_text else 0.0
    length_penalty = soft_length_penalty(
        text,
        script_args.length_penalty_chars,
        script_args.length_penalty_weight,
        script_args.length_penalty_max,
    )
    total_length_penalty = length_penalty + reasoning_penalty
    brevity_bonus = script_args.brevity_reward_weight * length_shape_score(text, target_low=40, target_high=120, hard_cap=160)
    result.update(
        {
            "evidence_program_alignment": evidence_alignment,
            "program_arg_precision": arg_precision,
            "operation_prior_score": op_prior,
            "ratio_divide_requirement_score": ratio_divide_score,
            "denominator_grounding_score": denominator_score,
            "table_argument_grounding_score": table_arg_score,
            "lookup_shortcut_score": lookup_score,
            "dsl_validity_score": dsl_score,
            "dsl_strict_validity_score": dsl_score,
            "program_arg_source_score": arg_source,
            "extra_number_penalty": extra_penalty,
            "scale_consistency_score": scale_score,
            "direct_lookup_gold": 1.0 if direct_lookup_gold_program(gold_prog) else 0.0,
            "calculation_shortcut": calculation_shortcut_score(pred_program, gold_prog),
        }
    )
    process_score = weighted_process_score(script_args, result) if script_args.process_reward_enabled else 0.0
    result["process_score"] = process_score
    result["semantic_process_score"] = semantic_process_score(result)
    process_adjustment = script_args.process_reward_weight * process_score if script_args.process_reward_enabled else 0.0
    if exact_match:
        core_score = CORRECT_EXECUTABLE_BASE_REWARD
        if script_args.outcome_first_reward:
            core_score += 0.03 * structure
            core_score += 0.02 * argument_coverage
            core_score += 0.01 * step_count
        else:
            core_score += 0.05 * structure
            core_score += 0.03 * argument_coverage
            core_score += 0.02 * step_count
        core_score += brevity_bonus
        core_score += process_adjustment
        if script_args.reward_mode == "program_first_answer_aux":
            core_score += 0.06 * answer_aux_score + 0.04 * program_answer_consistency
            if explicit_answer_num is None:
                core_score -= 0.05
            elif program_answer_consistency <= 0.0:
                core_score -= 0.15
        elif script_args.reward_mode == "program_gated_answer_aux":
            if explicit_answer_num is None:
                core_score -= 0.04
            elif program_answer_consistency <= 0.0:
                core_score = min(core_score, 0.25)
            else:
                core_score += 0.03 * answer_aux_score + 0.05 * program_answer_consistency
        core_score = max(script_args.correct_format_penalty_floor, core_score - format_penalty - total_length_penalty)
        if (
            script_args.reward_mode == "program_gated_answer_aux"
            and explicit_answer_num is not None
            and program_answer_consistency <= 0.0
        ):
            core_score = min(core_score, 0.25)
        if script_args.dense_reward_shaping and (
            program_has_recoverable_infix(pred_program)
            or post_program_text_for_gate
            or direct_lookup_wrapper_error(pred_program, gold_prog)
        ):
            core_score = min(core_score, CORRECT_EXECUTABLE_BASE_REWARD - min(0.35, max(0.02, format_penalty)))
        if conv_history_enabled and history_score < 1.0:
            history_penalty = script_args.history_grounding_reward_weight * (1.0 - history_score)
            core_score = min(core_score, CORRECT_EXECUTABLE_BASE_REWARD - history_penalty)
    else:
        core_score = script_args.wrong_executable_reward_cap * numeric_closeness(
            str(executed_value), gold_ans, script_args.answer_abs_tol, script_args.answer_rel_tol
        )
        process_adjustment = -process_adjustment
        semantic_penalty = script_args.risk_semantic_penalty_weight * (1.0 - result["semantic_process_score"])
        core_score += process_adjustment
        if script_args.reward_mode == "program_first_answer_aux":
            core_score += 0.01 * answer_aux_score
        core_score = min(
            script_args.wrong_executable_reward_cap,
            max(0.0, core_score - semantic_penalty - format_penalty - total_length_penalty),
        )

    result.update(
        {
            "core_score": core_score,
            "executable": 1.0,
            "exact_match": 1.0 if exact_match else 0.0,
            "invalid": 0.0,
            "wrong_executable": 0.0 if exact_match else 1.0,
            "structure_score": structure,
            "argument_coverage": argument_coverage,
            "step_count_score": step_count,
            "length_penalty": length_penalty,
            "reasoning_penalty": reasoning_penalty,
            "reasoning_too_long": reasoning_too_long,
            "reasoning_chars": reasoning_info["reasoning_chars"],
            "reasoning_lines": reasoning_info["reasoning_lines"],
            "reasoning_number_count": reasoning_info["reasoning_number_count"],
            "process_score": process_score,
            "semantic_process_score": result["semantic_process_score"],
            "process_adjustment": process_adjustment,
            "answer_correctness_reward": (answer_aux_score if script_args.reward_mode == "program_first_answer_aux" else (answer_aux_score if script_args.reward_mode == "program_gated_answer_aux" and exact_match and program_answer_consistency > 0.0 else (1.0 if exact_match and script_args.reward_mode != "program_gated_answer_aux" else 0.0))),
            "program_execution_reward": 1.0 if exact_match else 0.0,
            "program_answer_consistency_reward": program_answer_consistency,
            "operation_match_reward": op_prior,
            "argument_grounding_reward": arg_source,
            "denominator_grounding_reward": denominator_score,
            "table_argument_grounding_reward": table_arg_score,
            "lookup_shortcut_reward": lookup_score,
            "scale_consistency_reward": scale_score,
            "evidence_grounding_reward": evidence_alignment,
            "history_grounding_reward": history_score,
            "structure_reward": structure,
            "answer_shortcut_rate": 1.0 if script_args.reward_mode == "program_gated_answer_aux" and answer_aux_score > 0.0 and (not exact_match or program_answer_consistency <= 0.0) else 0.0,
            "answer_correct_program_wrong_rate": 1.0 if script_args.reward_mode == "program_gated_answer_aux" and answer_aux_score > 0.0 and not exact_match else 0.0,
            "program_correct_answer_missing_rate": 1.0 if script_args.reward_mode == "program_gated_answer_aux" and exact_match and explicit_answer_num is None else 0.0,
            "program_correct_answer_inconsistent_rate": 1.0 if script_args.reward_mode == "program_gated_answer_aux" and exact_match and explicit_answer_num is not None and program_answer_consistency <= 0.0 else 0.0,
            "program_correct_answer_consistent_rate": 1.0 if script_args.reward_mode == "program_gated_answer_aux" and exact_match and program_answer_consistency > 0.0 else 0.0,
        }
    )
    return result


def bundle_metadata_value(metadata: Any, key: str) -> str:
    if not isinstance(metadata, dict):
        return "unknown"
    value = metadata.get(key)
    text_value = first_text(value).strip().lower()
    return text_value or "unknown"


def bundle_bucket_key(source: Any, metadata: Any, requires_history: Any, script_args: ScriptArguments) -> tuple:
    if not script_args.bundle_bucket_by_source:
        return ("all",)
    history_value = metadata_requires_history(metadata, requires_history)
    return (
        first_text(source).strip().lower() or "unknown_source",
        "history" if history_value else "single_turn",
        bundle_metadata_value(metadata, "question_type"),
        bundle_metadata_value(metadata, "operation_type"),
        bundle_metadata_value(metadata, "answer_scale"),
        bundle_metadata_value(metadata, "difficulty_bucket"),
        bundle_metadata_value(metadata, "error_bucket"),
    )


def _list_value(values: Any, index: int, default: Any = None) -> Any:
    if isinstance(values, list):
        return values[index] if index < len(values) else default
    return default


def record_bundle_diagnostics(
    diagnostic_results: List[Dict[str, Any]],
    source_dataset: List[Any],
    metadata: List[Any],
    requires_history: List[Any],
    script_args: ScriptArguments,
    diagnostics: Optional[ProgramDiagnostics],
) -> None:
    if diagnostics is None or not script_args.bundle_diagnostics_enabled:
        return
    group_size = script_args.bundle_num_generations
    bundle_size = script_args.bundle_size
    if group_size <= 0 or bundle_size <= 1 or not diagnostic_results:
        return

    prompt_groups = []
    for start in range(0, len(diagnostic_results), group_size):
        end = start + group_size
        if end > len(diagnostic_results):
            break
        group_results = diagnostic_results[start:end]
        prompt_groups.append(
            {
                "results": group_results,
                "source": _list_value(source_dataset, start, ""),
                "metadata": _list_value(metadata, start, {}),
                "requires_history": _list_value(requires_history, start, None),
            }
        )
    if not prompt_groups:
        diagnostics.add("program/bundle_count", 0.0)
        diagnostics.add("program/bundle_skipped_prompt_rate", 0.0)
        return

    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for group in prompt_groups:
        buckets[bundle_bucket_key(group["source"], group["metadata"], group["requires_history"], script_args)].append(group)

    bundle_scores: List[float] = []
    skipped_prompts = 0
    for groups in buckets.values():
        complete_count = (len(groups) // bundle_size) * bundle_size
        skipped_prompts += len(groups) - complete_count
        for start in range(0, complete_count, bundle_size):
            bundle_groups = groups[start : start + bundle_size]
            for generation_index in range(group_size):
                score = sum(
                    float(group["results"][generation_index].get("exact_match", 0.0))
                    for group in bundle_groups
                )
                bundle_scores.append(score)

    skipped_rate = skipped_prompts / len(prompt_groups) if prompt_groups else 0.0
    diagnostics.add("program/bundle_skipped_prompt_rate", skipped_rate)
    diagnostics.add("program/bundle_count", len(bundle_scores) / group_size if group_size else 0.0)
    if not bundle_scores:
        return

    mean_score = sum(bundle_scores) / len(bundle_scores)
    variance = sum((score - mean_score) ** 2 for score in bundle_scores) / len(bundle_scores)
    q20 = percentile(bundle_scores, 0.20)
    q80 = percentile(bundle_scores, 0.80)
    left_tail_scores = [score for score in bundle_scores if q20 is not None and score <= q20]
    left_tail_mean = sum(left_tail_scores) / len(left_tail_scores) if left_tail_scores else q20

    diagnostics.add("program/bundle_score_mean", mean_score)
    diagnostics.add("program/bundle_score_std", math.sqrt(variance))
    diagnostics.add("program/bundle_score_q20", q20)
    diagnostics.add("program/bundle_score_q80", q80)
    diagnostics.add("program/bundle_left_tail_score_mean", left_tail_mean)
    diagnostics.add("program/bundle_all_wrong_rate", sum(1.0 for score in bundle_scores if score <= 0.0) / len(bundle_scores))
    diagnostics.add("program/bundle_all_correct_rate", sum(1.0 for score in bundle_scores if score >= bundle_size) / len(bundle_scores))
    diagnostics.add("program/bundle_exact_rate", mean_score / bundle_size)



def _safe_tensor_std(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.ones((), device=values.device, dtype=values.dtype)
    std = values.std(unbiased=False)
    if torch.isnan(std) or std <= 1e-6:
        return torch.ones((), device=values.device, dtype=values.dtype)
    return std


def compute_bundle_mvar_advantages(
    rewards: torch.Tensor,
    base_advantages: torch.Tensor,
    contexts: List[Dict[str, Any]],
    script_args: ScriptArguments,
) -> tuple[torch.Tensor, Dict[str, float]]:
    group_size = script_args.bundle_num_generations
    bundle_size = script_args.bundle_size
    advantages = base_advantages.clone()
    metrics: Dict[str, float] = {
        "program/bundle_grpo_replaced_rate": 0.0,
        "program/bundle_grpo_fallback_rate": 1.0,
    }
    if not script_args.bundle_grpo_enabled or group_size <= 0 or bundle_size <= 1 or rewards.numel() == 0:
        return advantages, metrics

    total_prompt_groups = rewards.numel() // group_size
    if total_prompt_groups <= 0:
        return advantages, metrics
    usable = total_prompt_groups * group_size
    rewards = rewards[:usable]
    advantages = advantages[:usable].clone()
    contexts = contexts[:usable]

    prompt_groups = []
    for start in range(0, usable, group_size):
        context = contexts[start] if start < len(contexts) else {}
        prompt_groups.append(
            {
                "indices": list(range(start, start + group_size)),
                "source": context.get("source_dataset", ""),
                "metadata": context.get("metadata", {}) or {},
                "requires_history": context.get("requires_history", None),
            }
        )

    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for group in prompt_groups:
        buckets[bundle_bucket_key(group["source"], group["metadata"], group["requires_history"], script_args)].append(group)

    replaced = torch.zeros_like(rewards, dtype=torch.bool)
    paper_advantages: List[float] = []
    fallback_prompts = 0
    q_down_level = script_args.bundle_grpo_quantile_down
    q_up_level = script_args.bundle_grpo_quantile_up
    eps = 1e-6

    for groups in buckets.values():
        complete_count = (len(groups) // bundle_size) * bundle_size
        fallback_prompts += len(groups) - complete_count
        for bundle_start in range(0, complete_count, bundle_size):
            bundle_groups = groups[bundle_start : bundle_start + bundle_size]
            paper_scores = []
            paper_indices = []
            for generation_index in range(group_size):
                indices = [group["indices"][generation_index] for group in bundle_groups]
                paper_indices.append(indices)
                paper_scores.append(rewards[indices].sum())
            scores = torch.stack(paper_scores)
            std = scores.std(unbiased=False) if scores.numel() > 1 else torch.zeros((), device=scores.device, dtype=scores.dtype)
            if std <= eps or torch.isnan(std):
                fallback_prompts += len(bundle_groups)
                continue
            q_down = torch.quantile(scores.float(), q_down_level).to(scores.dtype)
            q_up = torch.quantile(scores.float(), q_up_level).to(scores.dtype)
            for score, indices in zip(scores, paper_indices):
                rvar_adv = (torch.clamp(score - q_down, min=0.0) - torch.clamp(score - q_up, min=0.0) + q_down - q_up)
                rvar_adv = rvar_adv / (q_up_level - q_down_level + eps)
                cvar_adv = -torch.clamp(q_down - score, min=0.0) / (q_down_level + eps)
                adv = (script_args.bundle_grpo_cvar_weight * cvar_adv + rvar_adv) / (script_args.bundle_grpo_cvar_weight + 1.0)
                if script_args.bundle_grpo_normalize_by_bundle_std:
                    adv = adv / (_safe_tensor_std(scores) + eps)
                for index in indices:
                    advantages[index] = adv
                    replaced[index] = True
                paper_advantages.append(float(adv.detach().cpu()))

    if not script_args.bundle_grpo_fallback_to_prompt_advantage:
        advantages[~replaced] = 0.0
    replaced_rate = replaced.float().mean().item() if replaced.numel() else 0.0
    metrics["program/bundle_grpo_replaced_rate"] = replaced_rate
    metrics["program/bundle_grpo_fallback_rate"] = 1.0 - replaced_rate
    if paper_advantages:
        mean_adv = sum(paper_advantages) / len(paper_advantages)
        var_adv = sum((value - mean_adv) ** 2 for value in paper_advantages) / len(paper_advantages)
        metrics["program/bundle_grpo_advantage_mean"] = mean_adv
        metrics["program/bundle_grpo_advantage_std"] = math.sqrt(var_adv)
    return advantages, metrics


def apply_group_risk_shaping(
    rewards: List[float],
    diagnostic_results: List[Dict[str, Any]],
    script_args: ScriptArguments,
    diagnostics: Optional[ProgramDiagnostics] = None,
) -> List[float]:
    group_size = script_args.risk_group_size
    if not script_args.risk_aware_reward or group_size <= 1:
        return rewards
    shaped = list(rewards)
    for start in range(0, len(shaped), group_size):
        end = min(start + group_size, len(shaped))
        if end - start <= 1:
            continue
        group_rewards = rewards[start:end]
        group_results = diagnostic_results[start:end]
        q25 = percentile(group_rewards, 0.25)
        reward_min = min(group_rewards)
        reward_mean = sum(group_rewards) / len(group_rewards)
        variance = sum((value - reward_mean) ** 2 for value in group_rewards) / len(group_rewards)
        invalid_rate = sum(result["invalid"] for result in group_results) / len(group_results)
        exact_rate = sum(result["exact_match"] for result in group_results) / len(group_results)
        exact_count = sum(1 for result in group_results if result["exact_match"] >= 1.0)
        wrong_executable_rate = sum(result["wrong_executable"] for result in group_results) / len(group_results)
        process_q25 = percentile([result.get("semantic_process_score", result.get("process_score", 0.0)) for result in group_results], 0.25)
        process_mean = sum(result.get("semantic_process_score", result.get("process_score", 0.0)) for result in group_results) / len(group_results)
        dsl_invalid_rate = sum(1.0 - result.get("dsl_strict_validity_score", result.get("dsl_validity_score", 0.0)) for result in group_results) / len(group_results)
        semantic_wrong_rate = sum(
            1.0
            for result in group_results
            if result["wrong_executable"] >= 1.0
            and result.get("semantic_process_score", result.get("process_score", 0.0)) < script_args.risk_process_q25_floor
        ) / len(group_results)
        scale_error_rate = sum(1.0 - result.get("scale_consistency_score", 1.0) for result in group_results) / len(group_results)
        extra_number_rate = sum(1.0 for result in group_results if result.get("extra_number_penalty", 0.0) > 0.0) / len(group_results)
        risky_group = (
            (q25 is not None and q25 < script_args.risk_q25_floor)
            or (q25 is not None and q25 <= script_args.wrong_executable_reward_cap)
            or invalid_rate > script_args.risk_invalid_threshold
            or wrong_executable_rate >= script_args.risk_wrong_executable_threshold
            or (process_q25 is not None and process_q25 < script_args.risk_process_q25_floor)
            or dsl_invalid_rate > 0.0
        )
        if diagnostics is not None:
            diagnostics.add("program/group_reward_min", reward_min)
            diagnostics.add("program/group_reward_q25", q25)
            diagnostics.add("program/group_reward_mean", reward_mean)
            diagnostics.add("program/group_reward_std", math.sqrt(variance))
            diagnostics.add("program/group_invalid_rate", invalid_rate)
            diagnostics.add("program/group_exact_match_rate", exact_rate)
            diagnostics.add("program/group_wrong_executable_rate", wrong_executable_rate)
            diagnostics.add("program/group_all_wrong_rate", 1.0 if exact_count == 0 else 0.0)
            diagnostics.add("program/group_single_correct_rate", 1.0 if exact_count == 1 else 0.0)
            diagnostics.add("program/group_process_q25", process_q25)
            diagnostics.add("program/group_process_mean", process_mean)
            diagnostics.add("program/group_dsl_invalid_rate", dsl_invalid_rate)
            diagnostics.add("program/group_semantic_wrong_rate", semantic_wrong_rate)
            diagnostics.add("program/group_scale_error_rate", scale_error_rate)
            diagnostics.add("program/group_extra_number_rate", extra_number_rate)
            diagnostics.add("program/group_risky_rate", 1.0 if risky_group else 0.0)
        if not risky_group:
            continue
        if exact_count == 0 and script_args.risk_skip_all_wrong_groups:
            for index in range(start, end):
                shaped[index] = reward_mean
            continue
        for index in range(start, end):
            if diagnostic_results[index]["exact_match"] >= 1.0:
                if script_args.risk_exact_bonus_weight > 0:
                    shaped[index] = shaped[index] + script_args.risk_exact_bonus_weight
                if exact_count == 1 and script_args.risk_single_correct_advantage_clip > 0:
                    wrong_rewards = [
                        shaped[other_index]
                        for other_index in range(start, end)
                        if diagnostic_results[other_index]["exact_match"] < 1.0
                    ]
                    if wrong_rewards:
                        shaped[index] = min(
                            shaped[index],
                            max(wrong_rewards) + script_args.risk_single_correct_advantage_clip,
                        )
                continue
            semantic_gap = 1.0 - diagnostic_results[index].get(
                "semantic_process_score", diagnostic_results[index].get("process_score", 0.0)
            )
            penalty = script_args.risk_penalty_weight
            if diagnostic_results[index]["wrong_executable"] >= 1.0:
                penalty += script_args.risk_semantic_penalty_weight * max(0.0, semantic_gap)
            floor = HARD_INVALID_REWARD if diagnostic_results[index]["invalid"] >= 1.0 else 0.0
            shaped[index] = max(floor, shaped[index] - penalty)
    return shaped


def controlled_smoke_reward_log_path() -> str:
    return os.environ.get("V34R23_CONTROLLED_SMOKE_REWARD_LOG", "").strip()


def controlled_smoke_assert_groups_enabled() -> bool:
    return os.environ.get("V34R23_ASSERT_REWARD_GROUPS", "").strip().lower() in {"1", "true", "yes", "y", "on"}


def controlled_smoke_group_summary(
    rewards: List[float],
    completions: List[Any],
    diagnostic_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    values = [float(value) for value in rewards]
    mean = sum(values) / max(len(values), 1)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1) if values else 0.0
    texts = [completion_text(item) for item in completions]
    programs = [first_text(result.get("program")) for result in diagnostic_results if first_text(result.get("program"))]
    correct = sum(1 for result in diagnostic_results if float(result.get("exact_match", 0.0)) >= 1.0)
    executable = sum(1 for result in diagnostic_results if float(result.get("executable", 0.0)) >= 1.0)
    invalid = sum(1 for result in diagnostic_results if float(result.get("invalid", 0.0)) >= 1.0)
    wrong_executable = sum(1 for result in diagnostic_results if float(result.get("wrong_executable", 0.0)) >= 1.0)
    reward_std = math.sqrt(variance)
    return {
        "rewards": values,
        "reward_mean": mean,
        "reward_std": reward_std,
        "zero_std": reward_std <= 1e-8,
        "all_correct": bool(diagnostic_results) and correct == len(diagnostic_results),
        "all_wrong": bool(diagnostic_results) and correct == 0,
        "mixed_reward": len(set(values)) > 1 and correct > 0 and correct < len(diagnostic_results),
        "sampled_correct_rate": correct / max(len(diagnostic_results), 1),
        "sampled_executable_rate": executable / max(len(diagnostic_results), 1),
        "invalid_rate": invalid / max(len(diagnostic_results), 1),
        "wrong_executable_rate": wrong_executable / max(len(diagnostic_results), 1),
        "unique_program_ratio": len(set(programs)) / max(len(programs), 1),
        "reasoning_marker_rate": sum(1 for text in texts if "Reasoning:" in text) / max(len(texts), 1),
        "answer_marker_rate": sum(1 for text in texts if "Answer:" in text) / max(len(texts), 1),
        "normalized_answer_marker_rate": sum(1 for text in texts if "Normalized Answer:" in text) / max(len(texts), 1),
        "multiple_program_rate": sum(1 for text in texts if text.count("Program:") != 1) / max(len(texts), 1),
    }


def maybe_log_controlled_smoke_rewards(
    completions: List[Any],
    rewards: List[float],
    diagnostic_results: List[Dict[str, Any]],
    *,
    gold_answer: Sequence[Any],
    gold_program: Sequence[Any],
    input_prompt_raw: Sequence[Any],
    source_dataset: Sequence[Any],
    metadata: Sequence[Any],
    requires_history: Sequence[Any],
    record_ids: Sequence[Any],
) -> None:
    log_path = controlled_smoke_reward_log_path()
    if not log_path:
        return
    group_size = max(int(os.environ.get("V34R23_CONTROLLED_SMOKE_NUM_GENERATIONS", "8") or "8"), 1)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = sum(1 for _ in handle)
    with path.open("a", encoding="utf-8") as handle:
        for offset, start in enumerate(range(0, len(completions), group_size)):
            end = min(start + group_size, len(completions))
            meta = metadata[start] if start < len(metadata) and isinstance(metadata[start], dict) else {}
            prompt = first_text(input_prompt_raw[start] if start < len(input_prompt_raw) else "")
            group_record_ids = [first_text(item) for item in record_ids[start:end]]
            if controlled_smoke_assert_groups_enabled():
                if end - start != group_size:
                    raise RuntimeError(f"reward group size mismatch: expected {group_size}, got {end - start}")
                unique_ids = {item for item in group_record_ids if item}
                if len(unique_ids) != 1:
                    raise RuntimeError(f"reward group record_id corruption: {sorted(unique_ids)}")
            group_metadata = [item if isinstance(item, dict) else {} for item in metadata[start:end]]
            group_buckets = [
                first_text(item.get("v34r23_bucket") or item.get("bucket"))
                for item in group_metadata
            ]
            group_history = [
                bool(
                    item.get("v34r23_requires_history")
                    or item.get("requires_history")
                    or (requires_history[index] if index < len(requires_history) else False)
                )
                for index, item in enumerate(group_metadata, start=start)
            ]
            payload = {
                "call_index": existing + offset,
                "record_id": group_record_ids[0] if group_record_ids else first_text(meta.get("record_id")),
                "record_ids": group_record_ids,
                "training_bucket": first_text(meta.get("v34r23_bucket") or meta.get("bucket")),
                "training_buckets": group_buckets,
                "history_dependent": bool(
                    meta.get("v34r23_requires_history")
                    or meta.get("requires_history")
                    or (requires_history[start] if start < len(requires_history) else False)
                ),
                "history_dependent_values": group_history,
                "source_dataset": first_text(source_dataset[start] if start < len(source_dataset) else ""),
                "source_datasets": [first_text(item) for item in source_dataset[start:end]],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_sha256_values": [
                    hashlib.sha256(first_text(item).encode("utf-8")).hexdigest()
                    for item in input_prompt_raw[start:end]
                ],
                "gold_answer": first_text(gold_answer[start] if start < len(gold_answer) else ""),
                "gold_program": first_text(gold_program[start] if start < len(gold_program) else ""),
                "completion_count": end - start,
                "completions": [completion_text(item) for item in completions[start:end]],
                "diagnostics": diagnostic_results[start:end],
            }
            payload.update(controlled_smoke_group_summary(rewards[start:end], completions[start:end], diagnostic_results[start:end]))
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class V34R23RewardTraceEarlyStopCallback(TrainerCallback):
    def __init__(self, reward_log_path: str):
        self.reward_log_path = Path(reward_log_path)
        self.reason = ""

    def _read_rows(self) -> List[Dict[str, Any]]:
        if not self.reward_log_path.exists():
            return []
        rows = []
        with self.reward_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        rows = self._read_rows()
        recent = rows[-10:]
        last5 = rows[-5:]
        if len(last5) >= 5 and all(float(row.get("reward_std", 0.0)) <= 1e-8 for row in last5):
            self.reason = "grpo40_stopped_zero_reward_variance"
        elif len(recent) >= 10 and sum(1 for row in recent if row.get("zero_std")) / 10.0 >= 0.80:
            self.reason = "grpo40_stopped_zero_reward_variance"
        elif len(recent) >= 10:
            executable = sum(float(row.get("sampled_executable_rate", 0.0)) for row in recent) / 10.0
            if executable < 0.70:
                self.reason = "grpo40_stopped_execution_contract_regression"
        if rows:
            marker = max(
                float(row.get("reasoning_marker_rate", 0.0))
                + float(row.get("answer_marker_rate", 0.0))
                + float(row.get("normalized_answer_marker_rate", 0.0))
                for row in rows
            )
            if marker > 0.0:
                self.reason = "grpo40_stopped_execution_contract_regression"
        if len(recent) >= 3 and all(float(row.get("multiple_program_rate", 0.0)) > 0.0 for row in recent[-3:]):
            self.reason = "grpo40_stopped_execution_contract_regression"
        if len(last5) >= 5 and all(row.get("all_wrong") for row in last5):
            self.reason = "grpo40_stopped_zero_reward_variance"
        if any(str(value).lower() in {"nan", "inf", "-inf"} for value in logs.values()):
            self.reason = "grpo40_stopped_numerical_instability"
        if self.reason:
            control.should_training_stop = True
            if args.local_rank in [-1, 0]:
                reason_path = Path(args.output_dir) / "early_stop_reason.json"
                reason_path.write_text(
                    json.dumps({"reason": self.reason, "global_step": int(state.global_step)}, indent=2) + "\n",
                    encoding="utf-8",
                )
        return control


def make_reward_funcs(script_args: ScriptArguments, diagnostics: Optional[ProgramDiagnostics] = None):
    def reward_program_hard_gate(completions, gold_answer=None, gold_program=None, **kwargs):
        rewards = []
        diagnostic_results = []
        gold_answer = gold_answer or [""] * len(completions)
        gold_program = gold_program or [""] * len(completions)
        input_prompt_raw = kwargs.get("input_prompt_raw") or [""] * len(completions)
        source_dataset = kwargs.get("source_dataset") or [""] * len(completions)
        metadata = kwargs.get("metadata") or [None] * len(completions)
        requires_history = kwargs.get("requires_history") or [None] * len(completions)
        record_ids = kwargs.get("record_id") or [""] * len(completions)
        programs = []
        for completion, gold_ans, gold_prog, prompt, source, meta, req_history in zip(
            completions,
            gold_answer,
            gold_program,
            input_prompt_raw,
            source_dataset,
            metadata,
            requires_history,
        ):
            program = extract_anchor(completion_text(completion), "Program:")
            programs.append(program.strip())
            diagnostic_result = evaluate_program_completion(
                script_args,
                completion,
                gold_ans,
                gold_prog,
                prompt,
                source_dataset=source,
                metadata=meta if isinstance(meta, dict) else None,
                requires_history=req_history,
            )
            diagnostic_results.append(diagnostic_result)
            if diagnostics is not None:
                diagnostics.add("program/core_score", diagnostic_result["core_score"])
                diagnostics.add("program/executable_rate", diagnostic_result["executable"])
                diagnostics.add("program/exact_match_rate", diagnostic_result["exact_match"])
                diagnostics.add("program/invalid_rate", diagnostic_result["invalid"])
                diagnostics.add("program/wrong_executable_rate", diagnostic_result["wrong_executable"])
                diagnostics.add("program/structure_score", diagnostic_result["structure_score"])
                diagnostics.add("program/argument_coverage", diagnostic_result["argument_coverage"])
                diagnostics.add("program/step_count_score", diagnostic_result["step_count_score"])
                diagnostics.add("program/completion_words", diagnostic_result["completion_words"])
                diagnostics.add("program/completion_chars", diagnostic_result["completion_chars"])
                diagnostics.add("program/evidence_bullet_count", diagnostic_result["evidence_bullet_count"])
                diagnostics.add("program/evidence_chars", diagnostic_result["evidence_chars"])
                diagnostics.add("program/program_lines", diagnostic_result["program_lines"])
                diagnostics.add("program/program_chars", diagnostic_result["program_chars"])
                diagnostics.add("program/length_penalty", diagnostic_result["length_penalty"])
                diagnostics.add("program/reasoning_penalty", diagnostic_result["reasoning_penalty"])
                diagnostics.add("program/reasoning_too_long_rate", diagnostic_result["reasoning_too_long"])
                diagnostics.add("program/reasoning_chars", diagnostic_result["reasoning_chars"])
                diagnostics.add("program/reasoning_lines", diagnostic_result["reasoning_lines"])
                diagnostics.add("program/reasoning_number_count", diagnostic_result["reasoning_number_count"])
                diagnostics.add("program/forbidden_anchor_rate", diagnostic_result["forbidden_anchor"])
                diagnostics.add("program/multiple_program_rate", diagnostic_result["multiple_program"])
                diagnostics.add("program/post_program_text_rate", diagnostic_result["post_program_text"])
                diagnostics.add("program/schema_hard_gate_violation_rate", diagnostic_result["schema_hard_gate_violation"])
                diagnostics.add("program/prompt_contract_violation_rate", diagnostic_result["prompt_contract_violation"])
                diagnostics.add("program/answer_only_exact_match_rate", diagnostic_result["answer_only_exact_match"])
                diagnostics.add("program/answer_first_exact_match_rate", diagnostic_result["answer_first_exact_match"])
                diagnostics.add("program/answer_anchor_coverage_rate", diagnostic_result["answer_anchor_coverage"])
                diagnostics.add("program/program_answer_consistency_reward", diagnostic_result["program_answer_consistency_reward"])
                diagnostics.add("program/direct_lookup_literal_ok_rate", diagnostic_result["direct_lookup_literal_ok"])
                diagnostics.add("program/direct_lookup_wrapper_error_rate", diagnostic_result["direct_lookup_wrapper_error"])
                diagnostics.add("program/invalid_operator_get_rate", diagnostic_result["invalid_operator_get"])
                diagnostics.add("program/assignment_or_placeholder_rate", diagnostic_result["assignment_or_placeholder"])
                diagnostics.add("program/evidence_program_alignment", diagnostic_result["evidence_program_alignment"])
                diagnostics.add("program/program_arg_precision", diagnostic_result["program_arg_precision"])
                diagnostics.add("program/operation_prior_score", diagnostic_result["operation_prior_score"])
                diagnostics.add("program/ratio_divide_requirement_score", diagnostic_result["ratio_divide_requirement_score"])
                diagnostics.add("program/denominator_grounding_score", diagnostic_result["denominator_grounding_score"])
                diagnostics.add("program/table_argument_grounding_score", diagnostic_result["table_argument_grounding_score"])
                diagnostics.add("program/lookup_shortcut_score", diagnostic_result["lookup_shortcut_score"])
                diagnostics.add("program/dsl_validity_score", diagnostic_result["dsl_validity_score"])
                diagnostics.add("program/dsl_strict_validity_score", diagnostic_result["dsl_strict_validity_score"])
                diagnostics.add("program/symbolic_variable_rate", diagnostic_result["symbolic_variable_rate"])
                diagnostics.add("program/mixed_infix_rate", diagnostic_result["mixed_infix_rate"])
                diagnostics.add("program/recoverable_infix_rate", diagnostic_result["recoverable_infix_rate"])
                diagnostics.add("program/direct_lookup_gold_rate", diagnostic_result["direct_lookup_gold"])
                diagnostics.add("program/calculation_shortcut_rate", diagnostic_result["calculation_shortcut"])
                diagnostics.add("program/program_arg_source_score", diagnostic_result["program_arg_source_score"])
                diagnostics.add("program/extra_number_penalty", diagnostic_result["extra_number_penalty"])
                diagnostics.add("program/scale_consistency_score", diagnostic_result["scale_consistency_score"])
                diagnostics.add("program/semantic_process_score", diagnostic_result["semantic_process_score"])
                diagnostics.add("program/process_score", diagnostic_result["process_score"])
                diagnostics.add("program/process_adjustment", diagnostic_result["process_adjustment"])
                diagnostics.add("program/answer_correctness_reward", diagnostic_result["answer_correctness_reward"])
                diagnostics.add("program/answer_shortcut_rate", diagnostic_result["answer_shortcut_rate"])
                diagnostics.add("program/answer_correct_program_wrong_rate", diagnostic_result["answer_correct_program_wrong_rate"])
                diagnostics.add("program/program_correct_answer_missing_rate", diagnostic_result["program_correct_answer_missing_rate"])
                diagnostics.add("program/program_correct_answer_inconsistent_rate", diagnostic_result["program_correct_answer_inconsistent_rate"])
                diagnostics.add("program/program_correct_answer_consistent_rate", diagnostic_result["program_correct_answer_consistent_rate"])
                diagnostics.add("program/program_execution_reward", diagnostic_result["program_execution_reward"])
                diagnostics.add("program/operation_match_reward", diagnostic_result["operation_match_reward"])
                diagnostics.add("program/argument_grounding_reward", diagnostic_result["argument_grounding_reward"])
                diagnostics.add("program/denominator_grounding_reward", diagnostic_result["denominator_grounding_reward"])
                diagnostics.add("program/table_argument_grounding_reward", diagnostic_result["table_argument_grounding_reward"])
                diagnostics.add("program/lookup_shortcut_reward", diagnostic_result["lookup_shortcut_reward"])
                diagnostics.add("program/scale_consistency_reward", diagnostic_result["scale_consistency_reward"])
                diagnostics.add("program/evidence_grounding_reward", diagnostic_result["evidence_grounding_reward"])
                diagnostics.add("program/history_grounding_reward", diagnostic_result["history_grounding_reward"])
                diagnostics.add("program/structure_reward", diagnostic_result["structure_reward"])
            rewards.append(diagnostic_result["core_score"])
        record_bundle_diagnostics(
            diagnostic_results,
            source_dataset=source_dataset,
            metadata=metadata,
            requires_history=requires_history,
            script_args=script_args,
            diagnostics=diagnostics,
        )
        rewards = apply_group_risk_shaping(rewards, diagnostic_results, script_args, diagnostics=diagnostics)
        maybe_log_controlled_smoke_rewards(
            completions,
            rewards,
            diagnostic_results,
            gold_answer=gold_answer,
            gold_program=gold_program,
            input_prompt_raw=input_prompt_raw,
            source_dataset=source_dataset,
            metadata=metadata,
            requires_history=requires_history,
            record_ids=record_ids,
        )
        if diagnostics is not None and completions:
            non_empty_programs = [program for program in programs if program and program.upper() != "N/A"]
            diagnostics.add("program/has_program_rate", len(non_empty_programs) / len(completions))
            if non_empty_programs:
                diagnostics.add("program/unique_program_ratio", len(set(non_empty_programs)) / len(non_empty_programs))
        return rewards

    return [
        reward_program_hard_gate,
    ]


def prepare_dataset(dataset: Dataset, script_args: ScriptArguments, is_main_process: bool) -> Dataset:
    def validate_row(row: Dict[str, Any]) -> Dict[str, Any]:
        reward_profile = first_text(row.get("reward_profile"))
        if reward_profile != script_args.reward_profile_expected:
            raise ValueError(f"Unexpected reward_profile: {reward_profile}")
        if not first_text(row.get("input_prompt_raw")):
            raise ValueError("Missing input_prompt_raw")
        if not first_text(row.get("gold_answer")):
            raise ValueError("Missing gold_answer")
        if not first_text(row.get("gold_program")):
            raise ValueError("Missing gold_program")
        executed_value, _, _ = execute_prediction_program(first_text(row.get("gold_program")))
        if executed_value is None:
            raise ValueError("gold_program is not executable")
        if not numeric_equal(
            str(executed_value),
            first_text(row.get("gold_answer")),
            script_args.answer_abs_tol,
            script_args.answer_rel_tol,
        ):
            raise ValueError("gold_program execution does not match gold_answer")
        strict_program_only = script_args.reward_mode == "frontier_execution_calibration"
        input_prompt_raw = apply_cot_pot_output_format(
            first_text(row["input_prompt_raw"]),
            answer_first=script_args.reward_mode == "answer_first",
            program_first_answer=script_args.reward_mode in {"program_first_answer_aux", "program_gated_answer_aux"},
            strict_program_only=strict_program_only,
        )
        if script_args.reward_mode == "answer_first":
            system_prompt = ANSWER_FIRST_SYSTEM_PROMPT
        elif script_args.reward_mode in {"program_first_answer_aux", "program_gated_answer_aux"}:
            system_prompt = PROGRAM_FIRST_ANSWER_SYSTEM_PROMPT
        elif strict_program_only:
            system_prompt = STRICT_PROGRAM_SYSTEM_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT
        input_prompt_chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt_raw},
        ]
        return {
            "prompt": input_prompt_chat,
            "input_prompt_chat": input_prompt_chat,
            "input_prompt_raw": input_prompt_raw,
            "gold_answer": first_text(row["gold_answer"]),
            "gold_program": first_text(row["gold_program"]),
            "reward_profile": reward_profile,
            "record_id": first_text(row.get("record_id")),
            "source_dataset": first_text(row.get("source_dataset")),
            "metadata": row.get("metadata") or {},
            "requires_history": bool((row.get("metadata") or {}).get("requires_history")),
            "reference_response": first_text(row.get("reference_response")),
        }

    return dataset.map(
        validate_row,
        num_proc=script_args.preprocessing_num_workers,
        desc="Processing dataset" if is_main_process else None,
    )


def grpo_train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    is_main_process = training_args.local_rank in [-1, 0]
    train_file = ensure_local_path(script_args.train_file, "train_file")
    valid_file = ensure_local_path(script_args.valid_file, "valid_file")
    tokenizer_path = ensure_local_path(script_args.tokenizer_name_or_path or model_args.model_name_or_path, "tokenizer_name_or_path")
    model_path = ensure_local_path(model_args.model_name_or_path, "model_name_or_path")
    peft_path = ensure_local_path(script_args.peft_path, "peft_path") if script_args.peft_path else None
    output_dir = Path(training_args.output_dir)
    logging_dir = configure_tensorboard_reporting(training_args, script_args, is_main_process, "GRPO")
    training_args.log_completions = bool(script_args.log_prompt_completions)
    apply_grpo_training_arg_defaults(training_args)
    validate_grpo_reward_contract(script_args, training_args)

    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model args: {model_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(f"Training args: {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    raw_eval_dataset = load_dataset("json", data_files=str(valid_file), split="train")

    with training_args.main_process_first(desc="Dataset preparation"):
        train_dataset = prepare_dataset(raw_train_dataset, script_args, is_main_process)
        eval_dataset = prepare_dataset(raw_eval_dataset, script_args, is_main_process)

    train_audit = audit_program_numeric_rows(list(raw_train_dataset), script_args)
    eval_audit = audit_program_numeric_rows(list(raw_eval_dataset), script_args)

    torch_dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot both be set")

    quantization_config = None
    if model_args.load_in_4bit or model_args.load_in_8bit:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )
    num_gpus = torch.cuda.device_count()
    if ddp:
        model_kwargs["device_map"] = None
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            usable_mem = int(gpu_props.total_memory * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)

    if model_args.use_peft:
        if peft_path is not None:
            model = PeftModel.from_pretrained(model, str(peft_path), is_trainable=True)
            peft_config = getattr(model, "peft_config", None)
        else:
            target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
            if target_modules == "all" or (target_modules and "all" in target_modules):
                target_modules = find_all_linear_names(
                    model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit
                )
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
            )
            model = get_peft_model(model, peft_config)
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in filter(lambda p: p.requires_grad, model.parameters()):
                param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        peft_config = None

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "model_args": asdict(model_args),
        "script_args": asdict(script_args),
        "training_args_summary": {
            "output_dir": training_args.output_dir,
            "learning_rate": training_args.learning_rate,
            "beta": training_args.beta,
            "max_steps": training_args.max_steps,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "num_generations": training_args.num_generations,
            "max_completion_length": training_args.max_completion_length,
            "report_to": training_args.report_to,
            "logging_dir": training_args.logging_dir,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "bf16": training_args.bf16,
            "use_vllm": training_args.use_vllm,
            "temperature": training_args.temperature,
            "top_p": training_args.top_p,
            "min_p": training_args.min_p,
            "multi_objective_aggregation": training_args.multi_objective_aggregation,
            "scale_rewards": training_args.scale_rewards,
            "loss_type": training_args.loss_type,
        },
        "train_audit": train_audit,
        "eval_audit": eval_audit,
        "tokenizer_path": str(tokenizer_path),
        "model_path": str(model_path),
        "tensorboard_logging_dir": str(logging_dir) if logging_dir else None,
        "created_at": datetime.now().isoformat(),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    program_diagnostics = ProgramDiagnostics()
    reward_funcs = make_reward_funcs(script_args, diagnostics=program_diagnostics)
    trainer = ProgramDiagnosticGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        program_diagnostics=program_diagnostics,
        script_args=script_args,
    )
    reward_log_path = controlled_smoke_reward_log_path()
    if reward_log_path and os.environ.get("V34R23_GRPO_EARLY_STOP_ENABLED", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        trainer.add_callback(V34R23RewardTraceEarlyStopCallback(reward_log_path))

    last_checkpoint = get_checkpoint(training_args)
    resume_checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None and is_main_process:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}")
        logger.info(
            f"Starting GRPO training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for max_steps={training_args.max_steps}"
        )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
    training_args.distributed_state.wait_for_everyone()
    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.model.config.save_pretrained(training_args.output_dir)
        logger.info(f"Training complete. Model saved to {training_args.output_dir}")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()

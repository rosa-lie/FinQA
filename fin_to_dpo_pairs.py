#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert FinGPT-style data to MedicalGPT DPO pair format.

Output schema (compatible with dpo_training.py):
{
  "system": "",
  "history": [],
  "question": "...",
  "response_chosen": "...",
  "response_rejected": "..."
}

Construction strategy:
- closed-set classification / multiple-choice: build hard negatives from label space
- structured extraction (NER / relation extraction): perturb labels while preserving format
- open-ended QA: sample candidate answers from the current SFT model and pick a distinct hard negative
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTION_KEYS = ["question", "query", "prompt", "input", "context", "text"]
ANSWER_KEYS = ["output", "answer", "response", "completion", "label", "sentiment"]
DEFAULT_NER_TYPES = ["person", "organization", "location"]
YES_NO_PAIRS = {
    "yes": "No",
    "no": "Yes",
    "true": "False",
    "false": "True",
    "是": "否",
    "否": "是",
}
SENTIMENT_ORDER = [
    "strong negative",
    "moderately negative",
    "mildly negative",
    "negative",
    "neutral",
    "positive",
    "mildly positive",
    "moderately positive",
    "strong positive",
    "消极",
    "中性",
    "积极",
]


@dataclass
class PairResult:
    question: str
    chosen: str
    rejected: str
    task_type: str
    reject_source: str


def to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v).strip()
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def pick_first(rec: Dict[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        if k in rec:
            val = to_text(rec.get(k))
            if val:
                return val
    return ""


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_candidates(raw: str) -> List[str]:
    cands = re.split(r"[,/|;；、]", raw)
    return [c.strip() for c in cands if c.strip()]


def parse_options(text: str) -> List[str]:
    m = re.search(r"(?i)options?\s*[:：]\s*(.+)", text)
    if m:
        options = split_candidates(m.group(1))
        if options:
            return options

    b = re.search(r"\{([^{}]+)\}", text)
    if b:
        options = split_candidates(b.group(1))
        if options:
            return options

    s = re.search(r"\[([^\[\]]+)\]", text)
    if s:
        options = split_candidates(s.group(1))
        if options:
            return options
    return []


def parse_mcq_options(text: str) -> List[Tuple[str, str]]:
    pairs = re.findall(r"(?m)^\s*([A-Z])\.\s*(.+?)\s*$", text)
    return [(label.strip(), body.strip()) for label, body in pairs]


def make_question(rec: Dict[str, Any]) -> str:
    instruction = to_text(rec.get("instruction"))
    inp = to_text(rec.get("input"))
    q = pick_first(rec, QUESTION_KEYS)
    if instruction and inp:
        return f"{instruction}\n\n{inp}"
    if instruction and q and instruction != q:
        return f"{instruction}\n\n{q}"
    if instruction:
        return instruction
    return q


def choose_adjacent_option(chosen: str, options: Sequence[str]) -> str:
    if not options:
        return ""
    chosen_norm = normalize_label(chosen)
    normalized = [normalize_label(opt) for opt in options]
    if chosen_norm in normalized:
        idx = normalized.index(chosen_norm)
        if idx > 0:
            return options[idx - 1]
        if idx + 1 < len(options):
            return options[idx + 1]
    pool = [opt for opt in options if normalize_label(opt) != chosen_norm]
    if not pool:
        return ""
    return max(pool, key=lambda opt: SequenceMatcher(None, chosen_norm, normalize_label(opt)).ratio())


def make_classification_rejected(chosen: str, question: str) -> str:
    options = parse_options(question)
    chosen_norm = normalize_label(chosen)

    if chosen_norm in YES_NO_PAIRS:
        return YES_NO_PAIRS[chosen_norm]

    if options:
        return choose_adjacent_option(chosen, options)

    if chosen_norm in [normalize_label(x) for x in SENTIMENT_ORDER]:
        return choose_adjacent_option(chosen, SENTIMENT_ORDER)

    return ""


def make_multiple_choice_rejected(chosen: str, question: str) -> str:
    mcq_options = parse_mcq_options(question)
    if not mcq_options:
        return ""

    label_to_text = {label: f"{label}. {text}" for label, text in mcq_options}
    chosen_label_match = re.match(r"^\s*([A-Z])\.", chosen)
    if chosen_label_match:
        labels = [label for label, _ in mcq_options]
        chosen_label = chosen_label_match.group(1)
        if chosen_label in labels:
            idx = labels.index(chosen_label)
            alt_idx = idx - 1 if idx > 0 else (idx + 1 if idx + 1 < len(labels) else idx)
            alt_label = labels[alt_idx]
            if alt_label != chosen_label:
                return label_to_text[alt_label]

    pool = [text for text in label_to_text.values() if normalize_label(text) != normalize_label(chosen)]
    if not pool:
        return ""
    return pool[0]


def parse_relation_options(question: str) -> List[str]:
    options = parse_options(question)
    if options:
        return [opt.replace("/", "_").replace(" ", "_") for opt in options]
    rel_match = re.search(r"(?i)relations?\s+include\s*[:：]\s*(.+)", question)
    if rel_match:
        raw = rel_match.group(1)
        return [opt.replace("/", "_").replace(" ", "_") for opt in split_candidates(raw)]
    return []


def make_relation_extraction_rejected(chosen: str, question: str) -> str:
    relations = parse_relation_options(question)
    parts = [part.strip() for part in re.split(r";\s*", chosen) if part.strip()]
    if not parts:
        return ""

    rewritten = []
    changed = False
    for part in parts:
        if ":" not in part:
            rewritten.append(part)
            continue
        relation, payload = [x.strip() for x in part.split(":", 1)]
        if not changed and relations:
            alt = choose_adjacent_option(relation, relations)
            if alt and normalize_label(alt) != normalize_label(relation):
                rewritten.append(f"{alt}: {payload}")
                changed = True
                continue
        rewritten.append(part)

    if changed:
        return "; ".join(rewritten)
    if len(parts) > 1:
        return "; ".join(parts[:-1])
    return ""


def make_ner_rejected(chosen: str, question: str) -> str:
    type_options = parse_options(question) or DEFAULT_NER_TYPES
    segments = [seg.strip() for seg in re.split(r";\s*", chosen.replace(".", ";")) if seg.strip()]
    rewritten = []
    changed = False
    for seg in segments:
        m = re.match(r"^(.*?)\s+is\s+(?:an?\s+)?(\w+)\s*$", seg, flags=re.IGNORECASE)
        if not m:
            rewritten.append(seg)
            continue
        entity = m.group(1).strip()
        entity_type = m.group(2).strip()
        if not changed:
            alt = choose_adjacent_option(entity_type, type_options)
            if alt and normalize_label(alt) != normalize_label(entity_type):
                rewritten.append(f"{entity} is a {alt}")
                changed = True
                continue
        rewritten.append(seg)

    if changed:
        return "; ".join(rewritten) + "."
    if len(segments) > 1:
        return "; ".join(segments[:-1]) + "."
    return ""


def detect_task_type(question: str, chosen: str) -> str:
    q_lower = question.lower()
    if parse_mcq_options(question) and re.match(r"^\s*[A-Z]\.", chosen):
        return "multiple_choice"
    if "extract entities" in q_lower or "entity types" in q_lower:
        return "ner_extraction"
    if "extract the subject and object" in q_lower or "output format should be" in q_lower:
        return "relation_extraction"
    if parse_options(question):
        return "classification"
    if len(chosen) <= 64 and re.fullmatch(r"[\w\-\s\.]+", chosen):
        return "classification"
    return "open_qa"


class ModelSampler:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str],
        max_new_tokens: int,
        num_candidates: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_p = top_p
        self._tokenizer = None
        self._model = None

    def _lazy_init(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @torch.inference_mode()
    def sample(self, question: str) -> List[str]:
        self._lazy_init()
        messages = [{"role": "user", "content": question}]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs["attention_mask"].to(self._model.device)
        outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.num_candidates,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        prompt_len = input_ids.shape[1]
        results: List[str] = []
        for sequence in outputs:
            text = self._tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip()
            if text:
                results.append(text)
        return results


def choose_model_rejected(chosen: str, candidates: Sequence[str]) -> str:
    chosen_norm = normalize_label(chosen)
    scored = []
    for cand in candidates:
        cand_norm = normalize_label(cand)
        if not cand_norm or cand_norm == chosen_norm:
            continue
        score = SequenceMatcher(None, chosen_norm, cand_norm).ratio()
        scored.append((score, cand))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1].strip()


def make_pair(rec: Dict[str, Any], sampler: Optional[ModelSampler], allow_open_qa_skip: bool) -> Optional[PairResult]:
    q = make_question(rec)

    chosen = to_text(rec.get("response_chosen"))
    rejected = to_text(rec.get("response_rejected"))
    if chosen and rejected and q:
        return PairResult(q, chosen, rejected, "preference_pair", "provided")

    chosen = pick_first(rec, ANSWER_KEYS)
    if not q or not chosen:
        return None

    task_type = detect_task_type(q, chosen)
    rejected = ""
    reject_source = ""

    if task_type == "classification":
        rejected = make_classification_rejected(chosen, q)
        reject_source = "label_space"
    elif task_type == "multiple_choice":
        rejected = make_multiple_choice_rejected(chosen, q)
        reject_source = "label_space"
    elif task_type == "relation_extraction":
        rejected = make_relation_extraction_rejected(chosen, q)
        reject_source = "structured_hard_negative"
    elif task_type == "ner_extraction":
        rejected = make_ner_rejected(chosen, q)
        reject_source = "structured_hard_negative"
    elif task_type == "open_qa" and sampler is not None:
        candidates = sampler.sample(q)
        rejected = choose_model_rejected(chosen, candidates)
        reject_source = "sft_model_sample"

    if not rejected and task_type == "open_qa" and allow_open_qa_skip:
        return None
    if not rejected:
        return None
    return PairResult(q, chosen, rejected, task_type, reject_source)


def iter_records(ds: Dataset) -> Iterable[Dict[str, Any]]:
    for row in ds:
        yield dict(row)


def load_source(args: argparse.Namespace) -> Dataset:
    if args.source_file:
        ext = Path(args.source_file).suffix.lower()
        if ext not in {".json", ".jsonl"}:
            raise ValueError("--source_file currently supports .json/.jsonl only")
        return load_dataset("json", data_files=args.source_file, split="train")
    if not args.dataset_name:
        raise ValueError("Please provide --dataset_name or --source_file")
    return load_dataset(args.dataset_name, args.dataset_config, split=args.split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="FinGPT/fingpt-sentiment-train")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Current SFT model used to sample rejected answers for open QA tasks.",
    )
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--skip_open_qa_without_model",
        action="store_true",
        help="Skip open QA rows if no SFT model is provided for reject sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = load_source(args)
    sampler = None
    if args.model_name_or_path:
        sampler = ModelSampler(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_new_tokens=args.max_new_tokens,
            num_candidates=args.num_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    rows = []
    stats = {
        "kept": 0,
        "skipped": 0,
        "task_types": {},
        "reject_sources": {},
    }
    for rec in iter_records(ds):
        pair = make_pair(rec, sampler=sampler, allow_open_qa_skip=args.skip_open_qa_without_model)
        if pair is None:
            stats["skipped"] += 1
            continue
        stats["kept"] += 1
        stats["task_types"][pair.task_type] = stats["task_types"].get(pair.task_type, 0) + 1
        stats["reject_sources"][pair.reject_source] = stats["reject_sources"].get(pair.reject_source, 0) + 1
        rows.append(
            {
                "system": "",
                "history": [],
                "question": pair.question,
                "response_chosen": pair.chosen,
                "response_rejected": pair.rejected,
            }
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({"output_file": str(out_path), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

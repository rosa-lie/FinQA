#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_SYSTEM_PROMPT = (
    "你是一名金融数值推理教师模型。"
    "请严格按照以下结构输出，不要省略标题，不要输出 JSON，不要添加额外说明。\n\n"
    "问题分析：...\n"
    "关键证据：\n- ...\n"
    "推理程序：...\n"
    "最终答案：..."
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_temperatures(spec: str) -> List[float]:
    values = []
    for token in (spec or "0.0").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values or [0.0]


def read_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    if args.system_prompt:
        return args.system_prompt.strip()
    return DEFAULT_SYSTEM_PROMPT


def make_generation_key(
    row: Dict[str, Any],
    candidate_index: int,
    backend: str,
    provider: Optional[str],
    model: Optional[str],
    temperature: float,
) -> str:
    base = json.dumps({
        "record_id": row.get("record_id", ""),
        "source_dataset": row.get("source_dataset", ""),
        "task_name": row.get("task_name", ""),
        "prompt": row.get("prompt", ""),
        "candidate_index": candidate_index,
        "teacher_backend": backend,
        "teacher_provider": provider or "",
        "teacher_model": model or "",
        "teacher_temperature": temperature,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def render_user_prompt(prompt: str, template_text: str) -> str:
    template = template_text or "{{ prompt }}"
    if "{{ prompt }}" in template:
        return template.replace("{{ prompt }}", prompt)
    return prompt


def build_messages(system_prompt: str, prompt: str, template_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": render_user_prompt(prompt, template_text)})
    return messages


def create_client(args: argparse.Namespace):
    from role_play_data.llm_client import create_llm_client

    client, model_name = create_llm_client(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    return client, model_name


def generate_with_openai(client: Any, model_name: str, messages: Sequence[Dict[str, str]], temperature: float, max_tokens: int, top_p: float) -> Tuple[str, Dict[str, Any]]:
    response = client.chat.completions.create(
        model=model_name,
        messages=list(messages),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    raw = response.model_dump() if hasattr(response, "model_dump") else {}
    return content.strip(), raw


def generate_candidate(row: Dict[str, Any], args: argparse.Namespace, system_prompt: str, user_template_text: str, temperature: float, client_bundle: Optional[Tuple[Any, str]]) -> Tuple[str, Dict[str, Any]]:
    if args.backend == "gold":
        return str(row.get("gold_response") or "").strip(), {"backend": "gold"}
    if args.backend == "copy_gold_final":
        gold_answer = str(row.get("gold_answer") or "").strip()
        content = "\n".join([
            "问题分析：根据题目和材料定位所需财务指标。",
            "关键证据：",
            "- 依据题目相关表格、文本和历史线索提取关键数值。",
            f"推理程序：{str(row.get('gold_program') or '请依据材料逐步完成数值计算。').strip()}",
            f"最终答案：{gold_answer}",
        ])
        return content, {"backend": "copy_gold_final"}
    if client_bundle is None:
        raise ValueError("OpenAI-compatible backend requires a valid client.")
    client, model_name = client_bundle
    messages = build_messages(system_prompt, str(row.get("prompt") or ""), user_template_text)
    return generate_with_openai(client, model_name, messages, temperature, args.max_tokens, args.top_p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher candidates for financial distillation.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--backend", choices=["openai", "gold", "copy_gold_final"], default="openai")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--system_prompt_file", type=str, default="")
    parser.add_argument("--user_template_file", type=str, default="prompts/financial_distill_teacher_user.txt")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--temperature_schedule", type=str, default="0.6")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    existing_keys = set()
    if args.resume and output_path.exists():
        for row in load_jsonl(output_path):
            key = row.get("generation_key")
            if key:
                existing_keys.add(key)

    temperatures = parse_temperatures(args.temperature_schedule)
    system_prompt = read_system_prompt(args)
    user_template_text = Path(args.user_template_file).read_text(encoding="utf-8").strip() if args.user_template_file else "{{ prompt }}"
    client_bundle = None
    resolved_model = ""
    if args.backend == "openai":
        client_bundle = create_client(args)
        resolved_model = client_bundle[1]

    generated = 0
    skipped = 0
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as f:
        for row in rows:
            for candidate_index in range(args.num_candidates):
                temperature = temperatures[candidate_index % len(temperatures)]
                generation_key = make_generation_key(
                    row=row,
                    candidate_index=candidate_index,
                    backend=args.backend,
                    provider=args.provider,
                    model=resolved_model or args.model or "",
                    temperature=temperature,
                )
                if generation_key in existing_keys:
                    skipped += 1
                    continue
                response_text, raw_response = generate_candidate(row, args, system_prompt, user_template_text, temperature, client_bundle)
                out = {
                    **row,
                    "candidate_index": candidate_index,
                    "generation_key": generation_key,
                    "teacher_backend": args.backend,
                    "teacher_provider": args.provider,
                    "teacher_model": resolved_model or args.model or args.backend,
                    "teacher_temperature": temperature,
                    "teacher_top_p": args.top_p,
                    "teacher_max_tokens": args.max_tokens,
                    "response": response_text,
                    "raw_response": raw_response,
                    "generated_at": int(time.time()),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                generated += 1
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    print(json.dumps({
        "input_file": str(input_path),
        "output_file": str(output_path),
        "backend": args.backend,
        "model": resolved_model or args.model or args.backend,
        "rows": len(rows),
        "num_candidates": args.num_candidates,
        "generated_rows": generated,
        "skipped_existing_rows": skipped,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

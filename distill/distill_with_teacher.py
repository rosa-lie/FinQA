#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_SYSTEM_PROMPT = (
    "你是一名金融数值推理教师模型。"
    "请严格使用 <think> 和 <answer> 标签输出，不要输出 JSON，不要添加额外说明。\n\n"
    "<think>\n"
    "在这里写逐步推理过程。\n"
    "</think>\n"
    "<answer>\n"
    "在这里仅输出最终答案本身。\n"
    "</answer>"
)

THREAD_LOCAL = threading.local()


def strip_answer_prefix(text: str) -> str:
    text = (text or "").strip()
    for prefix in ("最终答案：", "最终答案:", "答案：", "答案:", "answer:", "Answer:"):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def wrap_structured_response(response_text: str) -> str:
    text = (response_text or "").strip()
    if not text:
        return text
    if "<think>" in text and "<answer>" in text:
        if "</answer>" not in text:
            return text + "\n</answer>"
        return text
    lines = text.splitlines()
    answer_start = None
    for i, line in enumerate(lines):
        if line.startswith("最终答案：") or line.startswith("最终答案:") or line.startswith("答案：") or line.startswith("答案:"):
            answer_start = i
            break
    if answer_start is None:
        think_body = text
        answer_body = "信息不足，暂不作答。"
    else:
        think_body = "\n".join(lines[:answer_start]).strip()
        answer_body = strip_answer_prefix("\n".join(lines[answer_start:]).strip())
    think_body = think_body or "请根据题意完成逐步推理。"
    answer_body = strip_answer_prefix(answer_body or "信息不足，暂不作答。")
    return f"<think>\n{think_body}\n</think>\n<answer>\n{answer_body}\n</answer>"


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


def make_generation_key(row: Dict[str, Any], candidate_index: int) -> str:
    base = json.dumps(
        {
            "record_id": row.get("record_id", ""),
            "prompt": row.get("prompt", ""),
            "candidate_index": candidate_index,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def render_user_prompt(row: Dict[str, Any], template_text: str) -> str:
    template = template_text or "{{ prompt }}"
    values = {
        "prompt": str(row.get("prompt") or ""),
        "gold_answer": str(row.get("gold_answer") or ""),
        "gold_program": str(row.get("gold_program") or ""),
        "gold_response": str(row.get("gold_response") or ""),
        "gold_supporting_facts": "\n".join(str(item) for item in (row.get("gold_supporting_facts") or []) if str(item).strip()),
        "record_id": str(row.get("record_id") or ""),
        "source_dataset": str(row.get("source_dataset") or ""),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", value)
    return rendered


def build_messages(system_prompt: str, row: Dict[str, Any], template_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": render_user_prompt(row, template_text)})
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


def get_thread_client_bundle(args: argparse.Namespace, shared_client_bundle: Optional[Tuple[Any, str]]) -> Optional[Tuple[Any, str]]:
    if args.backend != "openai":
        return shared_client_bundle
    if args.max_concurrency <= 1:
        if shared_client_bundle is None:
            raise ValueError("OpenAI-compatible backend requires a valid client.")
        return shared_client_bundle
    bundle = getattr(THREAD_LOCAL, "client_bundle", None)
    if bundle is None:
        bundle = create_client(args)
        THREAD_LOCAL.client_bundle = bundle
    return bundle


def generate_with_openai(client: Any, model_name: str, messages: Sequence[Dict[str, str]], temperature: float, max_tokens: int, top_p: float) -> Tuple[str, Dict[str, Any]]:
    response = client.chat.completions.create(
        model=model_name,
        messages=list(messages),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else {}
    content = response.choices[0].message.content or ""
    message = {}
    if raw.get("choices"):
        message = raw["choices"][0].get("message") or {}
    reasoning = str(message.get("reasoning_content") or "").strip()
    content = str(content or "").strip()
    if reasoning and content:
        merged = "\n".join([reasoning, content])
    else:
        merged = reasoning or content
    return merged.strip(), raw


def generate_candidate(row: Dict[str, Any], args: argparse.Namespace, system_prompt: str, user_template_text: str, temperature: float, client_bundle: Optional[Tuple[Any, str]]) -> Tuple[str, Dict[str, Any]]:
    if args.backend == "gold":
        gold_think = str(row.get("gold_response") or "").strip()
        gold_answer = strip_answer_prefix(str(row.get("gold_answer") or "").strip())
        return f"<think>\n{gold_think}\n</think>\n<answer>\n{gold_answer}\n</answer>", {"backend": "gold"}
    if args.backend == "copy_gold_final":
        gold_answer = strip_answer_prefix(str(row.get("gold_answer") or "").strip())
        think_body = str(row.get("gold_program") or "请依据材料逐步完成数值计算。").strip()
        content = f"<think>\n{think_body}\n</think>\n<answer>\n{gold_answer}\n</answer>"
        return content, {"backend": "copy_gold_final"}
    if client_bundle is None:
        raise ValueError("OpenAI-compatible backend requires a valid client.")
    client, model_name = client_bundle
    messages = build_messages(system_prompt, row, user_template_text)
    response_text, raw = generate_with_openai(client, model_name, messages, temperature, args.max_tokens, args.top_p)
    return wrap_structured_response(response_text), raw


def build_success_output(
    row: Dict[str, Any],
    args: argparse.Namespace,
    candidate_index: int,
    generation_key: str,
    temperature: float,
    response_text: str,
    raw_response: Dict[str, Any],
    model_name: str,
    retry_attempts: int,
) -> Dict[str, Any]:
    return {
        **row,
        "candidate_index": candidate_index,
        "generation_key": generation_key,
        "teacher_backend": args.backend,
        "teacher_provider": args.provider,
        "teacher_model": model_name,
        "teacher_temperature": temperature,
        "teacher_top_p": args.top_p,
        "teacher_max_tokens": args.max_tokens,
        "response": response_text,
        "raw_response": raw_response,
        "retry_attempts": retry_attempts,
        "generated_at": int(time.time()),
    }


def build_failed_output(
    row: Dict[str, Any],
    args: argparse.Namespace,
    candidate_index: int,
    generation_key: str,
    temperature: float,
    model_name: str,
    retry_attempts: int,
    error: Dict[str, str],
) -> Dict[str, Any]:
    return {
        **row,
        "candidate_index": candidate_index,
        "generation_key": generation_key,
        "teacher_backend": args.backend,
        "teacher_provider": args.provider,
        "teacher_model": model_name,
        "teacher_temperature": temperature,
        "teacher_top_p": args.top_p,
        "teacher_max_tokens": args.max_tokens,
        "retry_attempts": retry_attempts,
        "failed": True,
        "error": error,
        "failed_at": int(time.time()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher candidates for financial distillation.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--failed_output_file", type=str, default="")
    parser.add_argument("--backend", choices=["openai", "gold", "copy_gold_final"], default="openai")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--system_prompt_file", type=str, default="")
    parser.add_argument("--user_template_file", type=str, default="distill/prompts/program_conditioned_distill_user.txt")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--temperature_schedule", type=str, default="0.6")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--max_concurrency", type=int, default=1)
    parser.add_argument("--submit_window_size", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--retry_backoff", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    failed_output_path = Path(args.failed_output_file) if args.failed_output_file else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if failed_output_path is not None:
        failed_output_path.parent.mkdir(parents=True, exist_ok=True)

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

    jobs: List[Tuple[int, Dict[str, Any], int, str, float]] = []
    skipped = 0
    sequence_index = 0
    for row in rows:
        for candidate_index in range(args.num_candidates):
            generation_key = make_generation_key(row, candidate_index)
            if generation_key in existing_keys:
                skipped += 1
                continue
            temperature = temperatures[candidate_index % len(temperatures)]
            jobs.append((sequence_index, row, candidate_index, generation_key, temperature))
            sequence_index += 1

    def process_job(job: Tuple[int, Dict[str, Any], int, str, float]) -> Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        sequence_index, row, candidate_index, generation_key, temperature = job
        model_name = resolved_model or args.model or args.backend
        attempts = 0
        last_error: Dict[str, str] = {"type": "UnknownError", "message": "unknown"}
        while True:
            try:
                job_client_bundle = get_thread_client_bundle(args, client_bundle)
                model_name = job_client_bundle[1] if (args.backend == "openai" and job_client_bundle is not None) else model_name
                response_text, raw_response = generate_candidate(
                    row=row,
                    args=args,
                    system_prompt=system_prompt,
                    user_template_text=user_template_text,
                    temperature=temperature,
                    client_bundle=job_client_bundle,
                )
                success = build_success_output(
                    row=row,
                    args=args,
                    candidate_index=candidate_index,
                    generation_key=generation_key,
                    temperature=temperature,
                    response_text=response_text,
                    raw_response=raw_response,
                    model_name=model_name,
                    retry_attempts=attempts,
                )
                return sequence_index, success, None
            except Exception as exc:
                last_error = {"type": exc.__class__.__name__, "message": str(exc)}
                if attempts >= args.max_retries:
                    failure = build_failed_output(
                        row=row,
                        args=args,
                        candidate_index=candidate_index,
                        generation_key=generation_key,
                        temperature=temperature,
                        model_name=model_name,
                        retry_attempts=attempts,
                        error=last_error,
                    )
                    print(json.dumps({"event": "failed", "generation_key": generation_key, "candidate_index": candidate_index, "error": last_error}, ensure_ascii=False), file=sys.stderr, flush=True)
                    return sequence_index, None, failure
                sleep_seconds = args.retry_sleep_seconds * (args.retry_backoff ** attempts)
                print(
                    json.dumps(
                        {
                            "event": "retry",
                            "generation_key": generation_key,
                            "candidate_index": candidate_index,
                            "attempt": attempts + 1,
                            "sleep_seconds": sleep_seconds,
                            "error": last_error,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                attempts += 1

    generated = 0
    failed = 0
    submit_window_size = args.submit_window_size if args.submit_window_size > 0 else max(args.max_concurrency * 2, 1)

    def consume_result(result: Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], success_file, failed_handle) -> Tuple[int, int]:
        _, success, failure = result
        generated_delta = 0
        failed_delta = 0
        if success is not None:
            success_file.write(json.dumps(success, ensure_ascii=False) + "\n")
            generated_delta = 1
        if failure is not None and failed_handle is not None:
            failed_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            failed_delta = 1
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        return generated_delta, failed_delta

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as success_file:
        failed_handle = failed_output_path.open("a" if args.resume else "w", encoding="utf-8") if failed_output_path else None
        try:
            if args.max_concurrency <= 1 or len(jobs) <= 1:
                for job in jobs:
                    generated_delta, failed_delta = consume_result(process_job(job), success_file, failed_handle)
                    generated += generated_delta
                    failed += failed_delta
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                    pending: Dict[concurrent.futures.Future, Tuple[int, Dict[str, Any], int, str, float]] = {}
                    job_iter = iter(jobs)

                    def submit_next() -> bool:
                        try:
                            next_job = next(job_iter)
                        except StopIteration:
                            return False
                        future = executor.submit(process_job, next_job)
                        pending[future] = next_job
                        return True

                    while len(pending) < submit_window_size and submit_next():
                        pass

                    while pending:
                        done, _ = concurrent.futures.wait(
                            pending.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            pending.pop(future, None)
                            generated_delta, failed_delta = consume_result(future.result(), success_file, failed_handle)
                            generated += generated_delta
                            failed += failed_delta
                        while len(pending) < submit_window_size and submit_next():
                            pass
        finally:
            if failed_handle is not None:
                failed_handle.close()

    print(json.dumps({
        "input_file": str(input_path),
        "output_file": str(output_path),
        "failed_output_file": str(failed_output_path) if failed_output_path else "",
        "backend": args.backend,
        "model": resolved_model or args.model or args.backend,
        "rows": len(rows),
        "num_candidates": args.num_candidates,
        "generated_rows": generated,
        "failed_rows": failed,
        "skipped_existing_rows": skipped,
        "max_concurrency": args.max_concurrency,
        "submit_window_size": submit_window_size,
        "max_retries": args.max_retries,
        "retry_sleep_seconds": args.retry_sleep_seconds,
        "retry_backoff": args.retry_backoff,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

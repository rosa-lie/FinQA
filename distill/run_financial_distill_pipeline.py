#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def run_command(cmd: List[str]) -> None:
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def load_jsonl_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def iter_chunks(rows: List[Dict[str, object]], chunk_size: int) -> Iterable[List[Dict[str, object]]]:
    if chunk_size <= 0:
        yield rows
        return
    for i in range(0, len(rows), chunk_size):
        yield rows[i:i + chunk_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run end-to-end financial distillation pipeline.')
    parser.add_argument('--source_spec', action='append', required=True, help='family=/path/to/file.{json,jsonl}')
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--teacher_backend', choices=['openai', 'gold', 'copy_gold_final'], default='openai')
    parser.add_argument('--teacher_provider', type=str, default='deepseek')
    parser.add_argument('--teacher_model', type=str, default='deepseek-reasoner')
    parser.add_argument('--teacher_num_candidates', type=int, default=4)
    parser.add_argument('--teacher_temperature_schedule', type=str, default='0.6')
    parser.add_argument('--teacher_max_tokens', type=int, default=512)
    parser.add_argument('--teacher_max_concurrency', type=int, default=4)
    parser.add_argument('--teacher_max_retries', type=int, default=3)
    parser.add_argument('--teacher_retry_sleep_seconds', type=float, default=2.0)
    parser.add_argument('--teacher_retry_backoff', type=float, default=2.0)
    parser.add_argument('--teacher_user_template_file', type=str, default='')
    parser.add_argument('--checkpoint_rows', type=int, default=512)
    parser.add_argument('--max_samples_per_family', type=int, default=0)
    parser.add_argument('--convfinqa_keep_final_only', type=str, default='true')
    parser.add_argument('--enable_reasoning_judge', action='store_true')
    parser.add_argument('--judge_provider', type=str, default='deepseek')
    parser.add_argument('--judge_model', type=str, default='deepseek-chat')
    parser.add_argument('--judge_prompt_file', type=str, default='distill/prompts/financial_reasoning_judge.txt')
    parser.add_argument('--summary_csv', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_file = work_dir / 'distill_input.jsonl'
    candidates_file = work_dir / 'distill_candidates.jsonl'
    audit_file = work_dir / 'distill_audit.jsonl'
    sft_file = work_dir / 'distill_sft.jsonl'
    dpo_file = work_dir / 'distill_dpo.jsonl'
    failed_file = work_dir / 'distill_failed.jsonl'
    summary_file = work_dir / 'distill_summary.jsonl'
    summary_csv_file = work_dir / 'distill_summary.csv'
    checkpoint_dir = work_dir / 'checkpoints'
    checkpoint_input_dir = checkpoint_dir / 'inputs'
    checkpoint_candidate_dir = checkpoint_dir / 'candidates'
    checkpoint_failed_dir = checkpoint_dir / 'failed'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_input_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_failed_dir.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        sys.executable,
        'distill/build_financial_distill_dataset.py',
        '--output_file', str(input_file),
        '--max_samples_per_family', str(args.max_samples_per_family),
        '--convfinqa_keep_final_only', str(args.convfinqa_keep_final_only),
    ]
    for spec in args.source_spec:
        build_cmd.extend(['--source_spec', spec])
    run_command(build_cmd)

    input_rows = load_jsonl_rows(input_file)
    checkpoint_plan = []
    for chunk_index, chunk_rows in enumerate(iter_chunks(input_rows, args.checkpoint_rows)):
        chunk_input_file = checkpoint_input_dir / f'part_{chunk_index:05d}.jsonl'
        chunk_candidate_file = checkpoint_candidate_dir / f'part_{chunk_index:05d}.jsonl'
        chunk_failed_file = checkpoint_failed_dir / f'part_{chunk_index:05d}.jsonl'
        write_jsonl_rows(chunk_input_file, chunk_rows)
        expected_candidates = len(chunk_rows) * args.teacher_num_candidates
        existing_candidates = count_jsonl_rows(chunk_candidate_file)
        checkpoint_plan.append({
            'chunk_index': chunk_index,
            'input_file': str(chunk_input_file),
            'candidate_file': str(chunk_candidate_file),
            'failed_file': str(chunk_failed_file),
            'input_rows': len(chunk_rows),
            'expected_candidates': expected_candidates,
            'existing_candidates': existing_candidates,
        })
        if existing_candidates >= expected_candidates and expected_candidates > 0:
            print(json.dumps({'event': 'checkpoint_skip', 'chunk_index': chunk_index, 'existing_candidates': existing_candidates, 'expected_candidates': expected_candidates}, ensure_ascii=False))
            continue

        teacher_cmd = [
            sys.executable,
            'distill/distill_with_teacher.py',
            '--input_file', str(chunk_input_file),
            '--output_file', str(chunk_candidate_file),
            '--failed_output_file', str(chunk_failed_file),
            '--backend', args.teacher_backend,
            '--num_candidates', str(args.teacher_num_candidates),
            '--temperature_schedule', args.teacher_temperature_schedule,
            '--max_tokens', str(args.teacher_max_tokens),
            '--max_concurrency', str(args.teacher_max_concurrency),
            '--max_retries', str(args.teacher_max_retries),
            '--retry_sleep_seconds', str(args.teacher_retry_sleep_seconds),
            '--retry_backoff', str(args.teacher_retry_backoff),
            '--resume',
        ]
        if args.teacher_backend == 'openai':
            teacher_cmd.extend(['--provider', args.teacher_provider])
            if args.teacher_model:
                teacher_cmd.extend(['--model', args.teacher_model])
            if args.teacher_user_template_file:
                teacher_cmd.extend(['--user_template_file', args.teacher_user_template_file])
        run_command(teacher_cmd)

    with candidates_file.open('w', encoding='utf-8') as merged:
        for plan in checkpoint_plan:
            chunk_candidate_file = Path(plan['candidate_file'])
            if not chunk_candidate_file.exists():
                continue
            with chunk_candidate_file.open('r', encoding='utf-8') as src:
                shutil.copyfileobj(src, merged)

    with failed_file.open('w', encoding='utf-8') as merged_failed:
        for plan in checkpoint_plan:
            chunk_failed_file = Path(plan['failed_file'])
            if not chunk_failed_file.exists():
                continue
            with chunk_failed_file.open('r', encoding='utf-8') as src:
                shutil.copyfileobj(src, merged_failed)

    score_cmd = [
        sys.executable,
        'distill/score_distill_candidates.py',
        '--input_file', str(candidates_file),
        '--audit_output_file', str(audit_file),
        '--sft_output_file', str(sft_file),
        '--dpo_output_file', str(dpo_file),
        '--summary_output_file', str(summary_file),
    ]
    if args.summary_csv:
        score_cmd.extend(['--summary_csv_file', str(summary_csv_file)])
    if args.enable_reasoning_judge:
        score_cmd.extend([
            '--enable_reasoning_judge',
            '--judge_provider', args.judge_provider,
            '--judge_model', args.judge_model,
            '--judge_prompt_file', args.judge_prompt_file,
        ])
    run_command(score_cmd)

    manifest = {
        'work_dir': str(work_dir),
        'input_file': str(input_file),
        'candidates_file': str(candidates_file),
        'audit_file': str(audit_file),
        'sft_file': str(sft_file),
        'dpo_file': str(dpo_file),
        'failed_file': str(failed_file),
        'summary_file': str(summary_file),
        'summary_csv_file': str(summary_csv_file) if args.summary_csv else '',
        'checkpoint_dir': str(checkpoint_dir),
        'checkpoint_rows': args.checkpoint_rows,
        'checkpoint_count': len(checkpoint_plan),
        'teacher_backend': args.teacher_backend,
        'teacher_provider': args.teacher_provider,
        'teacher_model': args.teacher_model,
        'teacher_max_concurrency': args.teacher_max_concurrency,
        'teacher_max_retries': args.teacher_max_retries,
        'teacher_retry_sleep_seconds': args.teacher_retry_sleep_seconds,
        'teacher_retry_backoff': args.teacher_retry_backoff,
        'teacher_user_template_file': args.teacher_user_template_file,
        'enable_reasoning_judge': args.enable_reasoning_judge,
        'judge_provider': args.judge_provider,
        'judge_model': args.judge_model,
        'checkpoints': checkpoint_plan,
    }
    (work_dir / 'distill_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

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
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(cmd: List[str]) -> None:
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


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
    summary_file = work_dir / 'distill_summary.jsonl'
    summary_csv_file = work_dir / 'distill_summary.csv'

    build_cmd = [sys.executable, 'distill/build_financial_distill_dataset.py', '--output_file', str(input_file), '--max_samples_per_family', str(args.max_samples_per_family), '--convfinqa_keep_final_only', str(args.convfinqa_keep_final_only)]
    for spec in args.source_spec:
        build_cmd.extend(['--source_spec', spec])
    run_command(build_cmd)

    teacher_cmd = [
        sys.executable,
        'distill/distill_with_teacher.py',
        '--input_file', str(input_file),
        '--output_file', str(candidates_file),
        '--backend', args.teacher_backend,
        '--num_candidates', str(args.teacher_num_candidates),
        '--temperature_schedule', args.teacher_temperature_schedule,
        '--max_tokens', str(args.teacher_max_tokens),
    ]
    if args.teacher_backend == 'openai':
        teacher_cmd.extend(['--provider', args.teacher_provider])
        if args.teacher_model:
            teacher_cmd.extend(['--model', args.teacher_model])
    run_command(teacher_cmd)

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
        'summary_file': str(summary_file),
        'summary_csv_file': str(summary_csv_file) if args.summary_csv else '',
        'teacher_backend': args.teacher_backend,
        'teacher_provider': args.teacher_provider,
        'teacher_model': args.teacher_model,
        'enable_reasoning_judge': args.enable_reasoning_judge,
        'judge_provider': args.judge_provider,
        'judge_model': args.judge_model,
    }
    (work_dir / 'distill_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List


def run_command(cmd: List[str]) -> None:
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def append_file(src: Path, dst_handle) -> None:
    if not src.exists():
        return
    with src.open('r', encoding='utf-8') as f:
        shutil.copyfileobj(f, dst_handle)




def normalize_distill_output_path(path_str: str) -> Path:
    path = Path(path_str)
    normalized = str(path)
    normalized = normalized.replace('/outputs/financial_reasoning/', '/data/financial_reasoning/')
    normalized = normalized.replace('/outputs/financial_reasoning', '/data/financial_reasoning')
    return Path(normalized)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Score distill checkpoint shards and write results under checkpoints/.')
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--candidate_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--summary_csv', action='store_true')
    parser.add_argument('--generate_synthetic_rejected', action='store_true')
    parser.add_argument('--synthetic_rejected_modes', type=str, default='answer_perturb,think_empty,think_program')
    parser.add_argument('--synthetic_rejected_max_per_group', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--numeric_abs_tol', type=float, default=1e-4)
    parser.add_argument('--numeric_rel_tol', type=float, default=1e-4)
    parser.add_argument('--require_program_match_for_positive', action='store_true')
    parser.add_argument('--run_reasoning_selection_on_failed_answer_check', action='store_true')
    parser.add_argument('--enable_reasoning_judge', action='store_true')
    parser.add_argument('--judge_provider', type=str, default=None)
    parser.add_argument('--judge_api_key', type=str, default=None)
    parser.add_argument('--judge_base_url', type=str, default=None)
    parser.add_argument('--judge_model', type=str, default=None)
    parser.add_argument('--judge_prompt_file', type=str, default='domain/financial/distill/prompts/financial_reasoning_judge.txt')
    parser.add_argument('--judge_max_tokens', type=int, default=256)
    parser.add_argument('--reasoning_total_threshold', type=int, default=8)
    parser.add_argument('--reasoning_min_instruction_alignment', type=int, default=1)
    parser.add_argument('--reasoning_min_task_relevance', type=int, default=1)
    parser.add_argument('--reasoning_min_logical_coherence', type=int, default=1)
    parser.add_argument('--reasoning_min_evidence_quality', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_work_dir = Path(args.work_dir)
    work_dir = normalize_distill_output_path(args.work_dir)
    checkpoint_dir = work_dir / 'checkpoints'
    source_checkpoint_dir = source_work_dir / 'checkpoints'
    candidate_dir = Path(args.candidate_dir) if args.candidate_dir else source_checkpoint_dir / 'candidates'
    output_dir = normalize_distill_output_path(args.output_dir) if args.output_dir else checkpoint_dir / 'scored'

    audit_dir = output_dir / 'audit'
    sft_dir = output_dir / 'sft'
    dpo_dir = output_dir / 'dpo'
    rejected_dir = output_dir / 'rejected'
    merged_dir = output_dir / 'merged'
    summary_dir = output_dir / 'summary'
    summary_csv_dir = output_dir / 'summary_csv'
    for d in [audit_dir, sft_dir, dpo_dir, rejected_dir, merged_dir, summary_dir, summary_csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    parts = sorted(candidate_dir.glob('part_*.jsonl'))
    if not parts:
        raise FileNotFoundError(f'No checkpoint candidate shards found in {candidate_dir}')

    summary_manifest = []
    for part in parts:
        stem = part.stem
        audit_file = audit_dir / f'{stem}.audit.jsonl'
        sft_file = sft_dir / f'{stem}.sft.jsonl'
        dpo_file = dpo_dir / f'{stem}.dpo.jsonl'
        summary_file = summary_dir / f'{stem}.summary.jsonl'
        summary_csv_file = summary_csv_dir / f'{stem}.summary.csv'

        if args.resume and all(p.exists() for p in [audit_file, sft_file, dpo_file, summary_file]):
            print(json.dumps({'event': 'score_checkpoint_skip', 'part': stem}, ensure_ascii=False))
            summary_manifest.append({
                'part': stem,
                'input_rows': count_jsonl_rows(part),
                'audit_rows': count_jsonl_rows(audit_file),
                'sft_rows': count_jsonl_rows(sft_file),
                'dpo_rows': count_jsonl_rows(dpo_file),
                'summary_rows': count_jsonl_rows(summary_file),
                'skipped': True,
            })
            continue

        cmd = [
            sys.executable,
            '-m', 'domain.financial.distill.score_distill_candidates',
            '--input_file', str(part),
            '--audit_output_file', str(audit_file),
            '--sft_output_file', str(sft_file),
            '--dpo_output_file', str(dpo_file),
            '--summary_output_file', str(summary_file),
            '--numeric_abs_tol', str(args.numeric_abs_tol),
            '--numeric_rel_tol', str(args.numeric_rel_tol),
            '--judge_max_tokens', str(args.judge_max_tokens),
            '--reasoning_total_threshold', str(args.reasoning_total_threshold),
            '--reasoning_min_instruction_alignment', str(args.reasoning_min_instruction_alignment),
            '--reasoning_min_task_relevance', str(args.reasoning_min_task_relevance),
            '--reasoning_min_logical_coherence', str(args.reasoning_min_logical_coherence),
            '--reasoning_min_evidence_quality', str(args.reasoning_min_evidence_quality),
        ]
        if args.summary_csv:
            cmd.extend(['--summary_csv_file', str(summary_csv_file)])
        if args.require_program_match_for_positive:
            cmd.append('--require_program_match_for_positive')
        if args.run_reasoning_selection_on_failed_answer_check:
            cmd.append('--run_reasoning_selection_on_failed_answer_check')
        if args.enable_reasoning_judge:
            cmd.extend([
                '--enable_reasoning_judge',
                '--judge_provider', str(args.judge_provider or ''),
                '--judge_prompt_file', str(args.judge_prompt_file),
            ])
            if args.judge_api_key:
                cmd.extend(['--judge_api_key', args.judge_api_key])
            if args.judge_base_url:
                cmd.extend(['--judge_base_url', args.judge_base_url])
            if args.judge_model:
                cmd.extend(['--judge_model', args.judge_model])
        if not args.generate_synthetic_rejected:
            run_command(cmd)
        else:
            cmd.extend(['--skip_dpo'])
            run_command(cmd)

            rejected_file = rejected_dir / f'{stem}.rejected.jsonl'
            rejected_cmd = [
                sys.executable,
                '-m', 'domain.financial.distill.build_rejected_candidates',
                '--input_file', str(audit_file),
                '--output_file', str(rejected_file),
                '--modes', args.synthetic_rejected_modes,
                '--max_per_group', str(args.synthetic_rejected_max_per_group),
            ]
            run_command(rejected_cmd)

            merged_file = merged_dir / f'{stem}.merged.jsonl'
            with merged_file.open('w', encoding='utf-8') as out:
                append_file(part, out)
                append_file(rejected_file, out)

            dpo_cmd = [
                sys.executable,
                '-m', 'domain.financial.distill.score_distill_candidates',
                '--input_file', str(merged_file),
                '--audit_output_file', str(audit_file),
                '--sft_output_file', str(sft_file),
                '--dpo_output_file', str(dpo_file),
                '--summary_output_file', str(summary_file),
                '--skip_sft',
                '--skip_audit',
                '--skip_summary',
            ]
            if args.summary_csv:
                dpo_cmd.extend(['--summary_csv_file', str(summary_csv_file)])
            if args.require_program_match_for_positive:
                dpo_cmd.append('--require_program_match_for_positive')
            if args.run_reasoning_selection_on_failed_answer_check:
                dpo_cmd.append('--run_reasoning_selection_on_failed_answer_check')
            if args.enable_reasoning_judge:
                dpo_cmd.extend([
                    '--enable_reasoning_judge',
                    '--judge_provider', str(args.judge_provider or ''),
                    '--judge_prompt_file', str(args.judge_prompt_file),
                ])
                if args.judge_api_key:
                    dpo_cmd.extend(['--judge_api_key', args.judge_api_key])
                if args.judge_base_url:
                    dpo_cmd.extend(['--judge_base_url', args.judge_base_url])
                if args.judge_model:
                    dpo_cmd.extend(['--judge_model', args.judge_model])
            run_command(dpo_cmd)

        summary_manifest.append({
            'part': stem,
            'input_rows': count_jsonl_rows(part),
            'audit_rows': count_jsonl_rows(audit_file),
            'sft_rows': count_jsonl_rows(sft_file),
            'dpo_rows': count_jsonl_rows(dpo_file),
            'summary_rows': count_jsonl_rows(summary_file),
            'skipped': False,
        })

    merged_audit = work_dir / 'distill_audit.jsonl'
    merged_sft = work_dir / 'distill_sft.jsonl'
    merged_dpo = work_dir / 'distill_dpo.jsonl'
    merged_summary = work_dir / 'distill_summary.jsonl'

    with merged_audit.open('w', encoding='utf-8') as out:
        for src in sorted(audit_dir.glob('part_*.audit.jsonl')):
            append_file(src, out)
    with merged_sft.open('w', encoding='utf-8') as out:
        for src in sorted(sft_dir.glob('part_*.sft.jsonl')):
            append_file(src, out)
    with merged_dpo.open('w', encoding='utf-8') as out:
        for src in sorted(dpo_dir.glob('part_*.dpo.jsonl')):
            append_file(src, out)
    with merged_summary.open('w', encoding='utf-8') as out:
        for src in sorted(summary_dir.glob('part_*.summary.jsonl')):
            append_file(src, out)

    manifest_file = output_dir / 'score_manifest.json'
    manifest = {
        'work_dir': str(work_dir),
        'candidate_dir': str(candidate_dir),
        'output_dir': str(output_dir),
        'parts': summary_manifest,
        'merged_audit': str(merged_audit),
        'merged_sft': str(merged_sft),
        'merged_dpo': str(merged_dpo),
        'merged_summary': str(merged_summary),
        'total_parts': len(summary_manifest),
        'total_input_rows': sum(item['input_rows'] for item in summary_manifest),
        'total_sft_rows': sum(item['sft_rows'] for item in summary_manifest),
        'total_dpo_rows': sum(item['dpo_rows'] for item in summary_manifest),
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

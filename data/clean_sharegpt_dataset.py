#!/usr/bin/env python3
"""Clean ShareGPT-style SFT jsonl data for MedicalGPT training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--min_turns', default=2, type=int)
    parser.add_argument('--max_turns', default=20, type=int)
    parser.add_argument('--max_total_chars', default=4000, type=int)
    parser.add_argument('--max_single_value_chars', default=2000, type=int)
    parser.add_argument('--drop_odd_turns', action='store_true', default=True)
    parser.add_argument('--keep_long_multiturn', action='store_true', help='Keep long multi-turn samples that exceed thresholds.')
    return parser.parse_args()


def normalize_turn(turn: Dict[str, Any]) -> Dict[str, str] | None:
    if not isinstance(turn, dict):
        return None
    role = turn.get('from')
    value = turn.get('value')
    if role not in {'human', 'gpt'}:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return {'from': role, 'value': value}


def main() -> None:
    args = parse_args()
    src = Path(args.source_file)
    dst = Path(args.output_file)
    dst.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    kept_turn_hist = Counter()
    kept_char_samples: List[int] = []

    with src.open('r', encoding='utf-8') as rf, dst.open('w', encoding='utf-8') as wf:
        for line in rf:
            line = line.strip()
            if not line:
                stats['skip_blank_line'] += 1
                continue
            stats['raw_records'] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats['drop_bad_json'] += 1
                continue

            conv = obj.get('conversations')
            if not isinstance(conv, list) or not conv:
                stats['drop_missing_conversations'] += 1
                continue

            cleaned: List[Dict[str, str]] = []
            bad_turn = False
            for idx, turn in enumerate(conv):
                normalized = normalize_turn(turn)
                if normalized is None:
                    stats['drop_invalid_turn'] += 1
                    bad_turn = True
                    break
                expected_role = 'human' if idx % 2 == 0 else 'gpt'
                if normalized['from'] != expected_role:
                    stats['drop_role_order'] += 1
                    bad_turn = True
                    break
                if len(normalized['value']) > args.max_single_value_chars:
                    stats['drop_overlong_turn'] += 1
                    bad_turn = True
                    break
                cleaned.append(normalized)
            if bad_turn:
                continue

            if len(cleaned) < args.min_turns:
                stats['drop_too_few_turns'] += 1
                continue
            if args.drop_odd_turns and len(cleaned) % 2 != 0:
                stats['drop_odd_turns'] += 1
                continue
            if len(cleaned) > args.max_turns and not args.keep_long_multiturn:
                stats['drop_too_many_turns'] += 1
                continue

            total_chars = sum(len(turn['value']) for turn in cleaned)
            if total_chars > args.max_total_chars and not args.keep_long_multiturn:
                stats['drop_overlong_sample'] += 1
                continue

            out = dict(obj)
            out['conversations'] = cleaned
            wf.write(json.dumps(out, ensure_ascii=False) + '\n')
            stats['kept_records'] += 1
            kept_turn_hist[len(cleaned)] += 1
            kept_char_samples.append(total_chars)

    report = {
        'source_file': str(src),
        'output_file': str(dst),
        'stats': dict(stats),
        'kept_turn_hist_top10': kept_turn_hist.most_common(10),
    }
    if kept_char_samples:
        xs = sorted(kept_char_samples)

        def pct(q: float) -> int:
            return xs[min(len(xs) - 1, int((len(xs) - 1) * q))]

        report['kept_total_chars'] = {
            'min': xs[0],
            'p50': pct(0.5),
            'p90': pct(0.9),
            'p95': pct(0.95),
            'p99': pct(0.99),
            'max': xs[-1],
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

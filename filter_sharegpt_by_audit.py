#!/usr/bin/env python3
"""Filter ShareGPT jsonl by audit review output."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', required=True)
    p.add_argument('--review_file', required=True)
    p.add_argument('--output_file', required=True)
    p.add_argument('--mode', default='conservative', choices=['conservative', 'strict'])
    return p.parse_args()


def load_review_map(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            out[int(obj['line_no'])] = obj
    return out


def should_drop(rec: Dict[str, Any], mode: str) -> bool:
    reasons: Set[str] = set(rec.get('reasons', []))
    if mode == 'strict':
        return True

    # conservative: only remove high-risk patterns
    if rec.get('severity') == 'high':
        return True
    if 'same_prompt_conflicting_labels' in reasons:
        return True
    if 'evidence_placeholder_text_1' in reasons:
        return True
    if 'long_prompt_short_answer' in reasons:
        return True
    if 'short_answer' in reasons and 'very_low_answer_prompt_ratio' in reasons:
        return True
    return False


def main() -> None:
    args = parse_args()
    inp = Path(args.input_file)
    rev = Path(args.review_file)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    review_map = load_review_map(rev)
    stats = Counter()
    reason_drop = Counter()

    with inp.open('r', encoding='utf-8') as rf, out.open('w', encoding='utf-8') as wf:
        for ln, line in enumerate(rf, 1):
            line = line.rstrip('\n')
            if not line:
                continue
            stats['total'] += 1
            rec = review_map.get(ln)
            if rec and should_drop(rec, args.mode):
                stats['dropped'] += 1
                for r in rec.get('reasons', []):
                    reason_drop[r] += 1
                continue
            wf.write(line + '\n')
            stats['kept'] += 1

    summary = {
        'input_file': str(inp),
        'review_file': str(rev),
        'output_file': str(out),
        'mode': args.mode,
        'stats': dict(stats),
        'drop_reasons': dict(reason_drop),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

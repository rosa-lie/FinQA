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
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        rows = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        return [dict(x) for x in data]
    if isinstance(data, dict):
        for key in ['data', 'records', 'examples', 'items']:
            if isinstance(data.get(key), list):
                return [dict(x) for x in data[key]]
        return [dict(data)]
    raise ValueError(f'Unsupported file format: {path}')


def to_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return json.dumps(value, ensure_ascii=False)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or '').lower())


def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def extract_field(row: Dict[str, Any], field: str) -> str:
    if field == 'prompt' and 'prompt' in row:
        return to_text(row.get('prompt'))
    value = row.get(field)
    if value is not None:
        return to_text(value)
    conv = row.get('conversations')
    if field == 'prompt' and isinstance(conv, list) and conv:
        return to_text(conv[0].get('value'))
    return ''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Decontaminate distillation data against reference prompts using n-gram overlap.')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--reference_file', action='append', required=True)
    parser.add_argument('--input_text_field', type=str, default='prompt')
    parser.add_argument('--reference_text_field', type=str, default='prompt')
    parser.add_argument('--ngram_size', type=int, default=8)
    parser.add_argument('--summary_file', type=str, default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_rows = load_rows(Path(args.input_file))
    reference_rows: List[Dict[str, Any]] = []
    for ref in args.reference_file:
        reference_rows.extend(load_rows(Path(ref)))

    reference_ngrams: Set[Tuple[str, ...]] = set()
    for row in reference_rows:
        text = extract_field(row, args.reference_text_field)
        reference_ngrams.update(ngrams(tokenize(text), args.ngram_size))

    kept: List[Dict[str, Any]] = []
    contaminated: List[Dict[str, Any]] = []
    for row in input_rows:
        text = extract_field(row, args.input_text_field)
        row_ngrams = ngrams(tokenize(text), args.ngram_size)
        if row_ngrams and row_ngrams.intersection(reference_ngrams):
            contaminated.append(row)
        else:
            kept.append(row)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = {
        'input_rows': len(input_rows),
        'reference_rows': len(reference_rows),
        'kept_rows': len(kept),
        'contaminated_rows': len(contaminated),
        'ngram_size': args.ngram_size,
        'input_file': args.input_file,
        'output_file': args.output_file,
        'reference_files': args.reference_file,
    }
    if args.summary_file:
        Path(args.summary_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

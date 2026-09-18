#!/usr/bin/env python3
"""Gate: 5 edge_multihop examples, visual inspection before the full 29."""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_rows import format_arxiv_row
from generate_qa import (
    sample_edge_multihop_pairs, generate_edge_multihop_qa,
    validate_edge_multihop, _client,
)
from evaluate_rag import load_pooled_samples

if __name__ == '__main__':
    client = _client()
    df = load_pooled_samples()
    print(f'pool: {len(df)} unique rows\n')

    pairs = sample_edge_multihop_pairs(df, n_pairs=5, seed=42)
    print(f'sampled {len(pairs)}/5 valid hop pairs\n')

    n_pass = 0
    for i, p in enumerate(pairs):
        anchor_text = format_arxiv_row(p['anchor'])
        target_text = format_arxiv_row(p['target'])
        print(f'=== pair {i} ===')
        print(f'anchor primary_id={p["anchor"]["primary_id"]}  (aggregate_id={p["anchor"]["aggregate_id"]})')
        print(f'target primary_id={p["target"]["primary_id"]}  (aggregate_id={p["target"]["aggregate_id"]})')
        print(f'shared_author: {p["shared_author"]}')

        if anchor_text is None or target_text is None:
            print('SKIP: null/short text on one side')
            print()
            continue

        try:
            qa = generate_edge_multihop_qa(anchor_text, target_text, p['shared_author'], client=client)
        except ValueError as e:
            print(f'FAILED: {e}')
            print()
            continue

        valid, reason = validate_edge_multihop(qa, p['shared_author'])
        print(f'Q: {qa["question"]}')
        print(f'A: {qa["answer"]}')
        print(f'validate_edge_multihop -> ({valid}, {reason!r})')
        if valid:
            n_pass += 1
        print()

    print(f'RESULT: {n_pass}/5 passed validation')

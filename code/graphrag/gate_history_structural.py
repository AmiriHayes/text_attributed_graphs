#!/usr/bin/env python3
"""History edge_multihop gate re-run using the structural_edges fallback."""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_rows import format_arxiv_row
from generate_qa import (
    generate_edge_multihop_qa, validate_edge_multihop, _client,
    sample_structural_edge_pairs, HISTORY_STRUCTURAL_EDGE_SYSTEM_PROMPT,
)
from evaluate_rag import load_pooled_samples

if __name__ == '__main__':
    client = _client()
    pool = load_pooled_samples(dataset='history')

    pairs = sample_structural_edge_pairs(pool, n_pairs=5, seed=42)
    print(f'sampled {len(pairs)}/5 structural_edges pairs\n')

    n_pass = 0
    for i, p in enumerate(pairs):
        anchor_text = format_arxiv_row(p['anchor'])
        target_text = format_arxiv_row(p['target'])
        print(f'--- pair {i} ---')
        print(f'anchor: {p["anchor"]["primary_id"]}  ({p["anchor"]["text_fidelity_b"]})')
        print(f'target: {p["target"]["primary_id"]}  ({p["target"]["text_fidelity_b"]})')
        if anchor_text is None or target_text is None:
            print('SKIP null/short text')
            continue
        try:
            qa = generate_edge_multihop_qa(
                anchor_text, target_text,
                shared_author='documented related work',
                client=client,
                system_prompt=HISTORY_STRUCTURAL_EDGE_SYSTEM_PROMPT,
                anchor_label='Book 1',
                target_label='Book 2 (documented related work)',
                connector_label='Connection',
            )
        except ValueError as e:
            print(f'FAILED: {e}')
            continue
        valid, reason = validate_edge_multihop(
            qa, 'documented related work',
            connector_words=['related work', 'both books', 'documented', 'connection', 'related'])
        print(f'Q: {qa["question"]}')
        print(f'A: {qa["answer"]}')
        print(f'-> ({valid}, {reason!r})')
        if valid:
            n_pass += 1
        print()

    print(f'GATE: {"PASS" if n_pass >= 3 else "FAIL"} ({n_pass}/5)')

#!/usr/bin/env python3
"""
Generic Amazon-schema question generation (full 102: 35 single /
33 aggregate / 34 edge_multihop) -- parameterized by --dataset so
electronics/toys (identical schema to amazon: scalar secondary_id,
broken binary categorical_label, aggregate_id family grouping) reuse
the exact validated Amazon pipeline without duplication.
Usage: python run_question_generation_amazon_schema.py --dataset electronics
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_rows import format_arxiv_row
from generate_qa import (
    generate_qa, generate_aggregate_qa, generate_edge_multihop_qa,
    validate_qa, validate_edge_multihop, classify_question_type,
    sample_cross_category_groups, sample_edge_multihop_pairs, _client,
    AMAZON_AGGREGATE_SYSTEM_PROMPT, AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
)
from evaluate_rag import load_pooled_samples

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

N_SINGLE_TARGET, N_SINGLE_BUDGET = 35, 100
N_AGG_TARGET, N_AGG_BUDGET = 33, 60
N_EDGE_TARGET, N_EDGE_BUDGET = 34, 110


def run_singlehop(df, client, seed=42):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    results, rejections, attempts = [], Counter(), 0
    for _, row in shuffled.iterrows():
        if len(results) >= N_SINGLE_TARGET or attempts >= N_SINGLE_BUDGET:
            break
        attempts += 1
        text = format_arxiv_row(row.to_dict())
        if text is None:
            rejections['formatter: null/short text'] += 1
            continue
        try:
            qa = generate_qa(text, client=client)
        except ValueError as e:
            rejections[f'API/parse: {type(e).__name__}'] += 1
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            rejections[reason] += 1
            continue
        results.append({
            'question': qa['question'], 'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'single_specific',
            'source_row_id': row['primary_id'], 'source_category_labels': None,
            'source_text_preview': text[:100], 'hop_type': 'single', 'hop_source_ids': None,
        })
    print(f'single-hop: {len(results)}/{N_SINGLE_TARGET} in {attempts} attempts')
    return results, rejections


def run_aggregate(df, client, seed=42):
    groups = sample_cross_category_groups(df, n_groups=N_AGG_BUDGET, group_size_range=(5, 10),
                                            min_categories=3, seed=seed)
    print(f'aggregate: found {len(groups)} candidate family-diverse groups (target budget {N_AGG_BUDGET})')
    results, rejections, attempts = [], Counter(), 0
    for g in groups:
        if len(results) >= N_AGG_TARGET or attempts >= N_AGG_BUDGET:
            break
        attempts += 1
        families = sorted(set(r['aggregate_id'] for r in g))
        try:
            qa = generate_aggregate_qa(g, client=client, system_prompt=AMAZON_AGGREGATE_SYSTEM_PROMPT,
                                        item_label='Product', text_fields=('text_fidelity_b',))
        except ValueError as e:
            rejections[f'API/parse: {type(e).__name__}'] += 1
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            rejections[reason] += 1
            continue
        results.append({
            'question': qa['question'], 'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'aggregate_cross_paper',
            'source_row_id': None, 'source_category_labels': families,
            'source_text_preview': None, 'hop_type': 'single', 'hop_source_ids': None,
        })
    print(f'aggregate: {len(results)}/{N_AGG_TARGET} in {attempts} attempts')
    return results, rejections


def run_edge_multihop(df, client, seed=42):
    pairs = sample_edge_multihop_pairs(df, n_pairs=N_EDGE_BUDGET, seed=seed)
    print(f'edge_multihop: sampled {len(pairs)} candidate reviewer-hop pairs')
    results, rejections, attempts = [], Counter(), 0
    for p in pairs:
        if len(results) >= N_EDGE_TARGET or attempts >= N_EDGE_BUDGET:
            break
        attempts += 1
        at = format_arxiv_row(p['anchor'])
        tt = format_arxiv_row(p['target'])
        if at is None or tt is None:
            rejections['null_text'] += 1
            continue
        try:
            qa = generate_edge_multihop_qa(
                at, tt, p['shared_author'], client=client,
                system_prompt=AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
                anchor_label='Product 1 (anchor)',
                target_label='Product 2 (connected via shared reviewer)',
                connector_label='Shared reviewer identifier',
            )
        except ValueError as e:
            rejections[f'API/parse: {type(e).__name__}'] += 1
            continue
        valid, reason = validate_edge_multihop(qa, p['shared_author'], connector_words=['reviewer'])
        if not valid:
            rejections[reason] += 1
            continue
        results.append({
            'question': qa['question'], 'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'edge_multihop',
            'source_row_id': None, 'source_category_labels': None,
            'source_text_preview': None, 'hop_type': 'single',
            'hop_source_ids': [p['anchor']['primary_id'], p['target']['primary_id']],
        })
    print(f'edge_multihop: {len(results)}/{N_EDGE_TARGET} in {attempts} attempts')
    return results, rejections


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    args = ap.parse_args()
    ds = args.dataset
    OUT_DIR = REPO_ROOT / f'data/{ds}'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = _client()
    print(f'Loading {ds} pool (sample_00-09, deduped)...')
    df = load_pooled_samples(dataset=ds)
    print(f'  {len(df)} unique rows\n')

    print('=== SINGLE-HOP ===')
    single_results, single_rej = run_singlehop(df, client)

    print('\n=== AGGREGATE ===')
    agg_results, agg_rej = run_aggregate(df, client)

    print('\n=== EDGE-MULTIHOP ===')
    edge_results, edge_rej = run_edge_multihop(df, client)

    all_results = []
    for i, r in enumerate(single_results):
        all_results.append({'id': f'{ds}_sh_{i:03d}', 'dataset': ds, **r, 'multi_hop_stub': None})
    for i, r in enumerate(agg_results):
        all_results.append({'id': f'{ds}_agg_{i:03d}', 'dataset': ds, **r, 'multi_hop_stub': None})
    for i, r in enumerate(edge_results):
        all_results.append({'id': f'{ds}_edge_{i:03d}', 'dataset': ds, **r, 'multi_hop_stub': None})

    out_df = pd.DataFrame(all_results, columns=[
        'id', 'dataset', 'question', 'reference_answer', 'question_type', 'question_subtype',
        'source_row_id', 'source_category_labels', 'source_text_preview', 'hop_type',
        'multi_hop_stub', 'hop_source_ids',
    ])
    out_path = OUT_DIR / 'questions.csv'
    out_df.to_csv(out_path, index=False)

    print('\n' + '=' * 20, 'FINAL REPORT', '=' * 20)
    print(f'Total: {len(single_results)} single / {len(agg_results)} aggregate / {len(edge_results)} edge_multihop '
          f'({len(all_results)}/102)')

    print('\nRejection breakdown:')
    for label, rej in [('single', single_rej), ('aggregate', agg_rej), ('edge_multihop', edge_rej)]:
        print(f'  {label}: {dict(rej) if rej else "none"}')

    print('\nQuestion type distribution:')
    for qt, n in Counter(r['question_type'] for r in all_results).most_common():
        print(f'  {qt:<12} {n}')

    print(f'\nSaved -> {out_path}')

    print('\n3 example questions per subtype:')
    for subtype, results in [('single_specific', single_results), ('aggregate_cross_paper', agg_results),
                              ('edge_multihop', edge_results)]:
        print(f'\n  {subtype}:')
        for r in results[:3]:
            print(f'    Q: {r["question"]}')

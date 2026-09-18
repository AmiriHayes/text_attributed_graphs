#!/usr/bin/env python3
"""
run_question_generation.py — Stage 3 QA generation, Step 6: generate the
full 50 for ArXiv (35 single-hop + 15 aggregate).

Output saved as CSV, not JSONL as originally specced — the user explicitly
asked for CSV for quick visual validation earlier in this pass, and that
still holds at this scale (50 rows); list-valued fields (source_category_labels)
are stored as their str() repr, readable in a spreadsheet and re-parseable
with ast.literal_eval if needed.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_rows import format_arxiv_row
from generate_qa import (
    generate_qa, generate_aggregate_qa, validate_qa, classify_question_type,
    sample_cross_category_groups, _client,
)
from evaluate_rag import load_pooled_samples

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / 'data/arxiv'

N_SINGLE_TARGET = 35
N_SINGLE_BUDGET = 100
N_AGG_TARGET = 15
N_AGG_BUDGET = 25


def run_singlehop_loop(df: pd.DataFrame, client, seed=42):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    results = []
    rejections = Counter()
    attempts = 0

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
            rejections[f'API/parse failure: {type(e).__name__}'] += 1
            print(f'  [single #{attempts}] API/parse failure: {e}')
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            rejections[reason] += 1
            continue
        results.append({
            'question': qa['question'],
            'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'single_specific',
            'source_row_id': row['primary_id'],
            'source_category_labels': None,
            'source_text_preview': text[:100],
            'hop_type': 'single',
        })

    print(f'Single-hop: {len(results)}/{N_SINGLE_TARGET} collected in {attempts} attempts')
    return results, rejections, attempts


def run_aggregate_loop(df: pd.DataFrame, client, seed=42):
    groups = sample_cross_category_groups(
        df, n_groups=N_AGG_BUDGET, group_size_range=(5, 10), min_categories=3, seed=seed)
    print(f'Aggregate: found {len(groups)} cross-category groups (>=3 categories, target budget {N_AGG_BUDGET})')

    results = []
    rejections = Counter()
    attempts = 0

    for g in groups:
        if len(results) >= N_AGG_TARGET or attempts >= N_AGG_BUDGET:
            break
        attempts += 1
        cats = sorted(set(r['aggregate_id'] for r in g))
        try:
            qa = generate_aggregate_qa(g, client=client)
        except ValueError as e:
            rejections[f'API/parse failure: {type(e).__name__}'] += 1
            print(f'  [aggregate #{attempts}] API/parse failure: {e}')
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            rejections[reason] += 1
            continue
        results.append({
            'question': qa['question'],
            'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'aggregate_cross_paper',
            'source_row_id': None,
            'source_category_labels': cats,
            'source_text_preview': None,
            'hop_type': 'single',  # per schema: hop_type is single/multi at the *retrieval* granularity level;
                                    # aggregate is a distinct axis (question_subtype), not the graph-multihop stub
        })

    print(f'Aggregate: {len(results)}/{N_AGG_TARGET} collected in {attempts} attempts')
    if len(results) < N_AGG_TARGET:
        print(f'  NOTE: did not reach {N_AGG_TARGET} within {N_AGG_BUDGET} group attempts — '
              f'per instructions, stopping here rather than lowering the 3-category minimum.')
    return results, rejections, attempts


if __name__ == '__main__':
    client = _client()

    print('Loading pooled sample_00-09...')
    df = load_pooled_samples()
    print(f'  {len(df)} unique rows after dedup on primary_id\n')

    print('=== SINGLE-HOP LOOP ===')
    single_results, single_rej, single_attempts = run_singlehop_loop(df, client)

    print('\n=== AGGREGATE LOOP ===')
    agg_results, agg_rej, agg_attempts = run_aggregate_loop(df, client)

    all_results = []
    for i, r in enumerate(single_results):
        all_results.append({'id': f'arxiv_sh_{i:03d}', 'dataset': 'arxiv', **r, 'multi_hop_stub': None})
    for i, r in enumerate(agg_results):
        all_results.append({'id': f'arxiv_agg_{i:03d}', 'dataset': 'arxiv', **r, 'multi_hop_stub': None})

    out_df = pd.DataFrame(all_results, columns=[
        'id', 'dataset', 'question', 'reference_answer', 'question_type',
        'question_subtype', 'source_row_id', 'source_category_labels',
        'source_text_preview', 'hop_type', 'multi_hop_stub',
    ])
    out_path = OUT_DIR / 'questions.csv'
    out_df.to_csv(out_path, index=False)

    print('\n' + '=' * 20 + ' STEP 6 REPORT ' + '=' * 20)
    print(f'Total generated: {len(single_results)} single-hop, {len(agg_results)} aggregate '
          f'({len(all_results)}/50 total)')

    print('\nSingle-hop rejection breakdown:')
    for reason, n in single_rej.most_common():
        print(f'  {n:3d}  {reason}')
    print('\nAggregate rejection breakdown:')
    for reason, n in agg_rej.most_common():
        print(f'  {n:3d}  {reason}')

    print('\nQuestion type distribution (across all 50):')
    for qt, n in Counter(r['question_type'] for r in all_results).most_common():
        print(f'  {qt:<12} {n}')

    print('\nCategory coverage (aggregate questions):')
    all_cats = Counter()
    for r in agg_results:
        for c in r['source_category_labels']:
            all_cats[c] += 1
    for c, n in all_cats.most_common():
        print(f'  {c:<12} appeared in {n} groups')

    print(f'\nSaved -> {out_path}')

#!/usr/bin/env python3
"""
extend_questions_to_150.py — Phase 0 of the final rerun: extend each
dataset's questions.csv from 102 -> 150 by APPENDING 48 new questions,
reusing the exact same generation pipeline (generate_qa.py), validation
(validate_qa), and RAGAS pilot gate (evaluate_rag.py) that produced the
original 102 -- same prompts, same schema, same gate threshold.

Priority order (cheapest/most-stable first, per spec):
  1. single_specific  (target 48)
  2. aggregate_cross_paper  (only if (1) falls short of 48)
  3. edge_multihop  (only if (1)+(2) still fall short of 48)

New questions draw from pool rows NOT already used as source_row_id in the
existing 102 (single_specific only -- aggregate/edge_multihop don't have a
single source_row_id), with a different sampling seed, so this never
regenerates a question that's already in the file. Never overwrites
existing rows -- pure append, continuing the id numbering
({ds}_sh_NNN / _agg_NNN / _edge_NNN) from the existing max.

RAGAS pilot gate: after generating the new batch, build the same plain
VectorStoreIndex baseline evaluate_rag.py uses, score the first 8 new
questions with RAGAS, and require mean faithfulness > 0.5 before accepting
the batch into questions.csv. Reported explicitly either way.

Reads:
  - data/{dataset}/questions.csv (existing 102)
  - data/{dataset}/train/samples/sample_00..09.jsonl (pool)

Writes:
  - data/{dataset}/questions.csv (150 rows, existing rows byte-identical)

Usage:
  python3 extend_questions_to_150.py --dataset arxiv
"""
import argparse
import sys
import types
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── ragas 0.2.15 / langchain_community incompatibility workaround ──────────
# ragas/llms/base.py unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai for a type reference it never
# actually instantiates unless VertexAI is requested (never, here). Current
# langchain-community no longer ships that module (moved to a separate
# langchain-google-vertexai package upstream). Stub it out rather than
# downgrade langchain-community, which cascades into a langchain-core
# version conflict with the installed langchain_openai/langgraph stack
# (see STAGE3_STATUS.md's prior note on this exact tradeoff).
_dummy = types.ModuleType('langchain_community.chat_models.vertexai')
class _ChatVertexAIStub:
    pass
_dummy.ChatVertexAI = _ChatVertexAIStub
sys.modules.setdefault('langchain_community.chat_models.vertexai', _dummy)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_rows import format_arxiv_row  # generic formatter, works on any dataset's canonical schema
from generate_qa import (
    generate_qa, generate_aggregate_qa, generate_edge_multihop_qa,
    validate_qa, validate_edge_multihop, classify_question_type,
    sample_cross_category_groups, sample_edge_multihop_pairs, _client,
    AMAZON_AGGREGATE_SYSTEM_PROMPT, AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
)
from evaluate_rag import load_pooled_samples, build_index, evaluate_with_ragas

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TARGET_NEW = 48
SEED = 142  # different from the original 102's seed=42, so shuffled order differs
PILOT_N = 8
PILOT_FAITHFULNESS_THRESHOLD = 0.5


def run_singlehop_extend(df, used_row_ids, client, target, seed=SEED):
    pool = df[~df['primary_id'].isin(used_row_ids)]
    shuffled = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    budget = target * 3
    results, rejections, attempts = [], Counter(), 0
    for _, row in shuffled.iterrows():
        if len(results) >= target or attempts >= budget:
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
    print(f'  single-hop: {len(results)}/{target} in {attempts} attempts (pool excl. {len(used_row_ids)} already-used rows)')
    return results, rejections


def run_aggregate_extend(df, client, target, seed=SEED):
    budget = target * 3
    groups = sample_cross_category_groups(df, n_groups=budget, group_size_range=(5, 10),
                                            min_categories=3, seed=seed)
    results, rejections, attempts = [], Counter(), 0
    for g in groups:
        if len(results) >= target or attempts >= budget:
            break
        attempts += 1
        families = sorted(set(r['aggregate_id'] for r in g))
        try:
            qa = generate_aggregate_qa(g, client=client, system_prompt=AMAZON_AGGREGATE_SYSTEM_PROMPT,
                                        item_label='Item', text_fields=('text_fidelity_b',))
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
    print(f'  aggregate: {len(results)}/{target} in {attempts} attempts')
    return results, rejections


def run_edge_multihop_extend(df, client, target, seed=SEED):
    budget = target * 3
    pairs = sample_edge_multihop_pairs(df, n_pairs=budget, seed=seed)
    results, rejections, attempts = [], Counter(), 0
    for p in pairs:
        if len(results) >= target or attempts >= budget:
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
                anchor_label='Item 1 (anchor)', target_label='Item 2 (connected)',
                connector_label='Shared connector identifier',
            )
        except ValueError as e:
            rejections[f'API/parse: {type(e).__name__}'] += 1
            continue
        valid, reason = validate_edge_multihop(qa, p['shared_author'], connector_words=['shared'])
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
    print(f'  edge_multihop: {len(results)}/{target} in {attempts} attempts')
    return results, rejections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--skip_pilot_gate', action='store_true',
                     help='debug only -- skip the RAGAS pilot gate and accept unconditionally')
    args = ap.parse_args()
    ds = args.dataset

    q_path = REPO_ROOT / f'data/{ds}/questions.csv'
    existing = pd.read_csv(q_path)
    n_original = len(existing)
    print(f'=== {ds}: {n_original} existing questions ===')
    print('  existing subtype distribution:', dict(existing['question_subtype'].value_counts()))

    used_row_ids = set(existing['source_row_id'].dropna().astype(str))
    max_idx = {
        'sh':   existing[existing['id'].str.contains('_sh_', na=False)]['id'].str.extract(r'_(\d+)$')[0].astype(int).max() if (existing['id'].str.contains('_sh_', na=False)).any() else -1,
        'agg':  existing[existing['id'].str.contains('_agg_', na=False)]['id'].str.extract(r'_(\d+)$')[0].astype(int).max() if (existing['id'].str.contains('_agg_', na=False)).any() else -1,
        'edge': existing[existing['id'].str.contains('_edge_', na=False)]['id'].str.extract(r'_(\d+)$')[0].astype(int).max() if (existing['id'].str.contains('_edge_', na=False)).any() else -1,
    }

    client = _client()
    print(f'Loading {ds} pool (sample_00-09, deduped)...')
    df = load_pooled_samples(dataset=ds)
    print(f'  {len(df)} unique rows\n')

    remaining = TARGET_NEW
    all_new = []

    print('=== SINGLE-HOP (priority 1) ===')
    single_results, single_rej = run_singlehop_extend(df, used_row_ids, client, target=remaining)
    all_new += [('sh', r) for r in single_results]
    remaining -= len(single_results)

    agg_results, edge_results = [], []
    if remaining > 0:
        print(f'\n=== AGGREGATE (priority 2) -- {remaining} more needed ===')
        agg_results, agg_rej = run_aggregate_extend(df, client, target=remaining)
        all_new += [('agg', r) for r in agg_results]
        remaining -= len(agg_results)

    if remaining > 0:
        print(f'\n=== EDGE-MULTIHOP (priority 3) -- {remaining} more needed ===')
        edge_results, edge_rej = run_edge_multihop_extend(df, client, target=remaining)
        all_new += [('edge', r) for r in edge_results]
        remaining -= len(edge_results)

    if remaining > 0:
        print(f'\n  WARNING: only collected {TARGET_NEW - remaining}/{TARGET_NEW} new questions '
              f'after exhausting all 3 subtypes -- reporting discrepancy, not silently short.')

    if not all_new:
        print('No new questions generated -- aborting without writing.')
        return

    # ── RAGAS pilot gate ──
    if not args.skip_pilot_gate:
        pilot = [r for _, r in all_new[:PILOT_N]]
        print(f'\n=== RAGAS PILOT GATE ({len(pilot)} of the new questions) ===')
        print('  building baseline VectorStoreIndex over the pool...')
        index = build_index(df)
        pilot_q = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                    'question_type': r['question_type']} for r in pilot]
        scored = evaluate_with_ragas(pilot_q, index)
        mean_faith = scored['faithfulness'].mean()
        n_nan = int(scored['faithfulness'].isna().sum())
        print(f'  pilot faithfulness scores: {scored["faithfulness"].tolist()}')
        print(f'  pilot mean faithfulness: {mean_faith}  (threshold: >{PILOT_FAITHFULNESS_THRESHOLD}, {n_nan}/{len(scored)} NaN)')
        # NaN must be treated as a hard failure, not silently pass a `<=` check
        # (nan <= threshold is False in Python/numpy, which would otherwise
        # let a fully-broken scoring run through as "GATE PASSED").
        if pd.isna(mean_faith) or n_nan > 0 or mean_faith <= PILOT_FAITHFULNESS_THRESHOLD:
            print(f'  GATE FAILED (mean={mean_faith}, {n_nan} NaN scores) -- not writing new questions to {q_path}. '
                  f'Reporting this honestly rather than lowering the threshold or treating NaN as a pass.')
            return
        print('  GATE PASSED.')
    else:
        print('\n=== RAGAS PILOT GATE SKIPPED (--skip_pilot_gate) ===')

    # ── assign continued ids, append ──
    new_rows = []
    counters = {'sh': max_idx['sh'] + 1, 'agg': max_idx['agg'] + 1, 'edge': max_idx['edge'] + 1}
    for kind, r in all_new:
        idx = counters[kind]
        counters[kind] += 1
        new_rows.append({'id': f'{ds}_{kind}_{idx:03d}', 'dataset': ds, **r, 'multi_hop_stub': None})

    new_df = pd.DataFrame(new_rows, columns=existing.columns)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(q_path, index=False)

    print(f'\n=== REPORT: {ds} ===')
    print(f'  original: {n_original}')
    print(f'  added:    {len(new_rows)}')
    print(f'  final:    {len(combined)}')
    print(f'  final subtype distribution: {dict(combined["question_subtype"].value_counts())}')
    print(f'  saved -> {q_path}')


if __name__ == '__main__':
    main()

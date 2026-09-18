#!/usr/bin/env python3
"""
run_full_ragas.py — Stage 3 QA generation, Step 7: full RAGAS eval on all 50.

Reads data/arxiv/questions.csv (Step 6 output), builds the VectorStoreIndex
from the same sample pool, scores all 50 with RAGAS, saves full per-question
results to data/arxiv/ragas_baseline.csv (CSV, not JSONL — see Step 6 note),
and reports single-hop / aggregate breakdowns separately.
"""
import ast
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_rag import load_pooled_samples, build_index, evaluate_with_ragas

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / 'data/arxiv'


def verdict(mean_faith: float) -> str:
    if mean_faith > 0.7:
        return 'STRONG'
    if mean_faith >= 0.5:
        return 'ACCEPTABLE'
    return 'NEEDS WORK'


def report_block(name: str, df: pd.DataFrame, extreme_metric: str):
    print(f'\n--- {name} ({len(df)} questions) ---')
    for metric in ['faithfulness', 'answer_relevance', 'context_relevance']:
        print(f'  {metric:<18} mean={df[metric].mean():.3f}  std={df[metric].std():.3f}')

    print(f'\n  3 LOWEST {extreme_metric}:')
    for _, r in df.nsmallest(3, extreme_metric).iterrows():
        print(f'    [{r[extreme_metric]:.3f}] {r["question"]}')
    print(f'  3 HIGHEST {extreme_metric}:')
    for _, r in df.nlargest(3, extreme_metric).iterrows():
        print(f'    [{r[extreme_metric]:.3f}] {r["question"]}')

    v = verdict(df['faithfulness'].mean())
    print(f'\n  VERDICT: {v}')
    return v


if __name__ == '__main__':
    print('Loading questions.csv...')
    q_df = pd.read_csv(OUT_DIR / 'questions.csv')
    print(f'  {len(q_df)} questions loaded')

    print('\nLoading pooled sample_00-09 + building VectorStoreIndex...')
    df = load_pooled_samples()
    index = build_index(df)

    questions = [
        {
            'question': r['question'],
            'reference_answer': r['reference_answer'],
            'question_type': r['question_type'],
        }
        for _, r in q_df.iterrows()
    ]

    print('\nRunning full RAGAS on all 50 questions (this will take a while)...')
    results = evaluate_with_ragas(questions, index)
    results['id'] = q_df['id'].values
    results['question_subtype'] = q_df['question_subtype'].values
    results['source_category_labels'] = q_df['source_category_labels'].values

    results.to_csv(OUT_DIR / 'ragas_baseline.csv', index=False)
    print(f'\nSaved -> {OUT_DIR / "ragas_baseline.csv"}')

    single_df = results[results['question_subtype'] == 'single_specific']
    agg_df = results[results['question_subtype'] == 'aggregate_cross_paper']

    print('\n' + '=' * 20 + ' STEP 7 REPORT ' + '=' * 20)
    v_single = report_block('SINGLE-HOP', single_df, 'answer_relevance')
    v_agg = report_block('AGGREGATE', agg_df, 'context_relevance')

    print('\n--- Hardest category combinations (aggregate, by context_relevance) ---')
    agg_sorted = agg_df.nsmallest(5, 'context_relevance')
    for _, r in agg_sorted.iterrows():
        cats = r['source_category_labels']
        print(f'  [ctx_rel={r["context_relevance"]:.3f}] {cats}')

    print('\n--- Cross-type comparison ---')
    print(f'  {"metric":<16}{"single_hop":>12}{"aggregate":>12}{"delta":>10}')
    for metric in ['faithfulness', 'answer_relevance', 'context_relevance']:
        s, a = single_df[metric].mean(), agg_df[metric].mean()
        print(f'  {metric:<16}{s:>12.3f}{a:>12.3f}{a-s:>10.3f}')

    print(f'\nOverall verdict: single-hop={v_single}  aggregate={v_agg}')

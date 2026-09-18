#!/usr/bin/env python3
"""
run_gate_v2.py — Stage 3 QA generation, gate rerun after prompt v2.

Three checks, run in sequence, each reported separately:
  1. LaTeX-corruption fix — targeted test on backslash-heavy rows.
  2. Single-hop gate v2 (harder prompt) — 10 questions, full RAGAS.
  3. Aggregate gate — 5 cross-paper questions, RAGAS, compared to (2).

Does not proceed to full-50 generation. Report-only.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_rows import format_arxiv_row
from generate_qa import (
    generate_qa, generate_aggregate_qa, validate_qa, classify_question_type,
    sample_category_groups, _client, _CONTROL_CHAR_RE,
)
from evaluate_rag import load_pooled_samples, build_index, evaluate_with_ragas

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / 'data/arxiv'


def check_latex_fix(df: pd.DataFrame, client, n=10, seed=42):
    print('\n' + '=' * 20 + ' CHECK 1 — LaTeX/JSON corruption fix ' + '=' * 20)
    has_backslash = df['text_fidelity_a'].apply(lambda t: isinstance(t, str) and '\\' in t)
    n_backslash = int(has_backslash.sum())
    print(f'Rows in pool with backslash in text_fidelity_a: {n_backslash} / {len(df)}')

    pool = df[has_backslash].sample(n=min(n, has_backslash.sum()), random_state=seed).reset_index(drop=True)

    n_corrupted = 0
    n_ok = 0
    n_other_fail = 0
    for _, row in pool.iterrows():
        text = format_arxiv_row(row.to_dict())
        if text is None:
            continue
        try:
            qa = generate_qa(text, client=client)
        except ValueError as e:
            n_other_fail += 1
            print(f'  API/parse FAILURE on row {row["primary_id"]}: {e}')
            continue
        if _CONTROL_CHAR_RE.search(qa['question']) or _CONTROL_CHAR_RE.search(qa['answer']):
            n_corrupted += 1
            print(f'  CORRUPTION on row {row["primary_id"]}:')
            print(f'    question: {qa["question"]!r}')
            print(f'    answer:   {qa["answer"]!r}')
        else:
            n_ok += 1

    print(f'\nResult: {n_ok} clean / {n_corrupted} corrupted / {n_other_fail} other failures (of {len(pool)} tested)')
    print('FIX CONFIRMED' if n_corrupted == 0 else 'FIX DID NOT FULLY HOLD — corruption still occurring')
    return n_corrupted == 0


def run_singlehop_gate_v2(df: pd.DataFrame, index, client, n=10, seed=43):
    print('\n' + '=' * 20 + ' CHECK 2 — single-hop gate v2 (harder prompt) ' + '=' * 20)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    questions = []
    for _, row in shuffled.iterrows():
        if len(questions) >= n:
            break
        text = format_arxiv_row(row.to_dict())
        if text is None:
            continue
        try:
            qa = generate_qa(text, client=client)
        except ValueError:
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            print(f'  rejected ({reason}): {qa.get("question","")[:80]}')
            continue
        questions.append({
            'question': qa['question'],
            'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'single_specific',
            'source_row_id': row['primary_id'],
        })
    print(f'Collected {len(questions)}/{n} valid v2 single-hop questions')

    results = evaluate_with_ragas(questions, index)
    results['question_subtype'] = 'single_specific'
    return results


def run_aggregate_gate(df: pd.DataFrame, index, client, n=5, seed=44):
    print('\n' + '=' * 20 + ' CHECK 3 — aggregate (cross-paper) gate ' + '=' * 20)
    groups = sample_category_groups(df, n_groups=n * 2, group_size_range=(5, 10), seed=seed)
    print(f'Found {len(groups)} candidate same-category groups (size>=5)')

    questions = []
    for g in groups:
        if len(questions) >= n:
            break
        try:
            qa = generate_aggregate_qa(g, client=client)
        except ValueError as e:
            print(f'  FAILED: {e}')
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            print(f'  rejected ({reason}): {qa.get("question","")[:80]}')
            continue
        questions.append({
            'question': qa['question'],
            'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'aggregate_cross_paper',
            'source_row_id': ','.join(str(r['primary_id']) for r in g),
        })
    print(f'Collected {len(questions)}/{n} valid aggregate questions')

    if not questions:
        return pd.DataFrame()

    results = evaluate_with_ragas(questions, index)
    results['question_subtype'] = 'aggregate_cross_paper'
    return results


def report(single_df: pd.DataFrame, agg_df: pd.DataFrame):
    print('\n' + '=' * 20 + ' REPORT ' + '=' * 20)

    print('\n--- Single-hop v2: answer_relevance distribution ---')
    print(f'  mean={single_df["answer_relevance"].mean():.3f}  std={single_df["answer_relevance"].std():.3f}')
    print(f'  faithfulness       mean={single_df["faithfulness"].mean():.3f}')
    print(f'  context_relevance  mean={single_df["context_relevance"].mean():.3f}  std={single_df["context_relevance"].std():.3f}')

    print('\n  3 LOWEST answer_relevance:')
    for _, r in single_df.nsmallest(3, 'answer_relevance').iterrows():
        print(f'    [{r["answer_relevance"]:.3f}] {r["question"]}')
    print('\n  3 HIGHEST answer_relevance:')
    for _, r in single_df.nlargest(3, 'answer_relevance').iterrows():
        print(f'    [{r["answer_relevance"]:.3f}] {r["question"]}')

    if len(agg_df):
        print('\n--- Aggregate vs single-hop v2 RAGAS comparison ---')
        for metric in ['faithfulness', 'answer_relevance', 'context_relevance']:
            s = single_df[metric].mean()
            a = agg_df[metric].mean()
            print(f'  {metric:<18} single={s:.3f}   aggregate={a:.3f}   delta={a-s:+.3f}')

        print('\n  Aggregate questions + scores:')
        for _, r in agg_df.iterrows():
            print(f'    [faith={r["faithfulness"]:.3f} ans_rel={r["answer_relevance"]:.3f} ctx_rel={r["context_relevance"]:.3f}] {r["question"]}')
    else:
        print('\n--- No valid aggregate questions collected — see failures above ---')


if __name__ == '__main__':
    client = _client()

    print('Loading pooled sample_00-09...')
    df = load_pooled_samples()
    print(f'  {len(df)} unique rows after dedup on primary_id')

    print('\nBuilding VectorStoreIndex (plain, no graph)...')
    index = build_index(df)

    latex_ok = check_latex_fix(df, client)

    single_df = run_singlehop_gate_v2(df, index, client)
    agg_df = run_aggregate_gate(df, index, client)

    report(single_df, agg_df)

    # Save full per-question dataframe (single-hop v2) for inspection
    single_path = OUT_DIR / 'ragas_gate_v2.jsonl'
    single_df.to_json(single_path, orient='records', lines=True)
    print(f'\nSaved single-hop v2 gate ({len(single_df)} rows) -> {single_path}')

    if len(agg_df):
        agg_path = OUT_DIR / 'ragas_gate_v2_aggregate.jsonl'
        agg_df.to_json(agg_path, orient='records', lines=True)
        print(f'Saved aggregate gate ({len(agg_df)} rows) -> {agg_path}')

    print(f'\nLaTeX fix: {"CONFIRMED" if latex_ok else "STILL BROKEN — see above"}')

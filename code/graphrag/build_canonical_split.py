#!/usr/bin/env python3
"""
Derives the canonical 150/150 question split for each dataset and brings the
existing output/run_final/ragas_results_{dataset}.csv `split` column into
agreement with it.

WHY THE EXISTING ROWS NEED TOUCHING
Phase 3 stamped a 75/75 split (stratified over the uneven 83/33/34 subtype
counts) into every row it wrote. The 300-question target is 150/150 with 50
of each subtype per side, which reassigns some of those original 150
questions. Leaving the old labels in place would mean the file disagrees
with graphrag_split.py about what "train" means.

WHAT IS AND IS NOT MODIFIED
Only the `split` column is rewritten. The four score columns
(faithfulness, answer_relevance, context_relevance, composite) and
wall_time_seconds are asserted bit-identical before and after -- compared as
raw strings, so no float round-trip can hide a change. A .bak copy is
written first. No RAGAS score is ever recomputed or overwritten.

Rows whose question_id is not in the canonical split (should not happen)
are reported and left with split='unknown' rather than silently dropped.

Reads:
  - data/{dataset}/questions.csv
  - output/run_final/ragas_results_{dataset}.csv

Writes:
  - output/run_final/question_split_{dataset}.csv
  - output/run_final/ragas_results_{dataset}.csv  (split column only)
  - output/run_final/ragas_results_{dataset}.csv.bak

Usage:
  python3 build_canonical_split.py              # all five
  python3 build_canonical_split.py --dataset arxiv
  python3 build_canonical_split.py --dry_run
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphrag_split import DATASETS, RUN_FINAL, get_split, split_map

SCORE_COLS = ['faithfulness', 'answer_relevance', 'context_relevance', 'composite',
              'wall_time_seconds']


def _fingerprint(df: pd.DataFrame) -> str:
    """Raw-string fingerprint of the columns that must not change. Compared as
    text rather than floats so a float round-trip cannot mask an edit."""
    cols = ['variant', 'question_id'] + [c for c in SCORE_COLS if c in df.columns]
    return df[cols].astype(str).to_csv(index=False)


def apply_to_dataset(dataset: str, dry_run: bool = False) -> None:
    print(f'\n=== {dataset} ===')
    get_split(dataset)                       # derives + writes the split file
    smap = split_map(dataset, verbose=False)

    results_path = RUN_FINAL / f'ragas_results_{dataset}.csv'
    if not results_path.exists():
        print(f'  no {results_path.name} yet -- split file written, nothing to reconcile')
        return

    df = pd.read_csv(results_path)
    before_fp = _fingerprint(df)
    old_split = df['split'].copy()

    new_split = df['question_id'].map(smap)
    missing = int(new_split.isna().sum())
    if missing:
        unknown_ids = sorted(df.loc[new_split.isna(), 'question_id'].unique())
        print(f'  WARNING: {missing} rows ({len(unknown_ids)} question_ids) are not in the '
              f'canonical split -- marking them split="unknown", not dropping. '
              f'First few: {unknown_ids[:5]}')
        new_split = new_split.fillna('unknown')

    changed = int((old_split != new_split).sum())
    changed_q = df.loc[old_split != new_split, 'question_id'].nunique()
    print(f'  rows={len(df)}  questions={df["question_id"].nunique()}  '
          f'variants={df["variant"].nunique()}')
    print(f'  split label changes: {changed} rows across {changed_q} questions')

    df['split'] = new_split
    after_fp = _fingerprint(df)
    if before_fp != after_fp:
        raise RuntimeError(f'{dataset}: score columns changed -- refusing to write')
    print('  verified: score columns bit-identical')

    if dry_run:
        print('  DRY RUN -- not writing')
        return

    shutil.copy2(results_path, results_path.with_suffix('.csv.bak'))
    df.to_csv(results_path, index=False)
    print(f'  backed up -> {results_path.name}.bak')
    print(f'  wrote -> {results_path.name}')
    counts = df.drop_duplicates('question_id').groupby(['split', 'question_subtype']).size()
    print(f'  per-question split/subtype counts now:\n{counts.to_string()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        apply_to_dataset(ds, dry_run=args.dry_run)


if __name__ == '__main__':
    main()

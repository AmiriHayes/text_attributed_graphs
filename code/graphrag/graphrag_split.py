#!/usr/bin/env python3
"""
Canonical GraphRAG question split -- single source of truth for which
questions are train vs test, shared by evaluation and DT analysis.

WHY THIS EXISTS
Phase 3 assigned a 75/75 split over 150 questions (stratified on the then-
uneven 83/33/34 subtype distribution) and wrote it into a `split` column
inside ragas_results_{dataset}.csv. Extending to 300 questions changes the
target to 150/150 with 50 of each subtype per side, which necessarily
reassigns some of the original 150 questions. Two files would then disagree
about what "train" means.

So the split lives in ONE place -- output/run_final/question_split_{ds}.csv
-- derived from data/{ds}/questions.csv, and both the evaluator and the
analysis read it rather than deriving their own. The `split` column inside
ragas_results_{ds}.csv is kept in sync for readability (see
build_canonical_split.py) but is never the authority.

FREEZING
The split file is written once and thereafter read back verbatim, so
results stay reproducible. It is only recomputed if the question id set in
questions.csv no longer matches the split file's id set (i.e. questions were
added or removed), and that recomputation is announced loudly.

Stratified on question_subtype, seed 42, test_size 0.5.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_FINAL = REPO_ROOT / 'output' / 'run_final'
SEED = 42
DATASETS = ['arxiv', 'amazon', 'history', 'electronics', 'toys']


def split_path(dataset: str) -> Path:
    return RUN_FINAL / f'question_split_{dataset}.csv'


def _derive(dataset: str) -> pd.DataFrame:
    q = pd.read_csv(REPO_ROOT / f'data/{dataset}/questions.csv')
    train_q, test_q = train_test_split(
        q, test_size=0.5, random_state=SEED, stratify=q['question_subtype'])
    out = pd.concat([
        pd.DataFrame({'question_id': train_q['id'], 'question_subtype': train_q['question_subtype'],
                      'split': 'train'}),
        pd.DataFrame({'question_id': test_q['id'], 'question_subtype': test_q['question_subtype'],
                      'split': 'test'}),
    ], ignore_index=True)
    # stable order by id so the written file is diff-friendly
    return out.sort_values('question_id').reset_index(drop=True)


def get_split(dataset: str, allow_recompute: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Returns the canonical split as a DataFrame
    [question_id, question_subtype, split]. Writes it on first use."""
    path = split_path(dataset)
    current_ids = set(pd.read_csv(REPO_ROOT / f'data/{dataset}/questions.csv')['id'])

    if path.exists():
        cached = pd.read_csv(path)
        if set(cached['question_id']) == current_ids:
            return cached
        if not allow_recompute:
            raise RuntimeError(
                f'{path.name} covers {len(cached)} questions but questions.csv now has '
                f'{len(current_ids)} -- refusing to recompute (allow_recompute=False)')
        if verbose:
            print(f'  [{dataset}] question set changed ({len(cached)} -> {len(current_ids)}) '
                  f'-- RECOMPUTING canonical split and overwriting {path.name}')

    out = _derive(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if verbose:
        counts = out.groupby(['split', 'question_subtype']).size().unstack(fill_value=0)
        print(f'  [{dataset}] wrote canonical split -> {path.name}')
        print(f'{counts.to_string()}')
    return out


def split_map(dataset: str, **kw) -> dict:
    """question_id -> 'train'|'test'."""
    s = get_split(dataset, **kw)
    return dict(zip(s['question_id'], s['split']))


if __name__ == '__main__':
    for ds in DATASETS:
        print(f'=== {ds} ===')
        s = get_split(ds)
        print(f'  total={len(s)}  train={(s["split"]=="train").sum()}  test={(s["split"]=="test").sum()}')

#!/usr/bin/env python3
"""
What does the decision tree contribute, and what does following it buy?

Table 3 reports rho = Spearman(train_tree.predict(variants), test_mean). Every
variant in that call was in the tree's training set, so the tree is scored on
constructions it has already seen. Two controls are missing.

  no_tree   Spearman(train_mean, test_mean) with no model at all. This is the
            floor: if the tree matches it, the reported rho measures how
            reproducible the proxy scores are across two random pools, not
            anything the meta-learner adds.

  lovo      Leave-one-variant-out. Refit the tree with one construction held
            out and predict it. This is the only setting in which the tree
            does the job the paper credits it with -- ranking a construction
            that was never scored -- and it is the result that justifies
            fitting a model rather than sorting the score table.

It also reports the practitioner payoff, which the paper never states: the
test-pool score of the construction the tree ranks first, against the oracle,
against the mean over all variants (picking blind), and against the obvious
default of primary entity with the richest text.

Writes:
  output/run_final/analysis/tree_ablation/tree_ablation.csv

Usage:
  python3 code/tree_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'code'))

import dt_consistency as D                                   # noqa: E402

OUT = REPO / 'output' / 'run_final' / 'analysis' / 'tree_ablation'

# Internal codes for "primary entity, richest text", the construction a
# practitioner would reach for without any search at all.
DEFAULT_NODE, DEFAULT_TEXT = 'N7', 'T12a'


def leave_one_variant_out(v: pd.DataFrame) -> np.ndarray:
    """Predict each variant from a tree refit without it."""
    pred = np.empty(len(v), dtype=float)
    for i in range(len(v)):
        model = D.fit(v.drop(v.index[i]), 'train_mean')
        pred[i] = D.apply_to(model, v.iloc[[i]])[0]
    return pred


def run() -> pd.DataFrame:
    rows = []
    for task, loader in D.TASKS.items():
        for ds in D.DATASETS:
            v = loader(ds)
            train = v['train_mean'].to_numpy(float)
            test = v['test_mean'].to_numpy(float)
            tree = D.apply_to(D.fit(v, 'train_mean'), v)
            lovo = leave_one_variant_out(v)

            oracle, blind = test.max(), test.mean()
            pick_tree = test[int(np.argmax(tree))]
            dflt = v[(v.Node_Idx == DEFAULT_NODE) & (v.Text_Idx == DEFAULT_TEXT)]
            default = (test[v.index.get_indexer(dflt.index)].mean()
                       if len(dflt) else np.nan)

            rows.append({
                'task': task, 'dataset': ds, 'n_variants': len(v),
                'rho_tree': round(float(spearmanr(tree, test).statistic), 3),
                'rho_no_tree': round(float(spearmanr(train, test).statistic), 3),
                'rho_lovo': round(float(spearmanr(lovo, test).statistic), 3),
                'blind_mean': round(float(blind), 2),
                'default_N1_T6a': (np.nan if np.isnan(default) else round(float(default), 2)),
                'taco_pick': round(float(pick_tree), 2),
                'oracle': round(float(oracle), 2),
                'pct_of_oracle_gain': (round(100 * (pick_tree - blind) / (oracle - blind), 1)
                                       if oracle > blind else np.nan),
                'beats_default': (np.nan if np.isnan(default) else bool(pick_tree >= default)),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = run()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'tree_ablation.csv', index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(df.to_string(index=False))
    print('\n--- mean over datasets ---')
    print(df.groupby('task')[['rho_tree', 'rho_no_tree', 'rho_lovo',
                              'pct_of_oracle_gain']].mean().round(3).to_string())
    print(f'\nwrote -> {OUT / "tree_ablation.csv"}')

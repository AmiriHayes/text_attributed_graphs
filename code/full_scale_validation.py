#!/usr/bin/env python3
"""
Does a ranking learned from 1,000-row subsets predict full-scale performance?

This is the experiment the paper's central claim requires and does not report.
dt_consistency.py measures subset-to-subset agreement: both pools are 1,000-row
subsets, so a high rho there says the proxy is self-consistent, not that it is
predictive of the thing a practitioner cares about. This script compares the
subset-based ranking against scores measured on the ENTIRE test pool
(20,775-50,000 rows) produced by code/full_scale_score.py.

Three rankings are compared against full-scale truth:
  train_mean  the practitioner's ranking: mean subset score over the train pool
  tree        the decision tree's prediction, fit on train-pool means
  test_mean   the paper's own "held-out" quantity, still at subset scale

FEASIBILITY
A construction can be cheap to score on a 1,000-row subset and impossible to
build on the full pool, because ground-truth edge rules that match on an
attribute grow quadratically in pool size while k-NN rules do not. Variants
that exceed the density guard at full scale are reported separately and
excluded from the correlations, since they have no full-scale score to
correlate against. Whether the proxy's top pick is among them is itself a
result.

Reads:
  output/run_final/full_scale/full_scale_scores_{dataset}.csv
  (plus whatever dt_consistency.py reads, for the subset side)

Writes:
  output/run_final/analysis/full_scale/full_scale_validation.csv
  output/run_final/analysis/full_scale/full_scale_per_variant_{dataset}.csv
  output/run_final/analysis/full_scale/density_subset_vs_full.csv

Usage:
  python3 code/full_scale_validation.py
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

FULL = REPO / 'output' / 'run_final' / 'full_scale'
# experiment_runner.MAX_EDGES: the pipeline's own degeneracy guard. A variant
# over this line is one the pipeline would refuse to train at full scale, which
# is the number that matters to a practitioner -- not the looser cap
# full_scale_score.py uses to get a score out of it anyway.
PIPELINE_MAX_EDGES = 200_000
OUT = REPO / 'output' / 'run_final' / 'analysis' / 'full_scale'
AXES = D.AXES


def _subset_side(ds: str) -> pd.DataFrame:
    """Per-variant train/test pool means plus the tree's prediction."""
    v = D.node_classification_variants(ds)
    model = D.fit(v, 'train_mean')
    v = v.copy()
    v['tree_pred'] = D.apply_to(model, v)
    return v


def _full_side(ds: str) -> pd.DataFrame | None:
    path = FULL / f'full_scale_scores_{ds}.csv'
    if not path.exists():
        return None
    f = pd.read_csv(path)
    f['full_score'] = f['S_GNN_step1'] * 100.0
    return f


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, per_variant, density = [], [], []

    for ds in D.DATASETS:
        full = _full_side(ds)
        if full is None:
            continue
        sub = _subset_side(ds)
        m = sub.merge(full, on=AXES, how='inner', suffixes=('', '_full'))

        n_enumerated = len(m)
        feasible = m[m['status'] == 'ok'].copy()
        dense = m[m['status'] == 'too_dense'].copy()

        # Is the proxy's own top pick even buildable at full scale?
        top_by_train = m.loc[m['train_mean'].idxmax()]
        top_by_tree = m.loc[m['tree_pred'].idxmax()]

        over_guard = int((m['n_edges'] > PIPELINE_MAX_EDGES).sum())
        top_train_over = bool(m.loc[m['train_mean'].idxmax(), 'n_edges']
                              > PIPELINE_MAX_EDGES)
        row = {
            'dataset': ds,
            'n_variants': n_enumerated,
            'n_feasible_full_scale': len(feasible),
            'n_too_dense_full_scale': len(dense),
            'n_over_pipeline_guard': over_guard,
            'top_pick_over_pipeline_guard': top_train_over,
            'top_train_mean_variant': f"{top_by_train.Node_Idx}/{top_by_train.Edge_Idx}/{top_by_train.Text_Idx}",
            'top_train_mean_feasible': top_by_train['status'] == 'ok',
            'top_tree_variant': f"{top_by_tree.Node_Idx}/{top_by_tree.Edge_Idx}/{top_by_tree.Text_Idx}",
            'top_tree_feasible': top_by_tree['status'] == 'ok',
        }

        if len(feasible) >= 3:
            y = feasible['full_score'].to_numpy(float)
            for name, col in (('train_mean', 'train_mean'),
                              ('tree', 'tree_pred'),
                              ('test_mean', 'test_mean')):
                r, p = spearmanr(feasible[col].to_numpy(float), y)
                row[f'rho_{name}_vs_full'] = round(float(r), 3)
                row[f'p_{name}_vs_full'] = float(p)

            # What does following the proxy actually buy at full scale?
            oracle = y.max()
            rand = y.mean()
            pick_tree = y[int(np.argmax(feasible['tree_pred'].to_numpy(float)))]
            pick_train = y[int(np.argmax(feasible['train_mean'].to_numpy(float)))]
            row.update(
                full_oracle=round(oracle, 2),
                full_random_mean=round(rand, 2),
                full_pick_tree=round(pick_tree, 2),
                full_pick_train_mean=round(pick_train, 2),
                pct_gain_tree=(round(100 * (pick_tree - rand) / (oracle - rand), 1)
                               if oracle > rand else np.nan),
            )
            # Absolute level shift between the two scales.
            row['mean_subset_score'] = round(float(feasible['train_mean'].mean()), 2)
            row['mean_full_score'] = round(float(y.mean()), 2)

        summary.append(row)

        keep = AXES + ['train_mean', 'test_mean', 'tree_pred', 'full_score',
                       'status', 'n_nodes', 'n_edges', 'isolated_frac', 'seconds']
        pv = m[[c for c in keep if c in m.columns]].copy()
        pv.insert(0, 'dataset', ds)
        pv['rank_subset'] = pv['train_mean'].rank(ascending=False)
        pv['rank_full'] = pv['full_score'].rank(ascending=False)
        per_variant.append(pv.sort_values('train_mean', ascending=False))

        for _, r in m.iterrows():
            density.append({
                'dataset': ds,
                'variant': f"{r.Node_Idx}/{r.Edge_Idx}/{r.Text_Idx}",
                'edge_rule': r.Edge_Idx,
                'full_n_nodes': r.get('n_nodes'),
                'full_n_edges': r.get('n_edges'),
                'full_mean_degree': r.get('mean_degree'),
                'full_isolated_frac': r.get('isolated_frac'),
                'status': r.get('status'),
            })

    return (pd.DataFrame(summary),
            pd.concat(per_variant, ignore_index=True) if per_variant else pd.DataFrame(),
            pd.DataFrame(density))


if __name__ == '__main__':
    s, pv, den = run()
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / 'full_scale_validation.csv', index=False)
    den.to_csv(OUT / 'density_subset_vs_full.csv', index=False)
    for ds, g in pv.groupby('dataset'):
        g.to_csv(OUT / f'full_scale_per_variant_{ds}.csv', index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(s.to_string(index=False))
    print(f'\nwrote -> {OUT}')

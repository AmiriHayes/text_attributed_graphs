#!/usr/bin/env python3
"""
Two diagnostics the construction tables cannot answer on their own.

(1) DOES SUBSAMPLING DISTORT THE EDGE AXIS?
A ground-truth edge rule that links rows sharing an attribute produces a number
of edges that grows roughly quadratically in pool size, while a k-NN rule
produces a number that grows linearly. A 1,000-row subset of a 20k-50k pool is
therefore not a scale model of the full graph: attribute-match rules look
pleasantly sparse at subset scale and are near-complete at full scale, and
participation rules lose most of their edges to the sampling. If the edge axis
is ranked at subset scale, that distortion is a confound, not a detail. This
script measures edges-per-node at both scales for every (node, edge) rule.

(2) ARE SCORES COMPARABLE ACROSS NODE TYPES?
Changing the node entity changes the node population, the label definition and
the class prior, so a score difference between N1 and N2 is not by itself a
statement about graph construction. McFadden pseudo-R2 is measured against an
intercept-only model and so absorbs part of the prior, but the raw accuracies
plotted in Figure 2 do not. This reports, per (dataset, node type), the node
count, the class count and the majority-class share, which is the context a
reader needs to interpret any cross-node-type comparison.

Writes:
  output/run_final/analysis/full_scale/edge_density_by_scale.csv
  output/run_final/analysis/full_scale/node_type_label_stats.csv

Usage:
  python3 code/scale_diagnostics.py
  python3 code/scale_diagnostics.py --datasets history toys --n_subsets 3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'code'))

from generic_data_manager import GenericDataManager          # noqa: E402
from label_factory import LabelFactory                       # noqa: E402
from tag_constructor import TAGConstructor                   # noqa: E402
from variant_registry import VariantRegistry                 # noqa: E402

OUT = REPO / 'output' / 'run_final' / 'analysis' / 'full_scale'
FULL = REPO / 'output' / 'run_final' / 'full_scale'
DATASETS = ['history', 'amazon', 'arxiv', 'electronics', 'toys']


def graph_stats(data) -> dict:
    n = data.num_nodes
    e = data.edge_index.shape[1] // 2
    deg = torch.zeros(n, dtype=torch.long)
    if data.edge_index.numel():
        deg.scatter_add_(0, data.edge_index[0],
                         torch.ones(data.edge_index.shape[1], dtype=torch.long))
    return {'n_nodes': n, 'n_edges': e,
            'edges_per_node': e / n if n else float('nan'),
            'isolated_frac': float((deg == 0).float().mean())}


def subset_density(ds: str, n_subsets: int, split: str) -> pd.DataFrame:
    dm = GenericDataManager(ds, base_path=str(REPO / 'data'))
    builder = TAGConstructor(dm)
    reg = VariantRegistry(ds, config_path=str(REPO / 'data' / 'configs'))
    # Density depends on (N, E) only, so one text fidelity per pair suffices.
    seen, rows = set(), []
    for v in reg.enumerate_variants():
        if v['M'] != 'M1':
            continue
        key = (v['N'], v['E'])
        if key in seen:
            continue
        seen.add(key)
        for i in range(n_subsets):
            try:
                st = graph_stats(builder.construct(v, dm.load_data(split, i), split))
            except Exception as exc:                            # noqa: BLE001
                print(f'  {ds} {key} subset {i}: {exc}')
                continue
            rows.append({'dataset': ds, 'Node_Idx': v['N'], 'Edge_Idx': v['E'],
                         'Text_Idx': v['T'], 'subset': i, **st})
    return pd.DataFrame(rows)


def node_type_labels(ds: str, split: str) -> pd.DataFrame:
    dm = GenericDataManager(ds, base_path=str(REPO / 'data'))
    reg = VariantRegistry(ds, config_path=str(REPO / 'data' / 'configs'))
    df = dm.load_data(split, None)
    rows = []
    for node_type in sorted({v['N'] for v in reg.enumerate_variants() if v['M'] == 'M1'}):
        try:
            node_list = dm.get_node_list(node_type, df)
            # Route through LabelFactory, not dm.get_category_labels: the
            # Amazon-family datasets carry a binary `categorical_label` column
            # and get their real 11-47 class targets from the m1_meta_url join,
            # which only LabelFactory.generate_labels applies. Reading the raw
            # column reports 2 classes for datasets the paper says have 47.
            y = LabelFactory.generate_labels('M1', node_type, dm, df, None,
                                             node_list)
            y = np.asarray(y)
            idx = y.argmax(axis=1) if y.ndim == 2 else y
            counts = Counter(idx.tolist())
            total = sum(counts.values())
            top = counts.most_common(1)[0][1] if counts else 0
            rows.append({'dataset': ds, 'node_type': node_type,
                         'n_nodes': len(node_list),
                         'n_classes_used': len(counts),
                         'n_classes_declared': y.shape[1] if y.ndim == 2 else np.nan,
                         'majority_class_share': round(top / total, 4) if total else np.nan})
        except Exception as exc:                                # noqa: BLE001
            rows.append({'dataset': ds, 'node_type': node_type, 'error': str(exc)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n_subsets', type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    dens, labs = [], []
    for ds in args.datasets:
        print(f'[{ds}] subset density ...', flush=True)
        dens.append(subset_density(ds, args.n_subsets, args.split))
        print(f'[{ds}] node-type label stats ...', flush=True)
        labs.append(node_type_labels(ds, args.split))

    sub = pd.concat(dens, ignore_index=True)
    agg = (sub.groupby(['dataset', 'Node_Idx', 'Edge_Idx'])
              .agg(subset_n_nodes=('n_nodes', 'mean'),
                   subset_n_edges=('n_edges', 'mean'),
                   subset_edges_per_node=('edges_per_node', 'mean'),
                   subset_isolated_frac=('isolated_frac', 'mean'))
              .reset_index())

    full_rows = []
    for ds in args.datasets:
        p = FULL / f'full_scale_scores_{ds}.csv'
        if p.exists():
            f = pd.read_csv(p)
            f = f.groupby(['dataset', 'Node_Idx', 'Edge_Idx']).agg(
                full_n_nodes=('n_nodes', 'mean'),
                full_n_edges=('n_edges', 'mean'),
                full_isolated_frac=('isolated_frac', 'mean')).reset_index()
            f['full_edges_per_node'] = f.full_n_edges / f.full_n_nodes
            full_rows.append(f)

    if full_rows:
        agg = agg.merge(pd.concat(full_rows, ignore_index=True),
                        on=['dataset', 'Node_Idx', 'Edge_Idx'], how='left')
        agg['density_ratio_full_over_subset'] = (
            agg.full_edges_per_node / agg.subset_edges_per_node)

    agg.to_csv(OUT / 'edge_density_by_scale.csv', index=False)
    lab = pd.concat(labs, ignore_index=True)
    lab.to_csv(OUT / 'node_type_label_stats.csv', index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print('\n=== edge density: subset (1k rows) vs full pool ===')
    print(agg.round(3).to_string(index=False))
    print('\n=== node-type label statistics (full pool) ===')
    print(lab.to_string(index=False))
    print(f'\nwrote -> {OUT}')


if __name__ == '__main__':
    main()

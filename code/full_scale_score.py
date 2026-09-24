#!/usr/bin/env python3
"""
Full-scale scoring: train every valid M1 variant ONCE on the entire pool.

WHY THIS EXISTS
Every number in the paper is measured on 1,000-row subsets. The train pool
and the test pool are both collections of 1,000-row subsets drawn from the
same per-split row pool, so "held-out" means a different subsample at the
SAME scale. Nothing in output/run_final/ establishes that a ranking learned
from subsets predicts performance on the full dataset, which is the claim the
method rests on. This script measures that directly.

WHAT IT DOES
For each dataset and each valid (M1, N, E, T) variant it loads the complete
data/{dataset}/test/raw.jsonl (20,776-50,000 rows rather than 1,000), builds
the graph, trains the same 2-layer GraphSAGE with the same hyperparameters
used for subset scoring, and records the same score, S_GNN_step1 =
max(0, McFadden pseudo-R2), on the same 70/15/15 node split.

The test pool is used, not the train pool, so the full-scale score is
measured on rows disjoint from the subsets the decision tree was fit on.

It also records n_nodes, n_edges and the isolated-node fraction at full scale
so they can be compared against the same quantities at subset scale. That
comparison is what tells you whether the subset proxy distorts the edge axis.

INFEASIBLE VARIANTS
experiment_runner.py skips graphs above MAX_EDGES = 200k as degenerate. At
full scale many more variants cross that line, and a variant that cannot be
built at full scale is a real result, not a missing value: the proxy ranks
constructions a practitioner could never deploy. Such variants are recorded
with status='too_dense' and their edge count, never silently dropped.

Reads:
  data/configs/{dataset}_{dataset,variants}.yaml
  data/{dataset}/test/raw.jsonl and data/{dataset}/test/embeddings/

Writes:
  output/run_final/full_scale/full_scale_scores_{dataset}.csv

Usage:
  python3 code/full_scale_score.py --datasets history
  python3 code/full_scale_score.py --datasets history toys electronics
  python3 code/full_scale_score.py --datasets amazon --max_edges 3000000
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'code'))

from generic_data_manager import GenericDataManager          # noqa: E402
from tag_constructor import TAGConstructor                   # noqa: E402
from variant_registry import VariantRegistry                 # noqa: E402
from models import ModelFactory                              # noqa: E402
from trainer import GNNTrainer, compute_step1_score          # noqa: E402

OUT = REPO / 'output' / 'run_final' / 'full_scale'
DUPLICATES_CSV = REPO / 'output' / 'run_final' / 'analysis' / 'edge_rule_duplicates.csv'
NO_TEXT_CONTROL = 'T12e'


def duplicate_rules(dataset: str) -> set[tuple[str, str]]:
    """(node_type, edge_rule) pairs that duplicate another rule's graph.

    Keeps the last rule by name within each colliding group, matching
    dt_consistency._drop_duplicate_edges, so the full-scale variant set lines
    up one-to-one with the subset variant set it is compared against.
    """
    if not DUPLICATES_CSV.exists():
        return set()
    import csv
    drop = set()
    with DUPLICATES_CSV.open() as fh:
        for row in csv.DictReader(fh):
            if row['dataset'] != dataset:
                continue
            rules = sorted(row['duplicate_rules'].split('+'))
            for rule in rules[:-1]:
                drop.add((row['node_type'], rule))
    return drop

# Same hyperparameters as experiment_runner.py, so the only difference
# between a subset score and a full-scale score is the number of rows.
HIDDEN_DIM = 256
DROPOUT = 0.5
LR = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 100
PATIENCE = 30

# experiment_runner uses 200k. Full-scale graphs are legitimately larger, so
# the cap is raised and crossings are recorded rather than treated as errors.
DEFAULT_MAX_EDGES = 2_000_000


def pick_device() -> str:
    """Same auto-selection experiment_runner.py uses, so the subset scores and
    the full-scale scores come off the same backend."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def score_dataset(dataset: str, max_edges: int, split: str = 'test',
                  device: str = 'cpu', dedupe: bool = False,
                  skip_control: bool = False) -> pd.DataFrame:
    dm = GenericDataManager(dataset, base_path=str(REPO / 'data'))
    builder = TAGConstructor(dm)
    registry = VariantRegistry(dataset, config_path=str(REPO / 'data' / 'configs'))

    df = dm.load_data(split, None)
    print(f'[{dataset}] full {split} pool: {len(df):,} rows', flush=True)

    variants = [v for v in registry.enumerate_variants() if v['M'] == 'M1']
    n_all = len(variants)
    if dedupe:
        drop = duplicate_rules(dataset)
        variants = [v for v in variants if (v['N'], v['E']) not in drop]
    if skip_control:
        variants = [v for v in variants if v['T'] != NO_TEXT_CONTROL]
    print(f'[{dataset}] {len(variants)} of {n_all} M1 variants to score'
          f'{" (duplicates collapsed)" if dedupe else ""}'
          f'{" (control skipped)" if skip_control else ""}', flush=True)

    rows = []
    for i, variant in enumerate(variants, 1):
        N, E, T = variant['N'], variant['E'], variant['T']
        tag = f"{N}/{E}/{T}"
        rec = {'dataset': dataset, 'Node_Idx': N, 'Edge_Idx': E, 'Text_Idx': T,
               'variant': tag, 'split': split, 'n_rows': len(df)}
        t0 = time.time()
        try:
            data = builder.construct(variant, df, split)
            n_edges = data.edge_index.shape[1] // 2
            n_nodes = data.num_nodes
            deg = torch.zeros(n_nodes, dtype=torch.long)
            if data.edge_index.numel():
                deg.scatter_add_(0, data.edge_index[0],
                                 torch.ones(data.edge_index.shape[1], dtype=torch.long))
            rec.update(n_nodes=n_nodes, n_edges=n_edges,
                       isolated_frac=float((deg == 0).float().mean()),
                       mean_degree=float(deg.float().mean()))

            if n_edges > max_edges:
                rec.update(status='too_dense', S_GNN_step1=float('nan'),
                           pseudo_r2=float('nan'), seconds=time.time() - t0)
                print(f'  [{i}/{len(variants)}] {tag:18s} too_dense '
                      f'({n_edges:,} edges > {max_edges:,})', flush=True)
                rows.append(rec)
                continue

            out_dim = data.y.shape[-1]
            model = ModelFactory.create(task_type='categorical',
                                        in_dim=data.x.shape[1],
                                        hidden_dim=HIDDEN_DIM, out_dim=out_dim,
                                        dropout=DROPOUT)
            trainer = GNNTrainer(model=model, device=device, lr=LR,
                                 weight_decay=WEIGHT_DECAY, epochs=EPOCHS,
                                 patience=PATIENCE, task_type='categorical')
            metrics = trainer.train(data=data, num_classes=out_dim,
                                    verbose=False, early_stopping=True)
            s, psr2 = compute_step1_score(metrics, 'categorical')
            rec.update(status='ok', S_GNN_step1=s, pseudo_r2=psr2,
                       accuracy=metrics.get('accuracy', float('nan')),
                       n_classes=out_dim, seconds=time.time() - t0)
            print(f'  [{i}/{len(variants)}] {tag:18s} S={s:.4f}  '
                  f'acc={metrics.get("accuracy", float("nan")):.3f}  '
                  f'nodes={n_nodes:,} edges={n_edges:,}  '
                  f'{time.time() - t0:.0f}s', flush=True)

        except Exception as exc:                                   # noqa: BLE001
            rec.update(status=f'error: {type(exc).__name__}: {exc}',
                       S_GNN_step1=float('nan'), pseudo_r2=float('nan'),
                       seconds=time.time() - t0)
            print(f'  [{i}/{len(variants)}] {tag:18s} ERROR {exc}', flush=True)
            traceback.print_exc()
        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['history', 'toys', 'electronics', 'arxiv', 'amazon'])
    ap.add_argument('--split', default='test', choices=['train', 'test'])
    ap.add_argument('--max_edges', type=int, default=DEFAULT_MAX_EDGES)
    ap.add_argument('--device', default=None)
    ap.add_argument('--dedupe', action='store_true',
                    help='Skip edge rules that duplicate another rule\'s graph.')
    ap.add_argument('--skip_control', action='store_true',
                    help='Skip the no-text control, which dt_consistency '
                         'excludes by default anyway.')
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f'device: {device}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        t0 = time.time()
        out = OUT / f'full_scale_scores_{ds}.csv'
        df = score_dataset(ds, args.max_edges, args.split, device,
                           args.dedupe, args.skip_control)
        df.to_csv(out, index=False)
        print(f'[{ds}] done in {(time.time() - t0) / 60:.1f} min -> {out}\n',
              flush=True)


if __name__ == '__main__':
    main()

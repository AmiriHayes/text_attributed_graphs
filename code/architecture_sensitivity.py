#!/usr/bin/env python3
"""
Are the construction rankings a property of the data or of GraphSAGE?

Every score in the paper comes from one architecture: a 2-layer GraphSAGE with
mean aggregation. Holding it fixed is the right way to isolate construction
effects within a run, but it leaves open whether the learned rules would
survive a different message-passing operator. If N2 is a bad construction only
because SAGE handles it badly, the rule is about the model, not the graph.

This swaps SAGEConv for GCNConv and GATConv, keeps everything else identical
(same 2 layers, same BatchNorm, same MLP head, same hidden_dim, dropout,
optimizer, epochs, patience and 70/15/15 split), rescores every valid M1
variant on the first n train-pool subsets, and reports the rank correlation
between each architecture's per-variant ranking and GraphSAGE's.

Writes:
  output/run_final/analysis/architecture/arch_scores.csv     per (variant, subset, arch)
  output/run_final/analysis/architecture/arch_agreement.csv  rank correlations

Usage:
  python3 code/architecture_sensitivity.py --datasets history toys --n_subsets 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.nn import GATConv, GCNConv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'code'))

import dt_consistency as D                                   # noqa: E402
from generic_data_manager import GenericDataManager          # noqa: E402
from tag_constructor import TAGConstructor                   # noqa: E402
from variant_registry import VariantRegistry                 # noqa: E402
from trainer import GNNTrainer, compute_step1_score          # noqa: E402

OUT = REPO / 'output' / 'run_final' / 'analysis' / 'architecture'

HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY, EPOCHS, PATIENCE = 256, 0.5, 0.001, 5e-4, 100, 30
MAX_EDGES = 200_000          # same guard experiment_runner.py uses


class ConvPredictor(nn.Module):
    """models.SAGEPredictor with the convolution operator swapped out.

    Every other component is copied from it so that the only difference
    between architectures is the message-passing operator.
    """

    def __init__(self, conv: str, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        if conv == 'gcn':
            self.conv1, self.conv2 = GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)
        elif conv == 'gat':
            # 4 heads, concat off, so the hidden width matches the other two.
            self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        else:
            raise ValueError(conv)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, out_dim))
        self.dropout = dropout
        self.log_softmax_output = True

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))),
                      p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))),
                      p=self.dropout, training=self.training)
        return F.log_softmax(self.mlp(x), dim=-1)


def score(ds: str, n_subsets: int, archs: list[str], device: str) -> pd.DataFrame:
    dm = GenericDataManager(ds, base_path=str(REPO / 'data'))
    builder = TAGConstructor(dm)
    reg = VariantRegistry(ds, config_path=str(REPO / 'data' / 'configs'))
    variants = [v for v in reg.enumerate_variants() if v['M'] == 'M1']
    print(f'[{ds}] {len(variants)} variants x {n_subsets} subsets x {len(archs)} archs',
          flush=True)

    rows = []
    for i, v in enumerate(variants, 1):
        for s in range(n_subsets):
            try:
                data = builder.construct(v, dm.load_data('train', s), 'train')
            except Exception as exc:                            # noqa: BLE001
                print(f'  {v["N"]}/{v["E"]}/{v["T"]} subset {s}: {exc}', flush=True)
                continue
            if data.edge_index.shape[1] // 2 > MAX_EDGES:
                continue
            out_dim = data.y.shape[-1]
            for arch in archs:
                t0 = time.time()
                model = ConvPredictor(arch, data.x.shape[1], HIDDEN_DIM, out_dim, DROPOUT)
                tr = GNNTrainer(model=model, device=device, lr=LR,
                                weight_decay=WEIGHT_DECAY, epochs=EPOCHS,
                                patience=PATIENCE, task_type='categorical')
                try:
                    m = tr.train(data=data, num_classes=out_dim, verbose=False,
                                 early_stopping=True)
                    sc, _ = compute_step1_score(m, 'categorical')
                except Exception as exc:                        # noqa: BLE001
                    print(f'  {arch} {v["N"]}/{v["E"]}/{v["T"]} s{s}: {exc}', flush=True)
                    continue
                rows.append({'dataset': ds, 'Node_Idx': v['N'], 'Edge_Idx': v['E'],
                             'Text_Idx': v['T'], 'subset': s, 'arch': arch,
                             'score': sc, 'seconds': time.time() - t0})
        print(f'  [{i}/{len(variants)}] {v["N"]}/{v["E"]}/{v["T"]} done', flush=True)
    return pd.DataFrame(rows)


def agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation between each architecture and GraphSAGE, per dataset."""
    rows = []
    for ds, g in scores.groupby('dataset'):
        sage = D.node_classification_variants(ds)[D.AXES + ['train_mean']]
        means = (g.groupby(D.AXES + ['arch'])['score'].mean().reset_index())
        for arch, ga in means.groupby('arch'):
            m = sage.merge(ga, on=D.AXES, how='inner')
            if len(m) < 3:
                continue
            r, p = spearmanr(m['train_mean'], m['score'])
            top_sage = m.loc[m['train_mean'].idxmax()]
            top_arch = m.loc[m['score'].idxmax()]
            rows.append({
                'dataset': ds, 'arch': arch, 'n_variants': len(m),
                'rho_vs_sage': round(float(r), 3), 'p': float(p),
                'sage_top': f'{top_sage.Node_Idx}/{top_sage.Edge_Idx}/{top_sage.Text_Idx}',
                f'{arch}_top': f'{top_arch.Node_Idx}/{top_arch.Edge_Idx}/{top_arch.Text_Idx}',
                'same_top_pick': (top_sage.Node_Idx, top_sage.Edge_Idx, top_sage.Text_Idx)
                                 == (top_arch.Node_Idx, top_arch.Edge_Idx, top_arch.Text_Idx),
                'same_top_node_type': top_sage.Node_Idx == top_arch.Node_Idx,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['history', 'amazon'])
    ap.add_argument('--n_subsets', type=int, default=5)
    ap.add_argument('--archs', nargs='+', default=['gcn', 'gat'])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available()
                             else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    all_scores = []
    for ds in args.datasets:
        t0 = time.time()
        all_scores.append(score(ds, args.n_subsets, args.archs, device))
        print(f'[{ds}] {(time.time() - t0) / 60:.1f} min', flush=True)

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(OUT / 'arch_scores.csv', index=False)
    agr = agreement(scores)
    agr.to_csv(OUT / 'arch_agreement.csv', index=False)

    pd.set_option('display.width', 220)
    print('\n=== construction ranking agreement with GraphSAGE ===')
    print(agr.to_string(index=False))
    print(f'\nwrote -> {OUT}')


if __name__ == '__main__':
    main()

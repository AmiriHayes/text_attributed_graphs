#!/usr/bin/env python3
"""
Find edge rules that build the same graph as another edge rule.

code/README.md already records one case of this: E11c was byte-identical to
E10a and was retired. Nothing in the pipeline checks for it, so the next
collision goes unnoticed, and a collision is not cosmetic. Two rules that
produce identical edge sets are one construction counted twice: they inflate
the reported size of the search space, they make the two one-hot features
perfectly collinear so any tree split between them is training noise dressed
as a construction rule, and they add a guaranteed-consistent pair to every
rank correlation computed over variants.

This constructs one graph per (node type, edge rule) on a fixed subset, hashes
the sorted edge set, and reports every group of rules that collide. It is
cheap enough to run as a gate before scoring.

Writes:
  output/run_final/analysis/edge_rule_duplicates.csv

Usage:
  python3 code/detect_duplicate_edge_rules.py
  python3 code/detect_duplicate_edge_rules.py --subset 3 --split train
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'code'))

from generic_data_manager import GenericDataManager          # noqa: E402
from tag_constructor import TAGConstructor                   # noqa: E402
from variant_registry import VariantRegistry                 # noqa: E402

OUT = REPO / 'output' / 'run_final' / 'analysis'
DATASETS = ['history', 'amazon', 'arxiv', 'electronics', 'toys']


def edge_hash(edge_index: torch.Tensor) -> tuple[str, int]:
    """Order-independent fingerprint of an undirected edge set."""
    if edge_index.numel() == 0:
        return 'empty', 0
    a = edge_index.min(dim=0).values
    b = edge_index.max(dim=0).values
    pairs = torch.stack([a, b], dim=1)
    pairs = torch.unique(pairs, dim=0)
    order = torch.argsort(pairs[:, 0] * (int(pairs.max()) + 1) + pairs[:, 1])
    buf = pairs[order].contiguous().numpy().tobytes()
    return hashlib.sha1(buf).hexdigest()[:16], int(pairs.shape[0])


def run(datasets: list[str], subset: int, split: str) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        dm = GenericDataManager(ds, base_path=str(REPO / 'data'))
        builder = TAGConstructor(dm)
        reg = VariantRegistry(ds, config_path=str(REPO / 'data' / 'configs'))
        df = dm.load_data(split, subset)

        # Density depends on (N, E); one text fidelity per pair is enough.
        seen = set()
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        counts: dict[tuple[str, str, str], int] = {}
        for v in reg.enumerate_variants():
            if v['M'] != 'M1':
                continue
            key = (v['N'], v['E'])
            if key in seen:
                continue
            seen.add(key)
            try:
                g = builder.construct(v, df, split)
            except Exception as exc:                            # noqa: BLE001
                print(f'  {ds} {key}: {exc}', flush=True)
                continue
            h, n = edge_hash(g.edge_index)
            buckets[(v['N'], h)].append(v['E'])
            counts[(v['N'], h, v['E'])] = n

        for (node, h), rules in sorted(buckets.items()):
            if len(rules) < 2 or h == 'empty':
                continue
            n_edges = counts[(node, h, rules[0])]
            rows.append({'dataset': ds, 'node_type': node,
                         'duplicate_rules': '+'.join(sorted(rules)),
                         'n_rules': len(rules), 'n_edges': n_edges,
                         'edge_hash': h})
            print(f'  DUPLICATE  {ds:12s} {node}  {"+".join(sorted(rules))}  '
                  f'({n_edges:,} edges)', flush=True)
        print(f'[{ds}] checked {len(seen)} (node, edge) pairs', flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--subset', type=int, default=0)
    ap.add_argument('--split', default='train')
    args = ap.parse_args()

    dup = run(args.datasets, args.subset, args.split)
    OUT.mkdir(parents=True, exist_ok=True)
    dup.to_csv(OUT / 'edge_rule_duplicates.csv', index=False)

    pd.set_option('display.width', 200)
    if dup.empty:
        print('\nNo duplicate edge rules found.')
    else:
        print('\n=== edge rules that build identical graphs ===')
        print(dup.to_string(index=False))
        aff = dup.groupby('dataset')['node_type'].count()
        print('\nAffected (dataset, node type) pairs:')
        print(aff.to_string())
    print(f'\nwrote -> {OUT / "edge_rule_duplicates.csv"}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step 1 (redesign v2) — build the 5 selected ArXiv M1 configs as TAG graphs
on the sample_00-09 pool (NOT the full corpus) — the pool the questions
were generated from.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'code'))
from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from evaluate_rag import load_pooled_samples

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPHS_DIR = REPO_ROOT / 'data/graphrag/graphs'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    import torch

    configs = pd.read_csv(REPO_ROOT / 'data/graphrag/arxiv_selected_configs.csv')
    print(configs.to_string(index=False))
    print()

    pool_df = load_pooled_samples()
    print(f'sample pool (sample_00-09, deduped on primary_id): {len(pool_df)} unique rows\n')

    dm = GenericDataManager('arxiv')
    tc = TAGConstructor(dm)

    results = []
    for _, row in configs.iterrows():
        band, N, E, T = row['band'], row['node_idx'], row['edge_idx'], row['text_idx']
        variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
        print(f'=== {band}  N={N} E={E} T={T} ===')

        t0 = time.time()
        data = tc.construct(variant, pool_df, 'train')
        build_time = time.time() - t0

        n_nodes = data.num_nodes
        n_edges = data.edge_index.shape[1] // 2

        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        ei = data.edge_index.numpy()
        g.add_edges_from(zip(ei[0], ei[1]))
        largest_cc = max(nx.connected_components(g), key=len)
        cc_pct = 100 * len(largest_cc) / n_nodes

        flag = ' <<< FLAG: largest CC < 30%' if cc_pct < 30 else ''
        print(f'  nodes={n_nodes}  edges={n_edges}  largest_CC={cc_pct:.1f}%  build_time={build_time:.1f}s{flag}')

        out_path = GRAPHS_DIR / f'{band}_{N}_{E}_{T}.pt'
        torch.save(data, out_path)
        print(f'  saved -> {out_path}\n')

        results.append({
            'band': band, 'node_idx': N, 'edge_idx': E, 'text_idx': T,
            'num_nodes': n_nodes, 'num_edges': n_edges,
            'largest_cc_pct': round(cc_pct, 2), 'build_time_s': round(build_time, 1),
            'flagged': cc_pct < 30, 'file_size_mb': round(out_path.stat().st_size / 1e6, 1),
        })

    summary = pd.DataFrame(results)
    summary.to_csv(REPO_ROOT / 'data/graphrag/step1_graph_stats.csv', index=False)
    print('=' * 20, 'STEP 1 SUMMARY', '=' * 20)
    print(summary.to_string(index=False))
    print(f'\n{summary["flagged"].sum()} of {len(summary)} configs flagged (largest CC < 30%)')
    print(f'total .pt footprint: {summary["file_size_mb"].sum():.1f} MB')

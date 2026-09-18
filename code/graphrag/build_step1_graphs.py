#!/usr/bin/env python3
"""
Step 1 — build all 10 selected ArXiv M1 configs as full-corpus TAG graphs.
Persists each to data/graphrag/graphs/{band}.pt (memory safety — only
~3.7GB free, and Steps 2-4 need headroom for 20 vector indices + RAGAS).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'code'))
from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPHS_DIR = REPO_ROOT / 'data/graphrag/graphs'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    import torch

    configs = pd.read_csv(REPO_ROOT / 'data/graphrag/arxiv_selected_configs.csv')

    dm = GenericDataManager('arxiv')
    df = dm.load_data('train')
    tc = TAGConstructor(dm)

    results = []
    for _, row in configs.iterrows():
        band, N, E, T = row['band'], row['node_idx'], row['edge_idx'], row['text_idx']
        variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
        print(f'=== {band}  N={N} E={E} T={T} ===')

        t0 = time.time()
        data = tc.construct(variant, df, 'train')
        build_time = time.time() - t0

        n_nodes = data.num_nodes
        n_edges = data.edge_index.shape[1] // 2  # undirected, PyG stores both directions

        # largest connected component via networkx on the (undirected) edge list
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        ei = data.edge_index.numpy()
        g.add_edges_from(zip(ei[0], ei[1]))
        largest_cc = max(nx.connected_components(g), key=len)
        cc_pct = 100 * len(largest_cc) / n_nodes

        flag = ' <<< FLAG: largest CC < 30%' if cc_pct < 30 else ''
        print(f'  nodes={n_nodes}  edges={n_edges}  largest_CC={cc_pct:.1f}%  build_time={build_time:.1f}s{flag}')

        out_path = GRAPHS_DIR / f'{band}.pt'
        torch.save(data, out_path)
        print(f'  saved -> {out_path}\n')

        results.append({
            'band': band, 'node_idx': N, 'edge_idx': E, 'text_idx': T,
            'num_nodes': n_nodes, 'num_edges': n_edges,
            'largest_cc_pct': round(cc_pct, 2), 'build_time_s': round(build_time, 1),
            'flagged': cc_pct < 30,
        })

    summary = pd.DataFrame(results)
    summary.to_csv(REPO_ROOT / 'data/graphrag/step1_graph_stats.csv', index=False)
    print('=' * 20, 'STEP 1 SUMMARY', '=' * 20)
    print(summary.to_string(index=False))
    n_flagged = summary['flagged'].sum()
    print(f'\n{n_flagged} of {len(summary)} configs flagged (largest CC < 30%)')

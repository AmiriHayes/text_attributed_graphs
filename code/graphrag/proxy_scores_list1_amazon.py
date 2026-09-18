#!/usr/bin/env python3
"""List 1 -- graph-only proxy scores for Amazon M1 variants (zero API cost).
Ported from proxy_scores_list1.py (arxiv): string-typed IDs (no float
rounding needed), N7/N8/N9 all present, and defensive handling for the
6 zero-edge N8/E10b+E10c variants (ground-truth edges don't connect
reviewer nodes for amazon -- a real structural fact, not a bug)."""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score

from evaluate_rag import load_pooled_samples

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_ap = argparse.ArgumentParser()
_ap.add_argument('--dataset', default='amazon')
DATASET = _ap.parse_args().dataset
GRAPHS_DIR = REPO_ROOT / f'data/graphrag/graphs/{DATASET}'

DIAMETER_CC_LIMIT = 10_000
BETWEENNESS_NODE_LIMIT = 5_000
SEED = 42


def to_networkx(data) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.numpy()
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return g


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float('nan')
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def link_prediction_auc(data, seed=SEED) -> float:
    rng = np.random.RandomState(seed)
    ei = data.edge_index.numpy()
    pairs = set()
    for a, b in zip(ei[0], ei[1]):
        if a == b:
            continue
        pairs.add((int(a), int(b)) if a < b else (int(b), int(a)))
    pairs = list(pairs)
    if len(pairs) < 10:
        return float('nan')
    n_mask = max(1, int(0.10 * len(pairs)))
    idx = rng.choice(len(pairs), size=n_mask, replace=False)
    pos_pairs = [pairs[i] for i in idx]

    n = data.num_nodes
    existing = set(pairs)
    neg_pairs = []
    attempts = 0
    while len(neg_pairs) < n_mask and attempts < n_mask * 50:
        attempts += 1
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a == b:
            continue
        p = (a, b) if a < b else (b, a)
        if p in existing:
            continue
        neg_pairs.append(p)

    x = data.x.numpy()
    def score(pairs_):
        return np.array([float(np.dot(x[a], x[b])) for a, b in pairs_])

    pos_scores = score(pos_pairs)
    neg_scores = score(neg_pairs)
    if len(neg_scores) == 0:
        return float('nan')
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    if np.isnan(s).any() or len(set(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def compute_list1(variant: str) -> dict:
    pt_path = GRAPHS_DIR / f'{variant}.pt'
    if not pt_path.exists():
        return None
    data = torch.load(pt_path, weights_only=False)
    g = to_networkx(data)
    n = g.number_of_nodes()
    n_edges_total = g.number_of_edges()

    row = {'variant': variant, 'num_nodes': n, 'num_edges': n_edges_total}

    degrees = np.array([d for _, d in g.degree()])
    row['mean_degree'] = float(degrees.mean())
    row['isolated_node_fraction'] = float((degrees == 0).mean())

    ccs = list(nx.connected_components(g))
    largest_cc = max(ccs, key=len)
    row['largest_cc_fraction'] = len(largest_cc) / n

    if len(largest_cc) <= DIAMETER_CC_LIMIT:
        sub = g.subgraph(largest_cc)
        try:
            row['diameter'] = float(nx.diameter(sub))
        except Exception:
            row['diameter'] = float('nan')
    else:
        row['diameter'] = float('nan')

    # ── Community structure (Louvain) -- guard for the zero-edge graphs ──
    t0 = time.time()
    if n_edges_total == 0:
        # every node its own community; modularity undefined for an edgeless graph
        communities = [{i} for i in range(n)]
        row['n_communities'] = n
        row['modularity'] = float('nan')
        row['mean_community_size'] = 1.0
        row['largest_community_fraction'] = 1.0 / n
        row['singleton_fraction'] = 1.0
    else:
        communities = nx.algorithms.community.louvain_communities(g, seed=SEED)
        row['n_communities'] = len(communities)
        try:
            row['modularity'] = float(nx.algorithms.community.modularity(g, communities))
        except Exception:
            row['modularity'] = float('nan')
        sizes = np.array([len(c) for c in communities])
        row['mean_community_size'] = float(sizes.mean())
        row['largest_community_fraction'] = float(sizes.max() / n)
        row['singleton_fraction'] = float((sizes == 1).mean())
    louvain_t = time.time() - t0

    t0 = time.time()
    pr = nx.pagerank(g)
    pr_vals = np.array(list(pr.values()))
    row['pagerank_mean'] = float(pr_vals.mean())
    row['pagerank_std'] = float(pr_vals.std())
    row['pagerank_gini'] = gini(pr_vals)
    pagerank_t = time.time() - t0

    t0 = time.time()
    clustering = nx.clustering(g)
    row['clustering_coeff_mean'] = float(np.mean(list(clustering.values()))) if n > 0 else float('nan')
    clustering_t = time.time() - t0

    if n <= BETWEENNESS_NODE_LIMIT and n_edges_total > 0:
        t0 = time.time()
        bc = nx.betweenness_centrality(g, seed=SEED)
        row['betweenness_mean'] = float(np.mean(list(bc.values())))
        bc_t = time.time() - t0
    else:
        row['betweenness_mean'] = float('nan')
        bc_t = 0.0

    row['link_pred_auc'] = link_prediction_auc(data, seed=SEED)

    row['_timing'] = f'louvain={louvain_t:.1f}s pagerank={pagerank_t:.1f}s clustering={clustering_t:.1f}s betweenness={bc_t:.1f}s'
    return row


def compute_homophily(variant: str, node_type: str, node_lists: dict, label_maps: dict) -> float:
    """Only meaningful for N7 (products carry categorical_label directly).
    Note: amazon's categorical_label is known-degenerate (binary, 2 values
    dataset-wide) -- computed as specified but likely low-information."""
    if node_type != 'N7':
        return float('nan')
    pt_path = GRAPHS_DIR / f'{variant}.pt'
    data = torch.load(pt_path, weights_only=False)
    node_list = node_lists['N7']
    labels = label_maps['N7']
    node_labels = np.array([labels.get(pid, -1) for pid in node_list])

    ei = data.edge_index.numpy()
    a, b = ei[0], ei[1]
    same = 0
    total = 0
    seen = set()
    for i, j in zip(a, b):
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        li, lj = node_labels[i], node_labels[j]
        if li == -1 or lj == -1:
            continue
        total += 1
        if li == lj:
            same += 1
    return float(same / total) if total > 0 else float('nan')


if __name__ == '__main__':
    print(f'Loading {DATASET} pool for node-order / label reconstruction...')
    pool = load_pooled_samples(dataset=DATASET)
    print(f'  {len(pool)} unique rows')

    node_lists = {
        'N7': list(dict.fromkeys(pool['primary_id'].tolist())),
    }
    label_maps = {
        'N7': dict(zip(pool['primary_id'], pool['categorical_label'])),
    }

    variants = sorted(p.stem for p in GRAPHS_DIR.glob('*.pt'))
    print(f'{len(variants)} variant .pt files found\n')

    rows = []
    for v in variants:
        node_type = v.split('_')[0]
        print(f'--- {v} ---')
        t0 = time.time()
        row = compute_list1(v)
        if row is None:
            print('  MISSING .pt file, skipping')
            continue
        row['node_type'] = node_type
        row['homophily'] = compute_homophily(v, node_type, node_lists, label_maps)
        print(f'  n={row["num_nodes"]} e={row["num_edges"]} '
              f'largest_cc={row["largest_cc_fraction"]:.3f} modularity={row["modularity"]} '
              f'pagerank_gini={row["pagerank_gini"]:.3f} link_auc={row["link_pred_auc"]}  '
              f'homophily={row["homophily"]}  ({time.time()-t0:.1f}s)  {row["_timing"]}')
        del row['_timing']
        rows.append(row)

    df = pd.DataFrame(rows)
    out = REPO_ROOT / f'data/graphrag/proxy_list1_{DATASET}.csv'
    df.to_csv(out, index=False)
    print(f'\nsaved -> {out}')

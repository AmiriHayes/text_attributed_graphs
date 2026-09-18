#!/usr/bin/env python3
"""List 2 -- question-aware proxy scores for Amazon M1 variants (no LLM,
no API). Ported from proxy_scores_list2.py (arxiv): string-typed IDs (no
float rounding needed). source_row_id / hop_source_ids are product ASINs
(primary_id) -- these will only resolve against N7 graphs; N8 (reviewer)
and N9 (family, unless primary_id==aggregate_id) structurally skip most
or all of these, same pattern as arxiv's N8. Handled generically via
node-membership lookup, not special-cased.
"""
import argparse
import ast
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
from sentence_transformers import SentenceTransformer

from evaluate_rag import load_pooled_samples

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_ap = argparse.ArgumentParser()
_ap.add_argument('--dataset', default='amazon')
DATASET = _ap.parse_args().dataset
GRAPHS_DIR = REPO_ROOT / f'data/graphrag/graphs/{DATASET}'


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float('nan')
    return float(np.dot(a, b) / (na * nb))


def to_networkx(data) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.numpy()
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return g


if __name__ == '__main__':
    print(f'Loading {DATASET} pool + questions + SBERT model...')
    pool = load_pooled_samples(dataset=DATASET)
    q = pd.read_csv(REPO_ROOT / f'data/graphrag/{DATASET}_20q_subset.csv')
    single = q[q['question_subtype'] == 'single_specific'].copy()
    edge_q = q[q['question_subtype'] == 'edge_multihop'].copy()
    print(f'  pool={len(pool)}  single_specific={len(single)}  edge_multihop={len(edge_q)}')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    single_q_emb = model.encode(single['question'].tolist(), show_progress_bar=False)
    single_ids = single['source_row_id'].tolist()

    edge_q['hop_pair'] = edge_q['hop_source_ids'].apply(lambda s: ast.literal_eval(s))

    node_lists_by_type = {
        'N7': list(dict.fromkeys(pool['primary_id'].tolist())),
        'N8': sorted(pool['secondary_id'].dropna().unique().tolist()),
        'N9': sorted(pool['aggregate_id'].dropna().unique().tolist()),
    }
    idx_by_type = {nt: {pid: i for i, pid in enumerate(lst)} for nt, lst in node_lists_by_type.items()}

    variants = sorted(p.stem for p in GRAPHS_DIR.glob('*.pt'))
    rows = []

    for v in variants:
        node_type = v.split('_')[0]
        t0 = time.time()
        data = torch.load(GRAPHS_DIR / f'{v}.pt', weights_only=False)
        x = data.x.numpy()
        g = to_networkx(data)

        pid_to_idx = idx_by_type.get(node_type, {})

        # ── Q: source node retrievability (precision@1/3/5) ─────────────
        p1s, p3s, p5s = [], [], []
        n_skip_prec = 0
        if pid_to_idx:
            x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            for qi, pid in enumerate(single_ids):
                if pid not in pid_to_idx:
                    n_skip_prec += 1
                    continue
                src_idx = pid_to_idx[pid]
                qvec = single_q_emb[qi]
                qvec_n = qvec / (np.linalg.norm(qvec) + 1e-12)
                sims = x_norm @ qvec_n
                order = np.argsort(-sims)
                rank = int(np.where(order == src_idx)[0][0])
                p1s.append(1.0 if rank < 1 else 0.0)
                p3s.append(1.0 if rank < 3 else 0.0)
                p5s.append(1.0 if rank < 5 else 0.0)
        else:
            n_skip_prec = len(single_ids)

        # ── Q: hop reachability (edge_multihop) ─────────────────────────
        hop_lens, reachable = [], []
        n_skip_hop = 0
        if pid_to_idx:
            for pair in edge_q['hop_pair']:
                if pair[0] not in pid_to_idx or pair[1] not in pid_to_idx:
                    n_skip_hop += 1
                    continue
                a, b = pid_to_idx[pair[0]], pid_to_idx[pair[1]]
                try:
                    d = nx.shortest_path_length(g, a, b)
                    hop_lens.append(d)
                    reachable.append(1.0)
                except nx.NetworkXNoPath:
                    reachable.append(0.0)
        else:
            n_skip_hop = len(edge_q)

        # ── Q: neighborhood semantic coherence (all nodes) ──────────────
        coherences = []
        for node in g.nodes():
            neighbors = list(g.neighbors(node))
            if not neighbors:
                continue
            nv = x[node]
            sims = [cos_sim(nv, x[nb]) for nb in neighbors]
            sims = [s for s in sims if not np.isnan(s)]
            if sims:
                coherences.append(np.mean(sims))
        coherences = np.array(coherences)

        # ── Q: query-graph alignment delta (single_specific) ────────────
        deltas = []
        n_skip_align = 0
        if pid_to_idx:
            for qi, pid in enumerate(single_ids):
                if pid not in pid_to_idx:
                    n_skip_align += 1
                    continue
                src_idx = pid_to_idx[pid]
                neighbors = list(g.neighbors(src_idx))
                if not neighbors:
                    n_skip_align += 1
                    continue
                src_emb = x[src_idx]
                neighbor_mean = x[neighbors].mean(axis=0)
                smoothed = (src_emb + neighbor_mean) / 2.0
                qvec = single_q_emb[qi]
                raw_sim = cos_sim(qvec, src_emb)
                smoothed_sim = cos_sim(qvec, smoothed)
                if not (np.isnan(raw_sim) or np.isnan(smoothed_sim)):
                    deltas.append(smoothed_sim - raw_sim)
        else:
            n_skip_align = len(single_ids)

        row = {
            'variant': v, 'node_type': node_type,
            'precision_at_1': np.mean(p1s) if p1s else float('nan'),
            'precision_at_3': np.mean(p3s) if p3s else float('nan'),
            'precision_at_5': np.mean(p5s) if p5s else float('nan'),
            'precision_skip_rate': n_skip_prec / len(single_ids),
            'mean_hop_distance': np.mean(hop_lens) if hop_lens else float('nan'),
            'hop_reachability_fraction': np.mean(reachable) if reachable else float('nan'),
            'hop_skip_rate': n_skip_hop / len(edge_q),
            'neighborhood_coherence_mean': float(coherences.mean()) if len(coherences) else float('nan'),
            'neighborhood_coherence_std': float(coherences.std()) if len(coherences) else float('nan'),
            'query_alignment_delta': np.mean(deltas) if deltas else float('nan'),
            'align_skip_rate': n_skip_align / len(single_ids),
        }
        rows.append(row)
        print(f'{v}: p@1={row["precision_at_1"]}  hop_reach={row["hop_reachability_fraction"]}  '
              f'coherence={row["neighborhood_coherence_mean"]:.3f}  align_delta={row["query_alignment_delta"]}  '
              f'({time.time()-t0:.1f}s)')

    df = pd.DataFrame(rows)
    out = REPO_ROOT / f'data/graphrag/proxy_list2_{DATASET}.csv'
    df.to_csv(out, index=False)
    print(f'\nsaved -> {out}')

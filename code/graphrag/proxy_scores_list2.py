#!/usr/bin/env python3
"""
List 2 -- question-aware proxy scores for ArXiv M1 variants (no LLM, no
API, no new generation). Reuses existing .pt graph node embeddings
(data.x) and the local all-MiniLM-L6-v2 SBERT model (same model used to
build the on-disk node embeddings) to embed the 20 question texts fresh.
"""
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
GRAPHS_DIR = REPO_ROOT / 'data/graphrag/graphs/all_variants'
ROUND = 6


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
    print('Loading arxiv pool + questions + SBERT model...')
    pool = load_pooled_samples(dataset='arxiv')
    q = pd.read_csv(REPO_ROOT / 'data/graphrag/arxiv_20q_subset.csv')
    single = q[q['question_subtype'] == 'single_specific'].copy()
    edge_q = q[q['question_subtype'] == 'edge_multihop'].copy()
    print(f'  pool={len(pool)}  single_specific={len(single)}  edge_multihop={len(edge_q)}')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    single_q_emb = model.encode(single['question'].tolist(), show_progress_bar=False)
    single_ids = single['source_row_id'].round(ROUND).tolist()

    edge_q['hop_pair'] = edge_q['hop_source_ids'].apply(lambda s: [round(float(x), ROUND) for x in ast.literal_eval(s)])

    n7_node_list = list(dict.fromkeys(pool['primary_id'].round(ROUND).tolist()))
    n7_pid_to_idx = {pid: i for i, pid in enumerate(n7_node_list)}

    variants = sorted(p.stem for p in GRAPHS_DIR.glob('*.pt'))
    rows = []

    for v in variants:
        node_type = v.split('_')[0]
        t0 = time.time()
        data = torch.load(GRAPHS_DIR / f'{v}.pt', weights_only=False)
        x = data.x.numpy()
        g = to_networkx(data)

        pid_to_idx = n7_pid_to_idx if node_type == 'N7' else {}

        # ── Q6: source node retrievability (precision@1/3/5) ────────────
        p1s, p3s, p5s = [], [], []
        n_skip_q6 = 0
        if pid_to_idx:
            # rank all nodes by cosine sim to each question, once per question (reuse across metric)
            x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            for qi, pid in enumerate(single_ids):
                if pid not in pid_to_idx:
                    n_skip_q6 += 1
                    continue
                src_idx = pid_to_idx[pid]
                qvec = single_q_emb[qi]
                qvec_n = qvec / (np.linalg.norm(qvec) + 1e-12)
                sims = x_norm @ qvec_n
                order = np.argsort(-sims)
                rank = int(np.where(order == src_idx)[0][0])  # 0-indexed rank of source node
                p1s.append(1.0 if rank < 1 else 0.0)
                p3s.append(1.0 if rank < 3 else 0.0)
                p5s.append(1.0 if rank < 5 else 0.0)
        else:
            n_skip_q6 = len(single_ids)

        # ── Q7: hop reachability (edge_multihop) ────────────────────────
        hop_lens, reachable, within3 = [], [], []
        n_skip_q7 = 0
        if pid_to_idx:
            for pair in edge_q['hop_pair']:
                if pair[0] not in pid_to_idx or pair[1] not in pid_to_idx:
                    n_skip_q7 += 1
                    continue
                a, b = pid_to_idx[pair[0]], pid_to_idx[pair[1]]
                try:
                    d = nx.shortest_path_length(g, a, b)
                    hop_lens.append(d)
                    reachable.append(1.0)
                    within3.append(1.0 if d <= 3 else 0.0)
                except nx.NetworkXNoPath:
                    reachable.append(0.0)
                    within3.append(0.0)
        else:
            n_skip_q7 = len(edge_q)

        # ── Q8: neighborhood semantic coherence (all nodes, any node type) ──
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

        # ── Q9: query-graph alignment (single_specific only) ────────────
        deltas = []
        n_skip_q9 = 0
        if pid_to_idx:
            for qi, pid in enumerate(single_ids):
                if pid not in pid_to_idx:
                    n_skip_q9 += 1
                    continue
                src_idx = pid_to_idx[pid]
                neighbors = list(g.neighbors(src_idx))
                if not neighbors:
                    n_skip_q9 += 1
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
            n_skip_q9 = len(single_ids)

        row = {
            'variant': v, 'node_type': node_type,
            'precision_at_1': np.mean(p1s) if p1s else float('nan'),
            'precision_at_3': np.mean(p3s) if p3s else float('nan'),
            'precision_at_5': np.mean(p5s) if p5s else float('nan'),
            'q6_skip_rate': n_skip_q6 / len(single_ids),
            'mean_hop_length': np.mean(hop_lens) if hop_lens else float('nan'),
            'hop_reachable_frac': np.mean(reachable) if reachable else float('nan'),
            'hop_within3_frac': np.mean(within3) if within3 else float('nan'),
            'q7_skip_rate': n_skip_q7 / len(edge_q),
            'coherence_mean': float(coherences.mean()) if len(coherences) else float('nan'),
            'coherence_std': float(coherences.std()) if len(coherences) else float('nan'),
            'query_alignment_delta': np.mean(deltas) if deltas else float('nan'),
            'q9_skip_rate': n_skip_q9 / len(single_ids),
        }
        rows.append(row)
        print(f'{v}: p@1={row["precision_at_1"]}  hop_reach={row["hop_reachable_frac"]}  '
              f'coherence={row["coherence_mean"]:.3f}  align_delta={row["query_alignment_delta"]}  '
              f'({time.time()-t0:.1f}s)')

    df = pd.DataFrame(rows)
    out = REPO_ROOT / 'data/graphrag/proxy_list2_arxiv.csv'
    df.to_csv(out, index=False)
    print(f'\nsaved -> {out}')

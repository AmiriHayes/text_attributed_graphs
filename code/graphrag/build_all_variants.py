#!/usr/bin/env python3
"""
Steps 3-5 — build TAG graphs for all 20 M1 variants on the sample pool,
build text-based-only GraphRAG systems, run RAGAS on the 20-question subset.
"""
import sys
import time
import shutil
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / 'code'))

from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from evaluate_rag import load_pooled_samples
from format_rows import format_arxiv_row
from generate_qa import _client
from run_step2_4 import evaluate_text_based

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

OLD_GRAPHS_DIR = REPO_ROOT / 'data/graphrag/graphs'
NEW_GRAPHS_DIR = REPO_ROOT / 'data/graphrag/graphs/all_variants'
NEW_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = REPO_ROOT / 'data/graphrag/ragas_results_arxiv_all_variants.csv'

# old quintile filename -> (N,E,T) it corresponds to, for reuse detection
OLD_QUINTILE_MAP = {
    'quintile_1_N8_E10c_T12a.pt': ('N8', 'E10c', 'T12a'),
    'quintile_2_N8_E10c_T12b.pt': ('N8', 'E10c', 'T12b'),
    'quintile_3_N8_E11a_T12b.pt': ('N8', 'E11a', 'T12b'),
    'quintile_4_N7_E11b_T12a.pt': ('N7', 'E11b', 'T12a'),
    'quintile_5_N7_E11b_T12b.pt': ('N7', 'E11b', 'T12b'),
}


def get_or_build_graph(N, E, T, pool_df, tc):
    new_path = NEW_GRAPHS_DIR / f'{N}_{E}_{T}.pt'
    if new_path.exists():
        return torch.load(new_path, weights_only=False), 'already_in_all_variants', 0.0

    for old_name, (oN, oE, oT) in OLD_QUINTILE_MAP.items():
        if (oN, oE, oT) == (N, E, T):
            old_path = OLD_GRAPHS_DIR / old_name
            if old_path.exists():
                shutil.copy(old_path, new_path)
                return torch.load(new_path, weights_only=False), 'reused_from_quintile', 0.0

    variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
    t0 = time.time()
    data = tc.construct(variant, pool_df, 'train')
    build_time = time.time() - t0
    torch.save(data, new_path)
    return data, 'built_fresh', build_time


if __name__ == '__main__':
    variants = pd.read_csv(REPO_ROOT / 'data/graphrag/all_20_m1_variants.csv')
    questions_df = pd.read_csv(REPO_ROOT / 'data/graphrag/arxiv_20q_subset.csv')
    print(f'{len(variants)} variants, {len(questions_df)} questions')

    dm = GenericDataManager('arxiv')
    pool_df = load_pooled_samples()
    tc = TAGConstructor(dm)
    client = _client()

    questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                  'question_type': r['question_type']} for r in questions_df.to_dict('records')]

    if OUT_CSV.exists():
        OUT_CSV.unlink()
    header_written = False

    step1_rows = []
    for _, row in variants.iterrows():
        N, E, T = row['Node_Idx'], row['Edge_Idx'], row['Text_Idx']
        train_mean = row['train_mean']
        print(f'\n{"="*20} {N}/{E}/{T}  train_mean={train_mean:.2f} {"="*20}')

        data, source, build_time = get_or_build_graph(N, E, T, pool_df, tc)
        n_nodes = data.num_nodes
        n_edges = data.edge_index.shape[1] // 2

        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        ei = data.edge_index.numpy()
        g.add_edges_from(zip(ei[0], ei[1]))
        largest_cc = max(nx.connected_components(g), key=len)
        cc_pct = 100 * len(largest_cc) / n_nodes
        flag = ' <<< FLAG CC<30%' if cc_pct < 30 else ''
        print(f'  [{source}] nodes={n_nodes} edges={n_edges} CC={cc_pct:.1f}% build_time={build_time:.1f}s{flag}')
        step1_rows.append({'node_idx': N, 'edge_idx': E, 'text_idx': T, 'train_mean': train_mean,
                            'num_nodes': n_nodes, 'num_edges': n_edges, 'largest_cc_pct': round(cc_pct, 2),
                            'source': source, 'flagged': cc_pct < 30})

        # ---- node texts ----
        node_list = dm.get_node_list(N, pool_df)
        if N == 'N7':
            pid_to_row = {r['primary_id']: r for r in pool_df.to_dict('records')}
            node_texts = {nid: (format_arxiv_row(pid_to_row[nid]) or '') for nid in node_list if nid in pid_to_row}
        else:  # N8
            sys.path.insert(0, str(REPO_ROOT / 'data/graphrag'))
            from generate_qa import build_author_hop_map
            hop_map = build_author_hop_map(pool_df)
            df_reset = pool_df.reset_index(drop=True)
            node_texts = {}
            for author in node_list:
                idxs = hop_map.get(author, [])[:5]
                parts = [format_arxiv_row(df_reset.iloc[i].to_dict()) for i in idxs]
                node_texts[author] = ' '.join(p for p in parts if p)
        node_texts_by_pos = [node_texts.get(nid, '') for nid in node_list]

        # ---- text-based system ----
        docs = [Document(text=t, metadata={'pos': pos}) for pos, t in enumerate(node_texts_by_pos) if t]
        t0 = time.time()
        index = VectorStoreIndex.from_documents(docs)
        print(f'  index built: {len(docs)} docs in {time.time()-t0:.1f}s')

        t0 = time.time()
        results = evaluate_text_based(questions, index, g, node_texts_by_pos, client)
        print(f'  RAGAS ({len(questions)} Q) done in {time.time()-t0:.1f}s  mean_composite={results[["faithfulness","answer_relevance","context_relevance"]].mean(axis=1).mean():.3f}')

        results['system_name'] = f'{N}_{E}_{T}_text_based'
        results['node_idx'], results['edge_idx'], results['text_idx'] = N, E, T
        results['system_type'] = 'text_based'
        results['train_mean'] = train_mean
        results['question_id'] = questions_df['id'].values
        results['question_subtype'] = questions_df['question_subtype'].values
        results['composite_score'] = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)

        cols = ['system_name', 'node_idx', 'edge_idx', 'text_idx', 'system_type', 'train_mean',
                'question_id', 'question_subtype', 'faithfulness', 'answer_relevance',
                'context_relevance', 'composite_score']
        results[cols].to_csv(OUT_CSV, mode='a', header=not header_written, index=False)
        header_written = True

        del index, docs

    pd.DataFrame(step1_rows).to_csv(REPO_ROOT / 'data/graphrag/step1_graph_stats_all_variants.csv', index=False)
    print('\n\nALL 20 VARIANTS COMPLETE')
    print(f'saved -> {OUT_CSV}')

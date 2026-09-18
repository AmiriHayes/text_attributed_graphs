#!/usr/bin/env python3
"""
Amazon Steps 3-5 — build TAG graphs for all 38 M1 variants on the sample
pool, build text-based-only GraphRAG systems, run RAGAS on the 20-question
subset. Adapted from build_all_variants.py (arxiv): dataset='amazon'
throughout, adds N9 node-text resolution (arxiv's M1 set had no surviving
N9 variants, amazon's does), scalar secondary_id (already generalized in
generate_qa.py from the amazon question-gen port).
"""
import sys
import time
from pathlib import Path

import torch
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / 'code'))

from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from evaluate_rag import load_pooled_samples
from format_rows import format_arxiv_row  # dataset-agnostic despite the name
from generate_qa import _client, build_author_hop_map
from run_step2_4 import evaluate_text_based

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

GRAPHS_DIR = REPO_ROOT / 'data/graphrag/graphs/amazon'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = REPO_ROOT / 'data/graphrag/ragas_results_amazon.csv'
MAX_ITEMS_PER_HUB_TEXT = 5


def get_node_texts(node_type, node_list, pool_df):
    """N7: direct row text. N8 (reviewer, scalar secondary_id): concat that
    reviewer's own products' text. N9 (product family): concat a few
    member products' text."""
    if node_type == 'N7':
        pid_to_row = {r['primary_id']: r for r in pool_df.to_dict('records')}
        return {nid: (format_arxiv_row(pid_to_row[nid]) or '') for nid in node_list if nid in pid_to_row}

    elif node_type == 'N8':
        hop_map = build_author_hop_map(pool_df)  # handles scalar secondary_id already
        df_reset = pool_df.reset_index(drop=True)
        texts = {}
        for reviewer in node_list:
            idxs = hop_map.get(reviewer, [])[:MAX_ITEMS_PER_HUB_TEXT]
            parts = [format_arxiv_row(df_reset.iloc[i].to_dict()) for i in idxs]
            texts[reviewer] = ' '.join(p for p in parts if p)
        return texts

    elif node_type == 'N9':
        fam_groups = pool_df.groupby('aggregate_id')
        texts = {}
        for family in node_list:
            if family not in fam_groups.groups:
                texts[family] = ''
                continue
            members = pool_df.loc[fam_groups.groups[family]].head(MAX_ITEMS_PER_HUB_TEXT)
            parts = [format_arxiv_row(r) for r in members.to_dict('records')]
            texts[family] = ' '.join(p for p in parts if p)
        return texts

    else:
        raise ValueError(f'unhandled node_type {node_type}')


if __name__ == '__main__':
    client = _client()
    variants = pd.read_csv(REPO_ROOT / 'data/graphrag/all_amazon_m1_variants.csv')
    questions_df = pd.read_csv(REPO_ROOT / 'data/graphrag/amazon_20q_subset.csv')
    print(f'{len(variants)} variants, {len(questions_df)} questions')

    dm = GenericDataManager('amazon')
    pool_df = load_pooled_samples(dataset='amazon')
    tc = TAGConstructor(dm)

    questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                  'question_type': r['question_type']} for r in questions_df.to_dict('records')]

    # Resume support: if a prior run was interrupted, don't wipe its saved
    # rows or redo completed variants -- only pick up where it left off.
    already_done = set()
    header_written = OUT_CSV.exists()
    if header_written:
        prior = pd.read_csv(OUT_CSV)
        already_done = set(prior['system_name'].unique())
        print(f'Resuming: {len(already_done)} variants already saved in {OUT_CSV.name}, skipping those.\n')

    step1_rows = []
    for _, row in variants.iterrows():
        N, E, T = row['Node_Idx'], row['Edge_Idx'], row['Text_Idx']
        train_mean = row['train_mean']
        system_name = f'{N}_{E}_{T}_text_based'
        if system_name in already_done:
            print(f'{"="*20} {N}/{E}/{T}  SKIP (already saved) {"="*20}')
            continue
        print(f'\n{"="*20} {N}/{E}/{T}  train_mean={train_mean:.2f} {"="*20}')

        pt_path = GRAPHS_DIR / f'{N}_{E}_{T}.pt'
        if pt_path.exists():
            data = torch.load(pt_path, weights_only=False)
            build_time = 0.0
            print('  [reused from disk]')
        else:
            variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
            t0 = time.time()
            data = tc.construct(variant, pool_df, 'train')
            build_time = time.time() - t0
            torch.save(data, pt_path)

        n_nodes = data.num_nodes
        n_edges = data.edge_index.shape[1] // 2
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        ei = data.edge_index.numpy()
        g.add_edges_from(zip(ei[0], ei[1]))
        largest_cc = max(nx.connected_components(g), key=len)
        cc_pct = 100 * len(largest_cc) / n_nodes
        flag = ' <<< FLAG CC<30%' if cc_pct < 30 else ''
        print(f'  nodes={n_nodes} edges={n_edges} CC={cc_pct:.1f}% build_time={build_time:.1f}s{flag}')
        step1_rows.append({'node_idx': N, 'edge_idx': E, 'text_idx': T, 'train_mean': train_mean,
                            'num_nodes': n_nodes, 'num_edges': n_edges, 'largest_cc_pct': round(cc_pct, 2),
                            'flagged': cc_pct < 30})

        node_list = dm.get_node_list(N, pool_df)
        node_texts = get_node_texts(N, node_list, pool_df)
        node_texts_by_pos = [node_texts.get(nid, '') for nid in node_list]

        docs = [Document(text=t, metadata={'pos': pos}) for pos, t in enumerate(node_texts_by_pos) if t]
        t0 = time.time()
        index = VectorStoreIndex.from_documents(docs)
        print(f'  index built: {len(docs)} docs in {time.time()-t0:.1f}s')

        t0 = time.time()
        results = evaluate_text_based(questions, index, g, node_texts_by_pos, client)
        mean_comp = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1).mean()
        print(f'  RAGAS ({len(questions)} Q) done in {time.time()-t0:.1f}s  mean_composite={mean_comp:.3f}')

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

    pd.DataFrame(step1_rows).to_csv(REPO_ROOT / 'data/graphrag/step1_graph_stats_amazon.csv', index=False)
    print('\n\nALL AMAZON VARIANTS COMPLETE')
    print(f'saved -> {OUT_CSV}')

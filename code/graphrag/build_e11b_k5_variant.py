#!/usr/bin/env python3
"""Build the E11b k=5 pooled graph for ArXiv N7/T12a (same pool-based
methodology as every other GraphRAG .pt file this session), then run the
same text-based RAGAS evaluation used for the k=50 baseline, on the same
20-question subset, for a direct comparison."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'code'))

import pandas as pd
import networkx as nx
import torch

from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from edge_factory import EdgeFactory
from evaluate_rag import load_pooled_samples
from format_rows import format_arxiv_row
from generate_qa import _client
from run_step2_4 import evaluate_text_based

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from dotenv import load_dotenv

load_dotenv()
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VARIANT = {'M': 'M1', 'N': 'N7', 'E': 'E11b', 'T': 'T12a'}

if __name__ == '__main__':
    client = _client()
    dm = GenericDataManager('arxiv')
    pool_df = load_pooled_samples(dataset='arxiv')
    tc = TAGConstructor(dm)
    print(f'pool: {len(pool_df)} rows')

    # ── build k=5 pooled graph (monkey-patch, same pattern as GNN test) ──
    original = EdgeFactory._build_structural_similarity
    def patched(node_list, base_graph, k=5, chunk_size=1000):
        return original(node_list, base_graph, k=5, chunk_size=chunk_size)
    EdgeFactory._build_structural_similarity = staticmethod(patched)
    try:
        t0 = time.time()
        data = tc.construct(VARIANT, pool_df, 'train')
    finally:
        EdgeFactory._build_structural_similarity = original

    n_nodes, n_edges = data.num_nodes, data.edge_index.shape[1] // 2
    print(f'k=5 pooled graph: nodes={n_nodes} edges={n_edges}  (k=50 baseline had 447,618 edges)  build_time={time.time()-t0:.1f}s')

    out_pt = REPO_ROOT / 'data/graphrag/graphs/all_variants/N7_E11b_T12a_k5.pt'
    torch.save(data, out_pt)
    print(f'saved -> {out_pt}')

    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    ei = data.edge_index.numpy()
    g.add_edges_from(zip(ei[0], ei[1]))
    largest_cc = max(nx.connected_components(g), key=len)
    cc_pct = 100 * len(largest_cc) / n_nodes
    print(f'largest_cc = {cc_pct:.1f}%')

    # ── node texts (N7 -> direct row text, same as build_all_variants.py) ──
    node_list = dm.get_node_list('N7', pool_df)
    pid_to_row = {r['primary_id']: r for r in pool_df.to_dict('records')}
    node_texts_by_pos = [format_arxiv_row(pid_to_row[nid]) or '' for nid in node_list if nid in pid_to_row]

    docs = [Document(text=t, metadata={'pos': pos}) for pos, t in enumerate(node_texts_by_pos) if t]
    t0 = time.time()
    index = VectorStoreIndex.from_documents(docs)
    print(f'index built: {len(docs)} docs in {time.time()-t0:.1f}s')

    questions_df = pd.read_csv(REPO_ROOT / 'data/graphrag/arxiv_20q_subset.csv')
    questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                  'question_type': r['question_type']} for r in questions_df.to_dict('records')]

    t0 = time.time()
    results = evaluate_text_based(questions, index, g, node_texts_by_pos, client)
    mean_comp = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1).mean()
    print(f'RAGAS ({len(questions)} Q) done in {time.time()-t0:.1f}s  mean_composite={mean_comp:.4f}')

    results['system_name'] = 'N7_E11b_T12a_k5_text_based'
    results['node_idx'], results['edge_idx'], results['text_idx'] = 'N7', 'E11b_k5', 'T12a'
    results['system_type'] = 'text_based'
    results['question_id'] = questions_df['id'].values
    results['question_subtype'] = questions_df['question_subtype'].values
    results['composite_score'] = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)

    cols = ['system_name', 'node_idx', 'edge_idx', 'text_idx', 'system_type',
            'question_id', 'question_subtype', 'faithfulness', 'answer_relevance',
            'context_relevance', 'composite_score']
    out_csv = REPO_ROOT / 'data/graphrag/ragas_results_arxiv_e11b_k5.csv'
    results[cols].to_csv(out_csv, index=False)
    print(f'saved -> {out_csv}')

    print(f'\n=== COMPARISON ===')
    print(f'k=50 baseline (existing): mean_composite=0.3585')
    print(f'k=5  (this run):          mean_composite={mean_comp:.4f}')
    print(f'delta (k5 - k50): {mean_comp - 0.3585:+.4f}')

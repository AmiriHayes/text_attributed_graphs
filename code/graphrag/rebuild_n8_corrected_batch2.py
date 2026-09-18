#!/usr/bin/env python3
"""Post-audit N8 rebuild, batch 2: Electronics and Toys (ArXiv/Amazon
already done in rebuild_n8_corrected.py). Same infrastructure, same
per-variant logic, no new code paths -- only the DATASETS config and the
T12e-included-but-separate handling are new.

T12e handling: T12e-for-N8 variants that survived the 95%-zero exclusion
filter are evaluated too (not silently dropped) -- they're written into
the same output CSV and stay naturally distinguishable via text_idx=='T12e'
for separate reporting, per instruction.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'code'))

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()

from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from evaluate_rag import load_pooled_samples
from format_rows import format_arxiv_row
from generate_qa import _client, build_author_hop_map
from run_step2_4 import evaluate_text_based
import proxy_scores_list1 as p1mod

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    'electronics': {
        'graphs_dir': REPO_ROOT / 'data/graphrag/graphs/electronics',
        'questions_csv': REPO_ROOT / 'data/graphrag/electronics_20q_subset.csv',
        'ragas_filename': 'ragas_results_electronics_corrected.csv',  # exact name from Step 2
    },
    'toys': {
        'graphs_dir': REPO_ROOT / 'data/graphrag/graphs/toys',
        'questions_csv': REPO_ROOT / 'data/graphrag/toys_20q_subset.csv',
        'ragas_filename': 'ragas_results_toys_corrected.csv',
    },
}


def get_n8_variants(dataset: str):
    """Returns (main_df, t12e_df). t12e_df holds T12e-for-N8 rows that
    survived the 95%-zero exclusion filter (evaluated, reported separately
    via text_idx=='T12e'), not silently dropped."""
    import importlib
    ra = importlib.import_module('run_analysis')
    ra.DATASET_RUNS[dataset] = REPO_ROOT / 'output/run_post_audit'
    raw = ra._load_csv(dataset)
    m1 = raw[raw['Task_Idx'] == 'M1']
    summ = ra._build_summary(m1, dataset, verbose=False)
    n8 = summ[summ['Node_Idx'] == 'N8']
    cols = ['Node_Idx', 'Edge_Idx', 'Text_Idx', 'train_mean']
    main = n8[n8['Text_Idx'] != 'T12e'][cols].reset_index(drop=True)
    t12e = n8[n8['Text_Idx'] == 'T12e'][cols].reset_index(drop=True)
    return main, t12e


def build_or_load_graph(N, E, T, graphs_dir, pool_df, tc):
    pt_path = graphs_dir / f'{N}_{E}_{T}.pt'
    if pt_path.exists():
        return torch.load(pt_path, weights_only=False), False
    variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
    data = tc.construct(variant, pool_df, 'train')
    torch.save(data, pt_path)
    return data, True


def process_variant(dataset, cfg, row, dm, pool_df, tc, node_list, questions, questions_df,
                     client, ragas_out, proxy_rows, already_done, header_written, tag=''):
    N, E, T, train_mean = row['Node_Idx'], row['Edge_Idx'], row['Text_Idx'], row['train_mean']
    system_name = f'{N}_{E}_{T}_text_based'
    if system_name in already_done:
        print(f'  SKIP (already done) {N}/{E}/{T}')
        return header_written

    print(f'\n--- {dataset} {N}/{E}/{T}  train_mean={train_mean:.2f}{tag} ---')
    t0 = time.time()
    data, rebuilt = build_or_load_graph(N, E, T, cfg['graphs_dir'], pool_df, tc)
    n_nodes, n_edges = data.num_nodes, data.edge_index.shape[1] // 2
    print(f'  graph: nodes={n_nodes} edges={n_edges} {"[REBUILT]" if rebuilt else "[reused]"} ({time.time()-t0:.1f}s)')

    # ── List-1 proxy scores on this corrected graph ─────────────────────
    p1mod.GRAPHS_DIR = cfg['graphs_dir']
    proxy_row = p1mod.compute_list1(f'{N}_{E}_{T}')
    proxy_row['node_type'] = N
    proxy_row['dataset'] = dataset
    proxy_row['homophily'] = float('nan')  # N8 homophily undefined (no direct categorical_label)
    proxy_rows.append(proxy_row)

    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    ei = data.edge_index.numpy()
    g.add_edges_from(zip(ei[0], ei[1]))

    # N8 nodes are reviewers (secondary entities) -- node_list holds
    # reviewer ids, NOT primary_ids, so text must come from each
    # reviewer's own products via the author/reviewer hop map (bug 4 fix).
    hop_map = build_author_hop_map(pool_df)
    df_reset = pool_df.reset_index(drop=True)
    node_texts = {}
    for entity in node_list:
        idxs = hop_map.get(entity, [])[:5]
        parts = [format_arxiv_row(df_reset.iloc[i].to_dict()) for i in idxs]
        node_texts[entity] = ' '.join(p for p in parts if p)
    node_texts_by_pos = [node_texts.get(nid, '') for nid in node_list]

    docs = [Document(text=t, metadata={'pos': pos}) for pos, t in enumerate(node_texts_by_pos) if t]
    t0 = time.time()
    index = VectorStoreIndex.from_documents(docs)
    print(f'  index built: {len(docs)} docs in {time.time()-t0:.1f}s')

    t0 = time.time()
    results = evaluate_text_based(questions, index, g, node_texts_by_pos, client)
    mean_comp = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1).mean()
    print(f'  RAGAS ({len(questions)} Q) done in {time.time()-t0:.1f}s  mean_composite={mean_comp:.3f}')

    results['system_name'] = system_name
    results['node_idx'], results['edge_idx'], results['text_idx'] = N, E, T
    results['system_type'] = 'text_based'
    results['train_mean'] = train_mean
    results['question_id'] = questions_df['id'].values
    results['question_subtype'] = questions_df['question_subtype'].values
    results['composite_score'] = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)

    cols = ['system_name', 'node_idx', 'edge_idx', 'text_idx', 'system_type', 'train_mean',
            'question_id', 'question_subtype', 'faithfulness', 'answer_relevance',
            'context_relevance', 'composite_score']
    results[cols].to_csv(ragas_out, mode='a', header=not header_written, index=False)
    del index, docs
    return True


if __name__ == '__main__':
    client = _client()

    for dataset, cfg in DATASETS.items():
        print(f'\n{"="*70}\n{dataset.upper()} N8 REBUILD (batch 2)\n{"="*70}')
        variants, t12e_variants = get_n8_variants(dataset)
        print(f'{len(variants)} surviving N8 M1 variants (main) + {len(t12e_variants)} T12e variants (reported separately)')

        dm = GenericDataManager(dataset)
        pool_df = load_pooled_samples(dataset=dataset)
        tc = TAGConstructor(dm)
        questions_df = pd.read_csv(cfg['questions_csv'])
        questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                      'question_type': r['question_type']} for r in questions_df.to_dict('records')]

        ragas_out = REPO_ROOT / f'data/graphrag/{cfg["ragas_filename"]}'
        proxy_out = REPO_ROOT / f'data/graphrag/proxy_list1_{dataset}_n8_corrected.csv'
        already_done = set()
        header_written = ragas_out.exists()
        if header_written:
            prior = pd.read_csv(ragas_out)
            already_done = set(prior['system_name'].unique())
            print(f'resuming: {len(already_done)} variants already done')

        proxy_rows = []
        node_list = dm.get_node_list('N8', pool_df)

        for _, row in variants.iterrows():
            hw = process_variant(dataset, cfg, row, dm, pool_df, tc, node_list, questions,
                                  questions_df, client, ragas_out, proxy_rows, already_done,
                                  header_written)
            if hw:
                header_written = True

        for _, row in t12e_variants.iterrows():
            hw = process_variant(dataset, cfg, row, dm, pool_df, tc, node_list, questions,
                                  questions_df, client, ragas_out, proxy_rows, already_done,
                                  header_written, tag='  [T12e -- report separately]')
            if hw:
                header_written = True

        if proxy_rows:
            pdf = pd.DataFrame(proxy_rows)
            if proxy_out.exists():
                pdf = pd.concat([pd.read_csv(proxy_out), pdf], ignore_index=True)
            pdf.to_csv(proxy_out, index=False)
            print(f'saved proxies -> {proxy_out}')

    print('\nDONE (batch 2: electronics + toys)')

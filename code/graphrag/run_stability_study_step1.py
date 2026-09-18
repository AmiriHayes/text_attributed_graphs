#!/usr/bin/env python3
"""
Step 1 of the full stability study: for ArXiv and Amazon, for every
surviving M1 variant (N7/N8/N9), at both Scale B (1000-node, sample_00) and
Scale C (full sample pool), build a text-based GraphRAG system via
LlamaIndex with 1-hop neighbor expansion and run ALL questions from
data/{dataset}/questions.csv, scoring with RAGAS.

Resumable: skips any (variant, scale) already fully present (all questions)
in the output CSV, so it's safe to kill and restart.

Reads:
  - output/run_post_audit/construction_performance_table_{arxiv,amazon}.csv
  - data/graphrag/graphs/all_variants_scaleB/*.pt, all_variants/*.pt (ArXiv)
  - data/graphrag/graphs/amazon_scaleB/*.pt, amazon/*.pt (Amazon)
  - data/{arxiv,amazon}/questions.csv

Writes (incrementally, per variant):
  - data/graphrag/ragas_stability_arxiv_B.csv, _arxiv_C.csv
  - data/graphrag/ragas_stability_amazon_B.csv, _amazon_C.csv

Usage:
  python data/graphrag/run_stability_study_step1.py
"""
import sys
import time
import importlib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

from generic_data_manager import GenericDataManager
from evaluate_rag import load_pooled_samples
from format_rows import format_arxiv_row
from generate_qa import build_author_hop_map, _client
from run_step2_4 import evaluate_text_based

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

ra = importlib.import_module('run_analysis')
ra.DATASET_RUNS['arxiv'] = REPO_ROOT / 'output/run_post_audit'
ra.DATASET_RUNS['amazon'] = REPO_ROOT / 'output/run_post_audit'

SCALES = {
    'B': {'arxiv': REPO_ROOT / 'data/graphrag/graphs/all_variants_scaleB',
          'amazon': REPO_ROOT / 'data/graphrag/graphs/amazon_scaleB'},
    'C': {'arxiv': REPO_ROOT / 'data/graphrag/graphs/all_variants',
          'amazon': REPO_ROOT / 'data/graphrag/graphs/amazon'},
}
MAX_HOP_ROWS = 5


def get_surviving_variants(dataset):
    raw = ra._load_csv(dataset)
    m1 = raw[raw['Task_Idx'] == 'M1']
    summ = ra._build_summary(m1, dataset, verbose=False)
    return summ[['Node_Idx', 'Edge_Idx', 'Text_Idx']].drop_duplicates().to_dict('records')


def get_node_texts(node_type, node_list, df):
    df_reset = df.reset_index(drop=True)
    if node_type == 'N7':
        pid_to_row = {}
        for r in df_reset.to_dict('records'):
            pid_to_row.setdefault(r['primary_id'], r)
        return {nid: (format_arxiv_row(pid_to_row[nid]) or '') for nid in node_list if nid in pid_to_row}

    hop_map = {}
    if node_type == 'N8':
        hop_map = build_author_hop_map(df_reset)
    elif node_type == 'N9':
        for i, agg in enumerate(df_reset['aggregate_id']):
            if agg is None or (isinstance(agg, float) and pd.isna(agg)):
                continue
            hop_map.setdefault(agg, []).append(i)
    else:
        raise ValueError(f'unknown node_type {node_type}')

    texts = {}
    for entity in node_list:
        idxs = hop_map.get(entity, [])[:MAX_HOP_ROWS]
        parts = [format_arxiv_row(df_reset.iloc[i].to_dict()) for i in idxs]
        texts[entity] = ' '.join(p for p in parts if p)
    return texts


def already_done(out_csv, variant_key, n_questions_expected):
    if not out_csv.exists():
        return False
    try:
        prior = pd.read_csv(out_csv, usecols=['variant'])
        n = (prior['variant'] == variant_key).sum()
        return n >= n_questions_expected
    except Exception:
        return False


def run_dataset(dataset, client):
    dm = GenericDataManager(dataset)
    df0 = dm.load_data('train', 0)          # Scale B source (1000 rows)
    pool_df = load_pooled_samples(dataset=dataset)  # Scale C source (full pool)
    scale_dfs = {'B': df0, 'C': pool_df}

    variants = get_surviving_variants(dataset)
    print(f'\n{"="*70}\n{dataset.upper()}: {len(variants)} surviving M1 variants x 2 scales\n{"="*70}')

    questions_df = pd.read_csv(REPO_ROOT / f'data/{dataset}/questions.csv')
    questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                  'question_type': r.get('question_type', ''),
                  'question_subtype': r['question_subtype'], 'id': r['id']}
                 for r in questions_df.to_dict('records')]
    n_q = len(questions)
    print(f'  {n_q} questions loaded from data/{dataset}/questions.csv')

    for scale in ['B', 'C']:
        out_csv = REPO_ROOT / f'data/graphrag/ragas_stability_{dataset}_{scale}.csv'
        graphs_dir = SCALES[scale][dataset]
        df_scale = scale_dfs[scale]
        header_written = out_csv.exists()

        for v in variants:
            N, E, T = v['Node_Idx'], v['Edge_Idx'], v['Text_Idx']
            variant_key = f'{N}_{E}_{T}'
            pt_path = graphs_dir / f'{variant_key}.pt'
            if not pt_path.exists():
                print(f'  [{scale}] SKIP {variant_key}: no .pt file at {pt_path}')
                continue
            if already_done(out_csv, variant_key, n_q):
                print(f'  [{scale}] SKIP (already done) {variant_key}')
                continue

            # Transient network/API errors (rate limits, connection drops) --
            # common over a many-hour run -- should not kill the whole job.
            # Retry the variant a few times with backoff before giving up.
            MAX_ATTEMPTS = 5
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    t0 = time.time()
                    data = torch.load(pt_path, weights_only=False)
                    node_list = dm.get_node_list(N, df_scale)
                    node_texts = get_node_texts(N, node_list, df_scale)
                    node_texts_by_pos = [node_texts.get(nid, '') for nid in node_list]

                    g = nx.Graph()
                    g.add_nodes_from(range(data.num_nodes))
                    ei = data.edge_index.numpy()
                    g.add_edges_from(zip(ei[0], ei[1]))

                    docs = [Document(text=t, metadata={'pos': pos}) for pos, t in enumerate(node_texts_by_pos) if t]
                    if not docs:
                        print(f'  [{scale}] SKIP {variant_key}: 0 non-empty node texts')
                        break
                    index = VectorStoreIndex.from_documents(docs)
                    build_t = time.time() - t0

                    t1 = time.time()
                    results = evaluate_text_based(questions, index, g, node_texts_by_pos, client)
                    eval_t = time.time() - t1

                    results['variant'] = variant_key
                    results['node_type'] = N
                    results['question_id'] = questions_df['id'].values
                    results['question_subtype'] = questions_df['question_subtype'].values
                    results['scale'] = scale
                    results['composite'] = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)

                    cols = ['variant', 'node_type', 'question_id', 'question_subtype', 'scale',
                            'faithfulness', 'answer_relevance', 'context_relevance', 'composite']
                    results[cols].to_csv(out_csv, mode='a', header=not header_written, index=False)
                    header_written = True

                    mean_c = results['composite'].mean()
                    print(f'  [{scale}] {variant_key} ({N}): nodes={data.num_nodes} docs={len(docs)} '
                          f'build={build_t:.1f}s eval={eval_t:.1f}s ({n_q}q) mean_composite={mean_c:.3f} '
                          f'total={time.time()-t0:.1f}s')
                    del index, docs
                    break
                except Exception as e:
                    wait = min(60, 5 * attempt)
                    print(f'  [{scale}] ERROR {variant_key} (attempt {attempt}/{MAX_ATTEMPTS}): '
                          f'{type(e).__name__}: {e} -- retrying in {wait}s')
                    if attempt == MAX_ATTEMPTS:
                        print(f'  [{scale}] GIVING UP on {variant_key} after {MAX_ATTEMPTS} attempts')
                    else:
                        time.sleep(wait)


if __name__ == '__main__':
    client = _client()
    overall_t0 = time.time()
    for dataset in ['arxiv', 'amazon']:
        try:
            run_dataset(dataset, client)
        except Exception as e:
            print(f'FATAL for dataset {dataset}: {type(e).__name__}: {e} -- continuing to next dataset')
    print(f'\n{"="*70}\nSTEP 1 COMPLETE. Total wall time: {(time.time()-overall_t0)/60:.1f} min\n{"="*70}')

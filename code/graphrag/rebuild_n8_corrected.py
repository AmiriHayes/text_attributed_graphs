#!/usr/bin/env python3
"""Post-audit N8 rebuild: for arxiv and amazon, build/reuse GraphRAG graphs
for every surviving N8 M1 variant (T12e excluded -- degenerate, no real
text), run RAGAS on the dataset's existing 20Q subset, then compute the
List-1 proxy battery (modularity, link_pred_auc, pagerank_std, and the
rest) on the same corrected graphs and correlate against the new RAGAS
scores plus the post-fix raw_gnn train_mean.

Resumable: skips any (dataset, variant) already present in the output
RAGAS CSV.

NOTE: already run successfully for arxiv + amazon (see
ragas_results_arxiv_n8_corrected.csv / ragas_results_amazon_n8_corrected.csv).
Electronics + toys were done in rebuild_n8_corrected_batch2.py instead of
re-running this file, to avoid re-triggering arxiv/amazon.
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
from generate_qa import _client
from run_step2_4 import evaluate_text_based
from proxy_scores_list1 import compute_list1, compute_homophily  # arxiv version, dataset-agnostic funcs

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    'arxiv': {
        'graphs_dir': REPO_ROOT / 'data/graphrag/graphs/all_variants',
        'questions_csv': REPO_ROOT / 'data/graphrag/arxiv_20q_subset.csv',
    },
    'amazon': {
        'graphs_dir': REPO_ROOT / 'data/graphrag/graphs/amazon',
        'questions_csv': REPO_ROOT / 'data/graphrag/amazon_20q_subset.csv',
    },
}

PROXY_COLS = ['modularity', 'link_pred_auc', 'n_communities', 'mean_degree',
              'mean_community_size', 'pagerank_std', 'homophily']


def get_n8_surviving_variants(dataset: str) -> pd.DataFrame:
    import importlib
    ra = importlib.import_module('run_analysis')
    ra.DATASET_RUNS[dataset] = REPO_ROOT / 'output/run_post_audit'
    raw = ra._load_csv(dataset)
    m1 = raw[raw['Task_Idx'] == 'M1']
    summ = ra._build_summary(m1, dataset, verbose=False)
    n8 = summ[(summ['Node_Idx'] == 'N8') & (summ['Text_Idx'] != 'T12e')]
    return n8[['Node_Idx', 'Edge_Idx', 'Text_Idx', 'train_mean']].reset_index(drop=True)


def build_or_load_graph(dataset, N, E, T, graphs_dir, dm, pool_df, tc):
    pt_path = graphs_dir / f'{N}_{E}_{T}.pt'
    if pt_path.exists():
        return torch.load(pt_path, weights_only=False), False
    variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
    data = tc.construct(variant, pool_df, 'train')
    torch.save(data, pt_path)
    return data, True


if __name__ == '__main__':
    client = _client()
    all_results = []

    for dataset, cfg in DATASETS.items():
        print(f'\n{"="*70}\n{dataset.upper()} N8 REBUILD\n{"="*70}')
        variants = get_n8_surviving_variants(dataset)
        print(f'{len(variants)} surviving N8 M1 variants (T12e excluded)')

        dm = GenericDataManager(dataset)
        pool_df = load_pooled_samples(dataset=dataset)
        tc = TAGConstructor(dm)
        questions_df = pd.read_csv(cfg['questions_csv'])
        questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                      'question_type': r['question_type']} for r in questions_df.to_dict('records')]

        ragas_out = REPO_ROOT / f'data/graphrag/ragas_results_{dataset}_n8_corrected.csv'
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
            N, E, T, train_mean = row['Node_Idx'], row['Edge_Idx'], row['Text_Idx'], row['train_mean']
            system_name = f'{N}_{E}_{T}_text_based'
            if system_name in already_done:
                print(f'  SKIP (already done) {N}/{E}/{T}')
                continue
            print(f'\n--- {dataset} {N}/{E}/{T}  train_mean={train_mean:.2f} ---')
            t0 = time.time()
            data, rebuilt = build_or_load_graph(dataset, N, E, T, cfg['graphs_dir'], dm, pool_df, tc)
            n_nodes, n_edges = data.num_nodes, data.edge_index.shape[1] // 2
            print(f'  graph: nodes={n_nodes} edges={n_edges} {"[REBUILT]" if rebuilt else "[reused]"} ({time.time()-t0:.1f}s)')

            # ── List-1 proxy scores on this corrected graph ──────────────
            # compute_list1 reads its GRAPHS_DIR module global at call time -- override per dataset.
            import proxy_scores_list1 as p1mod
            p1mod.GRAPHS_DIR = cfg['graphs_dir']
            proxy_row = p1mod.compute_list1(f'{N}_{E}_{T}')
            proxy_row['node_type'] = N
            proxy_row['dataset'] = dataset
            homophily = float('nan')  # N8 homophily undefined (no direct categorical_label), consistent w/ prior work
            proxy_row['homophily'] = homophily
            proxy_rows.append(proxy_row)

            g = nx.Graph()
            g.add_nodes_from(range(n_nodes))
            ei = data.edge_index.numpy()
            g.add_edges_from(zip(ei[0], ei[1]))

            # N8 nodes are secondary entities (arxiv: authors, amazon:
            # reviewers) -- node_list holds N8 identifiers, NOT primary_ids,
            # so text must come from each entity's associated N7 rows via
            # the author/reviewer hop map, for both datasets. (Bug fixed:
            # the first version of this script looked N8 ids up in a
            # primary_id-keyed dict, which is the wrong id space entirely --
            # it silently produced ~empty node_texts, hence the broken
            # faithfulness=0/context_relevance=0 first run.)
            from generate_qa import build_author_hop_map
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
            header_written = True
            del index, docs

        if proxy_rows:
            pdf = pd.DataFrame(proxy_rows)
            if proxy_out.exists():
                pdf = pd.concat([pd.read_csv(proxy_out), pdf], ignore_index=True)
            pdf.to_csv(proxy_out, index=False)
            print(f'saved proxies -> {proxy_out}')

    # ── FINAL CORRELATION ANALYSIS ────────────────────────────────────────
    print(f'\n\n{"="*70}\nFINAL CORRELATION (corrected N8, both datasets)\n{"="*70}')
    for dataset in DATASETS:
        ragas_out = REPO_ROOT / f'data/graphrag/ragas_results_{dataset}_n8_corrected.csv'
        proxy_out = REPO_ROOT / f'data/graphrag/proxy_list1_{dataset}_n8_corrected.csv'
        if not ragas_out.exists():
            continue
        ragas = pd.read_csv(ragas_out)
        target = ragas.groupby('system_name').agg(
            mean_ragas_composite=('composite_score', 'mean'), train_mean=('train_mean', 'first'),
            edge_idx=('edge_idx', 'first'), text_idx=('text_idx', 'first'),
        ).reset_index()
        target['variant'] = 'N8_' + target['edge_idx'] + '_' + target['text_idx']

        print(f'\n--- {dataset} N8 (corrected, n={len(target)}) ---')
        r, p = spearmanr(target['train_mean'], target['mean_ragas_composite'])
        print(f'  raw_gnn vs RAGAS: rho={r:+.3f} p={p:.4f} n={len(target)}')

        if proxy_out.exists():
            proxies = pd.read_csv(proxy_out)
            proxies['variant'] = 'N8_' + proxies['variant'].str.split('_', n=1).str[1]
            merged = target.merge(proxies, on='variant', how='left')
            for col in ['modularity', 'link_pred_auc', 'pagerank_std']:
                if col in merged.columns:
                    sub = merged[['mean_ragas_composite', col]].dropna()
                    if len(sub) >= 3 and sub[col].nunique() > 1:
                        r2, p2 = spearmanr(sub[col], sub['mean_ragas_composite'])
                        print(f'  {col} vs RAGAS: rho={r2:+.3f} p={p2:.4f} n={len(sub)}')

    print('\nDONE')

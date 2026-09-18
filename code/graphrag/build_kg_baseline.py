#!/usr/bin/env python3
"""
KG-based baseline (#11) — LLM-extracted (h,r,t) triples over a focused
corpus (question-source papers + 300 random background), independent of
any TAG construction choice. Bypasses PropertyGraphIndex (confirmed broken
in this llama-index-core version — extraction runs but the index doesn't
read triplets back out of node metadata) and instead aggregates each
paper's extracted triples into a text summary, indexed the same way as
the community-based/text-based systems.
"""
import sys
import ast
import time
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_rag import load_pooled_samples, evaluate_with_ragas
from format_rows import format_arxiv_row

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.schema import TextNode
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

OUT_CSV = REPO_ROOT / 'data/graphrag/ragas_results_arxiv.csv'

if __name__ == '__main__':
    print('Resolving focused corpus...')
    qdf = pd.read_csv(REPO_ROOT / 'data/arxiv/questions.csv')
    single_ids = set(qdf[qdf.question_subtype == 'single_specific']['source_row_id'].dropna().tolist())
    edge_ids = set()
    for v in qdf[qdf.question_subtype == 'edge_multihop']['hop_source_ids'].dropna():
        edge_ids.update(ast.literal_eval(v))
    question_source_ids = single_ids | edge_ids

    pool = load_pooled_samples()
    rng = random.Random(42)
    random_300 = set(rng.sample(pool['primary_id'].tolist(), 300))
    corpus_ids = question_source_ids | random_300
    print(f'  question-source: {len(question_source_ids)}  random: {len(random_300)}  total unique: {len(corpus_ids)}')

    pid_to_row = {r['primary_id']: r for r in pool.to_dict('records')}
    nodes = []
    for pid in corpus_ids:
        if pid not in pid_to_row:
            continue
        t = format_arxiv_row(pid_to_row[pid])
        if t:
            nodes.append(TextNode(text=t, metadata={'primary_id': pid}))
    print(f'  {len(nodes)} nodes with valid text')

    print('\nExtracting (h,r,t) triples via SimpleLLMPathExtractor...')
    extractor = SimpleLLMPathExtractor(llm=Settings.llm, max_paths_per_chunk=8, num_workers=4)
    t0 = time.time()
    import asyncio
    extracted_nodes = asyncio.run(extractor.acall(nodes))
    print(f'  extraction done in {time.time()-t0:.1f}s')

    n_with_triples = sum(1 for n in extracted_nodes if n.metadata.get('relations'))
    print(f'  {n_with_triples}/{len(extracted_nodes)} papers yielded >=1 triple')

    print('\nBuilding per-paper triple summaries + VectorStoreIndex...')
    docs = []
    for n in extracted_nodes:
        rels = n.metadata.get('relations', [])
        if not rels:
            continue
        summary = '; '.join(f'{r.source_id} {r.label} {r.target_id}' for r in rels)
        docs.append(Document(text=summary, metadata={'primary_id': n.metadata.get('primary_id')}))
    print(f'  {len(docs)} triple-summary documents')

    t0 = time.time()
    kg_index = VectorStoreIndex.from_documents(docs)
    print(f'  index built in {time.time()-t0:.1f}s')

    print('\nRunning 102 questions through RAGAS...')
    questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                  'question_type': r['question_type']} for r in qdf.to_dict('records')]
    t0 = time.time()
    results = evaluate_with_ragas(questions, kg_index)
    print(f'  RAGAS done in {time.time()-t0:.1f}s')

    results['system_name'] = 'kg_baseline'
    results['band'] = 'kg_baseline'
    results['node_idx'] = None
    results['edge_idx'] = None
    results['text_idx'] = None
    results['system_type'] = 'kg_baseline'
    results['question_id'] = qdf['id'].values
    results['question_subtype'] = qdf['question_subtype'].values
    results['composite_score'] = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)

    cols = ['system_name', 'band', 'node_idx', 'edge_idx', 'text_idx', 'system_type',
            'question_id', 'question_subtype', 'faithfulness', 'answer_relevance',
            'context_relevance', 'composite_score']
    results[cols].to_csv(OUT_CSV, mode='a', header=False, index=False)
    print(f'\nsaved -> {OUT_CSV}')

    print('\n=== KG BASELINE RESULT ===')
    print(f'mean composite: {results["composite_score"].mean():.3f}')
    for subtype in ['single_specific', 'aggregate_cross_paper', 'edge_multihop']:
        sub = results[results['question_subtype'] == subtype]
        print(f'  {subtype:<22} mean composite={sub["composite_score"].mean():.3f}  (n={len(sub)})')

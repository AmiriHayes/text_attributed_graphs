#!/usr/bin/env python3
"""
Steps 2-4 — build community-based and text-based GraphRAG systems for all
5 selected ArXiv configs (10 systems total), run RAGAS on all 102 questions
per system, save incrementally, then compute Claim 6 correlation.

Node-text resolution (needed for N8/author configs — text_fidelity_a is a
per-paper field, authors don't have one directly):
  N7 (paper):  text_fidelity_a of that row.
  N8 (author): concatenation of that author's own papers' text_fidelity_a
               (capped at 5 papers to bound prompt/embedding length for
               prolific authors), analogous to how author_embeddings.npy
               was itself computed as a mean over that author's papers.

Text-based retrieval: top-3 via vector similarity, then 1-hop graph-neighbor
expansion (using the actual TAG edges), all expanded node texts passed as
context, answer generated directly via one LLM call.

Community-based retrieval: standard top-3 over Louvain-community-aggregated
text chunks (reuses the existing query-engine-based evaluate_with_ragas).
"""
import sys
import time
import json
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / 'code'))

from evaluate_rag import load_pooled_samples, evaluate_with_ragas
from format_rows import format_arxiv_row
from generate_qa import build_author_hop_map, _client
from generic_data_manager import GenericDataManager

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

GRAPHS_DIR = REPO_ROOT / 'data/graphrag/graphs'
OUT_CSV = REPO_ROOT / 'data/graphrag/ragas_results_arxiv.csv'
MAX_PAPERS_PER_AUTHOR_TEXT = 5

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)


def get_node_texts(node_type: str, node_list: list, pool_df: pd.DataFrame) -> dict:
    """Returns {node_id: text} for the given node type."""
    if node_type == 'N7':
        pid_to_row = {r['primary_id']: r for r in pool_df.to_dict('records')}
        return {nid: (format_arxiv_row(pid_to_row[nid]) or '') for nid in node_list if nid in pid_to_row}

    elif node_type == 'N8':
        hop_map = build_author_hop_map(pool_df)
        df_reset = pool_df.reset_index(drop=True)
        texts = {}
        for author in node_list:
            row_idxs = hop_map.get(author, [])[:MAX_PAPERS_PER_AUTHOR_TEXT]
            parts = []
            for i in row_idxs:
                t = format_arxiv_row(df_reset.iloc[i].to_dict())
                if t:
                    parts.append(t)
            texts[author] = ' '.join(parts)
        return texts

    else:
        raise ValueError(f'get_node_texts not implemented for {node_type}')


def build_text_based_answer(question: str, context_texts: list, client) -> str:
    context_block = '\n\n'.join(f'[{i+1}] {t[:800]}' for i, t in enumerate(context_texts))
    prompt = (
        f'Answer the question using ONLY the numbered context passages below. '
        f'Be concise (1-3 sentences).\n\nContext:\n{context_block}\n\nQuestion: {question}\nAnswer:'
    )
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def evaluate_text_based(questions: list, index, retriever_graph: nx.Graph, node_texts_by_pos: list, client) -> pd.DataFrame:
    """
    node_texts_by_pos: list of texts indexed by the PyG positional node index
    (same order as retriever_graph's node ids 0..n-1), used to fetch 1-hop
    neighbor text after retrieval.
    """
    from ragas import evaluate as ragas_evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy, ContextRelevance
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    retriever = index.as_retriever(similarity_top_k=3)

    rows = []
    for i, q in enumerate(questions):
        nodes = retriever.retrieve(q['question'])
        top_pos = [n.node.metadata['pos'] for n in nodes]

        expanded_pos = set(top_pos)
        for p in top_pos:
            if p in retriever_graph:
                expanded_pos.update(retriever_graph.neighbors(p))
        # cap expansion so context doesn't explode on high-degree nodes
        expanded_pos = list(expanded_pos)[:15]

        context_texts = [node_texts_by_pos[p] for p in expanded_pos if node_texts_by_pos[p]]
        if not context_texts:
            context_texts = ['']

        generated_answer = build_text_based_answer(q['question'], context_texts, client)

        rows.append({
            'id': i, 'question': q['question'], 'reference_answer': q['reference_answer'],
            'generated_answer': generated_answer, 'retrieved_contexts': context_texts,
            'question_type': q['question_type'],
        })
        if (i + 1) % 10 == 0:
            print(f'    text-based retrieval+gen: {i+1}/{len(questions)}')

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model='gpt-4o-mini', temperature=0.0))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model='text-embedding-3-small'))
    eval_data = [{'user_input': r['question'], 'response': r['generated_answer'],
                  'retrieved_contexts': r['retrieved_contexts']} for r in rows]
    dataset = EvaluationDataset.from_list(eval_data)
    result = ragas_evaluate(dataset=dataset, metrics=[Faithfulness(), ResponseRelevancy(), ContextRelevance()],
                             llm=ragas_llm, embeddings=ragas_embeddings)
    scores_df = result.to_pandas()

    out = pd.DataFrame(rows)
    out['faithfulness'] = scores_df['faithfulness'].values
    out['answer_relevance'] = scores_df['answer_relevancy'].values
    out['context_relevance'] = scores_df['nv_context_relevance'].values
    return out


def build_community_index(graph: nx.Graph, node_texts_by_pos: list):
    communities = nx.algorithms.community.louvain_communities(graph, seed=42)
    docs = []
    for ci, comm in enumerate(communities):
        texts = [node_texts_by_pos[p] for p in comm if node_texts_by_pos[p]]
        if not texts:
            continue
        agg_text = ' '.join(texts)[:4000]  # cap aggregate length
        docs.append(Document(text=agg_text, metadata={'community_id': ci, 'size': len(comm)}))
    index = VectorStoreIndex.from_documents(docs)
    return index, len(communities)


if __name__ == '__main__':
    client = _client()
    configs = pd.read_csv(REPO_ROOT / 'data/graphrag/arxiv_selected_configs.csv')
    questions_df = pd.read_csv(REPO_ROOT / 'data/arxiv/questions.csv')
    questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                  'question_type': r['question_type'], 'question_subtype': r['question_subtype'],
                  'id': r['id']} for r in questions_df.to_dict('records')]

    dm = GenericDataManager('arxiv')
    pool_df = load_pooled_samples()

    all_results = []
    if OUT_CSV.exists():
        OUT_CSV.unlink()  # fresh run
    header_written = False

    for _, row in configs.iterrows():
        band, N, E, T = row['band'], row['node_idx'], row['edge_idx'], row['text_idx']
        print('\n' + '=' * 20, f'{band}  N={N} E={E} T={T}', '=' * 20)

        pt_files = list(GRAPHS_DIR.glob(f'{band}_{N}_{E}_{T}.pt'))
        data = torch.load(pt_files[0], weights_only=False)
        node_list = dm.get_node_list(N, pool_df)
        node_texts = get_node_texts(N, node_list, pool_df)
        node_texts_by_pos = [node_texts.get(nid, '') for nid in node_list]

        g = nx.Graph()
        g.add_nodes_from(range(data.num_nodes))
        ei = data.edge_index.numpy()
        g.add_edges_from(zip(ei[0], ei[1]))

        # ---- TEXT-BASED ----
        print(f'[{band}] building text-based index ({len(node_list)} nodes)...')
        docs = [Document(text=t, metadata={'pos': pos}) for pos, t in enumerate(node_texts_by_pos) if t]
        t0 = time.time()
        text_index = VectorStoreIndex.from_documents(docs)
        print(f'[{band}] text-based index built in {time.time()-t0:.1f}s ({len(docs)} docs)')

        t0 = time.time()
        text_results = evaluate_text_based(questions, text_index, g, node_texts_by_pos, client)
        print(f'[{band}] text-based RAGAS done in {time.time()-t0:.1f}s')

        text_results['system_name'] = f'{band}_{N}_{E}_{T}_text_based'
        text_results['band'], text_results['node_idx'], text_results['edge_idx'], text_results['text_idx'] = band, N, E, T
        text_results['system_type'] = 'text_based'
        text_results['question_id'] = questions_df['id'].values
        text_results['question_subtype'] = questions_df['question_subtype'].values
        text_results['composite_score'] = text_results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)

        cols = ['system_name', 'band', 'node_idx', 'edge_idx', 'text_idx', 'system_type',
                'question_id', 'question_subtype', 'faithfulness', 'answer_relevance',
                'context_relevance', 'composite_score']
        text_results[cols].to_csv(OUT_CSV, mode='a', header=not header_written, index=False)
        header_written = True
        all_results.append(text_results[cols])
        print(f'[{band}] text-based saved. mean composite={text_results["composite_score"].mean():.3f}')

        del text_index, docs

        # ---- COMMUNITY-BASED ----
        print(f'[{band}] running Louvain + building community-based index...')
        t0 = time.time()
        comm_index, n_communities = build_community_index(g, node_texts_by_pos)
        print(f'[{band}] community-based index built in {time.time()-t0:.1f}s ({n_communities} communities)')

        t0 = time.time()
        comm_questions = [{'question': q['question'], 'reference_answer': q['reference_answer'],
                            'question_type': q['question_type']} for q in questions]
        comm_results = evaluate_with_ragas(comm_questions, comm_index)
        print(f'[{band}] community-based RAGAS done in {time.time()-t0:.1f}s')

        comm_results['system_name'] = f'{band}_{N}_{E}_{T}_community_based'
        comm_results['band'], comm_results['node_idx'], comm_results['edge_idx'], comm_results['text_idx'] = band, N, E, T
        comm_results['system_type'] = 'community_based'
        comm_results['question_id'] = questions_df['id'].values
        comm_results['question_subtype'] = questions_df['question_subtype'].values
        comm_results['composite_score'] = comm_results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)

        comm_results[cols].to_csv(OUT_CSV, mode='a', header=False, index=False)
        all_results.append(comm_results[cols])
        print(f'[{band}] community-based saved. mean composite={comm_results["composite_score"].mean():.3f}')

        del comm_index

    print('\n' + '=' * 20, 'STEP 2-3 COMPLETE', '=' * 20)
    print(f'saved -> {OUT_CSV}')

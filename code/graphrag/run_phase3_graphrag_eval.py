#!/usr/bin/env python3
"""
Phase 3 of the final rerun — GraphRAG evaluation on the 150-question sets,
for every M1 variant surviving the 95% zero-exclusion filter (same
threshold as run_analysis.py), excluding T12e entirely (not a genuine
no-text control for GraphRAG -- format_rows.py always reads
text_fidelity_a regardless of T variant).

Reuses the real, already-validated infra rather than rebuilding it:
  - TAGConstructor (code/tag_constructor.py) -- same graph-construction
    class experiment_runner.py uses for GNN training -- to build each
    variant's graph fresh (no pre-saved .pt files needed).
  - evaluate_text_based (data/graphrag/run_step2_4.py) -- the real
    1-hop-neighbor-expansion GraphRAG retrieval + RAGAS scoring logic
    that produced the original ragas_stability_{arxiv,amazon}_C.csv.
  - get_node_texts pattern (data/graphrag/run_stability_study_step1.py)
    -- generalizes to N7/N8/N9 for any dataset via the generic schema.

Full sample pool (sample_00-09, pooled + deduped) is the graph source,
matching the prior "Scale C" convention and the spec's "Full sample pool
graphs" requirement.

Question split: 75 train / 75 test, seed 42, stratified by
question_subtype -- identical split logic to graphrag_dt_analysis.py, so
the `split` column written here is what Phase 4 reads back rather than
re-deriving.

Does NOT read or write data/graphrag/ragas_stability_*.csv -- those are
historical reference only (different question count, different scale
convention). This is a clean, separate, final-rerun output.

Reads:
  - output/run_final/construction_performance_table_{dataset}.csv (to
    determine surviving M1 variants)
  - data/{dataset}/train/samples/sample_00..09.jsonl (pool)
  - data/{dataset}/questions.csv (150 rows)

Writes (incrementally, resumable -- skips any variant already fully
present with all 150 questions):
  - output/run_final/ragas_results_{dataset}.csv
    columns: variant, node_type, question_id, question_subtype, split,
             faithfulness, answer_relevance, context_relevance,
             composite, wall_time_seconds

Usage:
  python3 run_phase3_graphrag_eval.py --dataset history
"""
import argparse
import sys
import time
import types
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── ragas 0.2.15(pinned elsewhere)/langchain_community vertexai shim --
# see extend_questions_to_150.py for the full explanation. Also fixes a
# real async-executor crash under Python 3.14 -- this script requires
# ragas>=0.4 to be installed (already upgraded this session).
_dummy = types.ModuleType('langchain_community.chat_models.vertexai')
class _ChatVertexAIStub:
    pass
_dummy.ChatVertexAI = _ChatVertexAIStub
sys.modules.setdefault('langchain_community.chat_models.vertexai', _dummy)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from evaluate_rag import load_pooled_samples
from format_rows import format_arxiv_row
from generate_qa import build_author_hop_map, _client
from run_step2_4 import evaluate_text_based

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

ZERO_EXCLUSION_THRESHOLD = 0.95
MAX_HOP_ROWS = 5
SEED = 42
MAX_ATTEMPTS = 5


def get_surviving_m1_variants(dataset: str) -> list:
    """Same zero-exclusion logic as run_analysis.py's _build_summary, applied
    to output/run_final's fresh construction_performance_table. Excludes
    T12e unconditionally (not just via the zero filter -- T12e is
    structurally invalid for GraphRAG regardless of its score)."""
    path = REPO_ROOT / f'output/run_final/construction_performance_table_{dataset}.csv'
    df = pd.read_csv(path)
    m1 = df[df['Task_Idx'] == 'M1']
    score_col = 'S_GNN_step1' if 'S_GNN_step1' in m1.columns else 'normalized_score'

    survivors = []
    for key, grp in m1.groupby(['Node_Idx', 'Edge_Idx', 'Text_Idx']):
        N, E, T = key
        if T == 'T12e':
            continue
        train_rows = grp[grp['run_split'] == 'train']
        train_valid = train_rows[score_col].dropna()
        n_total = len(train_rows)
        n_zero = int((train_valid == 0.0).sum())
        if n_total > 0 and n_zero / n_total > ZERO_EXCLUSION_THRESHOLD:
            continue
        if train_valid.isna().all():
            continue
        survivors.append({'Node_Idx': N, 'Edge_Idx': E, 'Text_Idx': T})
    return survivors


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


def already_done(out_csv: Path, variant_key: str, n_questions_expected: int) -> bool:
    if not out_csv.exists():
        return False
    try:
        prior = pd.read_csv(out_csv, usecols=['variant'])
        return (prior['variant'] == variant_key).sum() >= n_questions_expected
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    args = ap.parse_args()
    ds = args.dataset

    client = _client()
    out_csv = REPO_ROOT / f'output/run_final/ragas_results_{ds}.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    variants = get_surviving_m1_variants(ds)
    print(f'{ds}: {len(variants)} surviving M1 variants (T12e excluded, >{ZERO_EXCLUSION_THRESHOLD*100:.0f}% zero excluded)')
    for v in variants:
        print(f"  {v['Node_Idx']}/{v['Edge_Idx']}/{v['Text_Idx']}")

    q_df = pd.read_csv(REPO_ROOT / f'data/{ds}/questions.csv')
    n_q = len(q_df)
    assert n_q == 150, f'expected 150 questions for {ds}, found {n_q}'

    train_q, test_q = train_test_split(q_df, test_size=0.5, random_state=SEED, stratify=q_df['question_subtype'])
    split_map = {qid: 'train' for qid in train_q['id']}
    split_map.update({qid: 'test' for qid in test_q['id']})
    print(f'  question split: {len(train_q)} train / {len(test_q)} test (seed={SEED}, stratified)')

    questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                  'question_type': r.get('question_type', '')} for r in q_df.to_dict('records')]

    print(f'Loading {ds} full sample pool...')
    pool_df = load_pooled_samples(dataset=ds)
    print(f'  {len(pool_df)} unique rows\n')

    dm = GenericDataManager(ds, base_path=str(REPO_ROOT / 'data'))
    tc = TAGConstructor(dm)

    header_written = out_csv.exists()

    for v in variants:
        N, E, T = v['Node_Idx'], v['Edge_Idx'], v['Text_Idx']
        variant_key = f'{N}_{E}_{T}'

        if already_done(out_csv, variant_key, n_q):
            print(f'SKIP (already done) {variant_key}')
            continue

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                t0 = time.time()
                variant = {'M': 'M1', 'N': N, 'E': E, 'T': T}
                data = tc.construct(variant, pool_df, 'train')

                node_list = dm.get_node_list(N, pool_df)
                node_texts = get_node_texts(N, node_list, pool_df)
                node_texts_by_pos = [node_texts.get(nid, '') for nid in node_list]

                g = nx.Graph()
                g.add_nodes_from(range(data.num_nodes))
                ei = data.edge_index.numpy()
                g.add_edges_from(zip(ei[0], ei[1]))

                docs = [Document(text=t, metadata={'pos': pos}) for pos, t in enumerate(node_texts_by_pos) if t]
                if not docs:
                    print(f'SKIP {variant_key}: 0 non-empty node texts')
                    break
                index = VectorStoreIndex.from_documents(docs)

                results = evaluate_text_based(questions, index, g, node_texts_by_pos, client)
                wall_time = time.time() - t0

                results['variant'] = variant_key
                results['node_type'] = N
                results['question_id'] = q_df['id'].values
                results['question_subtype'] = q_df['question_subtype'].values
                results['split'] = q_df['id'].map(split_map).values
                results['composite'] = results[['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)
                results['wall_time_seconds'] = wall_time

                cols = ['variant', 'node_type', 'question_id', 'question_subtype', 'split',
                        'faithfulness', 'answer_relevance', 'context_relevance', 'composite', 'wall_time_seconds']
                results[cols].to_csv(out_csv, mode='a', header=not header_written, index=False)
                header_written = True

                mean_c = results['composite'].mean()
                anomalies = ((results['faithfulness'] == 0) | (results['context_relevance'] == 0)).sum()
                print(f'{variant_key} ({N}): nodes={data.num_nodes} docs={len(docs)} '
                      f'wall_time={wall_time:.1f}s mean_composite={mean_c:.3f} '
                      f'anomalous(faith=0 or ctx=0)={anomalies}/{n_q}')
                del index, docs
                break
            except Exception as e:
                wait = min(60, 5 * attempt)
                print(f'ERROR {variant_key} (attempt {attempt}/{MAX_ATTEMPTS}): {type(e).__name__}: {e} -- retrying in {wait}s')
                if attempt == MAX_ATTEMPTS:
                    print(f'GIVING UP on {variant_key} after {MAX_ATTEMPTS} attempts')
                else:
                    time.sleep(wait)

    final = pd.read_csv(out_csv)
    print(f'\n=== {ds} REPORT ===')
    print(f'  n variants evaluated: {final["variant"].nunique()}')
    print(f'  mean composite range: {final.groupby("variant")["composite"].mean().min():.3f} - '
          f'{final.groupby("variant")["composite"].mean().max():.3f}')
    print(f'  saved -> {out_csv}')


if __name__ == '__main__':
    main()

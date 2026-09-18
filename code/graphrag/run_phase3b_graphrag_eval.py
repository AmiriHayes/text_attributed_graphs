#!/usr/bin/env python3
"""
Phase 3b -- GraphRAG evaluation of the NEWLY ADDED questions (the 300-question
extension), for every M1 variant surviving the 95% zero-exclusion filter,
T12e excluded.

RELATIONSHIP TO run_phase3_graphrag_eval.py
Identical evaluation machinery -- same TAGConstructor graph construction, same
pooled sample_00-09 source, same LlamaIndex VectorStoreIndex with 1-hop
neighbour expansion, same gpt-4o-mini generation, same RAGAS scoring via
run_step2_4.evaluate_text_based. Only the bookkeeping differs:

  1. RESUME GRANULARITY. Phase 3 skipped a variant when its row count reached
     the expected question count. With the question set now larger than what
     is already scored, that test would re-evaluate every question for a
     variant that is merely incomplete. Here the skip is keyed on the
     (variant, question_id) PAIR, so each variant is evaluated on exactly the
     questions it is missing and nothing is scored twice.

  2. SPLIT SOURCE. The train/test assignment comes from graphrag_split.py's
     canonical split file, not from a split re-derived here. See
     graphrag_split.py for why.

All three RAGAS metrics (faithfulness, answer_relevance, context_relevance)
are computed and stored per question, plus their mean as `composite`.
answer_relevance is the canonical DT target, but the other two are retained
so any metric can be re-analysed without re-running the evaluation.

wall_time_seconds is the time to build the index and score THIS batch of
questions for that variant -- not a running total over all of the variant's
questions. Every row of a batch carries the same value, so a per-variant
total for the timing table is sum-over-distinct: group by
(variant, wall_time_seconds) and take one value per group. The column set is
deliberately unchanged from Phase 3 so the two runs' rows concatenate cleanly.

Reads:
  - output/run_final/construction_performance_table_{dataset}.csv
  - output/run_final/question_split_{dataset}.csv (via graphrag_split)
  - data/{dataset}/questions.csv
  - data/{dataset}/train/samples/sample_00..09.jsonl

Writes (append-only, resumable):
  - output/run_final/ragas_results_{dataset}.csv

Usage:
  python3 run_phase3b_graphrag_eval.py --dataset arxiv
  python3 run_phase3b_graphrag_eval.py --dataset arxiv --plan_only
"""
import argparse
import sys
import time
import types
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── ragas / langchain_community vertexai shim -- see
# extend_questions_to_150.py for the full explanation.
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

load_dotenv()

from generic_data_manager import GenericDataManager
from tag_constructor import TAGConstructor
from evaluate_rag import load_pooled_samples
from format_rows import format_arxiv_row
from generate_qa import build_author_hop_map, _client
from run_step2_4 import evaluate_text_based
from graphrag_split import split_map

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

ZERO_EXCLUSION_THRESHOLD = 0.95
MAX_HOP_ROWS = 5
MAX_ATTEMPTS = 5
OUT_COLS = ['variant', 'node_type', 'question_id', 'question_subtype', 'split',
            'faithfulness', 'answer_relevance', 'context_relevance', 'composite',
            'wall_time_seconds']


def get_surviving_m1_variants(dataset: str) -> list:
    """Same zero-exclusion logic as run_analysis.py / Phase 3. T12e is excluded
    unconditionally, not merely via the zero filter: format_rows.py always
    reads text_fidelity_a regardless of T variant, so T12e is not a genuine
    no-text control for GraphRAG whatever it scores."""
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
    """Unchanged from Phase 3 -- N7 rows directly, N8/N9 as concatenated
    text of up to MAX_HOP_ROWS associated rows."""
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


def load_done_pairs(out_csv: Path) -> set:
    """Already-scored (variant, question_id) pairs."""
    if not out_csv.exists():
        return set()
    prior = pd.read_csv(out_csv, usecols=['variant', 'question_id'])
    return set(zip(prior['variant'], prior['question_id']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--plan_only', action='store_true',
                    help='report what would be evaluated and exit without API calls')
    args = ap.parse_args()
    ds = args.dataset

    out_csv = REPO_ROOT / f'output/run_final/ragas_results_{ds}.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    variants = get_surviving_m1_variants(ds)
    q_df = pd.read_csv(REPO_ROOT / f'data/{ds}/questions.csv')
    smap = split_map(ds, verbose=False)
    q_df['split'] = q_df['id'].map(smap)
    if q_df['split'].isna().any():
        raise RuntimeError(f'{ds}: {int(q_df["split"].isna().sum())} questions missing from the '
                           f'canonical split -- run build_canonical_split.py first')

    done = load_done_pairs(out_csv)
    print(f'{ds}: {len(variants)} surviving M1 variants (T12e excluded), '
          f'{len(q_df)} questions, {len(done)} (variant,question) pairs already scored')
    print(f'  split: {(q_df["split"]=="train").sum()} train / {(q_df["split"]=="test").sum()} test')
    print(f'  subtypes: {dict(q_df["question_subtype"].value_counts())}')

    plan = []
    for v in variants:
        vk = f"{v['Node_Idx']}_{v['Edge_Idx']}_{v['Text_Idx']}"
        pending = q_df[~q_df['id'].map(lambda qid: (vk, qid) in done)]
        plan.append((v, vk, pending))
        print(f'  {vk:20s} pending={len(pending):4d}')
    total_pending = sum(len(p) for _, _, p in plan)
    print(f'  TOTAL pending question-evaluations: {total_pending}')

    if args.plan_only:
        print('  PLAN ONLY -- exiting without evaluating')
        return
    if total_pending == 0:
        print('  nothing to do')
        return

    print(f'\nLoading {ds} full sample pool...')
    pool_df = load_pooled_samples(dataset=ds)
    print(f'  {len(pool_df)} unique rows\n')

    client = _client()
    dm = GenericDataManager(ds, base_path=str(REPO_ROOT / 'data'))
    tc = TAGConstructor(dm)
    header_written = out_csv.exists()

    for v, vk, pending in plan:
        if len(pending) == 0:
            print(f'SKIP (complete) {vk}')
            continue
        N, E, T = v['Node_Idx'], v['Edge_Idx'], v['Text_Idx']

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                t0 = time.time()
                data = tc.construct({'M': 'M1', 'N': N, 'E': E, 'T': T}, pool_df, 'train')

                node_list = dm.get_node_list(N, pool_df)
                node_texts = get_node_texts(N, node_list, pool_df)
                node_texts_by_pos = [node_texts.get(nid, '') for nid in node_list]

                g = nx.Graph()
                g.add_nodes_from(range(data.num_nodes))
                ei = data.edge_index.numpy()
                g.add_edges_from(zip(ei[0], ei[1]))

                docs = [Document(text=t, metadata={'pos': pos})
                        for pos, t in enumerate(node_texts_by_pos) if t]
                if not docs:
                    print(f'SKIP {vk}: 0 non-empty node texts')
                    break
                index = VectorStoreIndex.from_documents(docs)

                questions = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                              'question_type': r.get('question_type', '')}
                             for r in pending.to_dict('records')]
                results = evaluate_text_based(questions, index, g, node_texts_by_pos, client)
                wall_time = time.time() - t0

                # evaluate_text_based preserves input order, so positional
                # alignment back onto `pending` is safe.
                results['variant'] = vk
                results['node_type'] = N
                results['question_id'] = pending['id'].values
                results['question_subtype'] = pending['question_subtype'].values
                results['split'] = pending['split'].values
                results['composite'] = results[
                    ['faithfulness', 'answer_relevance', 'context_relevance']].mean(axis=1)
                results['wall_time_seconds'] = wall_time

                results[OUT_COLS].to_csv(out_csv, mode='a', header=not header_written, index=False)
                header_written = True

                anomalies = ((results['faithfulness'] == 0) | (results['context_relevance'] == 0)).sum()
                print(f'{vk} ({N}): nodes={data.num_nodes} docs={len(docs)} n_new={len(pending)} '
                      f'wall_time={wall_time:.1f}s '
                      f'mean_answer_relevance={results["answer_relevance"].mean():.3f} '
                      f'mean_composite={results["composite"].mean():.3f} '
                      f'anomalous(faith=0 or ctx=0)={anomalies}/{len(pending)}')
                del index, docs
                break
            except Exception as e:
                wait = min(60, 5 * attempt)
                print(f'ERROR {vk} (attempt {attempt}/{MAX_ATTEMPTS}): {type(e).__name__}: {e} '
                      f'-- retrying in {wait}s')
                if attempt == MAX_ATTEMPTS:
                    print(f'GIVING UP on {vk} after {MAX_ATTEMPTS} attempts')
                else:
                    time.sleep(wait)

    final = pd.read_csv(out_csv)
    per_variant = final.groupby('variant')['question_id'].nunique()
    print(f'\n=== {ds} REPORT ===')
    print(f'  n variants evaluated: {final["variant"].nunique()}')
    print(f'  questions per variant: min={per_variant.min()} max={per_variant.max()}')
    ar = final.groupby('variant')['answer_relevance'].mean()
    print(f'  mean answer_relevance range across variants: {ar.min():.3f} - {ar.max():.3f}')
    comp = final.groupby('variant')['composite'].mean()
    print(f'  mean composite range across variants: {comp.min():.3f} - {comp.max():.3f}')
    zero_f = final.groupby('variant').apply(lambda g: int((g['faithfulness'] == 0).sum()))
    flagged = zero_f[zero_f > 0]
    if len(flagged):
        print(f'  variants with faithfulness=0 rows: {dict(flagged)}')
    print(f'  saved -> {out_csv}')


if __name__ == '__main__':
    main()

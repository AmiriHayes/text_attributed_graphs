#!/usr/bin/env python3
"""
evaluate_rag.py — Stage 3 QA generation, Step 5: validation gate.

ragas==0.2.15 pinned (see requirements.txt) — latest (0.4.x) fails to import
in this environment: ragas/llms/base.py unconditionally imports
langchain_community.chat_models.vertexai, which no longer exists in current
langchain-community (split out upstream as part of that package's
deprecation). 0.2.15 is the last line confirmed to import cleanly here.

Builds the SIMPLEST possible retrieval baseline — a plain LlamaIndex
VectorStoreIndex over the pooled arxiv/train sample pool, no graph — and
scores it with RAGAS (faithfulness, answer_relevance, context_relevance)
as a sanity gate BEFORE spending budget generating the full 50-question set.

RAGAS metric name mapping (ragas 0.2.15 -> spec naming):
    faithfulness         -> faithfulness
    answer_relevancy      -> answer_relevance   (ResponseRelevancy)
    nv_context_relevance -> context_relevance   (ContextRelevance)
All three are reference-free (no ground-truth answer required), matching
the original methodology proposal's metric choice.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_rows import format_arxiv_row
from generate_qa import generate_qa, validate_qa, classify_question_type, _client

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = REPO_ROOT / 'data/arxiv/train/samples'
GEN_MODEL = 'gpt-4o-mini'


def load_pooled_samples(n_files=10, dataset='arxiv') -> pd.DataFrame:
    """Pool sample_00..sample_09, dedupe on primary_id. dataset defaults to
    'arxiv' for backward compat; pass e.g. 'amazon' to pool that dataset's
    train samples instead."""
    samples_dir = REPO_ROOT / f'data/{dataset}/train/samples'
    frames = []
    for i in range(n_files):
        path = samples_dir / f'sample_{i:02d}.jsonl'
        frames.append(pd.read_json(path, lines=True))
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.drop_duplicates(subset='primary_id').reset_index(drop=True)
    return pooled


def build_index(df: pd.DataFrame):
    """Simplest possible retrieval baseline: plain VectorStoreIndex, no graph."""
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
    Settings.llm = LlamaOpenAI(model=GEN_MODEL, temperature=0.0)

    docs = []
    for _, row in df.iterrows():
        text = format_arxiv_row(row.to_dict())
        if text:
            docs.append(Document(text=text, metadata={'primary_id': row['primary_id']}))
    return VectorStoreIndex.from_documents(docs)


def generate_n_valid_questions(df: pd.DataFrame, n: int, seed=42, max_attempts=None) -> list[dict]:
    """Reuses Step 2-4 logic to collect n valid single-hop questions."""
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    max_attempts = max_attempts or (n * 3)
    client = _client()
    results = []
    attempts = 0
    for _, row in shuffled.iterrows():
        if len(results) >= n or attempts >= max_attempts:
            break
        attempts += 1
        text = format_arxiv_row(row.to_dict())
        if text is None:
            continue
        try:
            qa = generate_qa(text, client=client)
        except ValueError:
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            continue
        results.append({
            'question': qa['question'],
            'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'source_row_id': row['primary_id'],
            'source_text_preview': text[:100],
        })
    return results


def evaluate_with_ragas(questions: list[dict], index, llm_for_generation: str = GEN_MODEL) -> pd.DataFrame:
    """
    For each question: retrieve top-3 contexts + generate an answer via the
    index's query engine, then score with RAGAS (faithfulness, answer_relevance,
    context_relevance). Returns one row per question.
    """
    from ragas import evaluate as ragas_evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy, ContextRelevance
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    query_engine = index.as_query_engine(similarity_top_k=3)

    rows = []
    for i, q in enumerate(questions):
        resp = query_engine.query(q['question'])
        generated_answer = str(resp)
        retrieved_contexts = [n.node.get_content() for n in resp.source_nodes]
        rows.append({
            'id': i,
            'question': q['question'],
            'reference_answer': q['reference_answer'],
            'generated_answer': generated_answer,
            'retrieved_contexts': retrieved_contexts,
            'question_type': q['question_type'],
        })

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_for_generation, temperature=0.0))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model='text-embedding-3-small'))

    eval_data = [
        {
            'user_input': r['question'],
            'response': r['generated_answer'],
            'retrieved_contexts': r['retrieved_contexts'],
        }
        for r in rows
    ]
    dataset = EvaluationDataset.from_list(eval_data)
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), ContextRelevance()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    scores_df = result.to_pandas()

    out = pd.DataFrame(rows)
    out['faithfulness'] = scores_df['faithfulness'].values
    out['answer_relevance'] = scores_df['answer_relevancy'].values
    out['context_relevance'] = scores_df['nv_context_relevance'].values
    return out


def print_gate_report(df: pd.DataFrame, n_expected: int) -> bool:
    print(f'\n=== VALIDATION GATE — RAGAS on first {n_expected} questions ===')
    for metric in ['faithfulness', 'answer_relevance', 'context_relevance']:
        print(f'  {metric:<18} mean={df[metric].mean():.3f}  std={df[metric].std():.3f}')

    print('\n  3 lowest-faithfulness questions:')
    lowest = df.nsmallest(3, 'faithfulness')
    for _, r in lowest.iterrows():
        print(f'    [{r["faithfulness"]:.3f}] {r["question"]}')

    mean_faith = df['faithfulness'].mean()
    verdict = 'PASS' if mean_faith > 0.5 else 'FAIL'
    print(f'\n  VERDICT: {verdict} (mean faithfulness = {mean_faith:.3f}, threshold = 0.5)')
    return verdict == 'PASS'


if __name__ == '__main__':
    print('Loading pooled sample_00-09...')
    df = load_pooled_samples()
    print(f'  {len(df)} unique rows after dedup on primary_id')

    print('\nBuilding VectorStoreIndex (plain, no graph)...')
    index = build_index(df)

    print('\nGenerating 10 valid single-hop questions for the gate...')
    questions = generate_n_valid_questions(df, n=10)
    print(f'  {len(questions)}/10 collected')

    if len(questions) < 10:
        print('  WARNING: fewer than 10 valid questions collected — gate results may be noisy')

    print('\nRunning RAG + RAGAS...')
    results_df = evaluate_with_ragas(questions, index)

    # Gate preview saves — CSV for quick visual validation. These 10 rows get
    # superseded by the full 50 from Step 6 / Step 7; saved now so there's
    # something concrete to look at before committing to the full run.
    out_dir = REPO_ROOT / 'data/arxiv'
    ragas_preview_path = out_dir / 'ragas_baseline.csv'
    results_df.to_csv(ragas_preview_path, index=False)
    print(f'\nSaved {len(results_df)}-row RAGAS gate preview -> {ragas_preview_path}')

    questions_preview = pd.DataFrame([
        {
            'id': f'arxiv_sh_{i:03d}',
            'dataset': 'arxiv',
            'question': q['question'],
            'reference_answer': q['reference_answer'],
            'question_type': q['question_type'],
            'source_row_id': q['source_row_id'],
            'source_text_preview': q['source_text_preview'],
            'hop_type': 'single',
        }
        for i, q in enumerate(questions)
    ])
    questions_preview_path = out_dir / 'questions.csv'
    questions_preview.to_csv(questions_preview_path, index=False)
    print(f'Saved {len(questions_preview)}-row questions preview -> {questions_preview_path}')

    passed = print_gate_report(results_df, n_expected=len(questions))
    print(f'\n{"PROCEED to Step 6" if passed else "STOP — do not proceed to Step 6 until this is fixed"}')

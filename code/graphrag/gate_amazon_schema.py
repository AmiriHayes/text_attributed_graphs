#!/usr/bin/env python3
"""
Generic Amazon-schema gate runner (single-hop RAGAS + edge-multihop
reviewer-connection) -- parameterized by --dataset so electronics/toys
(both share Amazon's exact schema) reuse it without duplication.
Usage: python gate_amazon_schema.py --dataset electronics
"""
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_rows import format_arxiv_row
from generate_qa import (
    generate_qa, validate_qa, classify_question_type, _client,
    sample_edge_multihop_pairs, generate_edge_multihop_qa, validate_edge_multihop,
    AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
)
from evaluate_rag import load_pooled_samples, evaluate_with_ragas

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    args = ap.parse_args()
    ds = args.dataset

    client = _client()
    pool = load_pooled_samples(dataset=ds)
    print(f'{ds} pool: {len(pool)} unique rows\n')

    print('=' * 20, 'SINGLE-HOP GATE (10 questions)', '=' * 20)
    shuffled = pool.sample(frac=1, random_state=42).reset_index(drop=True)
    questions = []
    for _, row in shuffled.iterrows():
        if len(questions) >= 10:
            break
        text = format_arxiv_row(row.to_dict())
        if text is None:
            continue
        try:
            qa = generate_qa(text, client=client)
        except ValueError as e:
            print(f'  FAILED: {e}')
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            print(f'  rejected ({reason}): {qa.get("question","")[:80]}')
            continue
        questions.append({
            'question': qa['question'], 'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
        })
    print(f'collected {len(questions)}/10\n')

    print(f'Building VectorStoreIndex ({ds} pool)...')
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
    Settings.llm = LlamaOpenAI(model='gpt-4o-mini', temperature=0.0)

    docs = []
    for _, row in pool.iterrows():
        t = format_arxiv_row(row.to_dict())
        if t:
            docs.append(Document(text=t, metadata={'primary_id': row['primary_id']}))
    index = VectorStoreIndex.from_documents(docs)
    print(f'  index built: {len(docs)} docs')

    single_results = evaluate_with_ragas(questions, index)
    mean_faith = single_results['faithfulness'].mean()
    print('\n  faithfulness       mean={:.3f}'.format(mean_faith))
    print('  answer_relevance   mean={:.3f}'.format(single_results['answer_relevance'].mean()))
    print('  context_relevance  mean={:.3f}'.format(single_results['context_relevance'].mean()))
    single_pass = mean_faith >= 0.5
    print(f'\n  SINGLE-HOP GATE: {"PASS" if single_pass else "FAIL"} (mean faithfulness={mean_faith:.3f})')

    print('\n\n' + '=' * 20, 'EDGE-MULTIHOP GATE (5 questions, reviewer connection)', '=' * 20)
    pairs = sample_edge_multihop_pairs(pool, n_pairs=5, seed=42)
    print(f'sampled {len(pairs)}/5 reviewer-hop pairs\n')

    n_pass = 0
    for i, p in enumerate(pairs):
        at = format_arxiv_row(p['anchor'])
        tt = format_arxiv_row(p['target'])
        print(f'--- pair {i} ---  shared_reviewer={p["shared_author"]}')
        if at is None or tt is None:
            print('SKIP null/short text')
            continue
        try:
            qa = generate_edge_multihop_qa(
                at, tt, p['shared_author'], client=client,
                system_prompt=AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
                anchor_label='Product 1 (anchor)',
                target_label='Product 2 (connected via shared reviewer)',
                connector_label='Shared reviewer identifier',
            )
        except ValueError as e:
            print(f'FAILED: {e}')
            continue
        valid, reason = validate_edge_multihop(qa, p['shared_author'], connector_words=['reviewer'])
        print(f'Q: {qa["question"]}')
        print(f'A: {qa["answer"]}')
        print(f'-> ({valid}, {reason!r})')
        if valid:
            n_pass += 1
        print()

    edge_pass = n_pass >= 3
    print(f'EDGE-MULTIHOP GATE: {"PASS" if edge_pass else "FAIL"} ({n_pass}/5)')

    print('\n\n' + '=' * 20, 'SUMMARY', '=' * 20)
    print(f'{ds} single-hop gate: {"PASS" if single_pass else "FAIL"}')
    print(f'{ds} edge-multihop gate: {"PASS" if edge_pass else "FAIL"}')

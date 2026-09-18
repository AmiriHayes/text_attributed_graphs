#!/usr/bin/env python3
"""Amazon port: single-hop gate, edge-multihop gate, co-reviewer density check."""
import sys
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_rows import format_arxiv_row  # dataset-agnostic despite the name
from generate_qa import (
    generate_qa, validate_qa, classify_question_type, _client,
    sample_edge_multihop_pairs, generate_edge_multihop_qa, validate_edge_multihop,
    AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
)
from evaluate_rag import load_pooled_samples, evaluate_with_ragas

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if __name__ == '__main__':
    client = _client()
    pool = load_pooled_samples(dataset='amazon')
    print(f'amazon pool: {len(pool)} unique rows\n')

    # ============================================================
    # SINGLE-HOP GATE
    # ============================================================
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
            'source_row_id': row['primary_id'],
        })
    print(f'collected {len(questions)}/10 valid single-hop questions')

    print('\nBuilding VectorStoreIndex (amazon pool)...')
    # evaluate_rag.py's build_index() is arxiv-specific (hardcoded paths/format
    # function inside); re-implementing inline here rather than generalizing
    # it too, since this is a one-off gate script.
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
    amazon_index = VectorStoreIndex.from_documents(docs)
    print(f'  index built: {len(docs)} docs')

    single_results = evaluate_with_ragas(questions, amazon_index)
    print('\n  faithfulness       mean={:.3f}  std={:.3f}'.format(
        single_results['faithfulness'].mean(), single_results['faithfulness'].std()))
    print('  answer_relevance   mean={:.3f}  std={:.3f}'.format(
        single_results['answer_relevance'].mean(), single_results['answer_relevance'].std()))
    print('  context_relevance  mean={:.3f}  std={:.3f}'.format(
        single_results['context_relevance'].mean(), single_results['context_relevance'].std()))
    mean_faith = single_results['faithfulness'].mean()
    single_gate_pass = mean_faith >= 0.5
    print(f'\n  SINGLE-HOP GATE: {"PASS" if single_gate_pass else "FAIL"} (mean faithfulness={mean_faith:.3f}, threshold=0.5)')

    # ============================================================
    # EDGE-MULTIHOP GATE
    # ============================================================
    print('\n\n' + '=' * 20, 'EDGE-MULTIHOP GATE (5 questions)', '=' * 20)
    pairs = sample_edge_multihop_pairs(pool, n_pairs=5, seed=42)
    print(f'sampled {len(pairs)}/5 valid hop pairs\n')

    n_pass = 0
    for i, p in enumerate(pairs):
        anchor_text = format_arxiv_row(p['anchor'])
        target_text = format_arxiv_row(p['target'])
        print(f'--- pair {i} ---')
        print(f'anchor primary_id={p["anchor"]["primary_id"]}  target primary_id={p["target"]["primary_id"]}')
        print(f'shared reviewer: {p["shared_author"]}')
        if anchor_text is None or target_text is None:
            print('SKIP: null/short text')
            continue
        try:
            qa = generate_edge_multihop_qa(
                anchor_text, target_text, p['shared_author'], client=client,
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

    edge_gate_pass = n_pass >= 3
    print(f'EDGE-MULTIHOP GATE: {"PASS" if edge_gate_pass else "FAIL"} ({n_pass}/5, threshold 3/5)')

    # ============================================================
    # CO-REVIEWER PRODUCT-PAIR DENSITY CHECK
    # ============================================================
    print('\n\n' + '=' * 20, 'CO-REVIEWER PRODUCT-PAIR DENSITY', '=' * 20)
    reviewer_to_products = defaultdict(set)
    for _, row in pool.iterrows():
        sid = row['secondary_id']
        if isinstance(sid, str):
            reviewer_to_products[sid].add(row['primary_id'])

    pair_shared_count = Counter()
    for reviewer, products in reviewer_to_products.items():
        if len(products) < 2:
            continue
        for p1, p2 in combinations(sorted(products), 2):
            pair_shared_count[(p1, p2)] += 1

    n_pairs_total = len(pair_shared_count)
    n_pairs_ge2 = sum(1 for v in pair_shared_count.values() if v >= 2)
    n_pairs_ge3 = sum(1 for v in pair_shared_count.values() if v >= 3)
    print(f'reviewers with >=2 products reviewed: {sum(1 for p in reviewer_to_products.values() if len(p) >= 2)}')
    print(f'total distinct product-pairs sharing >=1 reviewer: {n_pairs_total}')
    print(f'product-pairs sharing >=2 reviewers: {n_pairs_ge2}')
    print(f'product-pairs sharing >=3 reviewers: {n_pairs_ge3}')

    print('\n' + '=' * 20, 'SUMMARY', '=' * 20)
    print(f'single-hop gate: {"PASS" if single_gate_pass else "FAIL"}')
    print(f'edge-multihop gate: {"PASS" if edge_gate_pass else "FAIL"}')
    print(f'co-reviewer pairs with >=2 shared reviewers: {n_pairs_ge2}')

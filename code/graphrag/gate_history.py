#!/usr/bin/env python3
"""History port: single-hop gate + category-bridge edge_multihop gate."""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_rows import format_arxiv_row
from generate_qa import (
    generate_qa, validate_qa, classify_question_type, _client,
    sample_category_bridge_pairs, generate_edge_multihop_qa, validate_edge_multihop,
    HISTORY_CATEGORY_BRIDGE_SYSTEM_PROMPT,
)
from evaluate_rag import load_pooled_samples, evaluate_with_ragas

if __name__ == '__main__':
    client = _client()
    pool = load_pooled_samples(dataset='history')
    print(f'history pool: {len(pool)} unique rows\n')

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

    print('Building VectorStoreIndex (history pool)...')
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
    print('\n  faithfulness       mean={:.3f}'.format(single_results['faithfulness'].mean()))
    print('  answer_relevance   mean={:.3f}'.format(single_results['answer_relevance'].mean()))
    print('  context_relevance  mean={:.3f}'.format(single_results['context_relevance'].mean()))
    mean_faith = single_results['faithfulness'].mean()
    single_pass = mean_faith >= 0.5
    print(f'\n  SINGLE-HOP GATE: {"PASS" if single_pass else "FAIL"} (mean faithfulness={mean_faith:.3f})')

    print('\n\n' + '=' * 20, 'CATEGORY-BRIDGE GATE (5 questions)', '=' * 20)
    pairs = sample_category_bridge_pairs(pool, n_pairs=5, seed=42)
    print(f'sampled {len(pairs)}/5 cross-category pairs\n')

    n_pass = 0
    for i, p in enumerate(pairs):
        anchor_text = format_arxiv_row(p['anchor'])
        target_text = format_arxiv_row(p['target'])
        print(f'--- pair {i} ---')
        print(f'anchor cat={p["category_1"]}  target cat={p["category_2"]}')
        if anchor_text is None or target_text is None:
            print('SKIP null/short text')
            continue
        try:
            qa = generate_edge_multihop_qa(
                anchor_text, target_text,
                shared_author=f'{p["category_1"]}|{p["category_2"]}',
                client=client,
                system_prompt=HISTORY_CATEGORY_BRIDGE_SYSTEM_PROMPT,
                anchor_label=f'Book 1 ({p["category_1"]})',
                target_label=f'Book 2 ({p["category_2"]})',
                connector_label='Connection',
            )
        except ValueError as e:
            print(f'FAILED: {e}')
            continue
        valid, reason = validate_edge_multihop(
            qa, f'{p["category_1"]}|{p["category_2"]}',
            connector_words=['category', 'subject', 'both books', 'related', 'connection'])
        print(f'Q: {qa["question"]}')
        print(f'A: {qa["answer"]}')
        print(f'-> ({valid}, {reason!r})')
        if valid:
            n_pass += 1
        print()

    edge_pass = n_pass >= 3
    print(f'CATEGORY-BRIDGE GATE: {"PASS" if edge_pass else "FAIL"} ({n_pass}/5)')

    print('\n\n' + '=' * 20, 'SUMMARY', '=' * 20)
    print(f'single-hop gate: {"PASS" if single_pass else "FAIL"}')
    print(f'category-bridge gate: {"PASS" if edge_pass else "FAIL"}')

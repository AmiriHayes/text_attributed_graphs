#!/usr/bin/env python3
"""
generate_qa.py — Stage 3 QA generation, Steps 2-4: LLM caller, validator,
question-type classifier. Dataset-agnostic (takes plain text in, no
dataset-specific logic here).

Uses the OpenAI SDK (key from .env / OPENAI_API_KEY) — this is what was
already installed and working in this environment, per the pass-1 spec.
"""
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = 'gpt-4o-mini'
SLEEP_SECONDS = 0.5

SYSTEM_PROMPT = (
    "You generate factual QA pairs for a retrieval benchmark. Given a "
    "text passage, produce one specific factual question and its answer. "
    "Requirements: the question must be answerable ONLY from the provided "
    "text, not from general world knowledge. The answer must be 1-2 "
    "sentences maximum. "
    "The question must NOT be answerable from the title or first sentence "
    "alone. It should require understanding a specific detail, nuance, "
    "method, result, or limitation discussed in the body of the text. "
    "Avoid questions that begin with 'What does this paper propose' or "
    "'What is the main contribution of'. Prefer questions about specific "
    "findings, constraints, experimental conditions, comparisons, or "
    "failure modes. "
    "Your response must be valid JSON. If the text contains LaTeX notation, "
    "backslashes, or mathematical symbols, do not include them in your "
    "question or answer — paraphrase using plain English instead. Never "
    "use literal backslashes in your JSON output. "
    "Return valid JSON only, no other text: "
    '{"question": "...", "answer": "..."}'
)

AGGREGATE_SYSTEM_PROMPT = (
    "You are generating evaluation questions for a graph retrieval "
    "benchmark. Given multiple research paper abstracts from the same "
    "field, generate one question whose correct answer requires "
    "synthesizing information across at least three of the provided "
    "papers. The question should NOT be answerable from any single paper "
    "alone. Good question types: dominant methods across the field, "
    "common limitations mentioned by multiple authors, recurring "
    "experimental setups, contrasting approaches to the same problem. "
    "If the text contains LaTeX notation, backslashes, or mathematical "
    "symbols, do not include them in your question or answer — paraphrase "
    "using plain English instead. Never use literal backslashes in your "
    "JSON output. "
    'Return JSON only: {"question": "...", "answer": "..."}'
)

# Control characters that indicate JSON-escape corruption (a literal
# backslash from source LaTeX consumed as \n \f \t \r during json.loads,
# rather than being escaped by the model as \\n \\f \\t \\r).
_CONTROL_CHAR_RE = re.compile(r'[\x0c\x08\t\r]')

REFUSAL_PHRASES = ["I cannot", "As an AI", "I don't have", "I'm sorry"]


def _client() -> OpenAI:
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set (checked .env / environment)')
    return OpenAI(api_key=key)


def _call_llm_raw(system_prompt: str, user_content: str, client: OpenAI = None) -> str:
    """Low-level call, returns the raw (unparsed) response string. Exposed for
    debugging JSON-corruption cases where we need to see what the model
    actually returned before parsing."""
    client = client or _client()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ],
            temperature=0.3,
            response_format={'type': 'json_object'},
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise ValueError(f'API call failed: {type(e).__name__}: {e}')


def generate_qa(text: str, client: OpenAI = None) -> dict:
    """
    Calls the LLM on a single text passage. Returns {"question": str, "answer": str}.
    Raises ValueError if the API call fails or the response cannot be parsed.
    """
    raw = _call_llm_raw(SYSTEM_PROMPT, f'Text: {text}', client=client)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'Could not parse response as JSON: {e}\nraw={raw!r}')

    if 'question' not in parsed or 'answer' not in parsed:
        raise ValueError(f'Response JSON missing question/answer keys: {parsed!r}')

    time.sleep(SLEEP_SECONDS)
    return parsed


def generate_aggregate_qa(rows: list[dict], client: OpenAI = None, system_prompt: str = None,
                           item_label: str = 'Paper', text_fields: tuple = ('text_fidelity_b', 'text_fidelity_a')) -> dict:
    """
    Takes a list of 5-10 rows from the same grouping key and generates one
    question answerable only by synthesizing across >=3 of them. Returns
    {"question": str, "answer": str}. Raises ValueError on failure.
    text_fields controls which field(s) get concatenated per row (arxiv uses
    both title+abstract; amazon per its own guidance uses text_fidelity_b —
    product title — only, so pass text_fields=('text_fidelity_b',)).
    """
    system_prompt = system_prompt or AGGREGATE_SYSTEM_PROMPT
    listed = '\n'.join(
        f'{item_label} {i+1}: ' + ' — '.join(str(r.get(f, '') or '') for f in text_fields)
        for i, r in enumerate(rows)
    )
    raw = _call_llm_raw(system_prompt, listed, client=client)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'Could not parse response as JSON: {e}\nraw={raw!r}')

    if 'question' not in parsed or 'answer' not in parsed:
        raise ValueError(f'Response JSON missing question/answer keys: {parsed!r}')

    time.sleep(SLEEP_SECONDS)
    return parsed


def validate_qa(qa_dict: dict) -> tuple[bool, str]:
    """
    Checks, in order, first failure wins:
      - question/answer keys exist
      - question is a string with > 10 words
      - answer is a string with > 5 words
      - question ends with '?'
      - neither field contains a refusal phrase
      - neither field contains a stray control character (form feed,
        backspace, tab, carriage return) — a tell for JSON-escape corruption
        from unescaped LaTeX backslashes (\\n \\f \\t \\r consumed as JSON
        escapes during parsing rather than literal backslash+letter)
    """
    if 'question' not in qa_dict or 'answer' not in qa_dict:
        return False, 'missing question/answer key'

    q, a = qa_dict['question'], qa_dict['answer']

    if not isinstance(q, str) or len(q.split()) <= 10:
        return False, f'question too short ({len(q.split()) if isinstance(q, str) else "n/a"} words, need >10)'

    if not isinstance(a, str) or len(a.split()) <= 5:
        return False, f'answer too short ({len(a.split()) if isinstance(a, str) else "n/a"} words, need >5)'

    if not q.strip().endswith('?'):
        return False, 'question does not end with ?'

    for phrase in REFUSAL_PHRASES:
        if phrase.lower() in q.lower() or phrase.lower() in a.lower():
            return False, f'refusal phrase detected: "{phrase}"'

    if _CONTROL_CHAR_RE.search(q) or _CONTROL_CHAR_RE.search(a):
        return False, 'control_chars'

    return True, 'valid'


def sample_category_groups(df, n_groups: int, group_size_range=(5, 10), seed=42) -> list:
    """
    Groups rows by aggregate_id (== categorical_label for arxiv), keeps
    groups with >= group_size_range[0] members, and samples n_groups of them
    (group_size_range[1] rows each) for aggregate question generation.
    Returns a list of lists-of-row-dicts.
    """
    import random
    lo, hi = group_size_range
    groups = df.groupby('aggregate_id')
    candidates = [g for _, g in groups if len(g) >= lo]

    rng = random.Random(seed)
    rng.shuffle(candidates)

    out = []
    for g in candidates[:n_groups]:
        sample = g.sample(n=min(hi, len(g)), random_state=seed)
        out.append(sample.to_dict('records'))
    return out


EDGE_MULTIHOP_SYSTEM_PROMPT = (
    "You generate evaluation questions for a graph retrieval benchmark. "
    "You are given two research papers connected by a shared author. "
    "Generate one question whose complete answer requires information "
    "from BOTH papers — not answerable from either paper alone. "
    "The connection between the papers (the shared author) must be "
    "relevant to the question. Good question types: how an author's "
    "methodology evolved between two works, what different problems "
    "an author applied the same technique to, how findings in one "
    "paper relate to or contradict findings in the other. "
    "The question must explicitly reference that both papers are "
    "by the same researcher, and the author connection must be "
    "necessary to answer it — a question answerable purely through "
    "topical similarity without knowing the papers share an author "
    "is not acceptable. "
    "Do not generate questions answerable from general knowledge. "
    "If the text contains LaTeX notation, backslashes, or mathematical "
    "symbols, do not include them in your question or answer — paraphrase "
    "using plain English instead. Never use literal backslashes in your "
    "JSON output. "
    'Return JSON only: {"question": "...", "answer": "..."}'
)

AMAZON_AGGREGATE_SYSTEM_PROMPT = (
    "You are generating evaluation questions for a graph retrieval "
    "benchmark. Given multiple Amazon product titles (grouped by "
    "product family), generate one question whose correct answer "
    "requires synthesizing information across at least three of the "
    "provided products. The question should NOT be answerable from any "
    "single product alone. Good question types: common features or "
    "materials shared across the products, contrasting designs or use "
    "cases for a similar purpose, which product best fits a stated need "
    "given the group as a whole. "
    'Return JSON only: {"question": "...", "answer": "..."}'
)

AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT = (
    "You generate evaluation questions for a graph retrieval "
    "benchmark. You are given two product reviews connected by a "
    "shared reviewer. Generate one question whose complete answer "
    "requires information from BOTH reviews — not answerable from "
    "either review alone. The connection between the reviews (the "
    "shared reviewer) must be relevant to the question. Good question "
    "types: how the reviewer's opinion or experience differed between "
    "the two products, what different products the reviewer applied "
    "the same complaint or praise to, how the reviewer's use case for "
    "one product relates to or contrasts with their use case for the "
    "other. "
    "The question must explicitly reference that both products were "
    "reviewed by the same person, and the reviewer connection must be "
    "necessary to answer it — a question answerable purely through "
    "topical similarity without knowing the reviews share a reviewer "
    "is not acceptable. "
    "Do not generate questions answerable from general knowledge. "
    'Return JSON only: {"question": "...", "answer": "..."}'
)

HISTORY_CATEGORY_BRIDGE_SYSTEM_PROMPT = (
    "You generate evaluation questions for a book retrieval benchmark. "
    "You are given two history books from related subject categories. "
    "Generate one question whose complete answer requires information "
    "from BOTH books — not answerable from either book alone. The "
    "subject category connection between the books must be relevant "
    "to the question. Good question types: how events described in "
    "one book influenced events in the other, what contrasting "
    "perspectives two books offer on the same historical period, how "
    "a theme present in both books manifests differently across "
    "regions or time periods.\n"
    "Do not generate questions answerable from general historical "
    "knowledge alone.\n"
    'Return JSON only: {"question": "...", "answer": "..."}'
)


HISTORY_STRUCTURAL_EDGE_SYSTEM_PROMPT = (
    "You generate evaluation questions for a book retrieval benchmark. "
    "You are given two history books that are documented related works. "
    "Generate one question whose complete answer requires information "
    "from BOTH books — not answerable from either book alone. The "
    "documented relationship between the books must be relevant to "
    "the question. Good question types: what shared themes the two "
    "books explore, how the books offer contrasting perspectives on "
    "a related subject, how the books provide complementary coverage "
    "of the same historical period or region.\n"
    "Do not claim causal or temporal influence between the books. The "
    "connection is that they are documented related works — ask about "
    "shared themes, contrasting perspectives, or complementary coverage "
    "of the same subject.\n"
    "Do not generate questions answerable from general historical "
    "knowledge alone.\n"
    'Return JSON only: {"question": "...", "answer": "..."}'
)


def sample_structural_edge_pairs(df, n_pairs: int, max_attempts=None, seed=42) -> list:
    """
    History's edge_multihop mechanism using structural_edges (E10d,
    pre-given curated related-book links) instead of the random
    category-bridge fallback — replaces invented connections with real,
    documented ones. For each random anchor row, checks structural_edges
    for a target ID also present in the pool. Returns list of dicts:
      {'anchor': row_dict, 'target': row_dict}
    """
    import random
    max_attempts = max_attempts or (n_pairs * 10)
    df_reset = df.reset_index(drop=True)
    pool_ids = set(df_reset['primary_id'].tolist())
    rng = random.Random(seed)
    n = len(df_reset)

    out = []
    attempts = 0
    while len(out) < n_pairs and attempts < max_attempts:
        attempts += 1
        i = rng.randrange(n)
        anchor = df_reset.iloc[i]
        se = anchor['structural_edges']
        if not isinstance(se, list) or not se:
            continue
        candidates = [t for t in se if t in pool_ids and t != anchor['primary_id']]
        if not candidates:
            continue
        target_id = rng.choice(candidates)
        target = df_reset[df_reset['primary_id'] == target_id].iloc[0]
        out.append({'anchor': anchor.to_dict(), 'target': target.to_dict()})
    return out


def sample_category_bridge_pairs(df, n_pairs: int, max_attempts=None, seed=42) -> list:
    """
    History's edge_multihop fallback (secondary_id fully absent): no shared-
    entity hop exists, so pairs are two random books from DIFFERENT
    aggregate_id (subject category) values instead — the LLM is trusted to
    construct the actual thematic bridge, same as the aggregate mechanism's
    cross-category sampling already does. Returns list of dicts:
      {'anchor': row_dict, 'target': row_dict, 'category_1': str, 'category_2': str}
    """
    import random
    max_attempts = max_attempts or (n_pairs * 10)
    df_reset = df.reset_index(drop=True)
    rng = random.Random(seed)
    n = len(df_reset)

    out = []
    attempts = 0
    while len(out) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        anchor, target = df_reset.iloc[i], df_reset.iloc[j]
        if anchor['aggregate_id'] == target['aggregate_id']:
            continue  # require an actual category boundary crossing
        out.append({
            'anchor': anchor.to_dict(), 'target': target.to_dict(),
            'category_1': anchor['aggregate_id'], 'category_2': target['aggregate_id'],
        })
    return out


def build_author_hop_map(df) -> dict:
    """Explode secondary_id into an entity-token -> [row positional index, ...]
    map, so 'shares an author/reviewer' is computed per-individual-entity, not
    by exact value equality. Handles both list-valued secondary_id (arxiv —
    multiple authors per row) and scalar secondary_id (amazon — one reviewer
    per row)."""
    hop_map = {}
    for i, sid in enumerate(df['secondary_id']):
        if isinstance(sid, list):
            tokens = sid
        elif sid is None or (isinstance(sid, float) and pd.isna(sid)):
            continue
        else:
            tokens = [sid]
        for token in tokens:
            hop_map.setdefault(token, []).append(i)
    return hop_map


def sample_edge_multihop_pairs(df, n_pairs: int, max_attempts=None, seed=42) -> list:
    """
    Samples up to n_pairs (anchor, target, shared_author) triples via genuine
    2-hop traversal: anchor -> shared secondary_id (author) -> target, where
    target's aggregate_id (top-level category) differs from anchor's (so the
    hop crosses a meaningful boundary, not just another paper in the same
    field by the same author). Anchors with <2 other rows sharing an author
    are skipped and resampled. Returns list of dicts:
      {'anchor': row_dict, 'target': row_dict, 'shared_author': str}
    """
    import random
    max_attempts = max_attempts or (n_pairs * 10)
    hop_map = build_author_hop_map(df)
    df_reset = df.reset_index(drop=True)
    rng = random.Random(seed)
    n = len(df_reset)

    out = []
    attempts = 0
    while len(out) < n_pairs and attempts < max_attempts:
        attempts += 1
        anchor_i = rng.randrange(n)
        anchor = df_reset.iloc[anchor_i]
        raw_sid = anchor['secondary_id']
        if isinstance(raw_sid, list):
            authors = raw_sid
        elif raw_sid is None or (isinstance(raw_sid, float) and pd.isna(raw_sid)):
            authors = []
        else:
            authors = [raw_sid]  # scalar secondary_id (e.g. amazon reviewer)
        if not authors:
            continue

        candidate_targets = []
        chosen_author = None
        for author in authors:
            connected = [j for j in hop_map.get(author, []) if j != anchor_i]
            if len(connected) >= 1:
                candidate_targets.extend((j, author) for j in connected)
        if len(candidate_targets) < 1:
            continue  # fewer than 2 rows total share an author with anchor -> nowhere to hop

        # prefer targets whose aggregate_id differs from anchor's
        cross_boundary = [(j, a) for j, a in candidate_targets
                           if df_reset.iloc[j]['aggregate_id'] != anchor['aggregate_id']]
        pool = cross_boundary if cross_boundary else candidate_targets
        target_i, chosen_author = rng.choice(pool)
        target = df_reset.iloc[target_i]

        out.append({
            'anchor': anchor.to_dict(),
            'target': target.to_dict(),
            'shared_author': chosen_author,
        })
    return out


def generate_edge_multihop_qa(anchor_text: str, target_text: str, shared_author: str, client: OpenAI = None,
                               system_prompt: str = None, anchor_label: str = 'Paper 1 (anchor)',
                               target_label: str = 'Paper 2 (connected via shared author)',
                               connector_label: str = 'Shared author identifier') -> dict:
    """
    Calls the LLM on an anchor/target item pair connected by a shared
    author/reviewer. Returns {"question": str, "answer": str}. Raises
    ValueError on failure. system_prompt/labels are overridable so the same
    function serves arxiv (papers/authors) and amazon (products/reviewers).
    """
    system_prompt = system_prompt or EDGE_MULTIHOP_SYSTEM_PROMPT
    user_content = (
        f'{anchor_label}: {anchor_text}\n'
        f'{target_label}: {target_text}\n'
        f'{connector_label}: {shared_author}'
    )
    raw = _call_llm_raw(system_prompt, user_content, client=client)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'Could not parse response as JSON: {e}\nraw={raw!r}')

    if 'question' not in parsed or 'answer' not in parsed:
        raise ValueError(f'Response JSON missing question/answer keys: {parsed!r}')

    time.sleep(SLEEP_SECONDS)
    return parsed


def validate_edge_multihop(qa_dict: dict, shared_author: str, connector_words: list = None) -> tuple[bool, str]:
    """
    Reuses validate_qa() (same 10-word floor etc.), then adds an
    edge_multihop-specific check: reject if none of connector_words nor any
    token from the shared_author identifier appears in the question — a
    question that ignores the connection isn't testing the hop.
    connector_words defaults to ['author'] (arxiv); pass ['reviewer'] for
    amazon, etc.
    """
    valid, reason = validate_qa(qa_dict)
    if not valid:
        return valid, reason

    connector_words = connector_words or ['author']
    q_lower = qa_dict['question'].lower()
    author_tokens = re.split(r'[\s|.]+', shared_author.lower())
    author_tokens = [t for t in author_tokens if len(t) > 1]  # drop bare initials
    mentions_connector_word = any(w in q_lower for w in connector_words)
    mentions_name_token = any(tok in q_lower for tok in author_tokens)

    if not (mentions_connector_word or mentions_name_token):
        return False, 'no_connection_referenced'

    return True, 'valid'


def validate_edge_multihop_titles(qa_dict: dict, anchor_title: str, target_title: str) -> tuple[bool, str]:
    """
    History structural_edges variant of validate_edge_multihop. There's no
    shared-entity identifier to check tokens against here (the connector is
    "documented related work", not a name) -- instead checks for a bare
    "both" plus at least one of the two book titles appearing in the
    question. Calibrated after gate_history_structural.py's mechanical
    check (which required the literal substring "both books") flagged 3/5
    real passes as failures: natural LLM phrasing is "both 'Title A' and
    'Title B'", not "both books". Direct inspection of all 5 gate pairs
    confirmed the content itself was clean (real curated relationships,
    no fabricated connections) -- this is a validator calibration fix, not
    a content-quality fix.
    """
    valid, reason = validate_qa(qa_dict)
    if not valid:
        return valid, reason

    q_lower = qa_dict['question'].lower()
    has_both = re.search(r'\bboth\b', q_lower) is not None
    title_hit = (bool(anchor_title) and anchor_title.lower() in q_lower) or \
                (bool(target_title) and target_title.lower() in q_lower)

    if not (has_both and title_hit):
        return False, 'no_connection_referenced'

    return True, 'valid'


HISTORY_AGGREGATE_SYSTEM_PROMPT = (
    "You are generating evaluation questions for a graph retrieval "
    "benchmark. Given multiple history book descriptions (spanning "
    "different subject categories), generate one question whose correct "
    "answer requires synthesizing information across at least three of "
    "the provided books. The question should NOT be answerable from any "
    "single book alone. Good question types: shared historical themes "
    "across the books, contrasting perspectives on related events or "
    "periods, recurring subject matter treated differently across "
    "regions or eras.\n"
    "Do not generate questions answerable from general historical "
    "knowledge alone.\n"
    'Return JSON only: {"question": "...", "answer": "..."}'
)


def sample_cross_category_groups(df, n_groups: int, group_size_range=(5, 10),
                                   min_categories=3, max_resamples=20, seed=42) -> list:
    """
    Samples n_groups groups of 5-10 rows EACH SPANNING >= min_categories
    distinct categorical_label values (the inverse of sample_category_groups,
    which deliberately groups by a single shared category). Same-category
    groups let retrieval get lucky (2/5 in the gate scored context_relevance
    =1.0); cross-category groups are harder to cover with top-3 retrieval.

    Returns a list of lists-of-row-dicts. Each attempt resamples up to
    max_resamples times if the 3-category minimum isn't met; groups that
    never meet it within the budget are skipped (not padded/faked).
    """
    lo, hi = group_size_range
    rng_seed = seed
    out = []
    for i in range(n_groups):
        found = None
        for attempt in range(max_resamples):
            size = lo + (attempt % (hi - lo + 1))
            sample = df.sample(n=min(size, len(df)), random_state=rng_seed)
            rng_seed += 1
            if sample['aggregate_id'].nunique() >= min_categories:
                found = sample
                break
        if found is not None:
            out.append(found.to_dict('records'))
    return out


def classify_question_type(question: str) -> str:
    """Pure heuristic, no API call. relational / categorical / descriptive."""
    q = question.lower()

    relational_words = ['who', 'whose', 'between', 'related', 'connected',
                         'collaboration', 'co-author', 'coauthor']
    categorical_words = ['what type', 'which category', 'what kind',
                          'what field', 'what domain']

    if any(w in q for w in relational_words):
        return 'relational'
    if any(w in q for w in categorical_words):
        return 'categorical'
    return 'descriptive'


if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from format_rows import format_arxiv_row

    data_path = Path(__file__).resolve().parent.parent.parent / 'data/arxiv/train/samples/sample_00.jsonl'
    rows = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            text = format_arxiv_row(row)
            if text:
                rows.append((row, text))
            if len(rows) >= 5:
                break

    client = _client()
    step2_outputs = []
    print('=' * 20, 'STEP 2 — LLM CALLER', '=' * 20)
    for i, (row, text) in enumerate(rows):
        print(f'\n--- row {i} (primary_id={row["primary_id"]}) ---')
        print(f'text preview: {text[:150]}...')
        try:
            qa = generate_qa(text, client=client)
            print(f'parsed output: {json.dumps(qa, indent=2)}')
            step2_outputs.append(qa)
        except ValueError as e:
            print(f'FAILED: {e}')
            step2_outputs.append(None)

    n_ok = sum(1 for x in step2_outputs if x is not None)
    print(f'\nStep 2: {n_ok}/5 returned valid JSON')

    print('\n' + '=' * 20, 'STEP 3 — VALIDATOR', '=' * 20)
    step3_results = []
    for i, qa in enumerate(step2_outputs):
        if qa is None:
            print(f'row {i}: SKIPPED (step 2 failed)')
            step3_results.append(None)
            continue
        valid, reason = validate_qa(qa)
        print(f'row {i}: ({valid}, {reason!r})  Q: {qa["question"]!r}')
        step3_results.append((valid, reason))

    print('\n' + '=' * 20, 'STEP 4 — QUESTION TYPE CLASSIFIER', '=' * 20)
    for i, qa in enumerate(step2_outputs):
        if qa is None or not step3_results[i] or not step3_results[i][0]:
            continue
        qtype = classify_question_type(qa['question'])
        print(f'[{qtype:<12}] {qa["question"]}')

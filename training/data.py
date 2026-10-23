"""Training-pair data handling: pure Python, no torch, fully testable offline.

A pair is one JSON object per line:
    {"query": "...", "source": "dsp_notes.pdf", "subject": "dsp", "page": 12, "origin": "synthetic"}

Four rules protect the project's data contract and the eval:
  1. Only documents the manifest marks shareable=yes are trained on (scope.md data rules).
  2. No training query may equal or nearly equal a frozen eval question (leak guard).
  3. By default the eval questions' gold pages are not trained on at all: a model that has seen
     those pages would look better on the eval than it will on new ones.
  4. Validation pages are held out of training entirely (split by page, not by query).
"""
import json
import random
import re
from collections import Counter, defaultdict

REQUIRED = ('query', 'source', 'subject', 'page')
_TOKEN = re.compile(r'[a-z0-9]+')


def tokens(text):
    return _TOKEN.findall(text.lower())


def norm(text):
    return ' '.join(tokens(text))


def page_key(pair):
    return (pair['source'].lower(), int(pair['page']))


def load_pairs(path):
    """Read and validate pairs. Raises ValueError naming the first bad line."""
    pairs = []
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                p = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'{path}:{n}: not valid JSON ({exc})') from None
            missing = [k for k in REQUIRED if p.get(k) in (None, '')]
            if missing:
                raise ValueError(f'{path}:{n}: missing {missing}')
            if not isinstance(p['query'], str) or len(tokens(p['query'])) < 3:
                raise ValueError(f'{path}:{n}: query must be a string of at least 3 words')
            if not isinstance(p['page'], int) or p['page'] < 1:
                raise ValueError(f'{path}:{n}: page must be an integer >= 1')
            pairs.append(p)
    return pairs


def write_pairs(path, pairs):
    with open(path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')


def load_eval_questions(path):
    """Full question dicts of the frozen eval set (same file eval.run_eval reads)."""
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe(pairs):
    """Drop repeated (query, page) pairs, keeping the first."""
    seen, out = set(), []
    for p in pairs:
        key = (norm(p['query']), page_key(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def eval_guard(pairs, eval_questions, threshold=0.8, exclude_eval_pages=True):
    """Return (kept, dropped) where dropped is a list of (pair, reason).

    Drops pairs whose query equals or nearly equals (token Jaccard >= threshold) an eval question,
    and, when exclude_eval_pages, every pair that targets a gold page of an eval question.
    """
    eval_tokens = [tokens(q['question']) for q in eval_questions]
    gold_pages = {(g['source'].lower(), int(g['page']))
                  for q in eval_questions for g in q.get('gold', [])} if exclude_eval_pages else set()
    kept, dropped = [], []
    for p in pairs:
        qt = tokens(p['query'])
        if any(jaccard(qt, et) >= threshold for et in eval_tokens):
            dropped.append((p, 'eval_question_overlap'))
        elif page_key(p) in gold_pages:
            dropped.append((p, 'eval_gold_page'))
        else:
            kept.append(p)
    return kept, dropped


def shareable_guard(pairs, manifest, allow_non_shareable=False):
    """Keep pairs whose (subject, file) is in the manifest and shareable=yes.

    `manifest` is a list of dicts with file/subject/shareable (e.g. from pandas .to_dict('records')).
    Pairs for unknown documents are an error: a typo must not silently shrink the training set.
    """
    status = {(m['subject'], m['file']): m['shareable'] for m in manifest}
    unknown = sorted({(p['subject'], p['source']) for p in pairs} - set(status))
    if unknown:
        raise ValueError(f'pairs reference documents not in the manifest: {unknown[:5]}')
    kept, dropped = [], []
    for p in pairs:
        if allow_non_shareable or status[(p['subject'], p['source'])] == 'yes':
            kept.append(p)
        else:
            dropped.append((p, 'not_shareable'))
    return kept, dropped


def split_pairs(pairs, val_fraction=0.1, seed=0):
    """Split by page so no validation page is ever seen in training. Returns (train, val)."""
    keys = sorted({page_key(p) for p in pairs})
    if val_fraction <= 0 or len(keys) < 2:
        return list(pairs), []
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:n_val])
    return ([p for p in pairs if page_key(p) not in val_keys],
            [p for p in pairs if page_key(p) in val_keys])


def unique_page_batches(pairs, batch_size, seed=0, epoch=0, min_batch=2):
    """Index batches for one epoch in which no page appears twice.

    In-batch contrastive loss treats every other page in the batch as a wrong answer, so two
    queries about the same page in one batch would punish the model for being right. Round r takes
    the r-th query of every page, so each round holds distinct pages; rounds are chunked into
    batches. A trailing chunk smaller than min_batch (no negatives to learn from) is dropped.
    """
    if batch_size < 2:
        raise ValueError('batch_size must be at least 2 for in-batch negatives')
    rng = random.Random(f'{seed}/{epoch}')
    by_page = defaultdict(list)
    for i, p in enumerate(pairs):
        by_page[page_key(p)].append(i)
    pages = sorted(by_page)
    rng.shuffle(pages)
    for idxs in by_page.values():
        rng.shuffle(idxs)
    batches, r = [], 0
    while True:
        rnd = [by_page[pg][r] for pg in pages if len(by_page[pg]) > r]
        if not rnd:
            break
        rng.shuffle(rnd)
        for j in range(0, len(rnd), batch_size):
            chunk = rnd[j:j + batch_size]
            if len(chunk) >= min_batch:
                batches.append(chunk)
        r += 1
    rng.shuffle(batches)
    return batches


def describe(pairs):
    pages = {page_key(p) for p in pairs}
    by_subject = Counter(p['subject'] for p in pairs)
    return (f'{len(pairs)} pairs over {len(pages)} pages; '
            + ', '.join(f'{s}={n}' for s, n in sorted(by_subject.items())))

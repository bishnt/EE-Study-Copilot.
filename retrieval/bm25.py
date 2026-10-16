import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.types import Hit

STOP = frozenset(
    'a an and are as at be by for from has have in is it its of on or that the this to was were '
    'what which who how why when where with does do did can could should would about into than then '
    'explain define derive state find calculate give write describe'.split())
_TOKEN = re.compile(r'[a-z0-9]+')


def _stem(tok):
    for suf in ('ing', 'ed', 'es', 's'):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[:-len(suf)]
    return tok


def tokenize(text):
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP]


class BM25Index:
    """Keyword search over the chunk file written by ingestion.build_index."""

    def __init__(self, chunks):
        if not chunks:
            raise ValueError('BM25Index needs at least one chunk')
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c['text']) for c in chunks])

    @classmethod
    def from_jsonl(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls([json.loads(line) for line in f if line.strip()])

    def search(self, query, k=20, subject=None):
        toks = tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        hits = []
        for i in np.argsort(-scores):
            if scores[i] <= 0 or len(hits) >= k:
                break
            c = self.chunks[i]
            if subject and c['meta'].get('subject') != subject:
                continue
            hits.append(Hit(id=c['id'], text=c['text'], meta=c['meta'],
                            bm25_score=float(scores[i])))
        return hits

"""Page-image retrieval with ColPali-style multi-vector (late interaction) embeddings.

Each PDF page is one passage made of many patch vectors. A query is a handful of token
vectors. The page score is MaxSim: for every query vector take its best match among the page
vectors, then average over query vectors. The mean keeps the score roughly comparable across
queries of different length, which is what the evidence gate (`Config.min_visual`) needs.

On-disk layout written by `write_index` (all in one directory):
    pages.jsonl   one row per page: id, source, subject, category, page, shareable, text
    emb.npy       float16 (total_vectors, dim), every page's vectors stacked
    offsets.npy   int64 (n_pages + 1,), page i is emb[offsets[i]:offsets[i + 1]]
    meta.json     {"model": ..., "dim": ..., "pages": ...}  (queries must use the same model)
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from retrieval.types import Hit

MAX_VECTORS_PER_CHUNK = 60_000   # bounds the (query_vectors x chunk) similarity matrix
MAX_TEXT_CHARS = 3000            # page text kept for the prompt


def page_id(source, page):
    return 'pg-' + hashlib.sha1(f'{source}|{page}|image'.encode()).hexdigest()[:16]


def write_index(path, pages, vectors, model):
    """Persist pages (list of dict) and their vectors (list of (n_i, dim) arrays)."""
    if len(pages) != len(vectors):
        raise ValueError('pages and vectors must have the same length')
    if not pages:
        raise ValueError('refusing to write an empty visual index')
    dims = {v.shape[1] for v in vectors}
    if len(dims) != 1:
        raise ValueError(f'inconsistent embedding dimensions: {sorted(dims)}')
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(v) for v in vectors], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.save(out / 'emb.npy', np.concatenate(vectors).astype(np.float16))
    np.save(out / 'offsets.npy', offsets)
    with open(out / 'pages.jsonl', 'w', encoding='utf-8') as f:
        for p in pages:
            f.write(json.dumps(p) + '\n')
    (out / 'meta.json').write_text(
        json.dumps({'model': model, 'dim': int(dims.pop()), 'pages': len(pages)}))


def read_index(path):
    """Return (pages, emb, offsets, meta). `emb` is memory-mapped, not loaded into RAM."""
    src = Path(path)
    if not (src / 'meta.json').is_file():
        raise FileNotFoundError(
            f'No visual index in {src}. Build it: python -m ingestion.build_visual --class')
    meta = json.loads((src / 'meta.json').read_text())
    with open(src / 'pages.jsonl', encoding='utf-8') as f:
        pages = [json.loads(line) for line in f if line.strip()]
    emb = np.load(src / 'emb.npy', mmap_mode='r')
    offsets = np.load(src / 'offsets.npy')
    if len(offsets) != len(pages) + 1:
        raise ValueError(f'Corrupt visual index in {src}: {len(pages)} pages, '
                         f'{len(offsets) - 1} offset ranges. Rebuild it.')
    return pages, emb, offsets, meta


class VisualIndex:
    """MaxSim search over page embeddings. `embed_query` maps str -> (n, dim) array."""

    def __init__(self, pages, emb, offsets, embed_query, model=None):
        self.pages, self._emb, self._off, self._embed = pages, emb, offsets, embed_query
        self.model = model
        self._dim = emb.shape[1]
        lengths = np.diff(offsets)
        self._usable = np.flatnonzero(lengths > 0)   # a page with no vectors can never match
        if len(self._usable) == 0:
            raise ValueError('VisualIndex needs at least one page with vectors')

    @classmethod
    def from_dir(cls, path, embed_query, expected_model=None):
        pages, emb, offsets, meta = read_index(path)
        if expected_model and meta.get('model') != expected_model:
            raise ValueError(
                f"Visual index was built with {meta.get('model')!r} but COPILOT_VISUAL_MODEL is "
                f'{expected_model!r}. Embeddings from different models are not comparable; '
                'rebuild the index or fix the setting.')
        return cls(pages, emb, offsets, embed_query, meta.get('model'))

    @classmethod
    def from_config(cls, cfg):
        from retrieval import colpali_model  # lazy: pulls in torch
        model, processor = colpali_model.load(cfg.visual_model)
        return cls.from_dir(cfg.visual_dir,
                            lambda q: colpali_model.embed_query(model, processor, q),
                            expected_model=cfg.visual_model)

    def _maxsim(self, q, idx):
        """Mean-MaxSim of query vectors `q` (n, dim) against pages `idx`; returns (len(idx),)."""
        scores = np.empty(len(idx), dtype=np.float32)
        lengths = self._off[idx + 1] - self._off[idx]
        start = 0
        while start < len(idx):
            end, total = start, 0
            while end < len(idx) and (end == start or total + lengths[end] <= MAX_VECTORS_PER_CHUNK):
                total += lengths[end]
                end += 1
            block = np.concatenate(
                [self._emb[self._off[i]:self._off[i + 1]] for i in idx[start:end]]
            ).astype(np.float32)
            sim = q @ block.T                                       # (n_query, total)
            starts = np.concatenate([[0], np.cumsum(lengths[start:end])[:-1]])
            best = np.maximum.reduceat(sim, starts, axis=1)         # (n_query, pages in chunk)
            scores[start:end] = best.mean(axis=0)
            start = end
        return scores

    def search(self, query, k=10, subject=None):
        q = np.asarray(self._embed(query), dtype=np.float32)
        if q.ndim != 2 or q.shape[0] == 0:
            return []
        if q.shape[1] != self._dim:
            raise ValueError(f'Query embedding has dim {q.shape[1]} but the index has {self._dim}; '
                             'the index was built with a different model.')
        idx = self._usable
        if subject:
            idx = np.array([i for i in idx if self.pages[i].get('subject') == subject],
                           dtype=np.int64)
            if len(idx) == 0:
                return []
        scores = self._maxsim(q, idx)
        hits = []
        for j in np.argsort(-scores)[:k]:
            p = self.pages[idx[j]]
            hits.append(Hit(
                id=p['id'], text=p.get('text', '')[:MAX_TEXT_CHARS],
                meta={'source': p['source'], 'subject': p['subject'],
                      'category': p.get('category', ''), 'page': p['page'],
                      'kind': 'page_image', 'chapter': 0, 'shareable': p['shareable']},
                visual_score=float(scores[j])))
        return hits

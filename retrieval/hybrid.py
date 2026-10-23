from retrieval.types import Hit


class StaleIndexError(RuntimeError):
    pass


def rrf_fuse(ranked_lists, k=60):
    """Reciprocal rank fusion. Merges per-id score fields from every list."""
    merged = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            m = merged.setdefault(h.id, Hit(id=h.id, text=h.text, meta=h.meta))
            m.score += 1.0 / (k + rank)
            if h.bm25_score is not None:
                m.bm25_score = h.bm25_score
            if h.dense_sim is not None:
                m.dense_sim = h.dense_sim
            if h.visual_score is not None:
                m.visual_score = h.visual_score
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)


def fold_visual_into_text(visual_hits, text_lists):
    """Re-key page-image hits that share a (source, page) with a text chunk to that chunk's id.

    RRF then adds the page score to the chunk instead of listing the same page twice, and the
    chunk carries `visual_score`. Pages with no text chunk (pure diagrams, handwriting) keep
    their own id and enter the fusion as page-level hits.
    """
    twin = {}
    for hits in text_lists:
        for h in hits:
            twin.setdefault((h.source, h.page), h.id)
    return [Hit(id=twin.get((v.source, v.page), v.id), text=v.text, meta=v.meta,
                visual_score=v.visual_score) for v in visual_hits]


class HybridRetriever:
    """BM25 + dense (+ optional ColPali page retrieval) fused by reciprocal rank fusion.

    `visual` is any object with search(query, k, subject) returning Hits with `visual_score`.
    `page_renderer(hit) -> {'label', 'media_type', 'data'} | None` lets the pipeline attach the
    page image to the model call. Note the optional cross-encoder reranker reads text only, so
    it can down-rank image-only pages; leave it off when visual retrieval is on.
    """

    def __init__(self, bm25, dense, pool=20, rrf_k=60, reranker=None, require_shareable=True,
                 visual=None, visual_pool=10, page_renderer=None):
        self.bm25, self.dense = bm25, dense
        self.pool, self.rrf_k = pool, rrf_k
        self.reranker = reranker
        self.require_shareable = require_shareable
        self.visual, self.visual_pool = visual, visual_pool
        self.page_renderer = page_renderer

    def search(self, query, k=5, subject=None):
        lists = [self.bm25.search(query, self.pool, subject),
                 self.dense.search(query, self.pool, subject)]
        if self.visual:
            lists.append(fold_visual_into_text(
                self.visual.search(query, self.visual_pool, subject), lists))
        fused = rrf_fuse(lists, self.rrf_k)
        if self.require_shareable:
            if fused and all('shareable' not in h.meta for h in fused):
                raise StaleIndexError(
                    'Index has no shareable flag; rebuild it: python -m ingestion.build_class')
            fused = [h for h in fused if h.meta.get('shareable') == 'yes']
        if self.reranker:
            return self.reranker.rerank(query, fused[:self.pool], k)
        return fused[:k]

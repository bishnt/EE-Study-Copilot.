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
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)


class HybridRetriever:
    def __init__(self, bm25, dense, pool=20, rrf_k=60, reranker=None, require_shareable=True):
        self.bm25, self.dense = bm25, dense
        self.pool, self.rrf_k = pool, rrf_k
        self.reranker = reranker
        self.require_shareable = require_shareable

    def search(self, query, k=5, subject=None):
        fused = rrf_fuse([self.bm25.search(query, self.pool, subject),
                          self.dense.search(query, self.pool, subject)], self.rrf_k)
        if self.require_shareable:
            if fused and all('shareable' not in h.meta for h in fused):
                raise StaleIndexError(
                    'Index has no shareable flag; rebuild it: python -m ingestion.build_class')
            fused = [h for h in fused if h.meta.get('shareable') == 'yes']
        if self.reranker:
            return self.reranker.rerank(query, fused[:self.pool], k)
        return fused[:k]

class CrossEncoderReranker:
    """Optional second stage. Off by default; enable once eval shows P@5 gains."""

    def __init__(self, model_name='BAAI/bge-reranker-base', scorer=None):
        if scorer is None:
            from sentence_transformers import CrossEncoder  # lazy
            model = CrossEncoder(model_name)
            scorer = lambda q, texts: model.predict([(q, t) for t in texts]).tolist()
        self._scorer = scorer

    def rerank(self, query, hits, k):
        if not hits:
            return []
        scores = self._scorer(query, [h.text for h in hits])
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

from dataclasses import dataclass, field


@dataclass
class Hit:
    id: str
    text: str
    meta: dict = field(default_factory=dict)
    bm25_score: float | None = None
    dense_sim: float | None = None  # cosine similarity in [-1, 1]
    score: float = 0.0              # final fused / reranked score

    @property
    def source(self):
        return self.meta.get('source', '')

    @property
    def page(self):
        return int(self.meta.get('page', 0))

    @property
    def subject(self):
        return self.meta.get('subject', '')

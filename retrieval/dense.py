from app.config import MODEL_NAME, QUERY_PREFIX
from retrieval.types import Hit


class DenseIndex:
    """Embedding search over the Chroma collection. `embed` maps str -> list[float]."""

    def __init__(self, collection, embed, query_prefix=QUERY_PREFIX):
        self._col = collection
        self._embed = embed
        self._prefix = query_prefix

    @classmethod
    def from_config(cls, cfg):
        import chromadb  # lazy: keep tests and BM25-only use light
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        col = chromadb.PersistentClient(path=cfg.db_path).get_collection(cfg.collection)
        return cls(col, lambda q: model.encode([q], normalize_embeddings=True)[0].tolist())

    def search(self, query, k=20, subject=None):
        res = self._col.query(
            query_embeddings=[self._embed(self._prefix + query)],
            n_results=k,
            where={'subject': subject} if subject else None,
            include=['documents', 'metadatas', 'distances'])
        return [Hit(id=i, text=d, meta=m, dense_sim=1.0 - float(dist))  # cosine space
                for i, d, m, dist in zip(res['ids'][0], res['documents'][0],
                                         res['metadatas'][0], res['distances'][0])]

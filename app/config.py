"""Single source of truth for constants and tunable thresholds (see scope.md)."""
import os
from dataclasses import dataclass

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
REFUSAL = 'I could not find this in the documents.'  # exact sentence, scope.md
SUBJECTS = ('dsp', 'dcs', 'swp', 'emd', 'tht', 'pe')


def _env(name, default, cast=str):
    raw = os.getenv(name)
    return default if raw is None or raw == '' else cast(raw)


@dataclass(frozen=True)
class Config:
    db_path: str = 'vectordb_class'
    chunks_path: str = 'data/chunks_class.jsonl'
    syllabus_path: str = 'data/syllabus.json'
    collection: str = 'ee_corpus'
    top_k: int = 5              # final context size, also k for R3 (P@5)
    pool_size: int = 20         # candidates taken from each retriever before fusion
    rrf_k: int = 60
    # Evidence gate. Starting values only: calibrate with `python -m eval.run_eval`.
    min_dense_sim: float = 0.50
    min_bm25: float = 4.0
    max_context_chars: int = 9000
    llm_model: str = 'claude-sonnet-5-5'
    require_shareable: bool = True  # never send shareable=no text to the API

    @classmethod
    def from_env(cls):
        d = cls()
        return cls(
            db_path=_env('COPILOT_DB', d.db_path),
            chunks_path=_env('COPILOT_CHUNKS', d.chunks_path),
            syllabus_path=_env('COPILOT_SYLLABUS', d.syllabus_path),
            top_k=_env('COPILOT_TOP_K', d.top_k, int),
            pool_size=_env('COPILOT_POOL', d.pool_size, int),
            min_dense_sim=_env('COPILOT_MIN_DENSE', d.min_dense_sim, float),
            min_bm25=_env('COPILOT_MIN_BM25', d.min_bm25, float),
            llm_model=_env('COPILOT_MODEL', d.llm_model),
        )

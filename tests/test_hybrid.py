import pytest

from retrieval.hybrid import HybridRetriever, StaleIndexError, rrf_fuse
from retrieval.types import Hit
from tests.conftest import CHUNKS, FakeDense, chunk


def test_rrf_merges_scores_and_fields():
    l1 = [Hit('x', 't', {}, bm25_score=5.0), Hit('y', 't', {}, bm25_score=1.0)]
    l2 = [Hit('y', 't', {}, dense_sim=0.9), Hit('x', 't', {}, dense_sim=0.6)]
    fused = rrf_fuse([l1, l2], k=60)
    assert {h.id for h in fused} == {'x', 'y'}
    x = next(h for h in fused if h.id == 'x')
    assert x.bm25_score == 5.0 and x.dense_sim == 0.6


def test_hybrid_combines_both_retrievers(bm25):
    dense = FakeDense(CHUNKS, {'b': 0.8, 'a': 0.7})
    r = HybridRetriever(bm25, dense)
    hits = r.search('region of convergence', k=2)
    assert hits[0].id == 'a'          # strong in BM25 and in dense
    assert hits[0].bm25_score and hits[0].dense_sim


def test_non_shareable_chunks_never_returned(bm25):
    dense = FakeDense(CHUNKS, {'d': 0.99, 'a': 0.5})
    hits = HybridRetriever(bm25, dense).search('FFT decimation in time', k=5)
    assert 'd' not in {h.id for h in hits}


def test_stale_index_without_shareable_flag_fails_closed():
    """Indexes built before the shareable flag existed must be rebuilt, not trusted."""
    from retrieval.bm25 import BM25Index
    old = [{'id': str(i), 'text': t, 'meta': {'source': 's.pdf', 'subject': 'dsp', 'page': i}}
           for i, t in enumerate(['z transform basics and properties', 'fir filter window design',
                                  'iir butterworth analog prototype', 'dft and fft algorithms'])]
    r = HybridRetriever(BM25Index(old), FakeDense([], {}))
    with pytest.raises(StaleIndexError):
        r.search('z transform')

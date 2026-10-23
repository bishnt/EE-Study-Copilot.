import numpy as np
import pytest

from retrieval.hybrid import HybridRetriever, fold_visual_into_text
from retrieval.types import Hit
from retrieval.visual import VisualIndex, page_id, read_index, write_index
from tests.conftest import CHUNKS, FakeDense

DIM = 4


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_pages():
    """Three pages. p1 'looks like' axis 0, p2 axis 1, p3 axis 2 (different subject)."""
    rows = [('dsp_notes.pdf', 'dsp', 12, 'yes'), ('dsp_notes.pdf', 'dsp', 13, 'yes'),
            ('swp_notes.pdf', 'swp', 5, 'yes')]
    pages = [{'id': page_id(s, p), 'source': s, 'subject': sub, 'category': 'notes',
              'page': p, 'shareable': sh, 'text': f'text of {s} p{p}'} for s, sub, p, sh in rows]
    vecs = [np.stack([unit(1, 0.1 * i, 0, 0) for i in range(5)]),
            np.stack([unit(0, 1, 0.1 * i, 0) for i in range(7)]),
            np.stack([unit(0, 0, 1, 0.1 * i) for i in range(3)])]
    return pages, [v.astype(np.float16) for v in vecs]


def index(query_vectors, pages=None, vecs=None):
    p, v = make_pages()
    pages, vecs = pages or p, vecs or v
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in vecs])])
    return VisualIndex(pages, np.concatenate(vecs), offsets, lambda q: query_vectors)


def naive(q, page):
    return float((q @ page.astype(np.float32).T).max(axis=1).mean())


def test_maxsim_matches_naive_reference():
    q = np.stack([unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(1, 1, 0, 0)])
    pages, vecs = make_pages()
    hits = index(q).search('anything', k=3)
    got = {h.id: h.visual_score for h in hits}
    for page, v in zip(pages, vecs):
        assert got[page['id']] == pytest.approx(naive(q, v), abs=1e-3)


def test_best_matching_page_ranks_first_and_carries_metadata():
    hits = index(np.stack([unit(0, 1, 0, 0)] * 3)).search('q', k=3)
    assert hits[0].source == 'dsp_notes.pdf' and hits[0].page == 13
    assert hits[0].meta['kind'] == 'page_image' and hits[0].meta['shareable'] == 'yes'
    assert hits[0].visual_score > hits[-1].visual_score
    assert hits[0].text == 'text of dsp_notes.pdf p13'


def test_subject_filter():
    hits = index(np.stack([unit(0, 0, 1, 0)])).search('q', k=5, subject='dsp')
    assert {h.subject for h in hits} == {'dsp'}
    assert index(np.stack([unit(1, 0, 0, 0)])).search('q', subject='emd') == []


def test_chunked_scoring_equals_unchunked(monkeypatch):
    q = np.stack([unit(1, 0.3, 0, 0), unit(0, 1, 0.2, 0)])
    full = {h.id: h.visual_score for h in index(q).search('q', k=3)}
    monkeypatch.setattr('retrieval.visual.MAX_VECTORS_PER_CHUNK', 6)  # forces 1 page per chunk
    small = {h.id: h.visual_score for h in index(q).search('q', k=3)}
    assert small == pytest.approx(full)


def test_dimension_mismatch_is_an_error_not_a_silent_miss():
    with pytest.raises(ValueError, match='different model'):
        index(np.ones((2, 8), dtype=np.float32)).search('q')


def test_empty_query_embedding_returns_nothing():
    assert index(np.zeros((0, DIM), dtype=np.float32)).search('q') == []


def test_write_read_roundtrip_and_model_check(tmp_path):
    pages, vecs = make_pages()
    write_index(tmp_path, pages, vecs, 'vidore/colpali-v1.3')
    got_pages, emb, offsets, meta = read_index(tmp_path)
    assert got_pages == pages and meta == {'model': 'vidore/colpali-v1.3', 'dim': DIM, 'pages': 3}
    assert emb.shape[0] == 15 and list(offsets) == [0, 5, 12, 15]
    q = np.stack([unit(0, 0, 1, 0)])
    idx = VisualIndex.from_dir(tmp_path, lambda _: q, expected_model='vidore/colpali-v1.3')
    assert idx.search('q', k=1)[0].source == 'swp_notes.pdf'
    with pytest.raises(ValueError, match='rebuild'):
        VisualIndex.from_dir(tmp_path, lambda _: q, expected_model='vidore/colqwen2-v1.0')


def test_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_visual'):
        read_index(tmp_path / 'nope')


def test_empty_index_is_refused(tmp_path):
    with pytest.raises(ValueError):
        write_index(tmp_path, [], [], 'm')


# ---- fusion with text retrieval --------------------------------------------------------

class FakeVisual:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=10, subject=None):
        return [h for h in self.hits if not subject or h.subject == subject][:k]


def vhit(source, page, score, subject='dsp', shareable='yes', text=''):
    return Hit(id=page_id(source, page), text=text,
               meta={'source': source, 'subject': subject, 'page': page,
                     'kind': 'page_image', 'shareable': shareable},
               visual_score=score)


def test_visual_hit_folds_into_text_chunk_on_same_page(bm25):
    # chunk 'a' is dsp_notes.pdf p.12; the visual hit is the same page
    dense = FakeDense(CHUNKS, {'a': 0.7})
    r = HybridRetriever(bm25, dense, visual=FakeVisual([vhit('dsp_notes.pdf', 12, 0.8)]))
    hits = r.search('region of convergence', k=5)
    on_page = [h for h in hits if (h.source, h.page) == ('dsp_notes.pdf', 12)]
    assert len(on_page) == 1                     # not listed twice
    assert on_page[0].id == 'a' and on_page[0].visual_score == 0.8
    assert on_page[0].bm25_score and on_page[0].dense_sim
    assert 'region of convergence' in on_page[0].text   # text comes from the chunk


def test_folding_boosts_a_page_both_retrievers_agree_on(bm25):
    dense = FakeDense(CHUNKS, {'a': 0.6, 'b': 0.6})
    plain = HybridRetriever(bm25, dense).search('z plane', k=5)
    with_vis = HybridRetriever(
        bm25, dense, visual=FakeVisual([vhit('dsp_notes.pdf', 30, 0.9)])).search('z plane', k=5)
    score = lambda hits: next(h.score for h in hits if h.id == 'b')
    assert score(with_vis) > score(plain)


def test_image_only_page_enters_fusion_as_its_own_hit(bm25):
    r = HybridRetriever(bm25, FakeDense(CHUNKS, {}),
                        visual=FakeVisual([vhit('dsp_notes.pdf', 99, 0.9)]))
    hits = r.search('circuit diagram', k=5)
    assert [(h.source, h.page, h.visual_score) for h in hits] == [('dsp_notes.pdf', 99, 0.9)]


def test_non_shareable_visual_page_never_returned(bm25):
    secret = vhit('dsp_textbook.pdf', 44, 0.99, shareable='no')
    hits = HybridRetriever(bm25, FakeDense(CHUNKS, {'a': 0.7}),
                           visual=FakeVisual([secret])).search('region of convergence', k=5)
    assert all(h.source != 'dsp_textbook.pdf' for h in hits)


def test_fold_helper_leaves_unmatched_pages_alone():
    text = [[Hit('c1', 't', {'source': 's.pdf', 'page': 1})]]
    out = fold_visual_into_text([vhit('s.pdf', 1, 0.5), vhit('s.pdf', 2, 0.4)], text)
    assert [h.id for h in out] == ['c1', page_id('s.pdf', 2)]

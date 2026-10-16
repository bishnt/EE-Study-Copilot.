from retrieval.bm25 import BM25Index, tokenize
from retrieval.router import SubjectRouter


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize('Explain the filters') == ['filter']


def test_bm25_finds_relevant_chunk_first(bm25):
    hits = bm25.search('region of convergence right-sided sequence', k=3)
    assert hits[0].id == 'a'
    assert hits[0].bm25_score > 0


def test_bm25_subject_filter(bm25):
    hits = bm25.search('relay fault current', k=5, subject='dsp')
    assert all(h.subject == 'dsp' for h in hits)


def test_bm25_no_match_returns_empty(bm25):
    assert bm25.search('quantum chromodynamics', k=5) == []


def test_router_picks_clear_subject_and_abstains_when_unclear():
    syl = [
        {'subject': 'dsp', 'name': 'Digital Signal Processing',
         'chapters': [{'title': 'Z-transform', 'topics': ['region of convergence', 'FIR window method']}]},
        {'subject': 'swp', 'name': 'Switchgear and Protection',
         'chapters': [{'title': 'Relays', 'topics': ['IDMT relay', 'fault current', 'window method']}]},
    ]
    r = SubjectRouter(syl)
    assert r.route('region of convergence of z-transform') == 'dsp'
    assert r.route('IDMT relay fault current') == 'swp'
    assert r.route('window method') is None      # shared vocabulary -> no clear winner
    assert r.route('capital of france') is None

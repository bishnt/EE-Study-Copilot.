from eval.metrics import aggregate, check_targets, judge_faithfulness, precision_at_k
from retrieval.types import Hit
from tests.conftest import FakeLLM


def H(src, page):
    return Hit('i', 't', {'source': src, 'page': page})


def test_precision_at_k_divides_by_k():
    hits = [H('a.pdf', 1), H('a.pdf', 2), H('b.pdf', 9)]
    gold = [{'source': 'a.pdf', 'page': 1}, {'source': 'A.pdf', 'page': 2}]
    assert precision_at_k(hits, gold, k=5) == 2 / 5
    assert precision_at_k(hits, gold, k=2) == 1.0
    assert precision_at_k(hits, [], k=5) == 0.0


def test_page_tolerance():
    assert precision_at_k([H('a.pdf', 4)], [{'source': 'a.pdf', 'page': 3}], k=1, page_tol=1) == 1.0


def row(**kw):
    base = dict(should_refuse=False, refused=False, citations_ok=True, p_at_k=0.8, faithfulness=5)
    return {**base, **kw}


def test_aggregate_and_targets():
    rows = [row(), row(citations_ok=False),
            row(should_refuse=True, refused=True, p_at_k=None, faithfulness=None, citations_ok=None),
            row(should_refuse=True, refused=False)]
    s = aggregate(rows)
    assert s['R1_citations'] == 2 / 3          # answered: rows 1, 2, 4
    assert s['R2_refusal'] == 0.5
    assert s['false_refusal'] == 0.0
    assert not check_targets(s)['R1_citations']


def test_missing_data_fails_target():
    assert not any(check_targets(aggregate([])).values())


def test_judge_parses_score():
    assert judge_faithfulness(FakeLLM('Score: 4'), 'q', 'a', []) == 4
    assert judge_faithfulness(FakeLLM('no idea'), 'q', 'a', []) is None

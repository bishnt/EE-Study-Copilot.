import json
import math

import pytest

from eval.retrieval_metrics import (calibrate_threshold, collapse_to_pages, compare_reports,
                                    mean_metrics, page_metrics, regressions)
from eval.run_retrieval_eval import (Shareable, calibration, load_questions, main, run_questions,
                                     summarize)
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.types import Hit
from tests.conftest import CHUNKS, FakeDense
from tests.test_visual import FakeVisual, vhit


def hit(source, page, **kw):
    return Hit(id=f'{source}{page}{kw.get("n", "")}', text='t',
               meta={'source': source, 'page': page, 'shareable': kw.get('shareable', 'yes')},
               visual_score=kw.get('visual'))


def gold(*pairs):
    return [{'source': s, 'page': p} for s, p in pairs]


# ---- page metrics ------------------------------------------------------------------------

def test_collapse_keeps_first_occurrence_and_ignores_case():
    hits = [hit('A.pdf', 1), hit('a.pdf', 1, n='b'), hit('B.pdf', 2), hit('A.pdf', 1, n='c')]
    assert collapse_to_pages(hits) == [('a.pdf', 1), ('b.pdf', 2)]


def test_page_metrics_hand_computed():
    # chunks: two on A p1 (not gold), then B p2 (gold), then C p3 (gold) -> pages A1, B2, C3
    hits = [hit('A', 1), hit('A', 1, n='x'), hit('B', 2), hit('C', 3)]
    m = page_metrics(hits, gold(('B', 2), ('C', 3)), ks=(1, 5))
    assert m['first_gold_rank'] == 2 and m['mrr'] == 0.5
    assert m['hit@1'] == 0.0 and m['hit@5'] == 1.0
    assert m['recall@1'] == 0.0 and m['recall@5'] == 1.0
    assert m['p@5'] == pytest.approx(2 / 5)           # divided by k, not by pages returned
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert m['ndcg@5'] == pytest.approx(dcg / ideal)


def test_page_metrics_when_gold_not_retrieved_or_nothing_retrieved():
    m = page_metrics([hit('A', 1)], gold(('B', 2)))
    assert m['mrr'] == 0.0 and m['first_gold_rank'] is None and m['recall@5'] == 0.0
    assert page_metrics([], gold(('B', 2)))['ndcg@5'] == 0.0


def test_page_metrics_perfect_ranking_scores_one():
    m = page_metrics([hit('B', 2), hit('C', 3)], gold(('B', 2), ('C', 3)))
    assert m['ndcg@5'] == pytest.approx(1.0) and m['mrr'] == 1.0 and m['recall@5'] == 1.0


def test_page_metrics_requires_gold():
    with pytest.raises(ValueError):
        page_metrics([hit('A', 1)], [])


def test_mean_metrics():
    rows = [page_metrics([hit('B', 2)], gold(('B', 2))), page_metrics([hit('A', 1)], gold(('B', 2)))]
    assert mean_metrics(rows)['mrr'] == 0.5 and 'first_gold_rank' not in mean_metrics(rows)


# ---- threshold calibration -----------------------------------------------------------------

def test_calibration_blocks_target_share_of_refusals():
    ans = [0.7, 0.8, 0.6, 0.65]
    ref = [0.2, 0.3, 0.45, 0.1, 0.35, 0.25, 0.15, 0.4, 0.05, 0.5]
    c = calibrate_threshold(ans, ref, refusal_target=0.9)
    assert c['t_refusal_refusal_blocked'] >= 0.9
    assert sum(s >= c['t_refusal'] for s in ref) <= 1          # at most 10% slip through
    assert c['t_refusal_answerable_pass'] == 1.0               # still passes every answerable one


def test_calibration_threshold_is_rounded_up_never_down():
    c = calibrate_threshold([0.9], [0.4561, 0.1], refusal_target=1.0)
    assert c['t_refusal'] == 0.457 and c['t_refusal_refusal_blocked'] == 1.0


def test_calibration_missing_scores_never_pass():
    c = calibrate_threshold([0.8, None], [None, None, 0.2], refusal_target=0.9)
    assert c['t_refusal_answerable_pass'] == 0.5               # the unscored answerable misses
    assert c['t_refusal_refusal_blocked'] == 1.0


def test_calibration_needs_both_groups():
    assert calibrate_threshold([0.8], [])['t_refusal'] is None
    assert calibrate_threshold([], [0.1])['t_refusal'] is None


def test_calibration_when_refusals_retrieve_nothing_uses_floor():
    c = calibrate_threshold([0.8, 0.7], [None, None])
    assert c['t_refusal'] == 0.0 and c['t_refusal_refusal_blocked'] == 1.0


# ---- compare ---------------------------------------------------------------------------------

def report(r5, mrr, mode='hybrid+visual'):
    return {'modes': {mode: {'overall': {'recall@5': r5, 'mrr': mrr, 'hit@1': 0.5, 'ndcg@5': 0.5}}}}


def test_compare_and_regression_detection():
    comp = compare_reports(report(0.70, 0.60), report(0.80, 0.55))
    assert comp['hybrid+visual']['recall@5']['delta'] == pytest.approx(0.10)
    assert regressions(comp, 'hybrid+visual', tolerance=0.02) == ['hybrid+visual mrr 0.600 -> 0.550']
    assert regressions(comp, 'hybrid+visual', tolerance=0.10) == []
    assert 'missing' in regressions(comp, 'bm25')[0]


def test_cli_compare_strict_exit_codes(tmp_path, capsys):
    base, good, bad = (tmp_path / n for n in ('base.json', 'good.json', 'bad.json'))
    base.write_text(json.dumps(report(0.70, 0.60)))
    good.write_text(json.dumps(report(0.75, 0.62)))
    bad.write_text(json.dumps(report(0.60, 0.40)))
    assert main(['--compare', str(base), str(good), '--strict']) == 0
    assert main(['--compare', str(base), str(bad), '--strict']) == 1
    assert main(['--compare', str(base), str(bad)]) == 0       # without --strict only reports
    assert 'REGRESSION' in capsys.readouterr().out


# ---- runner -------------------------------------------------------------------------------------

QUESTIONS = [
    {'id': 'q1', 'subject': 'dsp', 'question': 'region of convergence right-sided sequence',
     'should_refuse': False, 'gold': gold(('dsp_notes.pdf', 12))},
    {'id': 'q2', 'subject': 'dsp', 'question': 'bilinear transformation s-plane',
     'should_refuse': False, 'gold': gold(('dsp_notes.pdf', 30))},
    {'id': 'r1', 'subject': None, 'question': 'who won the premier league',
     'should_refuse': True, 'gold': []},
]


def retrievers(bm25):
    dense = FakeDense(CHUNKS, {'a': 0.7, 'b': 0.7})
    vis = FakeVisual([vhit('dsp_notes.pdf', 12, 0.8), vhit('dsp_notes.pdf', 30, 0.7)])
    return {'bm25': Shareable(bm25), 'hybrid': HybridRetriever(bm25, dense),
            'hybrid+visual': HybridRetriever(bm25, dense, visual=vis), 'visual': Shareable(vis)}


def test_runner_scores_answerable_and_keeps_refusals_for_calibration(bm25):
    rows = run_questions(QUESTIONS, retrievers(bm25))
    assert len(rows) == 3 * 4
    refusal_rows = [r for r in rows if r['should_refuse']]
    assert refusal_rows and all('metrics' not in r for r in refusal_rows)
    s = summarize(rows)
    assert s['bm25']['n'] == 2
    # FakeVisual ignores the query and always ranks p.12 first: right for q1, second for q2.
    assert s['visual']['overall']['hit@1'] == 0.5 and s['visual']['overall']['hit@5'] == 1.0
    assert s['visual']['overall']['mrr'] == pytest.approx(0.75)
    assert s['bm25']['overall']['recall@5'] == 1.0
    assert 'dsp' in s['hybrid']['per_subject'] and 'chunk_p@5' in s['hybrid']['overall']


def test_calibration_uses_visual_mode_only(bm25):
    assert calibration(run_questions(QUESTIONS, {'bm25': Shareable(bm25)})) is None
    c = calibration(run_questions(QUESTIONS, retrievers(bm25)))
    assert c['n_answerable'] == 2 and c['n_refusal'] == 1


def test_shareable_wrapper_drops_non_shareable():
    class Inner:
        def search(self, q, k, subject):
            return [hit('a.pdf', 1), hit('b.pdf', 2, shareable='no')]
    assert [h.source for h in Shareable(Inner()).search('q', 5)] == ['a.pdf']
    assert len(Shareable(Inner(), require_shareable=False).search('q', 5)) == 2


def test_load_questions_rejects_answerable_without_gold(tmp_path):
    f = tmp_path / 'q.jsonl'
    f.write_text(json.dumps({'id': 'x', 'question': 'q', 'should_refuse': False, 'gold': []}) + '\n')
    with pytest.raises(ValueError, match="'x'"):
        load_questions(f)
    f.write_text(json.dumps({'id': 'y', 'question': 'q', 'should_refuse': True, 'gold': []}) + '\n')
    assert load_questions(f)[0]['id'] == 'y'


def test_main_runs_end_to_end_and_writes_report(tmp_path, monkeypatch, bm25, capsys):
    import eval.run_retrieval_eval as rre
    monkeypatch.setattr(rre, 'build_retrievers', lambda cfg, modes: {m: r for m, r in retrievers(bm25).items() if m in modes})
    q = tmp_path / 'q.jsonl'
    q.write_text('\n'.join(json.dumps(x) for x in QUESTIONS) + '\n')
    out = tmp_path / 'report.json'
    assert main(['--questions', str(q), '--out', str(out), '--modes', 'bm25,visual,hybrid+visual']) == 0
    report = json.loads(out.read_text())
    assert set(report['modes']) == {'bm25', 'visual', 'hybrid+visual'}
    assert report['n_questions'] == 3 and report['n_answerable'] == 2
    assert report['calibration']['n_refusal'] == 1 and report['config']['modes']
    assert len(report['rows']) == 3 * 3
    shown = capsys.readouterr().out
    assert 'recall@5' in shown and 'COPILOT_MIN_VISUAL suggestion' in shown


def test_main_reports_missing_visual_index_instead_of_crashing(tmp_path, monkeypatch, capsys):
    import eval.run_retrieval_eval as rre

    def boom(cfg, modes):
        raise FileNotFoundError('No visual index in x. Build it: python -m ingestion.build_visual --class')
    monkeypatch.setattr(rre, 'build_retrievers', boom)
    q = tmp_path / 'q.jsonl'
    q.write_text(json.dumps(QUESTIONS[0]) + '\n')
    assert main(['--questions', str(q), '--out', str(tmp_path / 'r.json')]) == 2
    assert 'build_visual' in capsys.readouterr().err


def test_main_rejects_unknown_modes(tmp_path):
    with pytest.raises(SystemExit):
        main(['--questions', str(tmp_path / 'q.jsonl'), '--modes', 'bm25,magic'])

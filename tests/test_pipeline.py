from app.config import REFUSAL, Config
from app.pipeline import answer_question
from app.scope import out_of_scope
from retrieval.hybrid import HybridRetriever
from tests.conftest import CHUNKS, FakeDense, FakeLLM

CFG = Config()
GOOD = 'The ROC of a right-sided sequence is the exterior of a circle. [dsp_notes.pdf, 12]'


def make(bm25, sims):
    return HybridRetriever(bm25, FakeDense(CHUNKS, sims))


def test_happy_path(bm25):
    llm = FakeLLM(GOOD)
    a = answer_question('region of convergence of right-sided sequence', make(bm25, {'a': 0.8}), llm, cfg=CFG)
    assert not a.refused and a.reason == 'ok' and a.text == GOOD
    assert ('dsp_notes.pdf', 12) in a.sources()


def test_no_evidence_refuses_without_calling_llm(bm25):
    llm = FakeLLM()
    a = answer_question('who won the world cup', make(bm25, {'b': 0.2}), llm, cfg=CFG)
    assert a.refused and a.reason == 'no_evidence' and a.text == REFUSAL and llm.calls == []


def test_out_of_scope_rules():
    assert out_of_scope('Explain ENEE 301 project scheduling')
    assert out_of_scope('predict the marks of the next exam paper')
    assert out_of_scope('write my assignment on FIR filters')
    assert out_of_scope('state the ROC of a right-sided sequence') is None


def test_out_of_scope_skips_retrieval(bm25):
    a = answer_question('Which questions are expected in the exam paper?', make(bm25, {'a': 0.9}),
                        FakeLLM(), cfg=CFG)
    assert a.refused and a.reason.startswith('out_of_scope')


def test_model_refusal_is_passed_through(bm25):
    a = answer_question('bilinear transformation', make(bm25, {'b': 0.8}), FakeLLM(REFUSAL), cfg=CFG)
    assert a.refused and a.reason == 'model_refused'


def test_repair_then_success(bm25):
    llm = FakeLLM('The ROC is outside a circle for such sequences.', GOOD)
    a = answer_question('region of convergence right-sided', make(bm25, {'a': 0.8}), llm, cfg=CFG)
    assert not a.refused and len(llm.calls) == 2 and 'Problems:' in llm.calls[1]


def test_invalid_citations_fail_closed(bm25):
    bad = 'The ROC is outside a circle for such sequences. [made_up.pdf, 7]'
    llm = FakeLLM(bad, bad)
    a = answer_question('region of convergence right-sided', make(bm25, {'a': 0.8}), llm, cfg=CFG)
    assert a.refused and a.reason == 'invalid_citations' and a.text == REFUSAL


def test_non_shareable_text_never_reaches_llm(bm25):
    llm = FakeLLM(GOOD)
    answer_question('FFT decimation in time', make(bm25, {'a': 0.8, 'd': 0.9}), llm, cfg=CFG)
    assert all('Textbook paragraph' not in prompt for prompt in llm.calls)

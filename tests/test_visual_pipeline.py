import pytest

from app.config import REFUSAL, Config
from app.pipeline import answer_question, has_evidence, page_images
from generation.llm import AnthropicLLM
from generation.prompt import VISUAL_NOTE
from retrieval.colpali_model import model_family
from retrieval.hybrid import HybridRetriever
from retrieval.types import Hit
from tests.conftest import CHUNKS, FakeDense, FakeLLM
from tests.test_visual import FakeVisual, vhit

CFG = Config(min_visual=0.5)
CITE = '[dsp_notes.pdf, 12]'
ANSWER = f'The ROC of a right-sided sequence is the exterior of a circle. {CITE}'


def renderer(calls=None, fail=False):
    def render(hit):
        if calls is not None:
            calls.append((hit.source, hit.page))
        if fail:
            return None
        return {'label': f'[{hit.source}, {hit.page}]', 'media_type': 'image/png', 'data': 'AAAA'}
    return render


def retriever(bm25, visual_hits, sims=None, render=None):
    return HybridRetriever(bm25, FakeDense(CHUNKS, sims or {}), visual=FakeVisual(visual_hits),
                           page_renderer=render)


def test_config_reads_visual_settings_from_env(monkeypatch):
    monkeypatch.setenv('COPILOT_VISUAL', 'true')
    monkeypatch.setenv('COPILOT_MIN_VISUAL', '0.62')
    monkeypatch.setenv('COPILOT_VISUAL_MODEL', 'vidore/colqwen2-v1.0')
    cfg = Config.from_env()
    assert cfg.visual_enabled and cfg.min_visual == 0.62 and cfg.visual_model.endswith('v1.0')
    monkeypatch.delenv('COPILOT_VISUAL')
    assert Config.from_env().visual_enabled is False      # opt-in by default


def test_visual_score_alone_counts_as_evidence():
    h = Hit('x', 't', {}, visual_score=0.7)
    assert has_evidence([h], CFG)
    assert not has_evidence([Hit('x', 't', {}, visual_score=0.3)], CFG)


def test_weak_visual_match_alone_refuses_without_calling_llm(bm25):
    llm = FakeLLM()
    r = retriever(bm25, [vhit('dsp_notes.pdf', 99, 0.2, text='x')], render=renderer())
    a = answer_question('unrelated question', r, llm, cfg=CFG)
    assert a.refused and a.reason == 'no_evidence' and llm.calls == []


def test_relevant_page_image_is_sent_to_the_model(bm25):
    llm, calls = FakeLLM(ANSWER), []
    r = retriever(bm25, [vhit('dsp_notes.pdf', 12, 0.8)], {'a': 0.7}, renderer(calls))
    a = answer_question('region of convergence right-sided', r, llm, cfg=CFG)
    assert not a.refused and calls == [('dsp_notes.pdf', 12)]
    assert [im['label'] for im in llm.images[0]] == [CITE]
    assert llm.systems[0].endswith(VISUAL_NOTE)


def test_no_images_when_visual_retrieval_is_off(bm25):
    llm = FakeLLM(ANSWER)
    r = HybridRetriever(bm25, FakeDense(CHUNKS, {'a': 0.8}))   # default pipeline, no visual
    answer_question('region of convergence right-sided', r, llm, cfg=CFG)
    assert llm.images == [None] and VISUAL_NOTE not in llm.systems[0]


def test_low_scoring_pages_are_not_rendered(bm25):
    llm, calls = FakeLLM(ANSWER), []
    r = retriever(bm25, [vhit('dsp_notes.pdf', 12, 0.3)], {'a': 0.8}, renderer(calls))
    answer_question('region of convergence right-sided', r, llm, cfg=CFG)
    assert calls == [] and llm.images == [None]


def test_image_cap_and_dedupe(bm25):
    hits = [Hit(str(i), 't', {'source': 's.pdf', 'page': i, 'shareable': 'yes'}, visual_score=0.9)
            for i in (1, 1, 2, 3, 4)]
    class R:
        page_renderer = staticmethod(renderer())
    got = page_images(hits, R, Config(min_visual=0.5, max_images=2))
    assert [im['key'] for im in got] == [('s.pdf', 1), ('s.pdf', 2)]
    assert page_images(hits, R, Config(max_images=0)) == []


def test_non_shareable_page_is_never_rendered():
    calls = []
    class R:
        page_renderer = staticmethod(renderer(calls))
    h = Hit('x', 't', {'source': 's.pdf', 'page': 1, 'shareable': 'no'}, visual_score=0.99)
    assert page_images([h], R, Config(min_visual=0.5)) == [] and calls == []


def test_image_only_page_that_cannot_be_rendered_is_dropped_and_refuses(bm25):
    llm = FakeLLM()
    r = retriever(bm25, [vhit('dsp_notes.pdf', 99, 0.9)], render=renderer(fail=True))
    a = answer_question('draw the filter circuit', r, llm, cfg=CFG)
    assert a.refused and a.reason == 'no_evidence' and a.text == REFUSAL and llm.calls == []


def test_dropped_page_does_not_let_weak_leftovers_through(bm25):
    # visual-only page (unrenderable) was the only real evidence; chunk 'b' is weak
    llm = FakeLLM()
    r = retriever(bm25, [vhit('dsp_notes.pdf', 99, 0.9)], {'b': 0.2}, renderer(fail=True))
    a = answer_question('bilinear', r, llm, cfg=CFG)
    assert a.refused and llm.calls == []


def test_citing_an_image_only_page_is_valid_when_its_image_was_sent(bm25):
    llm = FakeLLM('The circuit is a second-order low-pass section. [dsp_notes.pdf, 99]')
    r = retriever(bm25, [vhit('dsp_notes.pdf', 99, 0.9)], render=renderer())
    a = answer_question('what does the filter circuit look like', r, llm, cfg=CFG)
    assert not a.refused and ('dsp_notes.pdf', 99) in a.sources()


def test_repair_attempt_keeps_the_images(bm25):
    llm = FakeLLM('The ROC is outside a circle for such sequences.', ANSWER)
    r = retriever(bm25, [vhit('dsp_notes.pdf', 12, 0.8)], {'a': 0.7}, renderer())
    answer_question('region of convergence right-sided', r, llm, cfg=CFG)
    assert len(llm.images) == 2 and all(im for im in llm.images)


def test_anthropic_request_puts_labelled_images_before_the_question():
    content = AnthropicLLM._content('Q?', [{'label': CITE, 'media_type': 'image/png', 'data': 'AAAA'}])
    assert [b['type'] for b in content] == ['text', 'image', 'text']
    assert CITE in content[0]['text'] and content[1]['source']['type'] == 'base64'
    assert content[2]['text'] == 'Q?'
    assert AnthropicLLM._content('Q?', None) == 'Q?'      # text-only calls are unchanged


@pytest.mark.parametrize('name,family', [
    ('vidore/colpali-v1.3', 'ColPali'), ('vidore/colqwen2-v1.0', 'ColQwen2'),
    ('vidore/colqwen2.5-v0.2', 'ColQwen2_5'), ('vidore/colSmol-500M', 'ColIdefics3')])
def test_model_family_mapping(name, family):
    assert model_family(name)[0] == family


def test_unknown_visual_model_is_rejected():
    with pytest.raises(ValueError, match='Unsupported'):
        model_family('some/other-model')

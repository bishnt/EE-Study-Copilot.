from generation.citations import claims, extract_citations, validate
from retrieval.types import Hit

HITS = [Hit('a', 't', {'source': 'dsp_notes.pdf', 'page': 12})]


def test_extract_handles_page_prefix_and_path():
    cs = extract_citations('x [dsp_notes.pdf, 12] y [corpus/dsp/DSP_Notes.pdf, p. 3]')
    assert [(c.source, c.page) for c in cs] == [('dsp_notes.pdf', 12), ('dsp_notes.pdf', 3)]


def test_valid_answer_passes():
    v = validate('The ROC of a right-sided sequence lies outside a circle. [dsp_notes.pdf, 12]', HITS)
    assert v.ok


def test_hallucinated_citation_is_flagged():
    v = validate('The ROC lies outside a circle of radius r. [other_book.pdf, 99]', HITS)
    assert not v.ok and v.unknown


def test_uncited_claim_is_flagged():
    v = validate('The ROC lies outside a circle. [dsp_notes.pdf, 12]\n'
                 'Left-sided sequences converge inside a circle instead.', HITS)
    assert not v.ok and len(v.uncited) == 1


def test_math_lines_and_leadins_are_not_claims():
    text = 'Given:\nH(z) = 1 / (1 - 0.5 z^-1)\n## Result'
    assert claims(text) == []


def test_no_citation_at_all_fails():
    assert not validate('Short.', HITS).ok

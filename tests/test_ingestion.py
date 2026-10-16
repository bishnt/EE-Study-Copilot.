from ingestion.chunk import make_chunks
from ingestion.clean import clean_page, find_boilerplate


def test_chunks_carry_shareable_flag_and_fail_closed():
    text = 'The Z-transform converts a difference equation into an algebraic equation. ' * 3
    assert all(c['meta']['shareable'] == 'yes'
               for c in make_chunks('f.pdf', 'dsp', 'notes', 1, text, [], shareable='yes'))
    assert all(c['meta']['shareable'] == 'no' for c in make_chunks('f.pdf', 'dsp', 'notes', 1, text, []))


def test_chunk_ids_are_stable():
    t = 'Stable identifiers keep upserts idempotent across rebuilds of the index. ' * 2
    assert make_chunks('f.pdf', 'dsp', 'n', 3, t, [], 'yes') == make_chunks('f.pdf', 'dsp', 'n', 3, t, [], 'yes')


def test_boilerplate_removed_and_hyphens_joined():
    pages = [f'IOE Pulchowk\nbody {i} text\nPage footer' for i in range(5)]
    boiler = find_boilerplate(pages)
    out = clean_page('IOE Pulchowk\nconver-\ngence of the \ufb01lter\nPage footer', boiler)
    assert 'IOE Pulchowk' not in out and 'convergence' in out and 'filter' in out

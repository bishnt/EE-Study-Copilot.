import numpy as np
import pandas as pd
import pytest

fitz = pytest.importorskip('fitz')
pytest.importorskip('PIL')

from ingestion.build_visual import build  # noqa: E402
from ingestion.render import make_page_renderer, render_page_png  # noqa: E402
from retrieval.types import Hit  # noqa: E402
from retrieval.visual import read_index  # noqa: E402

DIM = 8


def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        doc.new_page(width=300, height=400).insert_text((40, 60), text)
    doc.save(path)
    doc.close()


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / 'corpus'
    (root / 'dsp').mkdir(parents=True)
    (root / 'swp').mkdir()
    make_pdf(root / 'dsp' / 'notes.pdf', ['Z transform ROC', 'Bilinear transform'])
    make_pdf(root / 'swp' / 'book.pdf', ['IDMT relay curve'])
    pd.DataFrame([
        {'file': 'notes.pdf', 'subject': 'dsp', 'category': 'notes', 'shareable': 'yes'},
        {'file': 'book.pdf', 'subject': 'swp', 'category': 'textbook', 'shareable': 'no'},
    ]).to_csv(root / 'manifest.csv', index=False)
    return root


class CountingEmbedder:
    def __init__(self):
        self.pages = 0

    def __call__(self, images):
        self.pages += len(images)
        rng = np.random.default_rng(0)
        return [rng.normal(size=(6, DIM)).astype(np.float16) for _ in images]


def run(corpus, out, embedder, **kw):
    return build(str(corpus / 'manifest.csv'), str(corpus), str(out), embedder=embedder,
                 log=lambda *_: None, **kw)


def test_build_writes_index_with_one_row_per_page(corpus, tmp_path):
    out = tmp_path / 'vis'
    assert run(corpus, out, CountingEmbedder()) == 3
    pages, emb, offsets, meta = read_index(out)
    assert [(p['source'], p['page']) for p in pages] == [
        ('notes.pdf', 1), ('notes.pdf', 2), ('book.pdf', 1)]
    assert 'ROC' in pages[0]['text'] and emb.shape == (18, DIM) and meta['pages'] == 3


def test_class_mode_excludes_non_shareable_documents(corpus, tmp_path):
    out = tmp_path / 'vis_class'
    run(corpus, out, CountingEmbedder(), shareable_only=True)
    pages, *_ = read_index(out)
    assert {p['source'] for p in pages} == {'notes.pdf'}
    assert all(p['shareable'] == 'yes' for p in pages)


def test_rerun_uses_cached_shards(corpus, tmp_path):
    out, emb = tmp_path / 'vis', CountingEmbedder()
    run(corpus, out, emb)
    assert emb.pages == 3
    run(corpus, out, emb)
    assert emb.pages == 3                       # nothing re-embedded


def test_changing_model_re_embeds(corpus, tmp_path):
    out, emb = tmp_path / 'vis', CountingEmbedder()
    run(corpus, out, emb, model_name='vidore/colpali-v1.3')
    run(corpus, out, emb, model_name='vidore/colqwen2-v1.0')
    assert emb.pages == 6


def test_manifest_edit_changes_shareable_without_re_embedding(corpus, tmp_path):
    out, emb = tmp_path / 'vis', CountingEmbedder()
    run(corpus, out, emb)
    df = pd.read_csv(corpus / 'manifest.csv')
    df.loc[df['file'] == 'book.pdf', 'shareable'] = 'yes'
    df.to_csv(corpus / 'manifest.csv', index=False)
    run(corpus, out, emb)
    pages, *_ = read_index(out)
    assert emb.pages == 3 and all(p['shareable'] == 'yes' for p in pages)


def test_corrupt_shard_is_re_embedded(corpus, tmp_path):
    out, emb = tmp_path / 'vis', CountingEmbedder()
    run(corpus, out, emb)
    for shard in (out / '.shards').glob('*.npz'):
        shard.write_bytes(b'not a zip')
    run(corpus, out, emb)
    assert emb.pages == 6


def test_missing_pdf_stops_before_any_embedding(corpus, tmp_path):
    (corpus / 'dsp' / 'notes.pdf').unlink()
    emb = CountingEmbedder()
    with pytest.raises(SystemExit, match='validate_corpus'):
        run(corpus, tmp_path / 'vis', emb)
    assert emb.pages == 0


def test_page_renderer_returns_png_block(corpus):
    block = make_page_renderer(corpus)(Hit('x', 't', {'source': 'notes.pdf', 'subject': 'dsp', 'page': 2}))
    import base64
    assert block['label'] == '[notes.pdf, 2]' and block['media_type'] == 'image/png'
    assert base64.b64decode(block['data'])[:8] == b'\x89PNG\r\n\x1a\n'


def test_page_renderer_stays_inside_corpus_and_survives_bad_pages(corpus, tmp_path):
    render = make_page_renderer(corpus)
    escape = Hit('x', 't', {'source': '../../etc/passwd', 'subject': 'dsp', 'page': 1})
    assert render(escape) is None
    assert render(Hit('x', 't', {'source': 'notes.pdf', 'subject': 'dsp', 'page': 9})) is None
    assert render(Hit('x', 't', {'source': 'gone.pdf', 'subject': 'dsp', 'page': 1})) is None
    assert render_page_png(corpus / 'dsp' / 'notes.pdf', 0) is None


def test_end_to_end_build_search_render_answer(corpus, tmp_path):
    """Real PDFs -> index -> ColPali-style search -> page rendered from the PDF -> model call."""
    import base64

    from app.config import Config
    from app.pipeline import answer_question
    from retrieval.bm25 import BM25Index
    from retrieval.hybrid import HybridRetriever
    from retrieval.visual import VisualIndex
    from tests.conftest import FakeDense, FakeLLM, chunk

    class OneHot:                       # page n gets axis n, so the 'query' can pick a page
        n = -1

        def __call__(self, images):
            out = []
            for _ in images:
                self.n += 1
                out.append(np.eye(DIM, dtype=np.float16)[[self.n]])
            return out

    out = tmp_path / 'vis'
    run(corpus, out, OneHot(), shareable_only=True)       # notes.pdf p1 -> e0, p2 -> e1
    query = np.eye(DIM, dtype=np.float32)[[1]]            # asks for the second page
    visual = VisualIndex.from_dir(out, lambda _: query)
    text_only = [chunk('t', 'Unrelated paragraph about transformers.', source='notes.pdf', page=1)]
    retriever = HybridRetriever(BM25Index(text_only), FakeDense(text_only, {}), visual=visual,
                                page_renderer=make_page_renderer(corpus))
    llm = FakeLLM('The bilinear transform maps the s-plane to the z-plane. [notes.pdf, 2]')
    a = answer_question('show the bilinear transform figure', retriever, llm,
                        cfg=Config(min_visual=0.5))
    assert not a.refused and a.reason == 'ok'
    (image,) = llm.images[0]
    assert image['label'] == '[notes.pdf, 2]'
    assert base64.b64decode(image['data'])[:4] == b'\x89PNG'

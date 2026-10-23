"""Render PDF pages to PNG with PyMuPDF (already a dependency; no poppler needed)."""
import base64
import logging
from pathlib import Path

log = logging.getLogger(__name__)
MAX_ZOOM = 3.0  # never upscale tiny pages more than this


def _png(page, max_side):
    import fitz  # PyMuPDF
    longest = max(page.rect.width, page.rect.height) or 1.0
    zoom = min(MAX_ZOOM, max_side / longest)
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).tobytes('png')


def iter_pages(pdf_path, max_side=1024):
    """Yield (page_number, png_bytes, text). Page numbers start at 1, like ingestion.extract."""
    import fitz
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            yield i, _png(page, max_side), page.get_text('text', sort=True)


def render_page_png(pdf_path, page_number, max_side=1400):
    """PNG bytes of one page (1-based), or None if the file or page is unavailable."""
    import fitz
    try:
        with fitz.open(pdf_path) as doc:
            if not 1 <= page_number <= len(doc):
                return None
            return _png(doc[page_number - 1], max_side)
    except Exception as exc:  # missing/corrupt PDF must not crash an answer
        log.warning('could not render %s p.%s: %s', pdf_path, page_number, exc)
        return None


def make_page_renderer(corpus_dir, max_side=1400):
    """Return render(hit) -> image block for the LLM, or None. Files stay inside corpus_dir."""
    root = Path(corpus_dir).resolve()

    def render(hit):
        path = (root / hit.subject / hit.source).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        png = render_page_png(path, hit.page, max_side)
        if png is None:
            return None
        return {'label': f'[{hit.source}, {hit.page}]', 'media_type': 'image/png',
                'data': base64.b64encode(png).decode('ascii')}

    return render

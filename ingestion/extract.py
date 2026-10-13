import fitz  # PyMuPDF

def table_to_markdown(rows):
    rows = [[(c or '').replace('\n', ' ').strip() for c in r] for r in rows]
    if len(rows) < 2 or len(rows[0]) < 2:
        return ''
    head, body = rows[0], rows[1:]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    lines += ['| ' + ' | '.join(r) + ' |' for r in body]
    return '\n'.join(lines)

def extract_pages(pdf_path):
    """Yield (page_number, text, tables). Page numbers start at 1."""
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text('text', sort=True)
        tables = []
        try:
            for t in page.find_tables().tables:
                md = table_to_markdown(t.extract())
                if md:
                    tables.append(md)
        except Exception:
            pass  # table detection can fail on odd layouts; keep the page text
        yield i, text, tables
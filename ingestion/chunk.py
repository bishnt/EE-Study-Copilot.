import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=120, separators=['\n\n', '\n', '. ', ' ', ''])

def _row(source, subject, category, page, tag, text, kind):
    cid = hashlib.sha1(f'{source}|{page}|{tag}'.encode()).hexdigest()[:16]
    return {'id': cid, 'text': text,
            'meta': {'source': source, 'subject': subject, 'category': category,
                     'page': page, 'kind': kind, 'chapter': 0}}

def make_chunks(source, subject, category, page, text, tables):
    out = []
    for idx, piece in enumerate(splitter.split_text(text)):
        if len(piece) >= 40:  # drop tiny fragments
            out.append(_row(source, subject, category, page, f't{idx}', piece, 'text'))
    for idx, md in enumerate(tables):
        out.append(_row(source, subject, category, page, f'tab{idx}', md[:3000], 'table'))
    return out
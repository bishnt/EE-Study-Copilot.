import pytest

from retrieval.bm25 import BM25Index


def chunk(cid, text, subject='dsp', source='dsp_notes.pdf', page=1, shareable='yes'):
    return {'id': cid, 'text': text,
            'meta': {'source': source, 'subject': subject, 'page': page,
                     'category': 'notes', 'kind': 'text', 'chapter': 0, 'shareable': shareable}}


CHUNKS = [
    chunk('a', 'The region of convergence of a right-sided sequence is the exterior of a circle in the z-plane.',
          page=12),
    chunk('b', 'The bilinear transformation maps the s-plane to the z-plane and avoids aliasing.',
          page=30),
    chunk('c', 'An IDMT relay operates faster when the fault current is larger.',
          subject='swp', source='swp_notes.pdf', page=5),
    chunk('d', 'Textbook paragraph about FFT decimation in time.', page=44,
          source='dsp_textbook.pdf', shareable='no'),
]


class FakeDense:
    """Returns canned hits with cosine sims; honours the subject filter like Chroma."""
    def __init__(self, chunks, sims):
        self.chunks, self.sims = chunks, sims

    def search(self, query, k=20, subject=None):
        from retrieval.types import Hit
        out = [Hit(id=c['id'], text=c['text'], meta=c['meta'], dense_sim=self.sims[c['id']])
               for c in self.chunks if c['id'] in self.sims
               and (not subject or c['meta']['subject'] == subject)]
        return sorted(out, key=lambda h: h.dense_sim, reverse=True)[:k]


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, user, max_tokens=1024):
        self.calls.append(user)
        return self.replies.pop(0)


@pytest.fixture
def chunks():
    return CHUNKS


@pytest.fixture
def bm25():
    return BM25Index(CHUNKS)

import json
from collections import defaultdict

from retrieval.bm25 import tokenize


class SubjectRouter:
    """Guess the subject from syllabus vocabulary; return None unless the signal is clear."""

    def __init__(self, syllabus):
        owners = defaultdict(set)
        for s in syllabus:
            texts = [s['name']] + [c['title'] + ' ' + ' '.join(c.get('topics', []))
                                   for c in s['chapters']]
            for tok in {t for text in texts for t in tokenize(text)}:
                owners[tok].add(s['subject'])
        self._owners = owners

    @classmethod
    def from_file(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    def route(self, query, min_score=1.0, min_margin=0.5):
        scores = defaultdict(float)
        for tok in set(tokenize(query)):
            subs = self._owners.get(tok, ())
            for s in subs:
                scores[s] += 1.0 / len(subs)  # words shared by many subjects count less
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return best[0] if best[1] >= min_score and best[1] - second >= min_margin else None

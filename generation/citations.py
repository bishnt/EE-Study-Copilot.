import re
from dataclasses import dataclass, field
from pathlib import PurePath

# [file.pdf, 12] or [file.pdf, p. 12]
CITE_RE = re.compile(r'\[\s*([^\[\]\n]+?)\s*,\s*(?:p\.?\s*)?(\d+)\s*\]')
_SENT_SPLIT = re.compile(r'(?<=[.!?:])\s+(?!\[)')


@dataclass(frozen=True)
class Citation:
    source: str
    page: int


def _norm(name):
    return PurePath(name.strip()).name.lower()


def extract_citations(text):
    return [Citation(_norm(m.group(1)), int(m.group(2))) for m in CITE_RE.finditer(text)]


def claims(text, min_words=4):
    """Sentences that make a claim. Headings, lead-ins ending in ':' and math-only lines are skipped."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        for sent in _SENT_SPLIT.split(line):
            stripped = CITE_RE.sub('', sent).strip(' -*>')
            words = re.findall(r'[A-Za-z]{2,}', stripped)
            if len(words) >= min_words and not stripped.endswith(':'):
                out.append(sent)
    return out


@dataclass
class Validation:
    citations: list = field(default_factory=list)
    unknown: list = field(default_factory=list)   # cited (source, page) not in the context
    uncited: list = field(default_factory=list)   # claim sentences without a citation

    @property
    def ok(self):
        return bool(self.citations) and not self.unknown and not self.uncited

    def problems(self):
        msgs = []
        if not self.citations:
            msgs.append('no citations')
        if self.unknown:
            msgs.append('citations not in the context: ' + ', '.join(
                f'[{c.source}, {c.page}]' for c in self.unknown))
        if self.uncited:
            msgs.append('uncited claims: ' + ' | '.join(s[:80] for s in self.uncited[:5]))
        return '; '.join(msgs)


def validate(text, hits, min_words=4):
    allowed = {(_norm(h.source), h.page) for h in hits}
    cites = extract_citations(text)
    unknown = [c for c in cites if (c.source, c.page) not in allowed]
    uncited = [s for s in claims(text, min_words) if not CITE_RE.search(s)]
    return Validation(citations=cites, unknown=unknown, uncited=uncited)

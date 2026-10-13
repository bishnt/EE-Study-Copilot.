import re
from collections import Counter

LIGATURES = {'\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff', '\ufb03': 'ffi', '\ufb04': 'ffl'}

def find_boilerplate(pages, min_frac=0.4):
    """Lines that repeat on many pages are headers or footers."""
    if len(pages) < 4:  # too few pages to judge repetition
        return set()
    counts = Counter()
    for text in pages:
        for line in {l.strip() for l in text.splitlines() if l.strip()}:
            counts[line] += 1
    n = len(pages)
    return {l for l, c in counts.items() if c / n >= min_frac and len(l) < 120}

def clean_page(text, boilerplate):
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = '\n'.join(l for l in text.splitlines() if l.strip() not in boilerplate)
    text = re.sub(r'-\n(?=[a-z])', '', text)            # join hyphenated words
    text = re.sub(r'(?<![.:;!?])\n(?!\n)', ' ', text)   # join soft line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
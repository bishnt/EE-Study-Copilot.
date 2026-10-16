"""Cheap pre-retrieval gate for requests scope.md lists as out of scope.
Everything else (other courses, general knowledge, missing evidence) is caught by the evidence gate."""
import re

_BLOCKED = [
    ('course_enee301', r'\benee\s*301\b|project engineering and management'),
    ('exam_prediction', r'\b(predict(ed)?|expected|likely|guess)\b.{0,40}\b(marks?|questions?|paper)\b'
                        r'|\bwill (this|it) come in (the )?exam\b|\bexam paper\b'),
    ('assignment', r'\b(write|do|complete|solve)\b.{0,20}\b(my|our)\b.{0,15}\b(assignment|homework|lab report)\b'),
    ('real_settings', r'\b(my|our|the actual|real)\b.{0,30}\b(substation|feeder|plant|site)\b.{0,40}\b(setting|settings|set the)\b'),
]
_COMPILED = [(name, re.compile(rx, re.I)) for name, rx in _BLOCKED]


def out_of_scope(question):
    """Return the matched rule name, or None."""
    for name, rx in _COMPILED:
        if rx.search(question):
            return name
    return None

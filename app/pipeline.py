from dataclasses import dataclass, field

from app.config import REFUSAL, Config
from app.scope import out_of_scope
from generation.citations import Validation, validate
from generation.prompt import SYSTEM, build_repair_prompt, build_user_prompt


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    reason: str = ''            # out_of_scope:<rule> | no_evidence | model_refused | invalid_citations | ok
    subject: str | None = None
    hits: list = field(default_factory=list)   # chunks actually sent to the model
    validation: Validation | None = None

    def sources(self):
        return sorted({(h.source, h.page) for h in self.hits})


def _refuse(question, reason, subject=None, hits=None, validation=None):
    return Answer(question, REFUSAL, True, reason, subject, hits or [], validation)


def has_evidence(hits, cfg):
    """Retrieval found something close enough to be worth answering from."""
    return any((h.dense_sim is not None and h.dense_sim >= cfg.min_dense_sim) or
               (h.bm25_score is not None and h.bm25_score >= cfg.min_bm25) for h in hits)


def answer_question(question, retriever, llm, router=None, subject=None, cfg=None):
    cfg = cfg or Config()
    rule = out_of_scope(question)
    if rule:
        return _refuse(question, f'out_of_scope:{rule}')

    subject = subject or (router.route(question) if router else None)
    hits = retriever.search(question, k=cfg.top_k, subject=subject)
    if not has_evidence(hits, cfg):
        return _refuse(question, 'no_evidence', subject, hits)

    prompt, used = build_user_prompt(question, hits, cfg.max_context_chars)
    text = llm.complete(SYSTEM, prompt).strip()
    if text.startswith(REFUSAL):
        return _refuse(question, 'model_refused', subject, used)

    check = validate(text, used)
    if not check.ok:  # one repair attempt, then fail closed
        text = llm.complete(SYSTEM, build_repair_prompt(prompt, text, check.problems())).strip()
        if text.startswith(REFUSAL):
            return _refuse(question, 'model_refused', subject, used)
        check = validate(text, used)
        if not check.ok:
            return _refuse(question, 'invalid_citations', subject, used, check)
    return Answer(question, text, False, 'ok', subject, used, check)

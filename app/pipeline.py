from dataclasses import dataclass, field

from app.config import REFUSAL, Config
from app.scope import out_of_scope
from generation.citations import Validation, validate
from generation.prompt import SYSTEM, VISUAL_NOTE, build_repair_prompt, build_user_prompt


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
               (h.bm25_score is not None and h.bm25_score >= cfg.min_bm25) or
               (h.visual_score is not None and h.visual_score >= cfg.min_visual) for h in hits)


def page_images(used, retriever, cfg):
    """Page images for the hits ColPali judged relevant. Empty unless visual retrieval is on.

    Only pages that clear the visual evidence gate are attached, and, like text, only pages
    from shareable documents: nothing else is ever sent to the API.
    """
    render = getattr(retriever, 'page_renderer', None)
    if render is None or cfg.max_images <= 0:
        return []
    images, seen = [], set()
    for h in used:
        key = (h.source, h.page)
        if key in seen or h.visual_score is None or h.visual_score < cfg.min_visual:
            continue
        if cfg.require_shareable and h.meta.get('shareable') != 'yes':
            continue
        seen.add(key)
        image = render(h)
        if image:
            images.append({**image, 'key': key})
        if len(images) >= cfg.max_images:
            break
    return images


def _complete(llm, prompt, images):
    if images:  # only vision calls pass `images`, so text-only LLM stubs keep working
        return llm.complete(SYSTEM + VISUAL_NOTE, prompt, images=images).strip()
    return llm.complete(SYSTEM, prompt).strip()


def answer_question(question, retriever, llm, router=None, subject=None, cfg=None):
    cfg = cfg or Config()
    rule = out_of_scope(question)
    if rule:
        return _refuse(question, f'out_of_scope:{rule}')

    subject = subject or (router.route(question) if router else None)
    hits = retriever.search(question, k=cfg.top_k, subject=subject)
    if not has_evidence(hits, cfg):
        return _refuse(question, 'no_evidence', subject, hits)

    images = page_images(hits, retriever, cfg)
    imaged = {im['key'] for im in images}
    # An image-only page whose image could not be rendered has nothing to cite: drop it.
    hits = [h for h in hits if h.text.strip() or (h.source, h.page) in imaged]
    if not has_evidence(hits, cfg):  # the dropped page may have been the only real evidence
        return _refuse(question, 'no_evidence', subject)

    prompt, used = build_user_prompt(question, hits, cfg.max_context_chars)
    images = [im for im in images if im['key'] in {(h.source, h.page) for h in used}]
    text = _complete(llm, prompt, images)
    if text.startswith(REFUSAL):
        return _refuse(question, 'model_refused', subject, used)

    check = validate(text, used)
    if not check.ok:  # one repair attempt, then fail closed
        text = _complete(llm, build_repair_prompt(prompt, text, check.problems()), images)
        if text.startswith(REFUSAL):
            return _refuse(question, 'model_refused', subject, used)
        check = validate(text, used)
        if not check.ok:
            return _refuse(question, 'invalid_citations', subject, used, check)
    return Answer(question, text, False, 'ok', subject, used, check)

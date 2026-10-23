"""Pure functions for the retrieval-only eval (no model, no API).

Text retrievers return chunks and ColPali returns pages, so every comparison here is done at
page level: hits are collapsed to unique (source, page) in rank order first. These numbers are
therefore not the chunk-level P@5 that run_eval reports for R3; they answer "did the right page
show up, and how high".
"""
import math

KS = (1, 5, 10)


def collapse_to_pages(hits):
    """Unique (source_lower, page) pairs in rank order, keeping the first (best) occurrence."""
    seen, pages = set(), []
    for h in hits:
        key = (h.source.lower(), h.page)
        if key not in seen:
            seen.add(key)
            pages.append(key)
    return pages


def _gold_set(gold):
    return {(g['source'].lower(), int(g['page'])) for g in gold}


def page_metrics(hits, gold, ks=KS):
    """Metrics for one answerable question. `gold` must list ALL relevant pages.

    recall@k  share of gold pages found in the top k pages
    p@k       share of the top-k slots holding a gold page (k, not len(found): short lists cost)
    hit@k     1 if any gold page is in the top k
    mrr       1 / rank of the first gold page (0 if none retrieved)
    ndcg@k    binary-relevance nDCG
    first_gold_rank  rank of the first gold page, or None
    """
    gold_pages = _gold_set(gold)
    if not gold_pages:
        raise ValueError('page_metrics needs at least one gold page')
    ranked = collapse_to_pages(hits)
    rel = [1 if p in gold_pages else 0 for p in ranked]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    out = {'mrr': 1.0 / first if first else 0.0, 'first_gold_rank': first}
    for k in ks:
        top = rel[:k]
        out[f'recall@{k}'] = sum(top) / len(gold_pages)
        out[f'p@{k}'] = sum(top) / k
        out[f'hit@{k}'] = 1.0 if any(top) else 0.0
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold_pages), k)))
        out[f'ndcg@{k}'] = dcg / ideal
    return out


def mean_metrics(rows, keys=None):
    """Average the numeric metrics over rows (dicts from page_metrics)."""
    if not rows:
        return {}
    keys = keys or [k for k in rows[0] if k != 'first_gold_rank']
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def calibrate_threshold(answerable_scores, refusal_scores, refusal_target=0.90):
    """Pick an evidence-gate threshold from top-1 scores of answerable vs should-refuse questions.

    t_refusal: the smallest threshold that keeps at least `refusal_target` of the should-refuse
               questions BELOW it (so the retriever alone cannot push them past the gate), which
               is what scope.md's R2 asks for. This is the recommendation.
    t_youden:  the threshold maximising TPR + TNR - 1, for comparison.
    A question with no score (nothing retrieved) can never clear the gate: it counts as blocked if
    it should refuse and as a miss if it is answerable. A pass means score >= threshold. Thresholds are rounded UP to 3 decimals so rounding can
    never let a should-refuse question through.
    """
    never = -math.inf
    ans = sorted(never if s is None else s for s in answerable_scores)
    ref = sorted(never if s is None else s for s in refusal_scores)
    if not ans or not ref:
        return {'t_refusal': None, 't_youden': None, 'n_answerable': len(ans), 'n_refusal': len(ref),
                'note': 'need both answerable and should-refuse questions'}

    def rates(t):
        tpr = sum(1 for s in ans if s >= t) / len(ans)       # answerable that clear the gate
        tnr = sum(1 for s in ref if s < t) / len(ref)        # should-refuse that are blocked
        return tpr, tnr

    def ceil3(x):
        return math.ceil(x * 1000) / 1000

    need = math.ceil(refusal_target * len(ref))              # how many refusals must be below t
    edge = ref[need - 1]
    t_ref = ceil3(math.nextafter(edge, math.inf)) if math.isfinite(edge) else 0.0
    candidates = sorted({s for s in ans + ref if math.isfinite(s)}) or [0.0]
    t_y = max(candidates, key=lambda t: (sum(rates(t)) - 1, t))
    out = {'n_answerable': len(ans), 'n_refusal': len(ref), 'refusal_target': refusal_target}
    for name, t in (('t_refusal', t_ref), ('t_youden', ceil3(t_y))):
        tpr, tnr = rates(t)
        out[name] = t
        out[f'{name}_answerable_pass'] = tpr
        out[f'{name}_refusal_blocked'] = tnr
    return out


def compare_reports(base, new, modes=None, keys=('recall@5', 'mrr', 'hit@1', 'ndcg@5')):
    """Per-mode metric deltas (new - base) between two retrieval reports."""
    out = {}
    for mode in (modes or [m for m in base['modes'] if m in new['modes']]):
        b, n = base['modes'][mode]['overall'], new['modes'][mode]['overall']
        out[mode] = {k: {'base': b.get(k), 'new': n.get(k),
                         'delta': None if b.get(k) is None or n.get(k) is None else n[k] - b[k]}
                     for k in keys}
    return out


def regressions(comparison, mode, keys=('recall@5', 'mrr'), tolerance=0.02):
    """Metrics of `mode` that dropped by more than `tolerance`."""
    if mode not in comparison:
        return [f'{mode}: missing from one of the reports']
    return [f'{mode} {k} {v["base"]:.3f} -> {v["new"]:.3f}'
            for k, v in comparison[mode].items()
            if k in keys and v['delta'] is not None and v['delta'] < -tolerance]

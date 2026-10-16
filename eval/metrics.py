"""Pure metric functions for the Week-1 targets in scope.md (R1-R4)."""
import re

TARGETS = {'R1_citations': 1.0, 'R2_refusal': 0.90, 'R3_p_at_5': 0.70, 'R4_faithfulness': 4.0}


def precision_at_k(hits, gold, k=5, page_tol=0):
    """Share of the top-k chunks sitting on a gold page. `gold` should list ALL relevant pages."""
    top = hits[:k]
    if not top or not gold:
        return 0.0
    rel = sum(1 for h in top if any(
        h.source.lower() == g['source'].lower() and abs(h.page - g['page']) <= page_tol
        for g in gold))
    return rel / k


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows):
    """rows: dicts with should_refuse, refused, citations_ok, p_at_k, faithfulness."""
    answered = [r for r in rows if not r['refused']]
    must_refuse = [r for r in rows if r['should_refuse']]
    answerable = [r for r in rows if not r['should_refuse']]
    return {
        'n': len(rows),
        'R1_citations': _mean([1.0 if r['citations_ok'] else 0.0 for r in answered]),
        'R2_refusal': _mean([1.0 if r['refused'] else 0.0 for r in must_refuse]),
        'false_refusal': _mean([1.0 if r['refused'] else 0.0 for r in answerable]),
        'R3_p_at_5': _mean([r['p_at_k'] for r in answerable if r['p_at_k'] is not None]),
        'R4_faithfulness': _mean([r['faithfulness'] for r in answered]),
    }


def check_targets(summary):
    """Return {metric: bool}. A metric with no data fails."""
    return {k: (summary.get(k) is not None and summary[k] >= t) for k, t in TARGETS.items()}


JUDGE_SYSTEM = ('You grade answers for faithfulness to source text. Score 1-5: 5 = every statement '
                'is supported by the context; 1 = mostly unsupported. Reply as "Score: N".')


def judge_faithfulness(llm, question, answer, hits):
    ctx = '\n\n'.join(f'[{h.source}, {h.page}]\n{h.text}' for h in hits)
    reply = llm.complete(JUDGE_SYSTEM, f'Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:\n{answer}',
                         max_tokens=20)
    m = re.search(r'\b([1-5])\b', reply)
    return int(m.group(1)) if m else None

"""Run the frozen eval set and compare against the targets in scope.md.

    python -m eval.run_eval --questions eval/source/questions.jsonl --out eval/report.json

Question format, one JSON object per line (see eval/questions.example.jsonl):
  id, subject, question, should_refuse (bool), gold: [{"source": file, "page": n}, ...]
Keep real questions and past papers in eval/source/ (git-ignored, never indexed).
"""
import argparse
import json
import sys

from dotenv import load_dotenv

from app.config import Config
from app.pipeline import answer_question
from eval.metrics import aggregate, check_targets, judge_faithfulness, precision_at_k


def run(questions, retriever, llm, router, cfg, judge=True):
    rows = []
    for q in questions:
        a = answer_question(q['question'], retriever, llm, router, q.get('subject'), cfg)
        refuse_expected = bool(q.get('should_refuse'))
        rows.append({
            'id': q['id'], 'subject': q.get('subject'), 'should_refuse': refuse_expected,
            'refused': a.refused, 'reason': a.reason,
            'citations_ok': bool(a.validation and a.validation.ok) if not a.refused else None,
            'p_at_k': None if refuse_expected else precision_at_k(a.hits, q.get('gold', []), cfg.top_k),
            'faithfulness': (judge_faithfulness(llm, q['question'], a.text, a.hits)
                             if judge and not a.refused else None),
            'top_dense': max((h.dense_sim for h in a.hits if h.dense_sim is not None), default=None),
            'top_bm25': max((h.bm25_score for h in a.hits if h.bm25_score is not None), default=None),
            'top_visual': max((h.visual_score for h in a.hits if h.visual_score is not None),
                              default=None),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--questions', default='eval/source/questions.jsonl')
    ap.add_argument('--out', default='eval/report.json')
    ap.add_argument('--no-judge', action='store_true', help='skip the LLM faithfulness judge (R4)')
    ap.add_argument('--strict', action='store_true', help='exit 1 if any target is missed')
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Config.from_env()
    from app.cli import build
    retriever, llm, router = build(cfg)
    with open(args.questions, encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()]

    rows = run(questions, retriever, llm, router, cfg, judge=not args.no_judge)
    summary = aggregate(rows)
    passed = check_targets(summary)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'passed': passed, 'rows': rows}, f, indent=2)

    for k, ok in passed.items():
        v = summary.get(k)
        print(f"{'PASS' if ok else 'FAIL'}  {k:16s} {'n/a' if v is None else round(v, 3)}")
    print('false_refusal   ', summary['false_refusal'])
    return 1 if args.strict and not all(passed.values()) else 0


if __name__ == '__main__':
    sys.exit(main())

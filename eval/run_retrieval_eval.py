"""Retrieval-only eval: which retriever puts the gold page highest? No LLM, no API key, cheap.

    python -m eval.run_retrieval_eval                          # all modes the index supports
    python -m eval.run_retrieval_eval --modes hybrid,hybrid+visual
    python -m eval.run_retrieval_eval --adapter adapters/ee-colpali --visual-dir vectordb_visual_ft \\
        --out eval/retrieval_report_ft.json                    # score a fine-tuned adapter
    python -m eval.run_retrieval_eval --compare eval/retrieval_report.json \\
        eval/retrieval_report_ft.json --strict                 # did fine-tuning help?

Uses the same question file as eval.run_eval (id, subject, question, should_refuse, gold).
Answerable questions give the ranking metrics (page-level, see eval/retrieval_metrics.py).
should_refuse questions are used only to calibrate COPILOT_MIN_VISUAL.
The question set is the frozen eval set: never train on it (training.data enforces that).
"""
import argparse
import dataclasses
import json
import sys

from dotenv import load_dotenv

from app.config import Config
from eval.metrics import precision_at_k
from eval.retrieval_metrics import (KS, calibrate_threshold, collapse_to_pages, compare_reports,
                                    mean_metrics, page_metrics, regressions)

MODES = ('bm25', 'dense', 'hybrid', 'visual', 'hybrid+visual')
VISUAL_MODES = ('visual', 'hybrid+visual')
TEXT_MODES = ('bm25', 'dense', 'hybrid', 'hybrid+visual')
TABLE_KEYS = ('recall@5', 'mrr', 'hit@1', 'ndcg@5', 'p@5')


class Shareable:
    """Wrap a single retriever so it obeys the same shareable-only rule as the pipeline."""

    def __init__(self, inner, require_shareable=True):
        self.inner, self.require = inner, require_shareable

    def search(self, query, k=5, subject=None):
        hits = self.inner.search(query, k, subject)
        return [h for h in hits if h.meta.get('shareable') == 'yes'] if self.require else hits


def load_questions(path):
    with open(path, encoding='utf-8') as f:
        qs = [json.loads(line) for line in f if line.strip()]
    missing = [q.get('id', '?') for q in qs if not q.get('should_refuse') and not q.get('gold')]
    if missing:
        raise ValueError(f'answerable questions need gold pages: {missing}')
    return qs


def run_questions(questions, retrievers, router=None, depth=30, ks=KS):
    """Return rows: one per (question, mode). `retrievers` maps mode -> object with .search()."""
    rows = []
    for q in questions:
        subject = q.get('subject') or (router.route(q['question']) if router else None)
        refuse = bool(q.get('should_refuse'))
        for mode, r in retrievers.items():
            hits = r.search(q['question'], depth, subject)
            row = {'id': q['id'], 'subject': q.get('subject'), 'mode': mode,
                   'should_refuse': refuse,
                   'top_pages': [f'{s}:{p}' for s, p in collapse_to_pages(hits)[:5]],
                   'top_visual': max((h.visual_score for h in hits if h.visual_score is not None),
                                     default=None)}
            if not refuse:
                row['metrics'] = page_metrics(hits, q['gold'], ks)
                row['chunk_p@5'] = precision_at_k(hits, q['gold'], 5)  # same definition as R3
            rows.append(row)
    return rows


def summarize(rows):
    """{mode: {'overall': metrics, 'per_subject': {subject: metrics}, 'n': answerable count}}"""
    out = {}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        mine = [r for r in rows if r['mode'] == mode and not r['should_refuse']]
        by_subject = {}
        for r in mine:
            by_subject.setdefault(r['subject'] or 'unknown', []).append(r['metrics'])
        overall = mean_metrics([r['metrics'] for r in mine])
        if mine:
            overall['chunk_p@5'] = sum(r['chunk_p@5'] for r in mine) / len(mine)
        out[mode] = {'n': len(mine), 'overall': overall,
                     'per_subject': {s: mean_metrics(ms) for s, ms in sorted(by_subject.items())}}
    return out


def calibration(rows):
    """min_visual suggestion from the visual-only mode (None if that mode did not run)."""
    vis = [r for r in rows if r['mode'] == 'visual']
    if not vis:
        return None
    return calibrate_threshold([r['top_visual'] for r in vis if not r['should_refuse']],
                               [r['top_visual'] for r in vis if r['should_refuse']])


def build_retrievers(cfg, modes):
    """Construct only what the requested modes need (the visual model is the expensive part)."""
    from retrieval.bm25 import BM25Index
    from retrieval.hybrid import HybridRetriever
    bm25 = BM25Index.from_jsonl(cfg.chunks_path) if set(modes) & set(TEXT_MODES) else None
    dense = None
    if set(modes) & {'dense', 'hybrid', 'hybrid+visual'}:
        from retrieval.dense import DenseIndex
        dense = DenseIndex.from_config(cfg)
    visual = None
    if set(modes) & set(VISUAL_MODES):
        from retrieval.visual import VisualIndex
        visual = VisualIndex.from_config(cfg)

    def hybrid(with_visual):
        return HybridRetriever(bm25, dense, pool=cfg.pool_size, rrf_k=cfg.rrf_k,
                               require_shareable=cfg.require_shareable,
                               visual=visual if with_visual else None, visual_pool=cfg.visual_pool)
    wrap = lambda r: Shareable(r, cfg.require_shareable)
    table = {'bm25': lambda: wrap(bm25), 'dense': lambda: wrap(dense),
             'hybrid': lambda: hybrid(False), 'visual': lambda: wrap(visual),
             'hybrid+visual': lambda: hybrid(True)}
    return {m: table[m]() for m in modes}


def print_table(summary):
    print(f"{'mode':16s}" + ''.join(f'{k:>10s}' for k in TABLE_KEYS) + f"{'n':>6s}")
    for mode, s in summary.items():
        o = s['overall']
        print(f'{mode:16s}' + ''.join(f"{o.get(k, float('nan')):10.3f}" for k in TABLE_KEYS)
              + f"{s['n']:6d}")


def print_comparison(comp):
    for mode, metrics in comp.items():
        print(mode)
        for k, v in metrics.items():
            d = 'n/a' if v['delta'] is None else f"{v['delta']:+.3f}"
            print(f"  {k:10s} {v['base']:.3f} -> {v['new']:.3f}  ({d})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--questions', default='eval/source/questions.jsonl')
    ap.add_argument('--out', default='eval/retrieval_report.json')
    ap.add_argument('--modes', help=f'comma list from: {", ".join(MODES)} (default: all)')
    ap.add_argument('--depth', type=int, default=30, help='chunks/pages requested per query')
    ap.add_argument('--adapter', help='LoRA adapter dir (overrides COPILOT_VISUAL_ADAPTER)')
    ap.add_argument('--visual-dir', help='visual index dir (overrides COPILOT_VISUAL_DIR)')
    ap.add_argument('--visual-model', help='overrides COPILOT_VISUAL_MODEL')
    ap.add_argument('--compare', nargs=2, metavar=('BASE', 'NEW'),
                    help='compare two reports instead of running')
    ap.add_argument('--mode', default='hybrid+visual', help='mode checked by --strict (compare)')
    ap.add_argument('--tolerance', type=float, default=0.02, help='allowed drop before --strict fails')
    ap.add_argument('--strict', action='store_true', help='compare: exit 1 on a regression')
    args = ap.parse_args(argv)

    if args.compare:
        base, new = (json.load(open(p, encoding='utf-8')) for p in args.compare)
        comp = compare_reports(base, new)
        print_comparison(comp)
        bad = regressions(comp, args.mode, tolerance=args.tolerance)
        for b in bad:
            print('REGRESSION', b)
        return 1 if args.strict and bad else 0

    load_dotenv()
    cfg = Config.from_env()
    over = {k: v for k, v in (('visual_adapter', args.adapter), ('visual_dir', args.visual_dir),
                              ('visual_model', args.visual_model)) if v}
    cfg = dataclasses.replace(cfg, **over)
    modes = args.modes.split(',') if args.modes else list(MODES)
    unknown = set(modes) - set(MODES)
    if unknown:
        ap.error(f'unknown modes: {sorted(unknown)}')

    from retrieval.router import SubjectRouter
    router = SubjectRouter.from_file(cfg.syllabus_path)
    questions = load_questions(args.questions)
    try:
        retrievers = build_retrievers(cfg, modes)
    except FileNotFoundError as exc:     # e.g. no visual index yet
        print(f'cannot build retrievers: {exc}', file=sys.stderr)
        return 2

    rows = run_questions(questions, retrievers, router, args.depth)
    summary = summarize(rows)
    calib = calibration(rows)
    from retrieval.colpali_model import model_tag
    report = {
        'config': {'visual_model': model_tag(cfg.visual_model, cfg.visual_adapter or None),
                   'visual_dir': cfg.visual_dir, 'depth': args.depth, 'modes': modes},
        'n_questions': len(questions),
        'n_answerable': sum(1 for q in questions if not q.get('should_refuse')),
        'modes': summary, 'calibration': calib, 'rows': rows}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print_table(summary)
    if calib and calib.get('t_refusal') is not None:
        print(f"\nCOPILOT_MIN_VISUAL suggestion: {calib['t_refusal']} "
              f"(blocks {calib['t_refusal_refusal_blocked']:.0%} of should-refuse questions, "
              f"{calib['t_refusal_answerable_pass']:.0%} of answerable ones still clear it; "
              f"Youden alternative {calib['t_youden']})")
    elif calib:
        print(f"\ncalibration skipped: {calib['note']}")
    print(f'report: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

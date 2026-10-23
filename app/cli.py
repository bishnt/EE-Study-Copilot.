"""python -m app.cli "state the ROC of a right-sided sequence" [--subject dsp] [--sources]"""
import argparse
import dataclasses
import sys

from dotenv import load_dotenv

from app.config import SUBJECTS, Config
from app.pipeline import answer_question


def build(cfg):
    from generation.llm import AnthropicLLM
    from retrieval.bm25 import BM25Index
    from retrieval.dense import DenseIndex
    from retrieval.hybrid import HybridRetriever
    from retrieval.router import SubjectRouter
    visual = renderer = None
    if cfg.visual_enabled:  # ColPali page retrieval + page images for the model
        from ingestion.render import make_page_renderer
        from retrieval.visual import VisualIndex
        visual = VisualIndex.from_config(cfg)
        renderer = make_page_renderer(cfg.corpus_dir)
    retriever = HybridRetriever(BM25Index.from_jsonl(cfg.chunks_path), DenseIndex.from_config(cfg),
                                pool=cfg.pool_size, rrf_k=cfg.rrf_k,
                                require_shareable=cfg.require_shareable,
                                visual=visual, visual_pool=cfg.visual_pool,
                                page_renderer=renderer)
    return retriever, AnthropicLLM(cfg.llm_model), SubjectRouter.from_file(cfg.syllabus_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('question', nargs='?', help='omit for an interactive prompt')
    ap.add_argument('--subject', choices=SUBJECTS)
    ap.add_argument('--sources', action='store_true', help='list retrieved chunks')
    ap.add_argument('--visual', action='store_true',
                    help='use ColPali page retrieval (same as COPILOT_VISUAL=1)')
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Config.from_env()
    if args.visual:
        cfg = dataclasses.replace(cfg, visual_enabled=True)
    retriever, llm, router = build(cfg)

    def ask(q):
        a = answer_question(q, retriever, llm, router, args.subject, cfg)
        print(a.text)
        if args.sources:
            print(f'\n[reason={a.reason} subject={a.subject}]')
            for h in a.hits:
                print(f'  {h.source} p.{h.page} dense={h.dense_sim} bm25={h.bm25_score}'
                      f' visual={h.visual_score}')

    if args.question:
        ask(args.question)
        return 0
    try:
        while True:
            q = input('\nquestion> ').strip()
            if q:
                ask(q)
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == '__main__':
    sys.exit(main())

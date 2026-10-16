"""python -m app.cli "state the ROC of a right-sided sequence" [--subject dsp] [--sources]"""
import argparse
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
    retriever = HybridRetriever(BM25Index.from_jsonl(cfg.chunks_path), DenseIndex.from_config(cfg),
                                pool=cfg.pool_size, rrf_k=cfg.rrf_k,
                                require_shareable=cfg.require_shareable)
    return retriever, AnthropicLLM(cfg.llm_model), SubjectRouter.from_file(cfg.syllabus_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('question', nargs='?', help='omit for an interactive prompt')
    ap.add_argument('--subject', choices=SUBJECTS)
    ap.add_argument('--sources', action='store_true', help='list retrieved chunks')
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Config.from_env()
    retriever, llm, router = build(cfg)

    def ask(q):
        a = answer_question(q, retriever, llm, router, args.subject, cfg)
        print(a.text)
        if args.sources:
            print(f'\n[reason={a.reason} subject={a.subject}]')
            for h in a.hits:
                print(f'  {h.source} p.{h.page} dense={h.dense_sim} bm25={h.bm25_score}')

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

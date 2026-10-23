"""Fine-tune a ColPali-family retriever on your corpus with LoRA.

    python -m training.make_pairs --out train/pairs.jsonl           # 1. questions -> pages (Claude)
    python -m training.train_colpali --pairs train/pairs.jsonl \\
        --out adapters/ee-colpali --epochs 2                        # 2. train (GPU)
    python -m ingestion.build_visual --class --adapter adapters/ee-colpali \\
        --out vectordb_visual_ft                                    # 3. re-embed with the adapter
    python -m eval.run_retrieval_eval --adapter adapters/ee-colpali \\
        --visual-dir vectordb_visual_ft --out eval/retrieval_report_ft.json
    python -m eval.run_retrieval_eval --compare eval/retrieval_report.json \\
        eval/retrieval_report_ft.json --strict                      # 4. adopt only if it helps

What it does: in-batch contrastive training (the library's ColbertLoss) where each query's page
is the positive and every other page in the batch is a negative. Batches never contain a page
twice. After each epoch it measures recall on held-out pages and keeps the best epoch.

Honest expectations: ColPali is already strong, a few thousand synthetic pairs from a small corpus
can overfit, and a validation gain here does not prove an end-to-end gain. The frozen eval
(eval.run_retrieval_eval) is the judge. Use --dry-run first: it validates the data and prints the
plan without loading any model.

"""
import argparse
import json
import math
import random
import re
import sys
import time
import uuid
from pathlib import Path

from training import data as D

# LoRA on the language model's attention + MLP projections and the final embedding projection.
# The vision tower is excluded by name. Verified at runtime: training aborts if nothing matches.
TARGET_REGEX = (r'^(?!.*(vision|visual|siglip)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
                r'|.*custom_text_proj$')
SAVE_KEY = ('recall@5', 'mrr')


def find_lora_targets(model, regex=TARGET_REGEX):
    """Names of nn.Linear modules the LoRA regex will wrap. Raises if there are none."""
    import torch
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and re.fullmatch(regex, n)]
    if not names:
        raise ValueError('LoRA target regex matched no Linear layers; this model family needs '
                         'a different TARGET_REGEX. First modules: '
                         + ', '.join(n for n, _ in list(model.named_modules())[:8]))
    return names


class PairCollator:
    """(query, image) examples -> model inputs, keys prefixed query_ / doc_.

    Mirrors colpali_engine's VisualRetrieverCollator (which needs the heavy `datasets` extra):
    queries go through processor.process_queries (prefix + augmentation tokens), pages through
    process_images, and ColPali's tokenizer pads on the right.
    """

    def __init__(self, processor):
        self.processor = processor
        tok = getattr(processor, 'tokenizer', None)
        if tok is not None and type(processor).__name__ == 'ColPaliProcessor':
            tok.padding_side = 'right'

    def __call__(self, examples):
        queries = self.processor.process_queries([e['query'] for e in examples])
        pages = self.processor.process_images([e['image'] for e in examples])
        return {**{f'query_{k}': v for k, v in queries.items()},
                **{f'doc_{k}': v for k, v in pages.items()}}


def _split_inputs(batch, prefix, device, dtype):
    out = {}
    for k, v in batch.items():
        if k.startswith(prefix):
            v = v.to(device)
            if k.endswith('pixel_values') and v.is_floating_point():
                v = v.to(dtype)
            out[k[len(prefix):]] = v
    return out


def evaluate_pairs(val_pairs, page_vectors, embed_query, ks=(1, 5, 10)):
    """Recall/MRR of val queries over a pool of pages ({(source, page): vectors}).

    Uses the same VisualIndex MaxSim as the app, so training-time numbers mean what serving does.
    """

    import numpy as np

    from eval.retrieval_metrics import mean_metrics, page_metrics
    from retrieval.visual import VisualIndex
    keys = list(page_vectors)
    pages = [{'id': f'{s}:{p}', 'source': s, 'subject': 'pool', 'page': p, 'shareable': 'yes',
              'text': ''} for s, p in keys]
    vecs = [np.asarray(page_vectors[k], dtype=np.float16) for k in keys]
    offsets = np.concatenate([[0], np.cumsum([len(v) for v in vecs])])
    index = VisualIndex(pages, np.concatenate(vecs), offsets, embed_query)
    rows = [page_metrics(index.search(p['query'], k=max(ks)),
                         [{'source': p['source'], 'page': p['page']}], ks) for p in val_pairs]
    return mean_metrics(rows)


def fit(model, train_pairs, make_example, collate, loss_fn, *, epochs=2, batch_size=8, lr=2e-5,
        weight_decay=0.01, warmup=0.1, grad_accum=1, max_grad_norm=1.0, seed=0,
        eval_fn=None, save_fn=None, log=print, log_every=10):
    """Train `model`; return {'baseline', 'history', 'best_epoch', 'steps'}.

    eval_fn(model) -> metrics dict (called before training and after each epoch);
    save_fn(model, epoch, metrics) is called whenever SAVE_KEY improves (every epoch if no eval_fn).
    """
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('no trainable parameters: LoRA was not applied')
    device = params[0].device
    base_dtype = next((p.dtype for p in model.parameters() if not p.requires_grad), params[0].dtype)

    per_epoch = len(D.unique_page_batches(train_pairs, batch_size, seed, 0))
    if per_epoch == 0:
        raise ValueError(f'no batches: need at least {batch_size} distinct pages')
    total = math.ceil(per_epoch / grad_accum) * epochs
    warm = max(1, int(warmup * total))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    torch.manual_seed(seed)

    baseline = eval_fn(model) if eval_fn else None
    if baseline:
        log(f'baseline (before training): {_fmt(baseline)}')
    history, best, best_epoch, step, t0 = [], None, None, 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses, opt_step = [], 0
        for i, idxs in enumerate(D.unique_page_batches(train_pairs, batch_size, seed, epoch)):
            batch = collate([make_example(train_pairs[j]) for j in idxs])
            q = model(**_split_inputs(batch, 'query_', device, base_dtype))
            d = model(**_split_inputs(batch, 'doc_', device, base_dtype))
            loss = loss_fn(q.float(), d.float())
            (loss / grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % grad_accum == 0 or i + 1 == per_epoch:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                opt_step += 1
                if opt_step % log_every == 0:
                    log(f'epoch {epoch} step {opt_step}: loss {sum(losses[-log_every:]) / len(losses[-log_every:]):.4f} '
                        f'lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)')
        metrics = eval_fn(model) if eval_fn else None
        entry = {'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'val': metrics}
        history.append(entry)
        log(f'epoch {epoch}: train loss {entry["train_loss"]:.4f}'
            + (f' | val {_fmt(metrics)}' if metrics else ''))
        score = tuple(metrics[k] for k in SAVE_KEY) if metrics else (epoch,)
        if best is None or score > best:
            best, best_epoch = score, epoch
            if save_fn:
                save_fn(model, epoch, metrics)
    return {'baseline': baseline, 'history': history, 'best_epoch': best_epoch, 'steps': step}


def _fmt(m):
    return ' '.join(f'{k}={m[k]:.3f}' for k in ('recall@1', 'recall@5', 'mrr') if k in m)


def prepare_data(args, manifest, log=print):
    """Load, validate and guard the pairs. Returns (train, val, report dict)."""
    pairs = D.dedupe(D.load_pairs(args.pairs))
    report = {'loaded': len(pairs)}
    pairs, dropped = D.shareable_guard(pairs, manifest, args.include_non_shareable)
    report['dropped_not_shareable'] = len(dropped)

    eval_path = Path(args.eval_questions)
    if eval_path.is_file():
        questions = D.load_eval_questions(eval_path)
        pairs, dropped = D.eval_guard(pairs, questions, exclude_eval_pages=not args.allow_eval_pages)
        report['dropped_eval_overlap'] = sum(1 for _, r in dropped if r == 'eval_question_overlap')
        report['dropped_eval_gold_page'] = sum(1 for _, r in dropped if r == 'eval_gold_page')
    elif args.no_eval_guard:
        log(f'WARNING: {eval_path} not found; training WITHOUT the eval leak guard.')
    else:
        raise SystemExit(f'{eval_path} not found. The leak guard needs the frozen eval questions '
                         '(--eval-questions PATH), or pass --no-eval-guard to proceed unguarded.')
    train, val = D.split_pairs(pairs, args.val_fraction, args.seed)
    report.update(train=len(train), val=len(val), kept=len(pairs))
    return train, val, report


def _page_image(corpus_dir, pair, max_side):
    from PIL import Image

    from ingestion.render import render_page_png
    import io
    root = Path(corpus_dir).resolve()
    path = (root / pair['subject'] / pair['source']).resolve()
    png = render_page_png(path, pair['page'], max_side) if path.is_relative_to(root) else None
    if png is None:
        raise RuntimeError(f"cannot render {pair['subject']}/{pair['source']} p.{pair['page']}")
    return Image.open(io.BytesIO(png)).convert('RGB')


def main(argv=None):
    import pandas as pd
    from dotenv import load_dotenv

    from app.config import Config
    load_dotenv()
    cfg = Config.from_env()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', default='train/pairs.jsonl')
    ap.add_argument('--out', default='adapters/ee-colpali')
    ap.add_argument('--model', default=cfg.visual_model)
    ap.add_argument('--manifest', default=f'{cfg.corpus_dir}/manifest.csv')
    ap.add_argument('--corpus', default=cfg.corpus_dir)
    ap.add_argument('--eval-questions', default='eval/source/questions.jsonl')
    ap.add_argument('--allow-eval-pages', action='store_true',
                    help='also train on the eval questions\' gold pages (eval will be optimistic)')
    ap.add_argument('--no-eval-guard', action='store_true', help='proceed without the eval file')
    ap.add_argument('--include-non-shareable', action='store_true',
                    help='train on shareable=no pages too (weights then encode that content)')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=8, help='= negatives per query; more is better')
    ap.add_argument('--grad-accum', type=int, default=1,
                    help='does NOT add negatives, only smooths the gradient')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--lora-r', type=int, default=32)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--val-fraction', type=float, default=0.1)
    ap.add_argument('--pool-distractors', type=int, default=150,
                    help='extra training pages mixed into the validation pool')
    ap.add_argument('--max-side', type=int, default=1024)
    ap.add_argument('--grad-checkpointing', action='store_true', help='save memory, cost speed')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='validate data, print the plan, stop')
    args = ap.parse_args(argv)
    if args.batch_size < 2:
        ap.error('--batch-size must be at least 2')

    manifest = pd.read_csv(args.manifest).to_dict('records')
    train, val, report = prepare_data(args, manifest)
    n_pages = len({D.page_key(p) for p in train})
    batches = len(D.unique_page_batches(train, args.batch_size, args.seed, 0))
    print('data:', json.dumps(report))
    print('train:', D.describe(train))
    print('val:  ', D.describe(val) if val else 'none (too few pages for a held-out split)')
    print(f'plan: {args.epochs} epochs x {batches} batches of {args.batch_size} '
          f'({n_pages} distinct training pages)')
    if n_pages < args.batch_size:
        raise SystemExit(f'need at least --batch-size ({args.batch_size}) distinct training pages')
    if args.dry_run:
        return 0

    import torch
    from colpali_engine.loss import ColbertLoss
    from peft import get_peft_model, LoraConfig

    from retrieval import colpali_model
    model, processor = colpali_model.load(args.model, trainable=True)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    targets = find_lora_targets(model)
    print(f'LoRA on {len(targets)} linear layers')
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout, target_modules=TARGET_REGEX))
    model.print_trainable_parameters()

    def image_of(pair):
        return _page_image(args.corpus, pair, args.max_side)

    @torch.no_grad()
    def eval_fn(m):
        m.eval()
        rng = random.Random(args.seed)
        pool_keys = {D.page_key(p): p for p in val}
        extra = sorted({D.page_key(p): p for p in train}.items())
        rng.shuffle(extra)
        for k, p in extra[:args.pool_distractors]:
            pool_keys.setdefault(k, p)
        vectors = {}
        for k, p in pool_keys.items():
            vectors[k] = colpali_model.embed_images(m, processor, [image_of(p)], 1)[0]
        metrics = evaluate_pairs(val, vectors, lambda q: colpali_model.embed_query(m, processor, q))
        m.train()
        return metrics

    run_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}'
    out = Path(args.out)
    meta = {'run_id': run_id, 'base_model': args.model, 'args': vars(args), 'data': report,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S')}

    def save_fn(m, epoch, metrics):
        out.mkdir(parents=True, exist_ok=True)
        m.save_pretrained(out)
        meta.update(epoch=epoch, val=metrics)
        (out / 'copilot_meta.json').write_text(json.dumps(meta, indent=2, default=str))
        print(f'saved adapter (epoch {epoch}) -> {out}')

    result = fit(model, train, lambda p: {'query': p['query'], 'image': image_of(p)},
                 PairCollator(processor), ColbertLoss(), epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, grad_accum=args.grad_accum, seed=args.seed,
                 eval_fn=eval_fn if val else None, save_fn=save_fn)
    meta.update(baseline=result['baseline'], history=result['history'], best_epoch=result['best_epoch'])
    (out / 'copilot_meta.json').write_text(json.dumps(meta, indent=2, default=str))

    base, best = result['baseline'], next(
        (h['val'] for h in result['history'] if h['epoch'] == result['best_epoch']), None)
    if base and best and tuple(best[k] for k in SAVE_KEY) <= tuple(base[k] for k in SAVE_KEY):
        print('\nWARNING: no validation gain over the base model. Do not adopt this adapter; '
              'more/better pairs, a lower lr or fewer epochs may help.')
    print(f'\nadapter: {out} (run {run_id}). Next: rebuild the visual index with --adapter and '
          'run eval.run_retrieval_eval; adopt only if the frozen eval improves.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

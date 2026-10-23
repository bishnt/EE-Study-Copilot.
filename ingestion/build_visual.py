"""Build the ColPali page-image index.

    python -m ingestion.build_visual --class     # class index: shareable=yes only (what the app uses)
    python -m ingestion.build_visual             # personal index: every manifest document

Each PDF page is rendered to an image and embedded as ~1000 patch vectors. Embedding is the slow
step (use a GPU; ColPali-3B on CPU takes many seconds per page), so every document is saved as a
shard in <out>/.shards and skipped on re-runs. Delete a shard to re-embed that document.
The manifest, not the shard, decides each page's `shareable` flag, so editing the manifest and
re-running never needs a re-embed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import VISUAL_MODEL_NAME
from ingestion.render import iter_pages
from retrieval.colpali_model import model_tag
from retrieval.visual import MAX_TEXT_CHARS, page_id, write_index


def _shard_path(out, subject, file):
    key = hashlib.sha1(f'{subject}/{file}'.encode()).hexdigest()[:16]
    return Path(out) / '.shards' / f'{key}.npz'


def _save_shard(path, subject, file, model, pages, texts, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, model=np.array(model), doc=np.array(f'{subject}/{file}'),
             pages=np.array(pages, dtype=np.int64), texts=np.array(json.dumps(texts)),
             lengths=np.array([len(v) for v in vectors], dtype=np.int64),
             vectors=np.concatenate(vectors).astype(np.float16))
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written shard


def _load_shard(path, subject, file, model):
    """Return (pages, texts, vectors) or None if absent, stale or built with another model."""
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z['model']) != model or str(z['doc']) != f'{subject}/{file}':
            return None
        cuts = np.cumsum(z['lengths'])[:-1]
        return (z['pages'].tolist(), json.loads(str(z['texts'])),
                np.split(z['vectors'], cuts))
    except (OSError, ValueError, KeyError):
        return None  # corrupt shard: re-embed


def _default_embedder(model_name, adapter, batch_size):
    from retrieval import colpali_model  # lazy: pulls in torch
    model, processor = colpali_model.load(model_name, adapter=adapter)
    return lambda images: colpali_model.embed_images(model, processor, images, batch_size)


def _embed_document(pdf, embed, batch_size, max_side):
    from PIL import Image
    pages, texts, vectors, buf = [], [], [], []

    def flush():
        images = [Image.open(io.BytesIO(png)).convert('RGB') for _, png, _ in buf]
        for (num, _, text), vec in zip(buf, embed(images)):
            pages.append(num)
            texts.append(text.strip()[:MAX_TEXT_CHARS])
            vectors.append(np.asarray(vec, dtype=np.float16))
        buf.clear()

    for item in iter_pages(pdf, max_side):
        buf.append(item)
        if len(buf) >= batch_size:
            flush()
    if buf:
        flush()
    return pages, texts, vectors


def build(manifest='corpus/manifest.csv', corpus='corpus', out='vectordb_visual',
          shareable_only=False, model_name=VISUAL_MODEL_NAME, batch_size=2, max_side=1024,
          subjects=None, embedder=None, log=print, adapter=None):
    """Embed every selected document, then write the index. `embedder(images) -> vectors` is
    injectable so tests can run without torch."""
    df = pd.read_csv(manifest)
    if shareable_only:
        df = df[df['shareable'] == 'yes']
    if subjects:
        df = df[df['subject'].isin(subjects)]
    if df.empty:
        raise SystemExit('Nothing to index: the manifest selection is empty.')

    missing = [str(Path(corpus) / r['subject'] / r['file']) for _, r in df.iterrows()
               if not (Path(corpus) / r['subject'] / r['file']).is_file()]
    if missing:
        raise SystemExit('Missing PDFs (run python -m ingestion.validate_corpus):\n'
                         + '\n'.join(missing))

    tag = model_tag(model_name, adapter)   # shards and the index are tied to model + adapter
    embed = embedder
    all_pages, all_vectors = [], []
    for _, r in df.iterrows():
        subject, file = r['subject'], r['file']
        shard = _shard_path(out, subject, file)
        got = _load_shard(shard, subject, file, tag)
        if got is None:
            if embed is None:  # load the model only when something actually needs embedding
                embed = _default_embedder(model_name, adapter, batch_size)
            log(f'embedding {subject}/{file}')
            got = _embed_document(Path(corpus) / subject / file, embed, batch_size, max_side)
            _save_shard(shard, subject, file, tag, *got)
        else:
            log(f'cached    {subject}/{file}')
        nums, texts, vectors = got
        for num, text, vec in zip(nums, texts, vectors):
            all_pages.append({'id': page_id(file, num), 'source': file, 'subject': subject,
                              'category': str(r['category']), 'page': int(num),
                              'shareable': str(r['shareable']), 'text': text})
            all_vectors.append(vec)

    write_index(out, all_pages, all_vectors, tag)
    log(f'visual index: {len(all_pages)} pages from {len(df)} documents -> {out}')
    return len(all_pages)


def main(argv=None):
    from dotenv import load_dotenv
    from app.config import Config
    load_dotenv()
    cfg = Config.from_env()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--class', dest='class_mode', action='store_true',
                    help='shareable=yes documents only; writes COPILOT_VISUAL_DIR '
                         f'(default {cfg.visual_dir})')
    ap.add_argument('--out', help='output directory (default: class index dir, or vectordb_visual)')
    ap.add_argument('--model', default=cfg.visual_model)
    ap.add_argument('--adapter', default=cfg.visual_adapter or None,
                    help='LoRA adapter dir from training.train_colpali (default COPILOT_VISUAL_ADAPTER)')
    ap.add_argument('--subject', action='append', help='limit to a subject (repeatable)')
    ap.add_argument('--batch-size', type=int, default=2, help='pages per forward pass')
    ap.add_argument('--max-side', type=int, default=1024, help='page image longest side, px')
    ap.add_argument('--manifest', default=f'{cfg.corpus_dir}/manifest.csv')
    args = ap.parse_args(argv)
    out = args.out or (cfg.visual_dir if args.class_mode else 'vectordb_visual')
    build(args.manifest, cfg.corpus_dir, out, args.class_mode, args.model, args.batch_size,
          args.max_side, args.subject, adapter=args.adapter)


if __name__ == '__main__':
    main()

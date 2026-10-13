import json
from pathlib import Path
import chromadb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from ingestion.extract import extract_pages
from ingestion.clean import find_boilerplate, clean_page
from ingestion.chunk import make_chunks

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
CHAPTER_MIN_SIM = 0.45  # tune with the check in Procedure 2.3

def chapter_vectors(model, syllabus):
    """One vector per chapter: title plus topic phrases."""
    out = {}
    for s in syllabus:
        texts = [c['title'] + '. ' + ', '.join(c.get('topics', [])) for c in s['chapters']]
        out[s['subject']] = ([c['no'] for c in s['chapters']],
                             model.encode(texts, normalize_embeddings=True))
    return out

def tag_chapter(vec, subject, chap_vecs):
    nos, vecs = chap_vecs[subject]
    sims = vecs @ vec
    best = int(np.argmax(sims))
    return int(nos[best]) if sims[best] >= CHAPTER_MIN_SIM else 0

def main(manifest='corpus/manifest.csv', db='vectordb',
         chunks_out='data/chunks.jsonl', batch=64, shareable_only=False):
    df = pd.read_csv(manifest)
    if shareable_only:  # class mode: use db='vectordb_class'
        df = df[df['shareable'] == 'yes']
    syllabus = json.load(open('data/syllabus.json'))
    model = SentenceTransformer(MODEL_NAME)
    chap_vecs = chapter_vectors(model, syllabus)
    col = chromadb.PersistentClient(path=db).get_or_create_collection(
        'ee_corpus', metadata={'hnsw:space': 'cosine'})

    chunks = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc='extract'):
        pages = list(extract_pages(Path('corpus') / r['subject'] / r['file']))
        boiler = find_boilerplate([p[1] for p in pages])
        for num, text, tables in pages:
            chunks += make_chunks(r['file'], r['subject'], r['category'], num,
                                  clean_page(text, boiler), tables)

    for i in tqdm(range(0, len(chunks), batch), desc='embed'):
        part = chunks[i:i + batch]
        vecs = model.encode([c['text'] for c in part], normalize_embeddings=True,
                            batch_size=batch)
        for c, v in zip(part, vecs):
            c['meta']['chapter'] = tag_chapter(v, c['meta']['subject'], chap_vecs)
        col.upsert(ids=[c['id'] for c in part], documents=[c['text'] for c in part],
                   embeddings=vecs.tolist(), metadatas=[c['meta'] for c in part])

    Path(chunks_out).parent.mkdir(exist_ok=True)
    with open(chunks_out, 'w') as f:  # BM25 on Day 5 reads this file
        for c in chunks:
            f.write(json.dumps(c) + '\n')
    print('chunks stored:', col.count())

if __name__ == '__main__':
    main()
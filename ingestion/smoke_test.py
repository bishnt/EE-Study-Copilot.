import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
docs = [
    'The Z-transform converts a difference equation into an algebraic equation.',
    'A single-phase full converter with resistive load has an average output voltage of 2Vm cos(alpha)/pi.',
    'An IDMT relay operates faster when the fault current is larger.',
]

client = chromadb.PersistentClient(path='tmp_db')
col = client.get_or_create_collection('smoke', metadata={'hnsw:space': 'cosine'})
col.upsert(
    ids=['a', 'b', 'c'],
    documents=docs,
    embeddings=model.encode(docs, normalize_embeddings=True).tolist(),
)

q = 'Represent this sentence for searching relevant passages: which relay has an inverse time characteristic?'
res = col.query(
    query_embeddings=model.encode([q], normalize_embeddings=True).tolist(),
    n_results=1,
)
print(res['documents'][0][0], res['distances'][0][0])
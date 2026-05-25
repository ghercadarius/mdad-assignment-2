from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, OLLAMA_MODEL
from src.data_loader import load_dataset, chunk_corpus
from src.indexer import get_client, create_collection, index_chunks, collection_name
from src.retriever import Retriever
from src.rag import RAGPipeline

DATASET = "scifact"
QUERY = "Does vitamin D supplementation reduce cancer risk?"

model = SentenceTransformer(EMBED_MODEL)
corpus, queries, qrels = load_dataset(DATASET)

bootstrap_client = get_client()
try:
    if not bootstrap_client.collections.exists(collection_name(DATASET)):
        print(f"[demo] Collection '{collection_name(DATASET)}' not found; indexing '{DATASET}'...")
        chunks = chunk_corpus(corpus)
        create_collection(bootstrap_client, DATASET)
        index_chunks(bootstrap_client, DATASET, chunks, model)
finally:
    bootstrap_client.close()

ret = Retriever(DATASET, model, alpha=0.5, top_k=5)

print("\n" + "="*60)
print(f"QUERY: {QUERY}")
print("="*60)

print("\n[1] BM25 results:")
for r in ret.bm25(QUERY)[:3]:
    print(f"  doc_id={r['doc_id']}  score={r['score']:.4f}")

print("\n[2] Dense results:")
for r in ret.dense(QUERY)[:3]:
    print(f"  doc_id={r['doc_id']}  score={r['score']:.4f}")

print("\n[3] Hybrid results:")
for r in ret.hybrid(QUERY)[:3]:
    print(f"  doc_id={r['doc_id']}  score={r['score']:.4f}")

print("\n[4] RAG answer:")
pipeline = RAGPipeline(DATASET, model, "hybrid", top_k=5)
try:
    result = pipeline.run(QUERY)
    print(result["answer"])
finally:
    ret.close()
    pipeline.close()

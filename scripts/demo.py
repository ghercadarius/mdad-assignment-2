from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, OLLAMA_MODEL
from src.data_loader import load_dataset
from src.retriever import Retriever
from src.rag import RAGPipeline

DATASET = "scifact"
QUERY = "Does vitamin D supplementation reduce cancer risk?"

model = SentenceTransformer(EMBED_MODEL)
_, queries, qrels = load_dataset(DATASET)
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
result = pipeline.run(QUERY)
print(result["answer"])

ret.close()
pipeline.close()

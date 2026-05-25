"""
Entry-point Cerința 3: RAG pe 10 query-uri din setul ales.
Setul recomandat: SciFact (cel mai interesant comportament al retrieverelor).
"""

import json
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, OLLAMA_MODEL
from src.data_loader import load_dataset
from src.rag import RAGPipeline

# ===== ALEGE SETUL DE DATE =====
CHOSEN_DATASET = "scifact"   # sau "nfcorpus" / "fiqa"
BEST_RETRIEVER = "hybrid"    # completează cu ce a ieșit cel mai bun în Cerința 2
N_QUERIES = 10
# ================================


def main():
    model = SentenceTransformer(EMBED_MODEL)
    _, queries, qrels = load_dataset(CHOSEN_DATASET)

    # Selectează primele N_QUERIES query-uri care au judecăți de relevanță
    eval_queries = [
        (qid, text) for qid, text in queries.items()
        if qid in qrels
    ][:N_QUERIES]

    pipeline = RAGPipeline(
        dataset=CHOSEN_DATASET,
        embed_model=model,
        retriever_type=BEST_RETRIEVER,
        top_k=10,
        llm_model=OLLAMA_MODEL,
    )

    results = []
    for qid, question in eval_queries:
        print(f"\n{'='*60}")
        print(f"Query [{qid}]: {question}")
        print(f"{'='*60}")

        result = pipeline.run(question)
        result["query_id"] = qid
        result["relevant_docs"] = list(qrels[qid].keys())

        print(f"Chunk-uri regăsite: {result['context_chunks']}")
        print(f"\nRăspuns LLM:\n{result['answer']}")

        results.append(result)

    # Salvare rezultate pentru analiză și raport
    with open("results/rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pipeline.close()
    print(f"\n\nRezultate salvate în results/rag_results.json")
    print("Analizează manual pentru halucinații!")


if __name__ == "__main__":
    main()

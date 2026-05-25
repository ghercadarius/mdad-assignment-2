from typing import List, Dict
import ollama

from src.config import OLLAMA_MODEL, TOP_K
from src.retriever import Retriever


RAG_PROMPT_TEMPLATE = """Ești un asistent util. Răspunde la întrebare DOAR pe baza contextului furnizat.
La finalul răspunsului, citează ID-urile chunk-urilor folosite în formatul [CITATIONS: id1, id2, ...].
Dacă contextul nu conține informația necesară, spune explicit că nu știi.

CONTEXT:
{context}

ÎNTREBARE: {question}

RĂSPUNS (cu citări):"""


def build_context(chunks: List[Dict]) -> str:
    """Construiește blocul de context din chunk-urile regăsite."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[{chunk['chunk_id']}] (doc: {chunk['doc_id']})\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str,
                    chunks: List[Dict],
                    model: str = OLLAMA_MODEL) -> Dict:
    """
    Generează răspuns RAG folosind Ollama.
    Returnează dict cu: question, context_chunks, prompt, answer
    """
    context = build_context(chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},   # temperatură mică pentru răspunsuri factuale
    )

    return {
        "question": question,
        "context_chunks": [c["chunk_id"] for c in chunks],
        "answer": response["message"]["content"],
    }


class RAGPipeline:
    """Pipeline complet: retrieval + generare."""

    def __init__(self, dataset: str, embed_model,
                 retriever_type: str = "hybrid",
                 alpha: float = 0.5,
                 top_k: int = TOP_K,
                 llm_model: str = OLLAMA_MODEL):
        self.retriever = Retriever(dataset, embed_model, alpha=alpha, top_k=top_k)
        self.retriever_type = retriever_type
        self.llm_model = llm_model
        self._top_k = top_k

    def _retrieve(self, query: str) -> List[Dict]:
        if self.retriever_type == "bm25":
            return self.retriever.bm25(query)
        elif self.retriever_type == "dense":
            return self.retriever.dense(query)
        else:
            return self.retriever.hybrid(query)

    def run(self, question: str) -> Dict:
        chunks = self._retrieve(question)
        # Păstrăm textul în chunks pentru context
        # (retrieve_* returnează doc_id și chunk_id; trebuie să adăugăm textul)
        # Facem un query suplimentar pentru a obține textul
        return generate_answer(question, chunks, self.llm_model)

    def close(self):
        self.retriever.close()

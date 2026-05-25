import math
from typing import List, Dict


def recall_at_k(retrieved_doc_ids: List[str],
                relevant_doc_ids: set,
                k: int = 10) -> float:
    """Recall@k: câți relevanți am recuperat din primii k."""
    if not relevant_doc_ids:
        return 0.0
    retrieved_top_k = set(retrieved_doc_ids[:k])
    return len(retrieved_top_k & relevant_doc_ids) / len(relevant_doc_ids)


def mrr(retrieved_doc_ids: List[str],
        relevant_doc_ids: set) -> float:
    """Mean Reciprocal Rank: 1/rang_primul_relevant."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: List[str],
              qrel: Dict[str, int],
              k: int = 10) -> float:
    """
    nDCG@k folosind scoruri de relevanță din qrel.
    qrel: {doc_id: relevance_score}
    """
    def dcg(scores):
        return sum(
            (2 ** s - 1) / math.log2(i + 2)
            for i, s in enumerate(scores)
        )

    # DCG actual
    actual_scores = [qrel.get(doc_id, 0) for doc_id in retrieved_doc_ids[:k]]
    actual_dcg = dcg(actual_scores)

    # IDCG (ideal)
    ideal_scores = sorted(qrel.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_retriever(retrieve_fn,
                       queries: Dict[str, str],
                       qrels: Dict[str, Dict[str, int]],
                       k: int = 10) -> Dict[str, float]:
    """
    Evaluează o funcție de retrieval pe toate query-urile.

    retrieve_fn(query_text) → List[{"doc_id": ..., ...}]
    Returnează dict cu mediile: recall@k, mrr, ndcg@k
    """
    recalls, mrrs, ndcgs = [], [], []

    for qid, query_text in queries.items():
        if qid not in qrels:
            continue
        qrel = qrels[qid]  # {doc_id: score}
        relevant = set(qrel.keys())

        results = retrieve_fn(query_text)
        retrieved_ids = [r["doc_id"] for r in results]

        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        mrrs.append(mrr(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, qrel, k))

    n = len(recalls)
    return {
        f"Recall@{k}": sum(recalls) / n if n else 0.0,
        "MRR": sum(mrrs) / n if n else 0.0,
        f"nDCG@{k}": sum(ndcgs) / n if n else 0.0,
        "n_queries": n,
    }

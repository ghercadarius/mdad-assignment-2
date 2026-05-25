from typing import List, Dict

import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

from src.config import TOP_K, ALPHA_DEFAULT
from src.indexer import collection_name, get_client


def _result_payload(obj) -> Dict:
    metadata = getattr(obj, "metadata", None)
    score = 0.0
    if metadata is not None:
        score = getattr(metadata, "score", None)
        if score is None:
            score = getattr(metadata, "certainty", None)
        score = score or 0.0

    properties = obj.properties
    return {
        "doc_id": properties["doc_id"],
        "chunk_id": properties["chunk_id"],
        "text": properties.get("text", ""),
        "title": properties.get("title", ""),
        "score": score,
    }


def retrieve_bm25(client: weaviate.WeaviateClient,
                  dataset: str,
                  query_text: str,
                  top_k: int = TOP_K) -> List[Dict]:
    """Retrieval sparse BM25."""
    collection = client.collections.get(collection_name(dataset))
    results = collection.query.bm25(
        query=query_text,
        limit=top_k,
        return_properties=["chunk_id", "doc_id", "text", "title"],
    )
    return [_result_payload(o) for o in results.objects]


def retrieve_dense(client: weaviate.WeaviateClient,
                   dataset: str,
                   query_vector: List[float],
                   top_k: int = TOP_K) -> List[Dict]:
    """Retrieval dens (nearest neighbor vectorial)."""
    collection = client.collections.get(collection_name(dataset))
    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_properties=["chunk_id", "doc_id", "text", "title"],
        return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True),
    )
    return [_result_payload(o) for o in results.objects]


def retrieve_hybrid(client: weaviate.WeaviateClient,
                    dataset: str,
                    query_text: str,
                    query_vector: List[float],
                    alpha: float = ALPHA_DEFAULT,
                    top_k: int = TOP_K) -> List[Dict]:
    """
    Retrieval hibrid: fuziune convexă BM25 + dens.
    alpha=0 → BM25 pur; alpha=1 → dens pur.
    """
    collection = client.collections.get(collection_name(dataset))
    results = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=alpha,
        limit=top_k,
        return_properties=["chunk_id", "doc_id", "text", "title"],
        return_metadata=wvc.query.MetadataQuery(score=True),
    )
    return [_result_payload(o) for o in results.objects]


class Retriever:
    """Wrapper convenabil pentru toate cele trei strategii."""

    def __init__(self, dataset: str, model: SentenceTransformer,
                 alpha: float = ALPHA_DEFAULT, top_k: int = TOP_K):
        self.dataset = dataset
        self.model = model
        self.alpha = alpha
        self.top_k = top_k
        self.client = get_client()

    def _embed(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def bm25(self, query: str) -> List[Dict]:
        return retrieve_bm25(self.client, self.dataset, query, self.top_k)

    def dense(self, query: str) -> List[Dict]:
        vec = self._embed(query)
        return retrieve_dense(self.client, self.dataset, vec, self.top_k)

    def hybrid(self, query: str) -> List[Dict]:
        vec = self._embed(query)
        return retrieve_hybrid(self.client, self.dataset,
                               query, vec, self.alpha, self.top_k)

    def close(self):
        self.client.close()

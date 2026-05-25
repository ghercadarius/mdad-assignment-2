import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict

from src.config import WEAVIATE_URL, EMBED_MODEL, EMBED_DIM


def get_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host="localhost", port=8080)


def collection_name(dataset: str) -> str:
    """Weaviate cere numele colecției să înceapă cu majusculă."""
    return dataset.capitalize()


def create_collection(client: weaviate.WeaviateClient, dataset: str,
                      recreate: bool = False):
    """Creează (sau recreează) colecția pentru un dataset."""
    name = collection_name(dataset)

    if client.collections.exists(name):
        if recreate:
            client.collections.delete(name)
        else:
            print(f"[indexer] Colecția '{name}' există deja, se sare.")
            return

    client.collections.create(
        name=name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE
        ),
        properties=[
            wvc.config.Property(name="chunk_id",
                                data_type=wvc.config.DataType.TEXT,
                                index_searchable=False),
            wvc.config.Property(name="doc_id",
                                data_type=wvc.config.DataType.TEXT,
                                index_searchable=False),
            wvc.config.Property(name="title",
                                data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="text",
                                data_type=wvc.config.DataType.TEXT),
        ],
    )
    print(f"[indexer] Colecție '{name}' creată.")


def index_chunks(client: weaviate.WeaviateClient,
                 dataset: str,
                 chunks: List[Dict],
                 model: SentenceTransformer,
                 batch_size: int = 128):
    """Vectorizează și inserează chunk-urile în Weaviate."""
    name = collection_name(dataset)
    collection = client.collections.get(name)

    texts = [c["text"] for c in chunks]

    print(f"[indexer] Vectorizare {len(texts)} chunk-uri pentru '{dataset}'...")
    # encode cu batch intern; GPU folosit automat dacă disponibil
    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=True,
                              normalize_embeddings=True)

    print(f"[indexer] Inserare în Weaviate...")
    with collection.batch.fixed_size(batch_size=200) as batch:
        for chunk, vec in tqdm(zip(chunks, embeddings), total=len(chunks)):
            batch.add_object(
                properties={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                },
                vector=vec.tolist(),
            )

    print(f"[indexer] '{dataset}': {len(chunks)} chunk-uri indexate.")

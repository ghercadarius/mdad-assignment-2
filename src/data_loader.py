import os
from typing import Dict, List, Tuple

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from src.config import BEIR_URL, DATASETS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_dataset(name: str) -> Tuple[dict, dict, dict]:
    """Descarcă (dacă nu există deja) și încarcă un set de date BEIR."""
    data_path = os.path.join(DATASETS_DIR, name)
    if not os.path.exists(data_path):
        url = BEIR_URL.format(name)
        util.download_and_unzip(url, DATASETS_DIR)
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="test")
    return corpus, queries, qrels


def chunk_document(doc_id: str, title: str, text: str,
                   chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Sparge un document în chunk-uri de dimensiune fixă (în cuvinte)
    cu overlap procentual.

    Returnează o listă de dicts cu câmpurile:
      chunk_id, doc_id, text, title
    """
    # Concatenăm titlul cu textul pentru context
    full_text = f"{title} {text}".strip() if title else text
    words = full_text.split()

    if len(words) <= chunk_size:
        return [{
            "chunk_id": f"{doc_id}_0",
            "doc_id": doc_id,
            "text": full_text,
            "title": title,
        }]

    chunks = []
    step = chunk_size - overlap
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"{doc_id}_{idx}",
            "doc_id": doc_id,
            "text": chunk_text,
            "title": title,
        })
        idx += 1
        start += step
    return chunks


def chunk_corpus(corpus: dict,
                 chunk_size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Aplică chunking pe întregul corpus."""
    all_chunks = []
    for doc_id, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        all_chunks.extend(chunk_document(doc_id, title, text,
                                         chunk_size, overlap))
    return all_chunks

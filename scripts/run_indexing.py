"""Entry-point pentru Cerința 1: indexare toate cele 3 seturi de date."""

from sentence_transformers import SentenceTransformer
from src.config import DATASETS, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.data_loader import load_dataset, chunk_corpus
from src.indexer import get_client, create_collection, index_chunks


def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = get_client()

    for dataset in DATASETS:
        print(f"\n{'='*50}")
        print(f"Procesare dataset: {dataset}")
        print(f"{'='*50}")

        corpus, queries, qrels = load_dataset(dataset)
        print(f"  Corpus: {len(corpus)} documente")

        chunks = chunk_corpus(corpus, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  Chunk-uri generate: {len(chunks)} "
              f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        create_collection(client, dataset, recreate=True)
        index_chunks(client, dataset, chunks, model)

    client.close()
    print("\nIndexare completă!")


if __name__ == "__main__":
    main()

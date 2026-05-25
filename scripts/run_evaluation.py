"""Entry-point pentru Cerința 2: tabel comparativ retrievere × datasets."""

import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import DATASETS, EMBED_MODEL, TOP_K, NDCG_K, ALPHA_DEFAULT
from src.data_loader import load_dataset
from src.retriever import Retriever
from src.evaluator import evaluate_retriever


def main():
    model = SentenceTransformer(EMBED_MODEL)
    rows = []

    for dataset in DATASETS:
        print(f"\n--- Evaluare {dataset} ---")
        _, queries, qrels = load_dataset(dataset)

        ret = Retriever(dataset, model, alpha=ALPHA_DEFAULT, top_k=TOP_K)

        for method, fn in [
            ("BM25",   ret.bm25),
            ("Dense",  ret.dense),
            ("Hybrid", ret.hybrid),
        ]:
            print(f"  {method}...", end=" ", flush=True)
            metrics = evaluate_retriever(fn, queries, qrels, k=TOP_K)
            print(f"nDCG@10={metrics[f'nDCG@{NDCG_K}']:.4f}")

            rows.append({
                "Dataset": dataset,
                "Retriever": method,
                **metrics,
            })

        ret.close()

    df = pd.DataFrame(rows)

    # Tabel pivot pentru raport
    pivot = df.pivot_table(
        index="Retriever",
        columns="Dataset",
        values=[f"Recall@{TOP_K}", "MRR", f"nDCG@{NDCG_K}"],
    )
    print("\n\n=== TABEL COMPARATIV ===")
    print(pivot.to_string(float_format="%.4f"))

    # Salvare CSV
    df.to_csv("results/metrics_table.csv", index=False)
    print("\nRezultate salvate în results/metrics_table.csv")


if __name__ == "__main__":
    main()

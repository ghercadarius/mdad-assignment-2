"""
Sweep pe: chunk_size, top_k, alpha.
Evaluare nDCG@10 pentru fiecare configurație.
"""

from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, ALPHA_DEFAULT, NDCG_K
from src.data_loader import load_dataset, chunk_corpus
from src.indexer import get_client, create_collection, index_chunks
from src.retriever import Retriever
from src.evaluator import evaluate_retriever


def sweep_alpha(dataset: str, queries: dict, qrels: dict,
                model: SentenceTransformer,
                alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
                top_k: int = 10) -> pd.DataFrame:
    """Sweep pe parametrul alpha al căutării hibride."""
    rows = []
    for alpha in alphas:
        ret = Retriever(dataset, model, alpha=alpha, top_k=top_k)
        metrics = evaluate_retriever(ret.hybrid, queries, qrels, k=NDCG_K)
        rows.append({"alpha": alpha, **metrics})
        ret.close()
        print(f"  alpha={alpha:.2f} → nDCG@10={metrics[f'nDCG@{NDCG_K}']:.4f}")
    return pd.DataFrame(rows)


def sweep_top_k(dataset: str, queries: dict, qrels: dict,
                model: SentenceTransformer,
                ks: List[int] = [3, 5, 10],
                alpha: float = ALPHA_DEFAULT) -> pd.DataFrame:
    """Sweep pe numărul de chunk-uri regăsite (top-k)."""
    rows = []
    for k in ks:
        ret = Retriever(dataset, model, alpha=alpha, top_k=k)
        metrics = evaluate_retriever(ret.hybrid, queries, qrels, k=k)
        rows.append({"top_k": k, **metrics})
        ret.close()
        print(f"  top_k={k} → nDCG@{k}={metrics[f'nDCG@{k}']:.4f}")
    return pd.DataFrame(rows)


def sweep_chunk_size(dataset: str, corpus: dict,
                     queries: dict, qrels: dict,
                     model: SentenceTransformer,
                     chunk_sizes: List[int] = [256, 512, 1024]) -> pd.DataFrame:
    """
    Sweep pe dimensiunea chunk-ului.
    ATENȚIE: reindexează Weaviate pentru fiecare dimensiune!
    """
    rows = []
    client = get_client()

    for cs in chunk_sizes:
        overlap = int(cs * 0.1)
        print(f"\n  Reindexare cu chunk_size={cs}, overlap={overlap}...")

        chunks = chunk_corpus(corpus, chunk_size=cs, overlap=overlap)
        create_collection(client, dataset, recreate=True)
        index_chunks(client, dataset, chunks, model)

        ret = Retriever(dataset, model, alpha=ALPHA_DEFAULT, top_k=10)
        metrics = evaluate_retriever(ret.hybrid, queries, qrels, k=NDCG_K)
        rows.append({"chunk_size": cs, **metrics})
        ret.close()
        print(f"  chunk_size={cs} → nDCG@10={metrics[f'nDCG@{NDCG_K}']:.4f}")

    client.close()
    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame, x_col: str, y_col: str,
               title: str, save_path: str = None):
    """Generează grafic pentru sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[x_col], df[y_col], marker="o", linewidth=2, markersize=8)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Marchează valoarea maximă
    best_idx = df[y_col].idxmax()
    best_x = df.loc[best_idx, x_col]
    best_y = df.loc[best_idx, y_col]
    ax.annotate(f"Best: {best_y:.4f}\n({x_col}={best_x})",
                xy=(best_x, best_y),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=10,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafic salvat: {save_path}")
    plt.show()

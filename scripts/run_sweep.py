"""Entry-point Cerința 4: hyperparameter sweep complet."""

from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL
from src.data_loader import load_dataset
from src.sweep import sweep_alpha, sweep_top_k, sweep_chunk_size, plot_sweep

CHOSEN_DATASET = "scifact"   # același ca în Cerința 3


def main():
    model = SentenceTransformer(EMBED_MODEL)
    corpus, queries, qrels = load_dataset(CHOSEN_DATASET)

    # --- 1. Sweep alpha ---
    print(f"\n{'='*50}")
    print("SWEEP: alpha (0=BM25 pur → 1=dens pur)")
    print(f"{'='*50}")
    df_alpha = sweep_alpha(CHOSEN_DATASET, queries, qrels, model)
    print(df_alpha.to_string(index=False))
    plot_sweep(df_alpha, "alpha", "nDCG@10",
               f"nDCG@10 vs Alpha — {CHOSEN_DATASET}",
               save_path="results/sweep_alpha.png")

    # --- 2. Sweep top-k ---
    print(f"\n{'='*50}")
    print("SWEEP: top-k (număr chunk-uri regăsite)")
    print(f"{'='*50}")
    df_k = sweep_top_k(CHOSEN_DATASET, queries, qrels, model)
    print(df_k.to_string(index=False))

    # --- 3. Sweep chunk size ---
    print(f"\n{'='*50}")
    print("SWEEP: chunk_size (256 / 512 / 1024)")
    print(f"{'='*50}")
    df_cs = sweep_chunk_size(CHOSEN_DATASET, corpus, queries, qrels, model)
    print(df_cs.to_string(index=False))
    plot_sweep(df_cs, "chunk_size", "nDCG@10",
               f"nDCG@10 vs Chunk Size — {CHOSEN_DATASET}",
               save_path="results/sweep_chunk_size.png")

    # Grafic combinat (opțional, arată bine în raport)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df_alpha["alpha"], df_alpha["nDCG@10"], "o-", linewidth=2)
    axes[0].set_xlabel("Alpha"); axes[0].set_ylabel("nDCG@10")
    axes[0].set_title("Alpha sweep"); axes[0].grid(alpha=0.3)

    axes[1].plot(df_cs["chunk_size"], df_cs["nDCG@10"], "s-", linewidth=2, color="orange")
    axes[1].set_xlabel("Chunk size"); axes[1].set_ylabel("nDCG@10")
    axes[1].set_title("Chunk size sweep"); axes[1].grid(alpha=0.3)

    plt.suptitle(f"Hyperparameter sweep — {CHOSEN_DATASET}", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/sweep_combined.png", dpi=150, bbox_inches="tight")
    print("\nGrafic combinat: results/sweep_combined.png")


if __name__ == "__main__":
    main()

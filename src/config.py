# Constante globale ale proiectului
DATASETS = ["nfcorpus", "scifact", "fiqa"]
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
DATASETS_DIR = "datasets"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384          # dimensiunea vectorului all-MiniLM-L6-v2

WEAVIATE_URL = "http://localhost:8080"

# Chunking
CHUNK_SIZE = 512         # tokens (valoarea default pentru sweep)
CHUNK_OVERLAP = 51       # ~10% din 512

# Retrieval
TOP_K = 10
ALPHA_DEFAULT = 0.5      # hibrid: 0=BM25 pur, 1=dens pur

# LLM
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"

# Evaluare
RECALL_K = 10
NDCG_K = 10
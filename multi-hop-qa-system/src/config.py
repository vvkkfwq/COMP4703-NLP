"""
Configure
"""

from pathlib import Path

# path
BASE_DIR = Path(__file__).parent.parent  # multi-hop-qa-system/
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_PATH = BASE_DIR / "bm25_index.pkl"
CORPUS_PATH = DATA_DIR / "sample-corpus.json"
RAG_PATH = DATA_DIR / "sample-rag.json"

# Embedding
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_MODELS: dict[str, dict] = {
    "bge-large": {
        "label": "BGE-Large",
        "model_name": "BAAI/bge-large-en-v1.5",
        "chroma_dir": CHROMA_DIR / "bge-large",
    },
    "llm-embedder": {
        "label": "LLM-Embedder",
        "model_name": "BAAI/llm-embedder",
        "chroma_dir": CHROMA_DIR / "llm-embedder",
    },
    "minilm-l6": {
        "label": "MiniLM-L6-v2",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "chroma_dir": CHROMA_DIR / "minilm-l6",
    },
    "me5-base": {
        "label": "mE5-Base",
        "model_name": "intfloat/multilingual-e5-base",
        "chroma_dir": CHROMA_DIR / "me5-base",
    },
}
EMBED_MODEL_DEFAULT = "bge-large"
EMBED_DEVICE = "mps"  # Apple Silicon；其他平台改 "cuda" 或 "cpu"

# Ingestion
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 64

# Retrieval
TOP_K = 10
RERANKER_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60

# LLM
LLM_MODEL = "gpt-4o-mini"

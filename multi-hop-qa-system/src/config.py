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

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
print(f"from config.py: project_root: {PROJECT_ROOT}")

use_GPU = True
is_STAGING = True

# Define data directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
STAGING_DIR = OUTPUT_DIR / "staging"
PRODUCTION_DIR = OUTPUT_DIR / "production"


# Ensure output directories exist
STAGING_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
if is_STAGING:
    CORPUS_FILE = DATA_DIR / "sample-corpus.json"
    QUERY_FILE = DATA_DIR / "sample-rag.json"
    OUTPUT_PATH = STAGING_DIR
else:
    CORPUS_FILE = DATA_DIR / "corpus.json"
    QUERY_FILE = DATA_DIR / "rag.json"
    OUTPUT_PATH = PRODUCTION_DIR

# Choose two best performance embedding model
RANKERS = {
    "llm-embedder-reranker": OUTPUT_PATH
    / "rerankerA.json",  # llm-embedder ranker + bge-reranker-base reranker
    "bge-large": OUTPUT_PATH / "rankerC.json",  # BAAI/bge-large-en-v1.5 ranker
}

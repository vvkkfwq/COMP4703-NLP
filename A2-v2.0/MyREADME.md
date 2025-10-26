# COMP4703 Assignment 2 - Multi-Hop RAG System

**Student:** Vickey Feng
**Course:** COMP4703 Natural Language Processing
**Project:** Multi-Hop Retrieval Augmented Generation (RAG) System Evaluation

## Project Overview

This project implements and evaluates a multi-stage Retrieval Augmented Generation (RAG) system for multi-hop question answering. The system explores performance trade-offs between different retrieval algorithms, reranking strategies, and Large Language Models (LLMs).

### System Architecture

The RAG pipeline consists of 3 stages:

1. **Stage 1 - Retrieval**: Four different embedding models retrieve top-k relevant documents
2. **Stage 2 - Reranking** (Optional): Cross-encoder reranker refines retrieval results
3. **Stage 3 - Generation**: Four different LLMs generate answers based on retrieved context

### Implemented Variants

**Retrieval Models (Stage 1):**

- Ranker A: `BAAI/llm-embedder`
- Ranker B: `sentence-transformers/all-MiniLM-L6-v2`
- Ranker C: `BAAI/bge-large-en-v1.5`
- Ranker D: `intfloat/multilingual-e5-large`

**Reranking Models (Stage 2):**

- Reranker A: llm-embedder + `BAAI/bge-reranker-base`
- Reranker B: all-MiniLM-L6-v2 + `BAAI/bge-reranker-base`

**Language Models (Stage 3):**

- RAG A: `meta-llama/Llama-2-7b-chat-hf`
- RAG B: `meta-llama/Meta-Llama-3-8B-Instruct`
- RAG C: `mistralai/Mistral-7B-Instruct-v0.3`
- RAG D: `Qwen/Qwen3-8B`

## Environment Setup

### Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure runtime settings:**

Edit [config.py](config.py) before running:

```python
use_GPU = True              # Set False for CPU-only execution
is_STAGING = False          # True: use sample data, False: use full dataset
```

**Important:** Always test with `is_STAGING = True` before production runs!

## Quick Start

### Option 1: Automated Pipeline (Recommended)

Run complete experimental workflow using shell scripts:

```bash
# Step 1: Run all retrieval variants (Stage 1)
bash run_rankers.sh

# Step 2: Evaluate retrieval outputs
bash evaluate_rankers.sh

# Step 3: Run all RAG configurations (Stage 3)
bash run_rag.sh

# Step 4: Evaluate RAG outputs
bash evaluate_rags.sh
```

**Note:** Scripts automatically detect STAGING mode from [config.py](config.py) and save logs to `logs/staging/` or `logs/production/`.

### Option 2: Manual Execution

**Stage 1 - Retrieval:**

```bash
python rankerA.py    # BAAI/llm-embedder
python rankerB.py    # sentence-transformers/all-MiniLM-L6-v2
python rankerC.py    # BAAI/bge-large-en-v1.5
python rankerD.py    # intfloat/multilingual-e5-large
```

**Stage 2 - Reranking:**

```bash
python rerankerA.py  # llm-embedder + bge-reranker-base
python rerankerB.py  # all-MiniLM-L6-v2 + bge-reranker-base
```

**Evaluate Retrieval:**

```bash
python MyRetEval.py --input output/production/rankerA.json
python MyRetEval.py --input output/production/rerankerA.json
# ... repeat for other rankers
```

**Stage 3 - RAG Generation:**

_Important:_ Edit [config.py](config.py) `RANKERS` dictionary to select which retrieval outputs to use.

```bash
python RAGA.py       # Llama-2-7b-chat
python RAGB.py       # Llama-3-8B-Instruct
python RAGC.py       # Mistral-7B-Instruct
python RAGD.py       # Qwen3-8B
```

**Evaluate RAG:**

```bash
python MyRagEval.py output/production/llm-embedder-reranker_llama2.json
python MyRagEval.py output/production/bge-large_llama3.json
# ... repeat for other RAG outputs
```

## Configuration Management

### Central Configuration ([config.py](config.py))

All runtime settings are managed through [config.py](config.py):

```python
# GPU/CPU toggle
use_GPU = True              # Uses torch.float16 on GPU to reduce memory

# Dataset selection
is_STAGING = False          # False: full dataset (~1000 docs, ~200 queries)
                            # True: sample data (10 docs, 5 queries)

# Auto-configured paths
OUTPUT_PATH                 # Automatically set to output/staging/ or output/production/

# RAG ranker selection
RANKERS = {
    "llm-embedder-reranker": OUTPUT_PATH / "rerankerA.json",
    "bge-large": OUTPUT_PATH / "rankerC.json",
}
```

**Never hardcode these values in individual scripts!**

### Output Naming Convention

- **Retrieval outputs:** `ranker[A-D].json`, `reranker[A-B].json`
- **RAG outputs:** `{ranker_name}_{llm_name}.json`
  - Example: `llm-embedder-reranker_llama2.json`

## Evaluation Metrics

### Retrieval Evaluation (MyRetEval.py)

Computes per-query and overall metrics:

- **Hits@10**: Proportion of queries with at least one relevant document in top-10
- **MAP@10**: Mean Average Precision at 10
- **MRR@10**: Mean Reciprocal Rank at 10
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10

### RAG Evaluation (MyRagEval.py)

Computes per-question-type and overall metrics:

- **Precision**: Token-level precision
- **Recall**: Token-level recall
- **F1**: Harmonic mean of precision and recall
- **Exact Match (EM)**: Binary exact match score

## Directory Structure

```
A2-v2.0/
├── config.py                      # Central configuration
├── MyREADME.md                    # This file
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── corpus.json               # Full corpus (~1000 documents)
│   ├── rag.json                  # Full queries (~200 questions)
│   ├── sample-corpus.json        # Test subset (10 documents)
│   └── sample-rag.json           # Test subset (5 queries)
│
├── output/
│   ├── staging/                  # Sample data outputs
│   │   ├── ranker*.json
│   │   ├── reranker*.json
│   │   └── *_*.json              # RAG outputs
│   └── production/               # Full dataset outputs
│       ├── ranker*.json
│       ├── reranker*.json
│       └── *_*.json              # RAG outputs
│
├── logs/
│   ├── staging/                  # Sample run logs
│   └── production/               # Production run logs
│
├── ranker[A-D].py                # 4 retrieval variants
├── reranker[A-B].py              # 2 reranking variants
├── RAG[A-D].py                   # 4 LLM variants
│
├── MyRetEval.py                  # Custom retrieval evaluator
├── MyRagEval.py                  # Custom RAG evaluator
│
├── run_rankers.sh                # Batch run all retrievers
├── run_rag.sh                    # Batch run all RAG configs
├── evaluate_rankers.sh           # Batch evaluate retrievers
└── evaluate_rags.sh              # Batch evaluate RAG outputs
```

## Estimated Runtime (Full Dataset)

Based on 24GB GPU (NVIDIA RTX 3090/4090):

| Stage | Task                         | Time   | Cumulative |
| ----- | ---------------------------- | ------ | ---------- |
| 1     | Retrieval (4 rankers)        | ~4h    | 4h         |
| 1     | Reranking (2 rerankers)      | ~2h    | 6h         |
| 2     | Evaluation (6 configs)       | ~10min | 6h         |
| 3     | RAG (8 configs, 2 LLMs each) | ~16h   | 22h        |
| 3     | Evaluation (8 configs)       | ~5min  | 22h        |

**Total GPU time:** ~22 hours
**Recommended buffer:** 40 hours (includes debugging/reruns)

## Testing Before Production

**Always test with sample data first:**

1. Set `is_STAGING = True` in [config.py](config.py)
2. Run: `bash run_rankers.sh`
3. Verify outputs in `output/staging/`
4. Check logs in `logs/staging/`
5. If successful, set `is_STAGING = False` and run production

## License

This code is provided for educational purposes only as part of COMP4703 coursework.

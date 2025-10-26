# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a COMP4703 (Natural Language Processing) Assignment 2 project focused on implementing and evaluating Retrieval Augmented Generation (RAG) systems for multi-hop question answering. The project compares different retrieval algorithms and Large Language Models (LLMs) to determine optimal configurations.

### Project Objective (CRITICAL - READ THIS FIRST)

**Core Goal:** Experimentally study the various **performance trade-offs** in RAG systems when solving the "Multi-Hop Retrieval" problem.

**Task Characteristics:**

- **Problem Type:** Factoid question answering that requires more than one document to get the correct answer
- **System Output:** Either answer the question succinctly based on instructions in a carefully crafted prompt, OR reply "Insufficient Data"
- **Dataset:** Small factoid QA dataset for Multi-Hop QA (~1000 documents, ~200 questions)

**RAG System Architecture (2-3 Stages):**

1. **Stage 1 (Required):** First-stage retrieval - finds top-k most relevant documents from corpus
2. **Stage 2 (Optional):** Reranking - refines first-stage results to improve effectiveness
3. **Stage 3 (Required):** LLM generation - generates answer to the query using top-k documents

**CRITICAL CHALLENGE - Non-orthogonal System Interactions:**

> **"The 'best' first stage and 'best' second stage may not produce the 'best' overall results when combined."**

This means:

- Complex system interactions can be orthogonal
- Subtle interactions between models are very unpredictable
- **The ONLY way to determine the best overall system is through extensive careful experimentation**

**Experimental Requirements:**

1. Explore several different alternatives for each stage
2. Write a **5-page experimental report** that includes:
   - Explanation of how each method works
   - Exhaustive experimental study comparing and contrasting approaches
   - Identification of the best overall system
   - Analysis of performance trade-offs and component interactions

**Report Goal:** Compare and contrast the approaches used in order to find the best overall system, with detailed analysis of why certain combinations work better than others despite individual component performance.

## Centralized Configuration System

**CRITICAL:** All runtime settings are managed through [config.py](config.py). Never hardcode these values in individual scripts.

```python
# config.py controls:
use_GPU = True              # Toggle GPU/CPU execution
is_STAGING = False          # True: sample data, False: full dataset
RANKERS = {...}             # Maps ranker names to output paths
OUTPUT_PATH                 # Auto-switches: output/staging/ or output/production/
```

**Before running experiments, always verify config.py settings:**

- Set `is_STAGING = True` for local testing with sample data
- Set `is_STAGING = False` only for final production runs on full dataset
- Set `use_GPU = True` on GPU server, `False` for local CPU development

## Key Commands

### Production Workflow Scripts

The project uses automated shell scripts for running complete experimental pipelines:

```bash
# Run all 4 rankers + 2 rerankers (Stage 1)
bash run_rankers.sh

# Evaluate all retrieval outputs
bash evaluate_rankers.sh

# Run all 8 RAG configurations (2 rankers × 4 LLMs)
bash run_rag.sh

# Evaluate all RAG outputs
bash evaluate_rags.sh
```

**Script Features:**

- Automatically detects STAGING mode from config.py
- Saves logs to `logs/staging/` or `logs/production/`
- Generates summary files with aggregated metrics
- Uses conda environment activation (update paths for your setup)

### Individual Component Execution

```bash
# Stage 1: Retrieval variants
python rankerA.py          # BAAI/llm-embedder
python rankerB.py          # sentence-transformers/all-MiniLM-L6-v2
python rankerC.py          # BAAI/bge-large-en-v1.5
python rankerD.py          # intfloat/multilingual-e5-large

# Stage 2: Reranking variants
python rerankerA.py        # llm-embedder + bge-reranker-base
python rerankerB.py        # sentence-transformers/all-MiniLM-L6-v2 + bge-reranker-base

# Stage 3: RAG variants (uses retrieval outputs from config.RANKERS)
python RAGA.py             # meta-llama/Llama-2-7b-chat-hf
python RAGB.py             # meta-llama/Meta-Llama-3-8B-Instruct
python RAGC.py             # mistralai/Mistral-7B-Instruct-v0.3
python RAGD.py             # Qwen/Qwen3-8B

# Evaluation
python MyRetEval.py --input <ranker_output.json>
python MyRagEval.py <rag_output.json>
```

### Baseline Reference Scripts

```bash
python example_retrieval.py   # Reference retrieval implementation
python example_reranker.py    # Reference reranker implementation
python example_rag.py         # Reference RAG implementation
python evaluate_stage_1.py    # Original retrieval evaluator
python evaluate_rag.py        # Original RAG evaluator
```

## Code Architecture

### File Naming Convention and Workflow

The project follows a systematic naming scheme:

1. **Retrieval Stage:**

   - `ranker[A-D].py` → produces `rankerA.json` (retrieval results only)
   - `reranker[A-B].py` → produces `rerankerA.json` (retrieval + reranking)

2. **RAG Stage:**

   - `RAG[A-D].py` → reads from `config.RANKERS` dictionary
   - Outputs named: `{ranker_name}_{llm_name}.json`
   - Example: `llm-embedder-reranker_llama2.json`

3. **Evaluation:**
   - `MyRetEval.py` → evaluates ranker/reranker outputs (Hits@k, MAP, MRR, NDCG)
   - `MyRagEval.py` → evaluates RAG outputs (Precision, Recall, F1, EM)

### Data Flow Pipeline

```
data/corpus.json ──┐
                   ├──> ranker*.py ──> ranker*.json ──┐
data/rag.json ─────┘                                   │
                                                       ├──> RAG*.py ──> {ranker}_{llm}.json ──> MyRagEval.py
                   ┌──> reranker*.py ──> reranker*.json ┘
                   │
        (Stage 1: Retrieval)     (Stage 2: Reranking)        (Stage 3: Generation)      (Evaluation)
```

### Core Implementation Patterns

**JSONReader Class** (in ranker\*.py)

- Parses corpus.json into LlamaIndex `Document` objects
- Preserves metadata: `title`, `source`, `published_at`
- Used consistently across all retrieval implementations

**Embedding Models** (Stage 1)

```python
from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/llm-embedder",
    query_instruction="Represent this query for retrieving relevant documents: "
)
```

**Reranking** (Stage 2)

```python
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-base",
    top_n=10
)
```

**LLM Generation** (Stage 3)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# RAG*.py files use HuggingFace transformers directly (not LlamaIndex LLM)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Output Format Requirements

**Retrieval Output (ranker/reranker\*.json):**

```json
{
  "query": "question text",
  "answer": "gold answer",
  "question_type": "inference_query",
  "retrieval_list": [
    {"text": "doc1 text with metadata", "score": 0.85},
    {"text": "doc2 text with metadata", "score": 0.76}
  ],
  "gold_list": ["evidence1", "evidence2", ...]
}
```

**RAG Output ({ranker}\_{llm}.json):**

```json
{
  "query": "question text",
  "model_answer": "predicted answer",
  "gold_answer": "gold answer",
  "question_type": "inference_query"
}
```

## Custom Evaluation Metrics

The project extends baseline metrics with two custom implementations:

1. **MyRetEval.py** - Adds NDCG@10 metric for retrieval evaluation

   - Computes normalized discounted cumulative gain
   - Accounts for ranking quality, not just binary relevance
   - Usage: `python MyRetEval.py --input output/production/rankerA.json`

2. **MyRagEval.py** - Adds Exact Match (EM) metric for RAG evaluation
   - Binary score: 1 if prediction exactly matches gold answer (case-insensitive)
   - Complements F1 score which gives partial credit
   - Usage: `python MyRagEval.py output/production/llm-embedder-reranker_llama2.json`

## Important Implementation Details

### Model Configuration

**Retrieval Models** (ranker\*.py):

- A: `BAAI/llm-embedder` (instruction-tuned, best for complex queries)
- B: `sentence-transformers/all-MiniLM-L6-v2` (lightweight, general-purpose)
- C: `BAAI/bge-large-en-v1.5` (large variant, high accuracy)
- D: `intfloat/multilingual-e5-large` (multilingual support, large variant)

**Reranking Model** (reranker\*.py):

- Uses `BAAI/bge-reranker-base` cross-encoder
- Reranks top-10 retrieved documents for improved precision

**LLM Models** (RAG\*.py):

- A: `meta-llama/Llama-2-7b-chat-hf` (7B, Meta)
- B: `meta-llama/Meta-Llama-3-8B-Instruct` (8B, Meta)
- C: `mistralai/Mistral-7B-Instruct-v0.3` (7B, Mistral)
- D: `Qwen/Qwen3-8B` (8B, Alibaba)

Models auto-download to `~/.cache/huggingface/hub` if not cached.

### GPU Considerations

- Set `use_GPU = True` in config.py for GPU execution
- Uses `torch.float16` to reduce memory from ~26GB to ~13GB
- Llama-2-7b requires ~13GB VRAM (needs 24GB GPU with safety margin)
- Monitor with `nvidia-smi` or `watch -n 1 nvidia-smi`
- Use `tmux` or `screen` for long-running jobs on remote servers

### Prompt Engineering

**Critical:** The instruction prompts in RAG\*.py are tuned for evaluation compatibility.

Default prompt structure:

```
[INST] <<SYS>>
You are a helpful assistant. Answer questions concisely based on the provided context.
If the context does not contain enough information, respond with "Insufficient Information".
<</SYS>>

Context: {retrieved_documents}

Question: {query}
[/INST]
```

**Warning:** Modifying prompts may change output format and break evaluation scripts. If you change prompts, update `MyRagEval.py` answer extraction logic accordingly.

### Answer Cleaning

RAG evaluation requires cleaning LLM outputs:

```python
# Remove model-specific tokens
answer = answer.replace("</s>", "").replace("[/INST]", "")
answer = answer.strip()
```

## Directory Structure

```
A2-v2.0/
├── config.py                      # CENTRAL CONFIGURATION
├── data/
│   ├── corpus.json               # Full corpus (~1000 docs)
│   ├── rag.json                  # Full queries (~200 questions)
│   ├── sample-corpus.json        # Testing subset (10 docs)
│   └── sample-rag.json           # Testing subset (5 queries)
├── output/
│   ├── staging/                  # Sample data outputs
│   └── production/               # Full dataset outputs
├── logs/
│   ├── staging/                  # Sample run logs
│   └── production/               # Production run logs
├── ranker[A-D].py                # 4 retrieval variants
├── reranker[A-B].py              # 2 reranking variants
├── RAG[A-D].py                   # 4 LLM variants
├── MyRetEval.py                  # Custom retrieval evaluator
├── MyRagEval.py                  # Custom RAG evaluator
├── run_rankers.sh                # Batch run all retrievers
├── run_rag.sh                    # Batch run all RAG configs
├── evaluate_rankers.sh           # Batch evaluate retrievers
├── evaluate_rags.sh              # Batch evaluate RAG outputs
├── example_*.py                  # Baseline reference scripts
└── evaluate_*.py                 # Original evaluation scripts
```

## Experimental Design Strategy

**Assignment Requirements:**

- 4 different retrieval algorithms ✓ (rankerA-D)
- 4 different LLMs ✓ (RAGA-D)
- 2 additional evaluation metrics ✓ (NDCG@10, Exact Match)

**Implemented Strategy:**

1. Evaluate all 4 rankers + 2 rerankers on full dataset
2. Select top 2 performers based on Hits@10, MAP@10, NDCG@10
3. Run 2 rankers × 4 LLMs = 8 RAG configurations
4. Evaluate with Precision, Recall, F1, Exact Match
5. Report findings in 5-page experimental report

**GPU Budget Management (40 hours total):**

- Retrieval (6 configs × 1h) = 6h
- RAG generation (8 configs × 2h) = 16h
- Buffer for debugging/reruns = 18h

**Best Practices:**

- Always test with `is_STAGING = True` before production runs
- Use `run_*.sh` scripts to avoid manual errors
- Monitor logs in real-time: `tail -f logs/production/RAGA.log`
- Validate output JSON format before evaluation

## Environment Setup

**GPU Server (NLPA2 environment):**

```bash
conda activate NLPA2
export CUDA_VISIBLE_DEVICES=0  # If multi-GPU system
```

**Local Development:**

```bash
conda activate COMP4703A2
# Ensure config.py has:
# use_GPU = False
# is_STAGING = True
```

**Dependencies** (pre-installed on GPU server):

- PyTorch 2.x (CUDA 12.8 or CPU variant)
- llama-index==0.9.40
- transformers, sentence-transformers
- FlagEmbedding (for reranking)
- tqdm, psutil, pandas

## Question Types

The dataset includes 4 question types (affects evaluation):

1. `inference_query` - Requires synthesizing info from multiple docs
2. `comparison_query` - Compares entities across docs
3. `temporal_query` - Requires temporal reasoning
4. `null_query` - Unanswerable (expects "Insufficient Information")

Evaluation scripts compute per-type and overall metrics.

## Common Pitfalls

1. **Forgetting to switch STAGING mode** - Always verify config.py before production runs
2. **Hardcoding GPU settings** - Use config.use_GPU instead
3. **Not cleaning LLM outputs** - Remove "</s>", "[/INST]" tokens for evaluation
4. **Running out of GPU memory** - Use torch.float16, not float32
5. **Session timeouts** - Use tmux/screen for long jobs
6. **Wrong output paths** - config.OUTPUT_PATH auto-handles staging/production
7. **Modifying prompts without updating evaluation** - Prompt changes may break metrics

## Submission Checklist

Final submission requires:

- [ ] `report.pdf` (5 pages max)
- [ ] `MyREADME.md` (execution instructions)
- [ ] `requirements.txt` (pip freeze output)
- [ ] `ranker[A-D].py` and `reranker[A-B].py`
- [ ] `RAG[A-D].py`
- [ ] `MyRetEval.py` and `MyRagEval.py`
- [ ] `output/production/ranker*.json` (6 files)
- [ ] `output/production/{ranker}_{llm}.json` (8 files)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a COMP4703 (Natural Language Processing) Assignment 2 project focused on implementing and evaluating Retrieval Augmented Generation (RAG) systems for multi-hop question answering. The project involves comparing different retrieval algorithms and Large Language Models (LLMs) to determine optimal configurations for answering factual questions that require information synthesis from multiple documents.

## Key Commands

### Running Experiments

**Stage 1: Retrieval**

```bash
python example_retrieval.py        # Run dense retrieval with embedding models
python evaluate_stage_1.py         # Evaluate retrieval performance (Hits@k, MAP, MRR)
```

**Stage 2: Reranking (Optional)**

```bash
python example_reranker.py         # Apply reranking to improve retrieval results
```

**Stage 3: RAG with LLM**

```bash
python example_rag.py              # Run full RAG pipeline with LLM generation
python evaluate_rag.py <output>    # Evaluate RAG results (Precision, Recall, F1)
```

**Test Runner Script**

```bash
bash test_runner.sh                # Run all three example scripts sequentially
```

### Environment Setup

The project uses a pre-configured conda environment (typically named `NLPA2` on the GPU server):

```bash
conda activate NLPA2
```

For local setup, dependencies include:

- PyTorch (GPU or CPU variant)
- llama-index==0.9.40
- transformers, sentence-transformers, FlagEmbedding
- rich, scikit-learn, tqdm, pandas, psutil

## Code Architecture

### Data Flow

1. **Corpus & Queries** → `data/corpus.json`, `data/rag.json` (full), or `data/sample-*.json` (for testing)
2. **Stage 1: Retrieval** → Embedding model creates vector index → Retrieves top-k documents
3. **Stage 2: Reranking** (optional) → Cross-encoder reranks top-k → Improves precision
4. **Stage 3: Generation** → LLM synthesizes answer from retrieved context → Outputs answer or "Insufficient Information"

### Core Components

**example_retrieval.py**

- Uses `JSONReader` to parse corpus documents with metadata (title, source, published_at)
- Implements `gen_stage_0()` function that:
  - Supports multiple embedding models (HuggingFace, OpenAI, Cohere, Voyage, Instructor)
  - Creates `VectorStoreIndex` with LlamaIndex
  - Retrieves top-k documents for each query
  - Optionally applies reranking via `FlagEmbeddingReranker`
  - Outputs JSON with retrieval_list (text + scores) and gold_list (evidence)
- Toggle `STAGING=True/False` to switch between sample and full datasets
- Key parameters: `topk=10`, `chunk_size=256`, embedding `model_name`

**example_rag.py**

- Uses HuggingFace transformers to load causal LLMs (e.g., Llama-2-7b-chat-hf)
- `run_query()` function handles chat template formatting and generation
- Default instruction prompt defines expected output format (direct answer or "Insufficient Information")
- Toggle `GPU=True/False` for device placement
- Reads Stage 1 retrieval results (`input_stage_1`), concatenates retrieved context
- Outputs JSON with query, prompt, model_answer, gold_answer, question_type

**evaluate_stage_1.py**

- Computes retrieval metrics: Hits@10, Hits@4, MAP@10, MRR@10
- Filters out `null_query` question types
- Matches retrieved text against gold evidence facts using substring matching

**evaluate_rag.py**

- Computes generation metrics: Precision, Recall, F1 (word-level overlap)
- Uses `count_overlap()` to tokenize and match predicted vs. gold answers
- Reports metrics per question_type and overall
- Handles answer extraction from verbose LLM responses

### Important Implementation Details

**Model Configuration**

- Retrieval models are specified in `rank_model_name` (e.g., "BAAI/llm-embedder")
- LLM models require HuggingFace model paths (e.g., "meta-llama/Llama-2-7b-chat-hf")
- Models are auto-downloaded to `~/.cache/huggingface/hub` if not cached

**GPU Considerations**

- Set `GPU=True` in example_rag.py when using GPU
- Use `torch.set_default_dtype(torch.float16)` for GPU to reduce memory usage
- Expected memory: ~13GB for Llama-2-7b (24GB GPU required with safety margin)
- Monitor with `nvidia-smi`

**Output Format Requirements**

- Stage 1 output must include: query, answer, question_type, retrieval_list, gold_list
- RAG output must include: query, model_answer, gold_answer, question_type
- Answers must be cleaned (remove "</s>", "[/INST]" tokens) for exact match evaluation

**Prompt Engineering**

- Default prompt in example_rag.py is carefully tested for evaluation compatibility
- Instructs model to return single word/entity or "Insufficient Information"
- Modifying prompts may break evaluation scripts if output format changes

**Data Handling**

- `JSONReader` wraps corpus documents as LlamaIndex `Document` objects with metadata
- `CustomExtractor` ensures metadata (title, source, published_at) is included in LLM context
- Question types: "inference_query", "comparison_query", "temporal_query", "null_query"

## Assignment Requirements Context

**Experimental Design**

- Must test 4 different retrieval algorithms (dense embedding models)
- Must test 4 different LLMs (7B-13B parameter range recommended)
- Typical strategy: Evaluate all 4 retrievers, select best 2, combine with 4 LLMs = 8 RAG configurations
- Must implement at least 2 additional custom evaluation metrics beyond provided ones

**GPU Budget**

- Total allocation: 40 hours
- Estimated breakdown: 4h retrieval + 16h RAG generation + margin for debugging
- Use `tmux` for long-running jobs to prevent session timeout

**Current Project Structure**

```
Assignment2/
├── data/                         # Dataset files
│   ├── corpus.json              # Full corpus (~1000 documents)
│   ├── rag.json                 # Full queries (~200 questions)
│   ├── sample-corpus.json       # Sample corpus for testing
│   └── sample-rag.json          # Sample queries for testing
├── docs/                         # Project documentation
│   ├── requirement.md           # Assignment requirements
│   └── roadmap.md               # Implementation roadmap
├── output/                       # Experiment outputs (gitignored)
├── example_retrieval.py          # Retrieval baseline implementation
├── example_reranker.py           # Reranking baseline implementation
├── example_rag.py                # RAG baseline implementation
├── evaluate_stage_1.py           # Retrieval evaluation script
├── evaluate_rag.py               # RAG evaluation script
├── test_runner.sh                # Quick test script for all stages
├── README.md                     # Project documentation
├── CLAUDE.md                     # This file - Claude Code guidance
├── llamaindex-tutorial.md        # LlamaIndex usage tutorial
└── A2-v2.0.pdf                   # Assignment specification
```

**Final Submission Structure**

```
S<student_id>/
├── report.pdf                    # 5-page experimental report
├── MyREADME.md                  # Execution instructions
├── requirements.txt              # pip freeze output
├── ranker[A-D].py               # 4 retrieval variants
├── rag[A-D].py                  # 4 LLM variants
├── MyRAGEval.py                 # Custom evaluation metrics (optional)
└── output/
    ├── ranker[A-D].json         # Retrieval results
    └── rag[A-D].json            # RAG results
```

**Critical Constraints**

- Use only STAGING=True for testing with sample data to conserve GPU time
- Switch to STAGING=False only for final full-dataset runs
- Each retrieval run ~1 hour, each LLM run ~2 hours on full dataset
- Total experiment time must fit within 40-hour GPU allocation

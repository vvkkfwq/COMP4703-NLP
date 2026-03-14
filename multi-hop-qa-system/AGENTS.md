# Repository Guidelines

## Project Overview

This repository is a compact multi-hop QA demo built around a fixed sample corpus and sample evaluation set. The app combines semantic retrieval, BM25 retrieval, reciprocal-rank-fusion hybrid retrieval, optional cross-encoder reranking, and OpenAI-backed generation behind a Streamlit UI.

The current implementation already has Milestones 1-3 largely in place. The next meaningful product work is Milestone 4 in [ROADMAP.md](/Users/vvkkfwq/Code/COMP4703-NLP/multi-hop-qa-system/ROADMAP.md): configuration switching, side-by-side comparison, and offline retrieval-metric summaries for each retrieval configuration.

## Project Structure

- `src/config.py`: central runtime configuration for paths, models, top-k, and device.
- `src/pipeline/ingest.py`: builds the Chroma vector store and serialized BM25 index from `data/sample-corpus.json`.
- `src/pipeline/rag.py`: end-to-end RAG pipeline that retrieves documents and calls OpenAI chat generation.
- `src/pipeline/demo.py`: CLI smoke test using the first question in `data/sample-rag.json`.
- `src/retriever/semantic.py`: Chroma-based dense retrieval using `BAAI/bge-large-en-v1.5`.
- `src/retriever/bm25_retriever.py`: disk-backed BM25 retrieval.
- `src/retriever/hybrid.py`: RRF fusion of semantic and BM25 retrieval, with optional reranking.
- `src/retriever/reranker.py`: cross-encoder reranker using `BAAI/bge-reranker-base`.
- `src/evaluation/metrics.py`: answer token-F1 and retrieval metrics (`NDCG@k`, `MAP@k`, `MRR@k`, `Hits@k`).
- `src/ui/app.py`: Streamlit application entry point.
- `data/`: fixed sample corpus and QA set.
- `chroma_db/` and `bm25_index.pkl`: local persisted retrieval indexes.

## Development Commands

Run commands from the repository root.

- `python -m src.pipeline.ingest`: rebuild Chroma and BM25 indexes from the sample corpus.
- `python -m src.pipeline.demo`: run a smoke test of the full RAG pipeline on the first preset sample.
- `streamlit run src/ui/app.py`: start the interactive QA UI.

Before running generation paths, copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Coding Conventions

- Keep Python changes small and localized. Follow existing PEP 8 style, type hints, and short module docstrings.
- Treat `src/config.py` as the single source of truth for paths, models, retrieval cutoffs, and device configuration. Do not hardcode duplicates elsewhere.
- Preserve the current module split: retrieval logic in `src/retriever`, orchestration in `src/pipeline`, metrics in `src/evaluation`, and presentation in `src/ui`.
- When adding retrieval metadata for display or evaluation, store transient scores in document metadata with underscore-prefixed keys such as `_score`, `_rrf_score`, and `_rerank_score`.
- Prefer additive changes that make configuration comparison easier rather than branching the code into separate pipelines.

## Working Notes

- This repository is currently sample-data oriented. `src/config.py` points to `data/sample-corpus.json` and `data/sample-rag.json`.
- The default retriever shown in the UI is hybrid retrieval with reranking. Milestone 4 should expose semantic-only vs hybrid retrieval and reranker on/off as first-class UI controls.
- Retrieval evaluation should use the existing `retrieval_metrics` helper rather than duplicating metric formulas in the UI.
- The app currently uses Apple Silicon defaults (`EMBED_DEVICE = "mps"`). If device behavior changes, update `src/config.py` rather than individual retrievers.

## Change Strategy

For Milestone 4 work, prefer this sequence:

1. Make retriever construction configurable from a small set of typed options.
2. Add UI sidebar controls for retrieval mode and reranker toggle.
3. Support running multiple configurations for the same question without duplicating prompt or rendering logic.
4. Precompute retrieval-only metrics across `sample-rag.json` and surface concise summaries in the sidebar.

Keep evaluation and retrieval-comparison logic deterministic and independent from OpenAI generation where possible.

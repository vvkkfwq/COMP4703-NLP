# CLAUDE.md

This repository uses [AGENTS.md](/Users/vvkkfwq/Code/COMP4703-NLP/multi-hop-qa-system/AGENTS.md) as the primary workspace guidance file.

## Quick Context

- App entry point: `streamlit run src/ui/app.py`
- Index build: `python -m src.pipeline.ingest`
- Smoke test: `python -m src.pipeline.demo`
- Core configuration: `src/config.py`
- Next planned work: Milestone 4 in [ROADMAP.md](/Users/vvkkfwq/Code/COMP4703-NLP/multi-hop-qa-system/ROADMAP.md)

## Implementation Notes

- Use the existing retriever split in `src/retriever/` rather than adding alternate one-off scripts.
- Reuse `src.evaluation.metrics.retrieval_metrics` for retrieval comparison features.
- Keep OpenAI configuration in `.env` and runtime settings in `src/config.py`.
- The repository already contains persisted local indexes in `chroma_db/` and `bm25_index.pkl`.

Refer to [AGENTS.md](/Users/vvkkfwq/Code/COMP4703-NLP/multi-hop-qa-system/AGENTS.md) for the full project guidance.

# Repository Guidelines

## Project Structure & Module Organization

The root directory hosts the runnable pipelines (`RAGA.py`, `RAGB.py`, `rankerA.py`, etc.) plus reference notebooks such as `example_rag.py`. Shared assets live under `data/`, with `sample-*` files for staging-sized runs and `corpus.json` / `rag.json` for production sweeps. Generated artifacts are separated into `output/staging` and `output/production`, while `logs/` mirrors that split for execution and evaluation logs. Long-form specs live in `docs/`, and `report/` tracks deliverables. Update `config.py` whenever you introduce a new dataset, GPU behavior, or ranker so downstream scripts inherit the paths automatically.

## Build, Test, and Development Commands

- `conda activate COMP4703A2 && python example_retrieval.py`: smoke-test retrieval settings against the sample corpus.
- `bash run_rankers.sh`: execute all ranker and reranker variants and stream logs into `logs/<stage>`.
- `bash run_rag.sh`: run the four configured RAG pipelines end-to-end.
- `bash evaluate_rankers.sh`: batch-evaluate ranker outputs via `MyRetEval.py`, persisting metric summaries to `logs/<stage>/evaluation_summary.txt`.
- `python evaluate_rag.py output/staging/RAGA.json data/sample-rag.json`: ad-hoc precision/recall/F1 calculation for a single RAG JSON (swap paths for production artifacts).

## Coding Style & Naming Conventions

Code is Python 3.10 and should follow PEP 8: four-space indentation, 88-ish character lines, `snake_case` functions, and `CamelCase` classes. Keep configuration toggles (`GPU`, `STAGING`, `use_GPU`) in uppercase to match existing modules, prefer type hints (see `example_rag.py`), and document non-obvious helpers with concise docstrings. Any new scripts should load shared paths from `config.py` instead of hardcoding directories.

## Testing & Evaluation Guidelines

Use the staging corpus (`data/sample-*`) for fast feedback before scaling to full `data/` assets. Rely on `evaluate_rankers.sh` and `evaluate_rags.sh` after every material change; both commands populate log files with Hits@K, MAP, MRR, NDCG, and overall micro metrics. For quick spot checks, call `python MyRagEval.py output/staging/llm-embedder-reranker_llama3.json` and review the printed overall block. Keep saved predictions in `output/<stage>/` and avoid editing generated JSONs manually—rerun the pipeline instead so metrics remain reproducible.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commits (`feat: ...`, `fix: ...`). Mirror that style, keep subjects under ~65 characters, and describe user-facing behavior (“feat: add rag evaluation”) rather than implementation details. Pull requests should include: (1) a summary of the pipeline(s) touched, (2) expected metric deltas or log snippets, (3) any dependency or config changes (e.g., new model names, GPU requirements), and (4) screenshots or log excerpts when output formatting changes. Reference issue IDs where applicable and check in only deterministic artifacts (never large checkpoints).

## Environment & Configuration Tips

`config.py` controls GPU usage, staging vs. production paths, and the mapping of ranker identifiers to output files. Flip `is_STAGING` to `True` before experimenting locally so logs flow into `logs/staging` and results stay isolated. Respect the existing directory contract when adding new pipelines: write to `OUTPUT_PATH` from the config and emit progress via `logs/<stage>/<script>.log` so the automation scripts and evaluators continue to pick up your run products.

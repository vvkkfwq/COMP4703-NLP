# Repository Guidelines

## Project Structure & Module Organization

- **Root scripts:** `example_retrieval.py`, `example_reranker.py`, and `example_rag.py` demonstrate retrieval, reranking, and end-to-end RAG flows. Toggle `GPU`/`STAGING` switches at the top to match your environment.
- **Evaluators:** `evaluate_stage_1.py` scores retrieval outputs, while `evaluate_rag.py` measures answer overlap. Both expect JSON predictions placed in `output/`.
- **Data:** `data/` stores gold corpora (`corpus.json`, `rag.json`) and lightweight samples for quick iterations. Add new fixtures as JSON with consistent keys (`query`, `model_answer`, `question_type`).
- **Docs & assets:** `docs/` houses assignment requirements and roadmap references; `A2-v2.0.pdf` captures the original brief. Generated artefacts should live under `output/` to stay git-ignored.

## Build, Test, and Development Commands

- `conda activate NLPA2` — align with the managed environment before running scripts.
- `python example_retrieval.py` — fetches candidate passages and writes them to `output/`.
- `python example_reranker.py` — refines retrieval lists with the configured reranker.
- `python example_rag.py` — runs generation; update checkpoints via `huggingface_hub`.
- `python evaluate_stage_1.py output/<file>.json` — report Hits@10/MAP for retrieval stage.
- `python evaluate_rag.py output/<file>.json data/sample-rag.json` — compute precision/recall/F1 for answers.
- `bash test_runner.sh` — orchestrates the three example pipelines; ensure `NLPA2` exists or edit the shebang variables.

## Coding Style & Naming Conventions

- Stick to Python 3.10, two-space indentation, and trailing-newline files to match existing sources.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and keep helper modules lowercase.
- Favour explicit docstrings and type hints for new public functions. Group imports as stdlib, third-party, then local.
- Persist outputs via `save_list_to_json`/`wr_dict` helpers to maintain consistent JSON structure.

## Testing Guidelines

- Cover both retrieval and generation paths: use `data/sample-*` fixtures for smoke checks, then full corpora for final scoring.
- Name intermediate JSONs as `output/<stage>-<model>.json` so evaluators can locate them.
- Before submitting, run both evaluation scripts and paste key metrics into your PR for traceability.

## Commit & Pull Request Guidelines

- Follow conventional commit prefixes seen in history (`feat:`, `refactor:`, `fix:`) with concise, imperative summaries.
- Limit commits to logical units (data prep, model tweak, evaluator change) to simplify reviews.
- PRs should summarize intent, list datasets/models touched, link assignment items, and include commands/metrics executed. Attach screenshots only when UI changes are involved.

## Environment & Data Tips

- Keep large checkpoints out of git; document any required models in the PR body.
- When iterating locally without a GPU, set `GPU = False` and reduce `max_new_tokens` to avoid OOM.
- Clear stale artefacts with `rm_file` utilities or manual deletion before re-running experiments to avoid mixing runs.

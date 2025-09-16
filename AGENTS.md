# Repository Guidelines

## Project Structure & Module Organization
- `predict.py` hosts the Cog `Predictor` class; keep inference helpers close to the class and gate large utilities behind clearly named functions within this module.
- `cog.yaml` describes the deployment image (Python 3.13, optional apt packages). Update it whenever dependencies or GPU needs change.
- `checkpoints/` stores pre-trained weights (`*.pth`, `*.pt`, configs). Never edit these in place—add new assets under versioned subfolders and document provenance.
- `requirements.txt` is intentionally minimal. Add exact versions for any runtime dependency you introduce and note optional extras with comments.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — recommended local environment before installing deps.
- `pip install -r requirements.txt` — install runtime and testing dependencies.
- `cog predict -i image=@/path/to/example.png scale=1.5` — execute the predictor end-to-end; adjust inputs to mirror your scenario.
- `python -m pytest` — run the test suite once `tests/` is populated; use `-k` to target specific behaviors.

## Coding Style & Naming Conventions
Use 4-space indentation and follow PEP 8 throughout. Name helper functions and files in `snake_case`, classes in `PascalCase`, and configuration constants in `UPPER_SNAKE_CASE`. Annotate new public APIs with type hints and include short docstrings stating inputs, side effects, and return types. Keep Cog inputs declared through `Input(...)` and document defaults inline.

## Testing Guidelines
Place unit tests under `tests/` mirroring the source tree (e.g., `tests/test_predict.py`). Prefer `pytest` fixtures for loading checkpoint snippets or sample waveforms, and mock heavyweight models when possible. Aim to cover input validation, shape/format conversions, and any custom preprocessing or postprocessing helpers. Record regression scenarios from previous issues as parametrized tests.

## Commit & Pull Request Guidelines
The mirror lacks historical Git metadata, so adopt concise, imperative commit subjects ("Add vocoder warm-up"), optionally followed by a one-line body. Each PR should include: goal-oriented summary, key implementation notes, any configuration or checkpoint changes, and manual or automated test evidence. Link related tickets and provide audio artifacts or logs when output changes.

## Model Assets & Security Notes
Large checkpoints are tracked already; do not re-commit binaries. Store new securely sourced weights under `checkpoints/<model>-<version>/` with checksums. Never embed secrets or access tokens in code or configs; rely on runtime environment variables and document required names in PRs.

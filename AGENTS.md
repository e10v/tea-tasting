# AGENTS

## Project structure

- `docs/`: documentation (for Material for MkDocs).
- `examples/`: examples as marimo notebooks, auto-generated from `docs/*.md` guides, excluded from linting and type checking; don't edit directly.
- `src/_internal`: docs tooling (marimo example generation, markdown extensions).
- `src/tea_tasting/`: public package code:
    - `src/tea_tasting/metrics/`: built-in metrics and metrics base classes.
    - `src/tea_tasting/aggr.py`: aggregated statistics helpers (count/mean/var/cov), Delta method for variance and covariance of ratio metrics.
    - `src/tea_tasting/config.py`: default parameter values used by other modules; can be set by users.
    - `src/tea_tasting/datasets.py`: simulated data generators.
    - `src/tea_tasting/experiment.py`: experiment orchestration (analyze/solve_power/simulate) and result classes.
    - `src/tea_tasting/multiplicity.py`: multiple testing corrections.
    - `src/tea_tasting/utils.py`: parameters validation, results formatting and rendering, graceful division by zero.
    - `src/tea_tasting/version.py`: package version extracted from `src/tea_tasting/_version.txt`.
- `tests/`: unit tests for pytest; mirrors `src/tea_tasting/` structure.

## Code style

- Follow PEP 8.
- Follow Google Python Style Guide.
- Exceptions:
    - Max line length: 88.
    - Use `uv run ruff check` for linting instead of `pylint`.
    - Use `uv run pyright` for type checking instead of `pytype`.
- When guides disagree, follow project tooling (Ruff/Pyright) and existing local conventions in this repo.
- Imports must satisfy Ruff isort rules (sorted within sections; two blank lines after imports).
- Type annotations are required for all modules including tests and internal tooling; keep type hints accurate and narrow.
- Using `cast` is usually discouraged: prefer correct annotations; if needed, use suppression (`# pyright: ignore`) over `cast`.
- All code in `src/tea_tasting/` should be covered by tests in `tests/`, and expected code coverage is 100%.
- Prefer pure functions and deterministic numeric behavior.

## Docs style

- When writing guides and docstrings, follow the Microsoft Writing Style Guide and Google developer documentation style guide.
- All classes and functions except tests and auto-generated examples must have Google-style docstrings.
- Exception: if the class has an explicit `__init__`, put the class documentation in `__init__`'s docstring (suppress rule `D101` when Ruff reports it); otherwise, use a class docstring.

## Code testing and checking

- Tests and checks for code changes:
    - Tests: `uv run coverage run -m pytest`
    - Code coverage (after tests): `uv run coverage report -m`
    - Full doctest: `uv run pytest --doctest-continue-on-failure --doctest-glob=*.md --doctest-modules --ignore=examples/ --ignore=tests/ --ignore-glob=src/_*`
    - Linting: `uv run ruff check`
    - Type checking: `uv run pyright`
- Run them after creating or updating Python modules; fix errors.
- Do not run a code formatter; `uv run ruff check --fix` is acceptable for lint autofixes.

## Docs testing and checking

- Tests and checks for docs changes:
    - Docs-only doctest: `uv run pytest --doctest-continue-on-failure --doctest-glob=*.md --ignore=tests/`
    - Markdown lint: `markdownlint-cli2 "*.md" "docs/*.md"`
- Run them after creating or updating markdown files; fix errors.

## Code versioning

- Develop only in the `dev` branch, never merge into `main`; leave all merges to me.
- Make atomic commits.
- Follow the Conventional Commits standard in commit messages but skip optional scope after type.

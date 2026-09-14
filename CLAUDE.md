# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project

`pkpdutils` is a python library for the pharmacokinetic and pharmacodynamic analysis of timecourses and parameters: non-compartmental analysis, uncertainty propagation, curve fitting, statistics on parameters (tests, bioequivalence, drug–drug interactions, meta-analysis) and figures. Pure library, no CLI entry points. Requires python >= 3.13, packaged with hatchling (version is read from `src/pkpdutils/__init__.py`). Runtime dependencies are `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib` and `rich`. It was `pkdb_analysis` until 0.3.1; the design of the rewrite is in `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`.

## Commands

```bash
# environment (uv based)
uv sync --extra dev
uv run pre-commit install

# tests
pytest                                   # all tests, in parallel with pytest-xdist
pytest -n 0                              # in one process, e.g. for --pdb
pytest tests/test_units.py               # single file
pytest tests/test_units.py::test_parse_unit  # single test
tox r -e py3.14                          # single tox env (py3.13, py3.14 available)
tox run-parallel                         # full matrix + ty

# lint / format / types
ruff check
ruff format
tox -e ty                       # ty type check (config in [tool.ty] in pyproject.toml)
uvx ty check                    # same check, straight from the working tree

# docs
uv run zensical build --clean
uv run python scripts/llms_txt.py
uv run zensical serve

# examples, they are modules of the `examples` package and not part of pkpdutils
python -m examples.timecourses
```

`develop` is the default branch and takes every change through a pull request; direct pushes are rejected by the rulesets in `.github/rulesets/` (applied with `.github/rulesets/apply.sh`), which require the `tests`, `ruff`, `ty` and `docs` checks. `main` only tracks the latest release and is fast-forwarded by the `sync-main` job of the release workflow, never by hand.

Release steps are in `docs/development.md`: the release is prepared on a branch, `uvx bump-my-version bump [dev|major|minor|patch]` updates `src/pkpdutils/__init__.py` and `CITATION.cff` and commits without tagging, and the tag is created on `develop` after the pull request was merged, which triggers the PyPI release workflow. A development version is finalized with `uvx bump-my-version bump dev` (`1.0.0.dev0` becomes `1.0.0`), while `major|minor|patch` start the next development cycle.

Documentation is [Zensical](https://zensical.org/): markdown sources in `docs/`, configured in `zensical.toml`, built into the gitignored `site/`. The API reference is rendered from the docstrings by mkdocstrings; a page in `docs/api/` is just `::: pkpdutils.<module>`. Formulas use `pymdownx.arithmatex` with MathJax. `scripts/llms_txt.py` runs after the build and writes `llms.txt`, `llms-full.txt` and the markdown of every page into `site/`.

## Architecture

**`units.py`.** One pint `UnitRegistry` per process, `ureg`; `Q_` creates quantities, `Quantity` and `Unit` are the annotation types. `check_dose_unit` validates dose units, `normalize_volume` and `normalize_clearance` convert results to `liter`/`liter/kg` and `liter/hour`/`liter/hour/kg`. Numerics run on plain arrays in the units of the input, pint is used at the boundaries.

**`timecourse.py`.** `Timecourse` is one curve (pydantic, frozen): `time`, `value`, units, optional `sd`/`se`/`n`, a `Dose` (`amount`, `unit`, `Route`, `time`, `duration`) and metadata. `Timecourses` wraps an `xarray.Dataset` with a `time` dimension plus any sample dimensions, the variables `value` and optionally `sd`, `se`, `n`, the dose as `dose_amount`/`dose_time`/`dose_duration` and units in `attrs["units"]`; `times` and `values` return the `(samples..., time)` arrays every analysis works on. `DosingRegimen` describes repeated dosing for steady state analyses.

`console.py` (rich console, for scripts and examples) and `log.py` provide the shared output/logging. Modules get their logger from the standard library with `logging.getLogger(__name__)`. The package never configures logging: `log.enable_rich_logging()` is the opt-in for scripts. Library code logs, it does not print, and log calls use lazy `%s` formatting rather than f-strings (enforced by ruff `G`).

## Conventions

- Type checking is done with [ty](https://docs.astral.sh/ty/). `[tool.ty.terminal] error-on-warning = true` means warnings fail the check, so the tree must stay at zero diagnostics. Suppress a diagnostic with a rule-specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`. ty also runs as a pre-commit hook (`--extra dev`).
- Every module, class and function of the package carries full type annotations and a google style docstring (ruff `D`); `examples/`, `tests/` and `scripts/` are exempt from the docstring rules. Docstrings of parameters and methods carry the formula and a citation key of `docs/references.md`.
- Results are `xarray.Dataset` objects with one variable per parameter over the sample dimensions of the input and `attrs["units"]` on every variable.
- `examples/` at the top level holds the runnable examples, they are not part of the package and are run as modules (`python -m examples.timecourses`). An example writes into the current working directory and never opens a window; the `conftest.py` at the root selects the `Agg` backend for the test session. `tests/examples/test_examples.py` runs them.
- Library code does not call `plt.show()`; figures are returned or saved.
- Test fixtures live in `tests/data/`; `tests/data/reference/nca_reference.json` holds the results of `pkdb_analysis` 0.3.1 on them and is the regression reference of the NCA.
- Markdown carries no hard line wraps: a paragraph, a list item or a table row is a single line. Code fences, headings and the rows of badges keep their line structure.
- Release notes go in `release-notes/` as part of a release commit.

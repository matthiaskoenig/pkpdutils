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
python -m examples.nca_single
python -m examples.nca_batch
python -m examples.steady_state
python -m examples.group_uncertainty
python -m examples.fitting_exponential
python -m examples.emax
python -m examples.dose_proportionality
python -m examples.covariate
```

`develop` is the default branch and takes every change through a pull request; direct pushes are rejected by the rulesets in `.github/rulesets/` (applied with `.github/rulesets/apply.sh`), which require the `tests`, `ruff`, `ty` and `docs` checks. `main` only tracks the latest release and is fast-forwarded by the `sync-main` job of the release workflow, never by hand.

Release steps are in `docs/development.md`: the release is prepared on a branch, `uvx bump-my-version bump [dev|major|minor|patch]` updates `src/pkpdutils/__init__.py` and `CITATION.cff` and commits without tagging, and the tag is created on `develop` after the pull request was merged, which triggers the PyPI release workflow. A development version is finalized with `uvx bump-my-version bump dev` (`1.0.0.dev0` becomes `1.0.0`), while `major|minor|patch` start the next development cycle.

Documentation is [Zensical](https://zensical.org/): markdown sources in `docs/`, configured in `zensical.toml`, built into the gitignored `site/`. The API reference is rendered from the docstrings by mkdocstrings; a page in `docs/api/` is just `::: pkpdutils.<module>`. Formulas use `pymdownx.arithmatex` with MathJax. `scripts/llms_txt.py` runs after the build and writes `llms.txt`, `llms-full.txt` and the markdown of every page into `site/`.

## Architecture

**`units.py`.** One pint `UnitRegistry` per process, `ureg`; `Q_` creates quantities, `Quantity` and `Unit` are the annotation types. `check_dose_unit` validates dose units, `normalize_volume` and `normalize_clearance` convert results to `liter`/`liter/kg` and `liter/hour`/`liter/hour/kg`. Numerics run on plain arrays in the units of the input, pint is used at the boundaries.

**`timecourse.py`.** `Timecourse` is one curve (pydantic, frozen): `time`, `value`, units, optional `sd`/`se`/`n`, a `Dose` (`amount`, `unit`, `Route`, `time`, `duration`) and metadata. `Timecourses` wraps an `xarray.Dataset` with a `time` dimension plus any sample dimensions, the variables `value` and optionally `sd`, `se`, `n`, the dose as `dose_amount`/`dose_time`/`dose_duration` and units in `attrs["units"]`; `times` and `values` return the `(samples..., time)` arrays every analysis works on. `DosingRegimen` describes repeated dosing for steady state analyses.

**`result.py`.** `ParameterResult` is the shared container of `NCAResult` and `FitResult`: an `xarray.Dataset` over the sample dimensions with `attrs["units"]` per variable and the integer `flags`, the classification of the variables (`parameters`, `derived_variables`, `point_variables`, with the suffixes of `base_name`), `to_quantities`, `to_dataframe` (one row per sample, point variables excluded), `flag_table` and `summarize(dim)`.

`console.py` (rich console, for scripts and examples) and `log.py` provide the shared output/logging. Modules get their logger from the standard library with `logging.getLogger(__name__)`. The package never configures logging: `log.enable_rich_logging()` is the opt-in for scripts. Library code logs, it does not print, and log calls use lazy `%s` formatting rather than f-strings (enforced by ruff `G`).

**`nca/` - non-compartmental analysis.** `nca(timecourses, options)` flattens the sample dimensions to `(N, n_time)` arrays and runs `compute_parameters` (`nca/nca.py`), the pure numpy core: `auc.py` packs the valid points of every row to the front (`pack_valid`) and sums the trapezoid segments (`segment_areas`, linear / linear-up-log-down / log), `terminal.py` evaluates every candidate window of the terminal regression at once with suffix sums (`window_statistics`) and picks it by the `TerminalPhase` rule (best adjusted R², last n, all after tmax, manual), `steady_state.py` adds the parameters of a dosing interval and `superposition`. `options.py` holds `NCAOptions`, `TerminalPhase`, the enumerations and `NCAFlag`; `result.py` holds `NCAResult` (an `xarray.Dataset` over the sample dims with `attrs["units"]` per variable and the integer `flags`) and `parameter_unit`, which derives the units with pint (`PARAMETER_UNITS` in `nca.py` maps every parameter to a unit expression). `nca_single` wraps one `Timecourse`. The rows run in chunks of `chunk_rows`, with `n_workers` over a `ProcessPoolExecutor`; the steady state path is chunked the same way. `tests/nca/test_reference.py` reproduces `pkdb_analysis` 0.3.1 with `AUCMethod.LINEAR` and `TerminalMethod.ALL_AFTER_TMAX`. `uncertainty.py` propagates `sd`/`se` of group curves: `bootstrap` resamples every point (`resample_values`), runs `run_rows` on the `N*B` replicate rows and reduces them (`reduce_replicates`); `delta` perturbs every point once and propagates `se` through the numerical Jacobian; `NCAOptions.resolve_uncertainty` picks the bootstrap by default for group data. Variables `x_sd`, `x_se`, `x_ci_low`, `x_ci_high`, `x_geomean`, `x_geocv`, `n`; `NCAResult.summarize(dim)` gives the same layout for individual results; `partial_auc` (`nca.py`) is the area between two times.

**`fit/` - curve fitting.** `model.py` holds `Model` (`predict`, `derived`, `initial_guess`), `ModelParameter` (name, unit expression, bounds, whether it is positive) and `parameter_unit_expression`, which derives a parameter unit from `[x]` and `[y]` without any normalization, so a fit reports its parameters in the raw units of the data. `options.py` holds `FitOptions`, `Weighting`, `ParameterScale` and `FitFlag`; the model library is `models_exponential.py` (`MonoExp`, `BiExp`, `TriExp`, `Bateman`, curve stripping for the guesses), `models_response.py` (`Emax`, `SigmoidEmax`, `Imax`, `SigmoidImax`) and `models_linear.py` (`Linear`, `LogLinear`, `Power`, `Allometric`), re-exported by `models.py`. `engine.py` is the core: `fit_row` searches in the scaled space (`log10` for positive parameters) with `scipy.optimize.least_squares` from `n_starts` Latin hypercube starts, derives the covariance from the Jacobian, transforms the t intervals back, propagates the derived parameters with the delta method, computes the statistics (`aic`, `aicc`, `bic` with `K = k + 1` estimated parameters) and runs the residual bootstrap; `fit_rows` draws one seed per row and maps them over a `ProcessPoolExecutor` for `n_workers > 1`; `build_result` assembles the dataset. `result.py` holds `FitResult(ParameterResult)` with `predict`, `predict_all` and `correlation`; `frontends.py` has `fit_timecourses` (a batch, times relative to the dose) and `fit_table` (a variable against a coordinate along one dimension of any dataset); `compare.py` ranks models by AICc and the Akaike weights; `proportionality.py` applies the confidence interval criterion of dose proportionality to a `Power` fit.

**`plot/` - figures.** `style.py` (`PlotStyle`), `timecourse.py` (`plot_timecourse`), `nca.py` (`plot_nca`, `plot_nca_grid`, `draw_nca_panel`), `fit.py` (`plot_fit`, `plot_goodness_of_fit`, `plot_dose_proportionality`). Functions return the `Figure`, take `ax`, never show.

## Conventions

- Type checking is done with [ty](https://docs.astral.sh/ty/). `[tool.ty.terminal] error-on-warning = true` means warnings fail the check, so the tree must stay at zero diagnostics. Suppress a diagnostic with a rule-specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`. ty also runs as a pre-commit hook (`--extra dev`).
- Every module, class and function of the package carries full type annotations and a google style docstring (ruff `D`); `examples/`, `tests/` and `scripts/` are exempt from the docstring rules. Docstrings of parameters and methods carry the formula and a citation key of `docs/references.md`.
- Results are `xarray.Dataset` objects with one variable per parameter over the sample dimensions of the input and `attrs["units"]` on every variable.
- `examples/` at the top level holds the runnable examples, they are not part of the package and are run as modules (`python -m examples.timecourses`). An example writes into the current working directory and never opens a window; the `conftest.py` at the root selects the `Agg` backend for the test session. `tests/examples/test_examples.py` runs them.
- Library code does not call `plt.show()`; figures are returned or saved.
- Test fixtures live in `tests/data/`; `tests/data/reference/nca_reference.json` holds the results of `pkdb_analysis` 0.3.1 on them and is the regression reference of the NCA.
- Markdown carries no hard line wraps: a paragraph, a list item or a table row is a single line. Code fences, headings and the rows of badges keep their line structure.
- Release notes go in `release-notes/` as part of a release commit.

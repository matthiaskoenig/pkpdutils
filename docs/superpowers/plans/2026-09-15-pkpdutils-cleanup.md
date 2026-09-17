# pkpdutils Cleanup and Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the bugs found by the audit, make the library fast on batches (caches, thread pool, chunking, vectorized readers), make the API consistent, add the publication tables and figures, and turn the documentation into runnable workflows with a gallery and diagrams; delivered as four pull requests into `develop`.

**Architecture:** No new subsystems. PR A touches every module locally (bug fixes, shared helpers `stats/sample.py`, `nca/auc.py`, `result.py`, `plot/_common.py`). PR B adds `pkpdutils/parallel.py` (shared executors, row splitting) and rewrites the hot paths in place. PR C changes signatures for consistency and adds `Timecourses` ergonomics, `summary_table` and the table/figure functions. PR D adds `scripts/render_examples.py`, `docs/gallery.md`, `docs/workflows.md`, the diagrams and the runnable snippets.

**Tech Stack:** numpy, scipy, pandas, xarray, pydantic, matplotlib, pytest; `concurrent.futures`; Zensical with mermaid.

**Spec:** `docs/superpowers/specs/2026-09-15-cleanup-and-usability-design.md` (binding). The survey reports with the reproductions, signatures, benchmarks and diagram code are session artifacts: `.superpowers/survey/api-bugs.md` (bugs B1-B33, inconsistencies, proposals U1-U10, code quality), `.superpowers/survey/performance.md` (benchmarks, findings F1-F14, scripts), `.superpowers/survey/docs-usability.md` (checklist, signatures, audit, mermaid code, gallery). Every task names the sections it consumes; when the reports are absent, the spec carries the decisions and the reproduction is rebuilt from its description.

## Global Constraints

- python 3.13 and 3.14; runtime dependencies unchanged
- ty `error-on-warning = true`: zero diagnostics; rule-specific ignores only; narrowing asserts in tests
- google docstrings with annotations on everything in `src`; formula docstrings `r"""` with single backslashes and a citation key
- library code logs (`logging.getLogger(__name__)`, lazy `%s`), never prints, never `plt.show()`
- `uv run pytest -q -W error` pristine; `uv run ruff check`, `uv run ruff format --check`, `uvx ty check`, `uv run zensical build --clean --strict` clean; `uv run tox run-parallel` green before every PR
- `tests/nca/test_reference.py` unchanged and passing
- every bug fix carries a regression test built from the audit's reproduction; every behaviour change is documented on its page and, when breaking, listed in the spec's "Breaking changes"
- never the em dash (U+2014); commit messages carry NO `Co-Authored-By` and end with `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`; markdown has no hard line wraps
- one branch per PR (`cleanup-bugs`, `cleanup-performance`, `cleanup-api`, `cleanup-docs`), each off the merged `develop` of the previous; PRs squash-merged with auto-merge when the checks pass
- parallel implementers commit only their own files by path; never `git add -A`

---

## PR A: bug fixes and shared helpers (branch `cleanup-bugs`)

### Task A1: `stats` fixes and shared primitives

**Files:** `src/pkpdutils/stats/sample.py`, `tests.py`, `ratio.py`, `bioequivalence.py`, `meta.py`, `ddi.py`; `tests/stats/*.py`
**Consumes:** api-bugs.md B1, B10, B11, B12, B13, B23, B24, B25, B26; code quality items on the stats duplicates (`_welch_df`, pooled SD, exponentiated t interval, `_pair`, `_labels_match`, `_log_positive`); inconsistencies rows "enum arguments", "stats keyword-only marker", "stats default kind", "stats to_dict/to_dataframe", "ParameterSample.n type", "multiple_comparison else".
**Produces:** public helpers in `stats/sample.py`: `welch_df(var_a, n_a, var_b, n_b)`, `pooled_sd(sd_a, n_a, sd_b, n_b)`, `cohen_d(mean_a, sd_a, n_a, mean_b, sd_b, n_b) -> (d, g)`, `exp_t_interval(center, se, df, ci_level) -> (low, high)`, `paired_values`, `paired_indices`, `labels_match`, `log_positive` (the private names removed, callers updated); `coerce(value, enum_type)` in `stats/sample.py` used at every public entry point so `compare(a, b, scale="log", test="paired_t")` and `multiple_comparison(p, "holm")` work (`ValueError` listing the members for an unknown string); degenerate inputs give NaN or a `ValueError` naming the sample, never `ZeroDivisionError` or a `RuntimeWarning` (`np.errstate` around the scipy calls that warn on identical samples); `ParameterSample(sd=-1)` raises; `PooledEffect.__eq__`; `to_dict` on `Heterogeneity`, `Study`, `MetaResult`, `BEResult` (`BEResult.to_dict` = its parameters' dicts keyed by name); `*` markers on `effects_from_arrays`, `multiple_comparison`, `substrate_sensitivity`; `effects_from_arrays(kind=HEDGES_G)` default; `multiple_comparison` raises on an unknown method; `ParameterSample.n: int | None`.
**Tests:** one regression test per bug (the reproduction of the report as the test body), `test_string_arguments_are_coerced`, and the helper unit tests. Commit: `Coerce string options, handle degenerate samples and share the stats primitives`.

### Task A2: NCA fixes and shared helpers

**Files:** `src/pkpdutils/nca/uncertainty.py`, `steady_state.py`, `nca.py`, `terminal.py`, `options.py`, `intervals.py`, `auc.py`, `src/pkpdutils/result.py`; `tests/nca/*.py`, `tests/test_result.py`
**Consumes:** api-bugs.md B2, B3, B17, B18, B19, B21, B29; code quality items `_take` x3, `decode_flags` x3, `repeat` x2, `delta`'s `model_copy` for the spread.
**Produces:** delta `x_geocv` = `sqrt(exp(sigma_log^2) - 1)` with `sigma_log^2 = log(1 + (x_sd / x)^2)` (test: within 5 % of the bootstrap on the survey's reproduction); mixed batches analysed row-wise by protocol (`is_multiple_dose` per row, not per batch; a single dose row in a multiple dose batch keeps `cl`/`cl_f` and has NaN steady state variables; test with one 5-dose and one 1-dose subject asserting both clearances); `summarize` skips `discrete_parameters` for the uncertainty variables (test: no `lambda_z_n_points_se`); `LAST_N` honours `exclude_cmax`; zero dose -> NaN `cl`, `auc_inf_dn`, `cmax_dn` with a debug log; `Kind.EFFECT` applies `lloq`/`blq` (documented: the area of an effect is always linear); `partial_auc` before the first sample: `C(0)=0` for `ORAL`, `c0` for `IV_BOLUS`, NaN for `IV_INFUSION` (test each); `take_rows(a, idx)` in `nca/auc.py` replacing the three `_take`; `decode_flags(flag_type, value)` in `result.py` used by `nca/options.py` and `fit/options.py`; `repeat_rows` helper in `uncertainty.py`; `delta(..., spread=...)` argument. Commit: `Fix the delta geocv, mixed dose batches, summarize of discrete parameters and share the NCA helpers`.

### Task A3: data model and io fixes

**Files:** `src/pkpdutils/timecourse.py`, `units.py`, `io.py`; `tests/test_timecourse.py`, `test_timecourses.py`, `test_io.py`, `test_units.py`
**Consumes:** api-bugs.md B4, B5, B6, B7, B8, B9, B20, B22, B30, B31, B32, B33; code quality items on the io duplicates (`_pad`, route/unit agreement, "collect the times of a subject" x3), the dead `or "mg"` fallback, the all-NaN `dose_duration` for non-infusion batches.
**Produces:** `unit=""` raises `ValueError("... use 'dimensionless'")`; `tissue` in `attrs["tissue"]` of the batch and restored by `_timecourse`; `write_events` columns `SD`, `SE`, `N` when present and `read_events(..., sd_col="SD", se_col="SE", n_col="N")`; `to_dataframe`/`from_dataframe` round trip of a ragged batch keeps the sample order and the grids (test: `from_dataframe(batch.to_dataframe(), ...) == batch` for a ragged batch); readers wrap the `Timecourse`/`Dosing` construction and raise `ValueError` naming the subject (single observation, duplicate times, missing infusion duration, NaN dose time); `Route` strings coerced in `from_arrays`, `from_dataset`, `Dose`, `Dosing`, the readers (`Route("oral")`, case-insensitive); `IU`/`IU/kg` doses accepted by `check_dose_unit` (the registry defines `IU`); a shared grid with NaN values stays a shared grid through `to_events` -> `from_events` (missing values become `MDV 1` rows with the time and are read back as NaN); io helpers shared with `timecourse.py` (`pad_protocols` public in `timecourse.py`, used by io); `dose_duration` present only for infusion batches (or documented as always present: choose "always present" if removing it breaks `Timecourses` invariants, and say so). Commit: `Fix the batch round trips, the reader errors and the dose unit checks`.

### Task A4: plot and fit fixes, `plot/_common.py`

**Files:** `src/pkpdutils/plot/_common.py` (new), `plot/timecourse.py`, `nca.py`, `fit.py`, `parameters.py`, `ratio.py`, `meta.py`, `src/pkpdutils/fit/engine.py` (docstring), `src/pkpdutils/nca/options.py` + `fit/options.py` (`n_workers` docstrings); `tests/plot/*.py`
**Consumes:** api-bugs.md B14, B15, B16, B27, B28; code quality items `_figure_of` shared, `_plain_log_ticks` placement, `_sample` duplicate, "label per sample" x3, `cmap(...)` x3, `(x > 0 if log_x else True)`, empty `ratios` mapping.
**Produces:** `plot/_common.py` with `figure_of(ax, figsize)`, `axes_of(axes, n, ...)`, `plain_log_ticks(axis)`, `sample_labels(result_or_batch)`, `sample_colors(n, cmap)`, `log_scale(ax, which)` (sets log and plain ticks, falls back to linear with a debug log when no positive value), used by every plot module; `plot_fit(log_x=True)` masks `x <= 0`; no plot function changes the layout engine of a passed `ax`'s figure; `plot_goodness_of_fit` legend only with labels; `plot_ratio({})` raises `ValueError`; the `n_workers` docstrings of `NCAOptions`, `FitOptions`, `run_rows`, `fit_rows` carry the `__main__` guard note for `spawn`/`forkserver`. Tests: one per bug, plus `test_common_helpers`. Commit: `Share the plot helpers and fix the log axes, legends and layout of the figures`.

### Task A5: gate and PR A

`uv run ruff check && uv run ruff format --check && uvx ty check && uv run pytest -q -W error && uv run tox run-parallel && uv run zensical build --clean --strict`; then:
```bash
git push -u origin cleanup-bugs
gh pr create --base develop --title "Fix the bugs of the audit and share the helpers" --body "..." # summary: the fixed bugs by area, the shared helpers, the Claude Code footer
gh pr merge --squash --auto --delete-branch
```
Wait for the merge, `git switch develop && git pull`.

---

## PR B: performance (branch `cleanup-performance`)

### Task B1: caches, percentiles, slices, benchmark script

**Files:** `src/pkpdutils/units.py`, `nca/result.py`, `nca/uncertainty.py`, `result.py`, `fit/engine.py`, `nca/nca.py` (`run_rows` slices), `scripts/benchmark.py` (new), `docs/development.md`
**Consumes:** performance.md F1, F2, F5, F7 and the scripts `bench_*.py`.
**Produces:** `functools.lru_cache` on `parse_unit`, `check_dose_unit`, `is_per_bodyweight` (`units.py`) and `parameter_unit` (`nca/result.py`) (pint objects are immutable; the cache is keyed by the strings); `nan_percentile(values, q, axis=-1)` in `result.py` (sort based, identical results, no `apply_along_axis`) used by `reduce_replicates`, `summarize`, `replicate_statistics`; `run_rows` passes slices of contiguous rows (`t[start:stop]`) instead of fancy-index copies; the bootstrap builds the replicate array per chunk; `scripts/benchmark.py` with the cases `nca-small`, `nca-large`, `nca-multiple`, `bootstrap`, `delta`, `fit`, `constructors`, `iterate` (each a function, `--repeat`, prints a markdown table with medians and peak RSS) and a section "Benchmarks" in `docs/development.md`. Tests: `nan_percentile` equals `np.nanpercentile` on random arrays with NaN; the caches do not change results (existing suite). Commit: `Cache the unit parsing, vectorize the nan percentiles and slice the chunks`.

### Task B2: shared executors, threads for the NCA, chunking, fit pool

**Files:** `src/pkpdutils/parallel.py` (new), `nca/nca.py`, `nca/uncertainty.py`, `fit/engine.py`, `nca/options.py`, `fit/options.py`, `docs/nca.md`/`fitting.md` (the `n_workers` paragraphs), `tests/test_parallel.py` (new), `tests/nca/test_nca.py`, `tests/fit/test_engine.py`
**Consumes:** performance.md F3, F4, F6, F12, F14 and `bench_extras.py`, `bench_firstpool.py`.
**Produces:** `pkpdutils.parallel`: `executor(kind: Literal["thread", "process"], n_workers: int) -> Executor` (module-level cache per kind and size, created lazily, `atexit.register(shutdown)`), `resolve_workers(n_workers: int | None, n_rows: int, *, threshold: int = 20_000, max_workers: int = 8) -> int` (`None`: 1 below the threshold else `min(os.cpu_count() or 1, max_workers)`; `1` serial), `split_rows(n_rows, n_workers, *, min_rows: int = 1_000) -> list[slice]`; `run_rows` uses the thread executor and `split_rows` (`chunk_rows` kept as an upper bound of a chunk for memory); `fit_rows` uses the process executor with `chunksize` and an initializer that holds the model and the options (rows send only their data); `NCAOptions.n_workers`/`FitOptions.n_workers` docstrings: `None` automatic, `1` serial, `> 1` that many. Tests: `resolve_workers`/`split_rows` unit tests, results equal serial for threads and processes, the executor is reused (same object on two calls). Benchmarks before/after in the PR description (`scripts/benchmark.py`). Commit: `Share the executors, run the NCA core in threads and split the chunks by worker`.

### Task B3: vectorized readers, cheaper construction and iteration, interval gather

**Files:** `src/pkpdutils/io.py`, `timecourse.py`, `nca/intervals.py`; `tests/test_io.py`, `test_timecourses.py`, `tests/nca/test_intervals.py`
**Consumes:** performance.md F8, F9, F10, F11 and the profiles.
**Produces:** `read_events` without scalar `Series.__getitem__` (vectorized dose expansion with `np.repeat`, per-subject arrays via `groupby(...).indices` and `np.split`; identical batches on the fixtures and on a 100 000-row table compared to the old implementation, which the test reconstructs from the merged commit's helper); `Timecourses.from_dataframe` builds the padded arrays directly (one `Timecourse` validation per batch is replaced by the batch validations: sorted times, duplicates, units) and yields identical batches; `__iter__`/`isel`/`sel` build `Timecourse` objects from the numpy arrays (`np.take`) without `ds.isel`; `compute_intervals` packs each row once and gathers the block of an interval by index (identical results on `tests/nca/test_intervals.py` and the reference tests). Targets: `from_events` 100 000 rows < 0.7 s, `from_dataframe` 10 000 subjects < 1 s, iteration of 10 000 curves < 0.8 s, multiple dose N=10 000 < 0.2 s serial (report the measured numbers in the PR). Commit: `Vectorize the event reader, the batch construction, the iteration and the interval gather`.

### Task B4: gate and PR B

As A5 with the branch `cleanup-performance`, title `Speed up the batches: caches, thread pool, chunking and vectorized readers`, the benchmark table before/after in the body.

---

## PR C: API consistency and usability (branch `cleanup-api`)

### Task C1: signatures, exports, conventions

**Files:** `src/pkpdutils/__init__.py`, `nca/__init__.py`, `nca/nca.py`, `nca/steady_state.py`, `fit/__init__.py`, `fit/engine.py` (`p_cv` fraction), `fit/proportionality.py` (`ProportionalityResult`), `stats/__init__.py`, `io.py`, `timecourse.py`, `plot/*.py`; every test, example and doc snippet that uses a changed signature
**Consumes:** api-bugs.md inconsistencies table (every row not marked "kept" in the spec's decisions), proposals U9 (`fit_timecourses(model, batch)` already exists; add `compare_models(models, timecourse)` accepting a `Timecourse`).
**Produces:** keyword-only `options` in `nca`, `nca_single`, `partial_auc`, `superposition`; top-level exports of the enumerations, the models, `plot` (module) and `summary_table` (placeholder import added in C2); `*_col` keywords and `covariates` on the three readers; plot signatures `f(data, *, ..., ax=None, style=DEFAULT_STYLE)`, `log_x`/`log_y` everywhere, `axes` for multi-panel figures, `draw_nca_panel` returns the `Axes` and is exported; `p_cv` as a fraction (docs, glossary, tests); `ProportionalityResult` dataclass; `Timecourses.relative_to_dose(which)`, `Timecourse.to_batch(dim, label)`; `auc_partial` in `PARAMETER_UNITS`; labels keep their dtype in `from_dataframe`; `stats` public helpers exported. Every change listed in the spec's "Breaking changes" section (append what is new). Commit: `Make the signatures, keywords and exports consistent across the package`.

### Task C2: batch ergonomics and the publication tables

**Files:** `src/pkpdutils/timecourse.py`, `result.py`, `nca/nca.py`, `nca/terminal.py`, `nca/options.py`, `stats/ratio.py`, `stats/ddi.py`, `fit/proportionality.py`, `src/pkpdutils/__init__.py`; `tests/test_timecourses.py`, `tests/test_result.py`, `tests/nca/test_terminal.py`, `tests/stats/*.py`, `tests/fit/test_proportionality.py`
**Consumes:** docs-usability.md "Proposed API additions" (signatures), api-bugs.md U1, U3; spec section C.
**Produces:** `Timecourses.select`, `groupby`, `mean`, `dose_normalized`; `summary_table` function and method with the spec's signature and layouts (numbers formatted with significant digits as strings; `stats` holds statistic names only; the "geometric mean [CV %]" convention is one `stats=("n", "geomean", "geocv")` table, documented); `ParameterResult.summarize` with `x_cv`, `x_min`, `x_max` and the `interval_*` reduction; `ratio_table`, `ddi_table`, `proportionality_table`; `lambda_z_t_last`, `lambda_z_span`, `NCAFlag.SPAN_LOW`, units and glossary rows. Tests: closed forms for `mean` (mean/sd/se/n of known curves), `select` by label/list/slice/coordinate, `groupby` partitions, `dose_normalized` units, `summary_table` cell contents on a known result, `lambda_z_span` on an analytic curve. Commit: `Add the batch selection and grouping, the publication tables and the terminal span`.

### Task C3: figures

**Files:** `src/pkpdutils/plot/timecourse.py`, `nca.py`, `fit.py`, `ratio.py`, `meta.py`, `_common.py`, `fit/frontends.py` (`attrs["x_name"]`, `["y_name"]`); `tests/plot/*.py`
**Consumes:** docs-usability.md proposals (`plot_mean_timecourse`, `plot_timecourse` `by`/`facet`/`max_legend`, `plot_troughs`, label fixes, annotations), api-bugs.md U4-U8, inconsistencies "Route handling in plots".
**Produces:** as the spec section C "Plots"; every figure rendered once into the scratchpad and viewed by the implementer. Commit: `Add the mean timecourse, trough and grouped figures and label the fit axes`.

### Task C4: gate and PR C

As A5 with the branch `cleanup-api`, title `Consistent signatures, batch selection, publication tables and figures`; the body lists the breaking changes.

---

## PR D: documentation (branch `cleanup-docs`)

### Task D1: render script, gallery, images, missing example figures

**Files:** `scripts/render_examples.py` (new), `docs/images/*.png` (committed), `docs/gallery.md`, `examples/timecourses.py`, `covariate.py`, `nca_from_sbmlsim.py` (figures), `examples/README.md`, `zensical.toml`, `docs/development.md`, the docs workflow is unchanged (the images are committed; the script is the documented way to refresh them)
**Consumes:** docs-usability.md "Example gallery", audit rows on images.
**Produces:** `uv run python scripts/render_examples.py` runs every example of `tests/examples/test_examples.py` in a temporary directory and copies the PNGs to `docs/images/`; every example writes at least one figure; `docs/gallery.md` with one card per example; the pages embed their figures (`![...](images/x.png)`). Commit: `Render the example figures into the docs and add the gallery`.

### Task D2: workflows, quickstart, diagrams, runnable snippets

**Files:** `docs/workflows.md` (new), `docs/index.md`, `timecourses.md`, `nca.md`, `uncertainty.md`, `fitting.md`, `pd.md`, `statistics.md`, `formats.md`, `plotting.md`, `glossary.md`, `development.md`, `zensical.toml`, `CLAUDE.md`, `README.md`
**Consumes:** docs-usability.md "Mermaid diagrams" (the six code blocks), "Documentation audit" (per page), "Inconsistencies"; the new API of PR C.
**Produces:** `docs/workflows.md` (study table -> `nca` by group -> `summary_table` -> `plot_mean_timecourse` -> csv; BE; DDI; steady state; dose proportionality; each runnable on `tests/data/formats/*.csv` or inline data, outputs shown as tables/images); the Quickstart replaced by the first workflow; six mermaid diagrams placed as the survey proposes; every page with one self-contained snippet and the imports unified; stale phrases removed; the `dose={...}` mapping and the coordinates form documented; glossary rows for the new variables; CLAUDE.md architecture updated (parallel.py, _common.py, summary_table, the plot signatures, the scripts); `docs/development.md` sections for the benchmark and render scripts and the snippet convention. Strict build clean. Commit: `Write the workflows, the diagrams and runnable snippets on every page`.

### Task D3: gate and PR D

As A5 with the branch `cleanup-docs`, title `Workflows, gallery, diagrams and runnable documentation`.

---

## Deviations from the spec

None intended; rulings during execution go into the ledger and the final report.

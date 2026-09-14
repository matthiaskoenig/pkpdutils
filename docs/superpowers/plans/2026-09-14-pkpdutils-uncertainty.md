# pkpdutils Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate the uncertainty of group timecourses (mean with `sd`/`se` and `n`) to the NCA parameters by bootstrap or delta method, summarize individual results over a sample dimension, add the partial AUC helper, and document all of it (phase 4 of the spec).

**Architecture:** The vectorized core already analyses any number of rows; the bootstrap builds `(N, B, n_time)` resampled values, flattens them to `N*B` rows, runs the chunked core once and reduces the replicates per parameter to `x_se`, `x_sd`, `x_ci_low`, `x_ci_high` (and geometric statistics for log-normal parameters). The delta method perturbs every time point of every row once (`N*(n_time+1)` rows), forms the numerical Jacobian of every continuous parameter and propagates `se` through it. Both live in `nca/uncertainty.py` and are called from `nca()`; the result variables follow the naming `x`, `x_sd`, `x_se`, `x_ci_low`, `x_ci_high`, `x_geomean`, `x_geocv`, plus `n`. `NCAResult.summarize(dim)` produces the same layout from individual results. `partial_auc` reuses the interpolation and the bounded `auc_aumc` of plan 2.

**Tech Stack:** numpy (`default_rng`, vectorized reductions), scipy.stats (`t`, `norm` quantiles), xarray, pint, pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`, section 2 "Uncertainty" and the `summarize`/partial AUC items; plan 2 (`docs/superpowers/plans/2026-09-14-pkpdutils-nca.md`) for the interfaces this plan consumes.

**Later plans:** fitting and PD models (5), statistics (6, consumes `x`, `x_sd`, `x_se`, `n` and individual values via `result.sample(...)`), release (7).

## Global Constraints

- package `pkpdutils`, python 3.13 and 3.14, runtime dependencies stay `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib`, `rich`
- ty `error-on-warning = true`: zero diagnostics on `src`, `tests`, `examples`, `scripts`; rule-specific `# ty: ignore[rule]` only, none unused; in tests prefer `assert x is not None` narrowing
- every module, class and function of `src/pkpdutils` has full type annotations and a google style docstring (ruff `D`) with the formula and a citation key of `docs/references.md` where a formula exists; `examples/`, `tests/`, `scripts/` exempt from `D`
- library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, never prints, never calls `plt.show()`
- test output pristine: `uv run pytest -q -W error` passes (numpy warnings wrapped in `np.errstate`)
- results are `xarray.Dataset` objects over the sample dimensions with `attrs["units"]` on every variable; uncertainty variables of a parameter `x` are `x_sd`, `x_se`, `x_ci_low`, `x_ci_high` (same unit as `x`), `x_geomean` (same unit), `x_geocv` (dimensionless); `n` (dimensionless) per sample
- log-normal parameters (geometric statistics reported): `auc_last`, `auc_inf_obs`, `auc_inf_pred`, `aumc_last`, `aumc_inf`, `auc_tau`, `cmax`, `cmax_ss`, `cmin`, `cmin_ss`, `ctrough`, `cavg`, `clast`, `c0`, `cl`, `cl_f`, `cl_ss`, `vz`, `vz_f`, `vss`, `thalf`, `lambda_z`, `mrt`, `auc_inf_dn`, `cmax_dn`; discrete parameters (no uncertainty, `NaN`): `tmax`, `tmin`, `tlast`, `tmax_half`, `temax`, `lambda_z_n_points`, `lambda_z_t_first`, `flags`
- text rules of the user's `~/.claude/CLAUDE.md`: never write the em dash character (U+2014) anywhere (use "-"); commit messages carry NO `Co-Authored-By` line; every commit message ends with the single line `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`
- markdown has no hard line wraps; formulas `\(...\)` inline and `\[...\]` display
- `develop` accepts only pull requests; work happens on the branch `uncertainty` off `develop`, merged through a pull request with the checks `tests`, `ruff`, `ty`, `docs`
- commands run from the repository root with `uv run`

## Interfaces of plans 1 and 2 this plan consumes

- `pkpdutils.timecourse.Timecourses`: `times`, `values` `(*sample_shape, n_time)` NaN padded; `sd`, `se` (same shape or `None`), `n` (`(*sample_shape)` or `None`); `dose_amount`, `dose_time`, `dose_duration`, `dose_unit`, `route`, `sample_dims`, `sample_shape`, `n_samples`, `n_time`, `time_unit`, `unit`, `substance`, `ds`, `has_uncertainty`; `from_arrays(time, values, *, time_unit, unit, dims, coords, sd, se, n, dose, route, substance)`, `from_timecourses`
- `pkpdutils.nca.nca`: `nca(timecourses, options) -> NCAResult`, `nca_single`, `compute_parameters(t, c, *, dose_amount, dose_time, dose_duration, route, options) -> dict[str, np.ndarray]`, `_compute_chunk(args)`, `_to_result(values, timecourses, shape) -> NCAResult`, `PARAMETER_UNITS`; `nca()` splits the rows into `ceil(n_rows / options.chunk_rows)` chunks and maps them serially or over a `ProcessPoolExecutor(max_workers=options.n_workers)`
- `pkpdutils.nca.auc`: `pack_valid`, `auc_aumc(tp, cp, n_valid, method, t_start=None, t_end=None)`, `interpolate_at(tp, cp, n_valid, t_query, method)`, `insert_point(tp, cp, n_valid, t_new, c_new)`
- `pkpdutils.nca.result`: `NCAResult(ds)` with `ds`, `sample_dims`, `parameters` (data variables except `flags`), `units(name)`, `__getitem__`, `__contains__`, `to_quantities(**indexers)`, `flags(**indexers)`, `to_dataframe()`, `flag_table()`; `parameter_unit(expression, *, unit, time_unit, dose_unit) -> (unit_str, factor)`
- `pkpdutils.nca.options`: `NCAOptions` (fields `kind`, `auc_method`, `terminal`, `lloq`, `blq`, `c0_method`, `extrapolation_warning`, `regimen`, `effect_threshold`, `n_workers`, `chunk_rows`), enums, `NCAFlag`

---

### Task 1: Branch and the em dash sweep

**Files:**
- Modify: `README.md`, `docs/index.md`, `CLAUDE.md`, `docs/references.md`

- [ ] **Step 1: Create the branch**

```bash
git switch uncertainty   # the branch exists and holds this plan
```

- [ ] **Step 2: Replace every em dash**

Run: `grep -rn $'\xe2\x80\x94' README.md docs/*.md CLAUDE.md src examples tests zensical.toml | cut -c1-120` (the em dash is written as its UTF-8 bytes so that this plan itself contains none)
Expected: 14 lines in `README.md` (6), `docs/index.md` (5), `CLAUDE.md` (2), `docs/references.md` (1). Replace each ` EM ` (em dash with spaces) by ` - ` and a bare em dash by `-` with `EM=$'\xe2\x80\x94'; sed -i "s/ $EM / - /g; s/$EM/-/g" README.md docs/index.md CLAUDE.md docs/references.md`, then read the changed lines (`git diff`) to confirm they read well; the feature lists in `README.md` and `docs/index.md` become `- **timecourse simulations** - concatenated ...` style lines.

- [ ] **Step 3: Verify and commit**

Run: `grep -rn $'\xe2\x80\x94' README.md docs CLAUDE.md src examples tests zensical.toml | grep -v docs/superpowers | wc -l` (expect `0`), `uv run pre-commit run --all-files`.

```bash
git add README.md docs/index.md CLAUDE.md docs/references.md
git commit -q -m "Replace the em dashes by plain dashes

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 2: Uncertainty options

**Files:**
- Modify: `src/pkpdutils/nca/options.py`, `src/pkpdutils/nca/__init__.py`
- Test: `tests/nca/test_options.py` (append)

**Interfaces:**
- Produces: `class UncertaintyMethod(StrEnum)`: `NONE = "none"`, `BOOTSTRAP = "bootstrap"`, `DELTA = "delta"`; `class BootstrapSpread(StrEnum)`: `SE = "se"`, `SD = "sd"`; `class BootstrapDistribution(StrEnum)`: `NORMAL = "normal"`, `LOGNORMAL = "lognormal"`; `NCAOptions` fields `uncertainty: UncertaintyMethod | None = None` (`None` = `BOOTSTRAP` when the batch carries `sd` or `se`, else `NONE`), `n_boot: int = Field(default=1000, ge=2)`, `seed: int | None = None`, `ci_level: float = Field(default=0.95, gt=0, lt=1)`, `bootstrap_spread: BootstrapSpread = BootstrapSpread.SE`, `bootstrap_distribution: BootstrapDistribution = BootstrapDistribution.NORMAL`, `delta_step: float = Field(default=0.01, gt=0, lt=1)` (relative perturbation of a point in units of its `se`); method `NCAOptions.resolve_uncertainty(has_uncertainty: bool) -> UncertaintyMethod`; exports of the three enums from `pkpdutils.nca`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/nca/test_options.py`:
```python
from pkpdutils.nca.options import BootstrapDistribution, BootstrapSpread, UncertaintyMethod


def test_uncertainty_defaults_and_resolution() -> None:
    options = NCAOptions()
    assert options.uncertainty is None
    assert options.n_boot == 1000
    assert options.seed is None
    assert options.ci_level == pytest.approx(0.95)
    assert options.bootstrap_spread is BootstrapSpread.SE
    assert options.bootstrap_distribution is BootstrapDistribution.NORMAL
    assert options.delta_step == pytest.approx(0.01)
    assert options.resolve_uncertainty(has_uncertainty=True) is UncertaintyMethod.BOOTSTRAP
    assert options.resolve_uncertainty(has_uncertainty=False) is UncertaintyMethod.NONE
    explicit = NCAOptions(uncertainty=UncertaintyMethod.DELTA)
    assert explicit.resolve_uncertainty(has_uncertainty=False) is UncertaintyMethod.DELTA


def test_uncertainty_validation() -> None:
    with pytest.raises(ValueError):
        NCAOptions(n_boot=1)
    with pytest.raises(ValueError):
        NCAOptions(ci_level=1.0)
    with pytest.raises(ValueError):
        NCAOptions(delta_step=0.0)
    assert NCAOptions(uncertainty="lognormal_bootstrap" if False else "bootstrap").uncertainty is UncertaintyMethod.BOOTSTRAP
```
(Simplify the last line to `assert NCAOptions(uncertainty="bootstrap").uncertainty is UncertaintyMethod.BOOTSTRAP`.) Add the three enums to the import at the top of the file instead of a mid-file import (ruff `E402`).

Run: `uv run pytest tests/nca/test_options.py -n 0 -q` → FAIL with `ImportError`.

- [ ] **Step 2: Implement**

In `src/pkpdutils/nca/options.py` add after `C0Method`:
```python
class UncertaintyMethod(StrEnum):
    """How the uncertainty of group timecourses is propagated to the parameters."""

    #: no uncertainty variables
    NONE = "none"
    #: parametric bootstrap: resample every time point, analyse the replicates
    BOOTSTRAP = "bootstrap"
    #: delta method: numerical Jacobian of every parameter with respect to the values
    DELTA = "delta"


class BootstrapSpread(StrEnum):
    """Which spread the bootstrap resamples every time point with."""

    #: the standard error of the mean: the uncertainty of the group mean curve
    SE = "se"
    #: the standard deviation: the spread of individual curves
    SD = "sd"


class BootstrapDistribution(StrEnum):
    """Distribution the bootstrap draws every time point from."""

    #: normal with the given mean and spread; draws below 0 are set to 0
    NORMAL = "normal"
    #: log-normal with the same mean and spread; positive by construction
    LOGNORMAL = "lognormal"
```
and the fields with docstring lines in `NCAOptions`:
```python
        uncertainty: propagation of `sd`/`se` to the parameters; `None` selects
            `BOOTSTRAP` when the batch carries an uncertainty and `NONE` otherwise
        n_boot: number of bootstrap replicates
        seed: seed of the bootstrap random generator, `None` for a fresh one
        ci_level: level of the confidence intervals
        bootstrap_spread: whether the replicates are drawn with `se` or `sd`
        bootstrap_distribution: normal or log-normal draws
        delta_step: relative perturbation of a point, in units of its `se`, for the delta method
```
```python
    uncertainty: UncertaintyMethod | None = None
    n_boot: int = Field(default=1000, ge=2)
    seed: int | None = None
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_spread: BootstrapSpread = BootstrapSpread.SE
    bootstrap_distribution: BootstrapDistribution = BootstrapDistribution.NORMAL
    delta_step: float = Field(default=0.01, gt=0.0, lt=1.0)

    def resolve_uncertainty(self, has_uncertainty: bool) -> UncertaintyMethod:
        """The uncertainty method of an analysis.

        Args:
            has_uncertainty: whether the batch carries `sd` or `se`

        Returns:
            `uncertainty` when set, else `BOOTSTRAP` for a batch with an
            uncertainty and `NONE` without.
        """
        if self.uncertainty is not None:
            return self.uncertainty
        return UncertaintyMethod.BOOTSTRAP if has_uncertainty else UncertaintyMethod.NONE
```
Export `BootstrapDistribution`, `BootstrapSpread`, `UncertaintyMethod` from `pkpdutils/nca/__init__.py` (import and `__all__`, sorted).

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/nca/test_options.py -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass.

```bash
git add src/pkpdutils/nca/options.py src/pkpdutils/nca/__init__.py tests/nca/test_options.py
git commit -q -m "Add the uncertainty options of the NCA

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 3: Row runner refactor and the partial AUC

**Files:**
- Modify: `src/pkpdutils/nca/nca.py`, `src/pkpdutils/nca/__init__.py`
- Test: `tests/nca/test_partial_auc.py`, `tests/nca/test_nca.py` (append one test)

**Interfaces:**
- Produces:
  - `run_rows(t, c, *, dose_amount, dose_time, dose_duration, route, options) -> dict[str, np.ndarray]`: the chunked serial/pool execution of the core on `(N, n)` arrays, extracted from `nca()`; `nca()` calls it. Uncertainty (Task 4/5) calls it with `N*B` and `N*(n+1)` rows.
  - `partial_auc(timecourses, t_start, t_end, options=None) -> xr.DataArray`: area of every sample between two times relative to the dose (values at the bounds interpolated with `options.auc_method`, `NaN` when a bound lies outside the observed range of a sample); dims = sample dims, `attrs["units"]` = unit of `auc_last`, name `auc_partial`. `t_start`/`t_end` are floats in the time unit of the batch.

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_partial_auc.py`:
```python
import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca, partial_auc

K, C0 = 0.5, 10.0


def batch() -> Timecourses:
    t = np.array([0.5, 1, 2, 4, 6, 8, 12])
    curves = [
        Timecourse(time=t, value=C0 * np.exp(-K * t), time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS), substance="x", label="a"),
        Timecourse(time=t, value=2 * C0 * np.exp(-K * t), time_unit="hr", unit="mg/l", dose=Dose(amount=200, unit="mg", route=Route.IV_BOLUS), substance="x", label="b"),
    ]
    return Timecourses.from_timecourses(curves)


def test_partial_auc_analytic_log_rule() -> None:
    area = partial_auc(batch(), 1.0, 6.0, NCAOptions(auc_method=AUCMethod.LOG))
    expected = C0 / K * (np.exp(-K * 1.0) - np.exp(-K * 6.0))
    np.testing.assert_allclose(area.values, [expected, 2 * expected], rtol=1e-9)
    assert area.dims == ("individual",)
    assert area.attrs["units"] == "hour * milligram / liter"
    assert area.name == "auc_partial"


def test_partial_auc_bounds_between_points_and_outside() -> None:
    area = partial_auc(batch(), 1.5, 5.0, NCAOptions(auc_method=AUCMethod.LOG))
    expected = C0 / K * (np.exp(-K * 1.5) - np.exp(-K * 5.0))
    assert float(area.values[0]) == pytest.approx(expected, rel=1e-9)
    outside = partial_auc(batch(), 1.0, 20.0)
    assert np.isnan(outside.values).all()
    before = partial_auc(batch(), 0.0, 4.0)
    assert np.isnan(before.values).all()


def test_partial_auc_whole_range_equals_auc_last() -> None:
    tcs = batch()
    options = NCAOptions(auc_method=AUCMethod.LINEAR)
    whole = partial_auc(tcs, 0.5, 12.0, options)
    # auc_last of a bolus includes the inserted (0, C0) segment, so compare against the area from the first sample
    result = nca(tcs.__class__.from_timecourses([tc.model_copy(update={"dose": None}) for tc in tcs]), options)
    np.testing.assert_allclose(whole.values, result["auc_last"].values, rtol=1e-12)


def test_partial_auc_relative_to_dose_time() -> None:
    t = np.array([10.5, 11, 12, 14, 16, 18, 22])
    tc = Timecourse(time=t, value=C0 * np.exp(-K * (t - 10)), time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS, time=10.0), substance="x")
    area = partial_auc(Timecourses.from_timecourses([tc]), 1.0, 6.0, NCAOptions(auc_method=AUCMethod.LOG))
    expected = C0 / K * (np.exp(-K * 1.0) - np.exp(-K * 6.0))
    assert float(area.values[0]) == pytest.approx(expected, rel=1e-9)


def test_partial_auc_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="t_end"):
        partial_auc(batch(), 4.0, 1.0)
```

Append to `tests/nca/test_nca.py` (add `from pkpdutils.nca.nca import run_rows` to the imports at the top of the file):
```python
def test_run_rows_matches_nca_flags_and_shapes() -> None:
    tc = oral_timecourse()
    batch = Timecourses.from_timecourses([tc, tc])
    values = run_rows(
        batch.times,
        batch.values,
        dose_amount=batch.dose_amount,
        dose_time=batch.dose_time,
        dose_duration=batch.dose_duration,
        route=batch.route,
        options=NCAOptions(),
    )
    result = nca(batch)
    assert set(values) == {*result.parameters, "flags", "n"} - {"n"}
    for name in result.parameters:
        assert values[name].shape == (2,)
    np.testing.assert_array_equal(values["flags"], result["flags"].values)
```
(`run_rows` returns the raw magnitudes; `nca()` adds `n` and the unit factors, so the test compares the key set, the shapes and the flags.)

Run: `uv run pytest tests/nca/test_partial_auc.py tests/nca/test_nca.py -n 0 -q -k "partial or run_rows"` → FAIL with `ImportError`.

- [ ] **Step 2: Refactor `nca()` and add `partial_auc`**

In `src/pkpdutils/nca/nca.py`, move the chunking block of `nca()` (from the comment "the rows are analysed in chunks" through the `values = {...}` line) into

```python
def run_rows(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Run the core on `(N, n)` arrays in chunks, serially or in the worker pool.

    The rows are analysed in chunks of at most `options.chunk_rows` rows, which
    bounds the memory of the vectorized core; with `options.n_workers > 1` the
    chunks are mapped in order over a `ProcessPoolExecutor`.

    Args:
        t: times `(N, n)`
        c: values `(N, n)`
        dose_amount: dose per row, `None` without doses
        dose_time: dose time per row, `None` for 0
        dose_duration: infusion duration per row, `None` for none
        route: route of the batch
        options: the options

    Returns:
        One `(N,)` array per parameter and `flags`.
    """
```
with the body of the block (`n_rows = t.shape[0]`, the jobs, the pool/serial map, the key assertion, the concatenation) and `return values`. `nca()` becomes: flatten, `values = run_rows(...)`, the flagged-samples log line, `_to_result`. Uncertainty hooks are added to `nca()` in Task 4.

Add to `nca.py`:
```python
def partial_auc(
    timecourses: Timecourses,
    t_start: float,
    t_end: float,
    options: NCAOptions | None = None,
) -> xr.DataArray:
    """Area under the curve of every sample between two times relative to the dose.

    The values at the bounds are interpolated with the trapezoid rule of
    `options.auc_method` (`pkpdutils.nca.auc.interpolate_at`) and the area is
    summed with the same rule; a sample whose observed range does not cover
    `[t_start, t_end]` gives `NaN`.

    Args:
        timecourses: the batch
        t_start: start of the interval, in the time unit of the batch, relative to the dose
        t_end: end of the interval, greater than `t_start`
        options: the options, defaults for `None`

    Returns:
        The areas over the sample dimensions, named `auc_partial`, with the unit of `auc_last`.

    Raises:
        ValueError: if `t_end <= t_start`.
    """
    if t_end <= t_start:
        raise ValueError(f"'t_end' ({t_end}) must be greater than 't_start' ({t_start})")
    options = options or NCAOptions()
    n_rows = timecourses.n_samples
    t = timecourses.times.reshape(n_rows, timecourses.n_time)
    c = timecourses.values.reshape(n_rows, timecourses.n_time)
    if timecourses.dose_time is not None:
        t = t - np.asarray(timecourses.dose_time, dtype=np.float64).reshape(n_rows)[:, None]
    tp, cp, n_valid = pack_valid(t, c)
    start = np.full(n_rows, float(t_start))
    end = np.full(n_rows, float(t_end))
    c_start = interpolate_at(tp, cp, n_valid, start, options.auc_method)
    c_end = interpolate_at(tp, cp, n_valid, end, options.auc_method)
    tp, cp, n_valid = insert_point(tp, cp, n_valid, start, c_start)
    tp, cp, n_valid = insert_point(tp, cp, n_valid, end, c_end)
    area, _ = auc_aumc(tp, cp, n_valid, options.auc_method, t_start=start, t_end=end)
    area = np.where(np.isfinite(c_start) & np.isfinite(c_end), area, np.nan)
    unit, factor = parameter_unit(
        PARAMETER_UNITS["auc_last"], unit=timecourses.unit, time_unit=timecourses.time_unit, dose_unit=timecourses.dose_unit
    )
    coords = {d: timecourses.ds[d] for d in timecourses.sample_dims if d in timecourses.ds.coords}
    return xr.DataArray(
        (area * factor).reshape(timecourses.sample_shape),
        dims=timecourses.sample_dims,
        coords=coords,
        name="auc_partial",
        attrs={"units": unit},
    )
```
`insert_point` with a NaN value leaves the row unchanged, so an out-of-range bound simply produces the `NaN` result through the mask. Import `interpolate_at` (already available in `auc.py`). Export `partial_auc` from `pkpdutils/nca/__init__.py`.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/nca -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. `test_partial_auc_whole_range_equals_auc_last`: with `dose=None` the curves have no route, no `(0, C0)` insertion, so `auc_last` (area to the last positive value) equals the partial area from the first to the last sample with the linear rule.

```bash
git add src/pkpdutils/nca tests/nca
git commit -q -m "Extract the row runner of the NCA and add the partial AUC

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 4: Bootstrap (`nca/uncertainty.py`) and the uncertainty variables of the result

**Files:**
- Create: `src/pkpdutils/nca/uncertainty.py`
- Modify: `src/pkpdutils/nca/nca.py` (`nca()` hook, `_to_result` unit lookup), `src/pkpdutils/nca/result.py` (`parameters`, `to_dataframe`, `has_uncertainty`)
- Test: `tests/nca/test_uncertainty.py`, `tests/nca/test_result.py` (append)

**Interfaces:**
- Produces (`uncertainty.py`):
  - `LOGNORMAL_PARAMETERS: frozenset[str]`, `DISCRETE_PARAMETERS: frozenset[str]` (the sets of the Global Constraints), `UNCERTAINTY_SUFFIXES = ("_sd", "_se", "_ci_low", "_ci_high", "_geomean", "_geocv")`, `SUMMARY_SUFFIXES = ("_median", "_q25", "_q75", "_n")`
  - `base_name(name: str) -> str | None`: the parameter a derived variable belongs to (`"auc_last_se"` → `"auc_last"`), `None` for a base name
  - `resolve_spread(timecourses, options) -> np.ndarray`: `(n_samples, n_time)` spread per point according to `options.bootstrap_spread`, derived from the other spread and `n` when needed; `ValueError` when it cannot be derived
  - `resample_values(c, spread, n_boot, rng, distribution) -> np.ndarray` `(N, B, n)`: normal draws clipped at 0, or log-normal draws with the same mean and spread; points without a finite spread are copied
  - `reduce_replicates(replicates: dict[str, np.ndarray], point: dict[str, np.ndarray], *, spread_kind: BootstrapSpread, n_subjects: np.ndarray | None, ci_level: float) -> dict[str, np.ndarray]`: per continuous parameter `x` the arrays `x_sd`, `x_se`, `x_ci_low`, `x_ci_high` (percentile interval) and, for log-normal parameters, `x_geomean`, `x_geocv`; nan-aware over the `B` axis
  - `bootstrap(timecourses, options, point: dict[str, np.ndarray]) -> dict[str, np.ndarray]`: resamples, runs `run_rows` on the `N*B` rows, reduces
- `nca()`: after `run_rows`, `method = options.resolve_uncertainty(timecourses.has_uncertainty)`; `BOOTSTRAP` → `values.update(bootstrap(...))`; `DELTA` → Task 5; both add `values["n"]` (`timecourses.n` or `NaN`); `_to_result` derives the unit of a derived variable from its base name (`_geocv` and `n` dimensionless)
- `NCAResult.parameters` = base parameters only (no derived variables, no `n`, no `flags`); `NCAResult.derived_variables: list[str]`; `NCAResult.has_uncertainty: bool` (`"..._se"` present for any parameter); `to_dataframe()` includes every data variable (sample dims, base parameters, derived variables, `n`, decoded `flags`)

Bootstrap definitions (Efron & Tibshirani 1993, ch. 6; parametric bootstrap of the mean curve): for every time point \(i\) draw \(C_i^{(b)} \sim \mathcal N(\bar C_i, s_i)\) with \(s_i = \mathrm{se}_i\) (`SE`, the uncertainty of the mean curve) or \(s_i = \mathrm{sd}_i\) (`SD`, the spread of individuals); log-normal draws use \(\sigma^2 = \ln(1 + s_i^2/\bar C_i^2)\), \(\mu = \ln \bar C_i - \sigma^2/2\). The replicates of a parameter give its standard error (`SE` draws) or standard deviation (`SD` draws), the other one through \(\mathrm{se} = \mathrm{sd}/\sqrt n\); the confidence interval is the percentile interval at `ci_level`; the geometric mean and geometric CV \(\sqrt{e^{\sigma_{\ln}^2} - 1}\) come from the logarithms of the positive replicates.

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_uncertainty.py`:
```python
import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, UncertaintyMethod, nca, nca_single
from pkpdutils.nca.options import BootstrapDistribution, BootstrapSpread
from pkpdutils.nca.uncertainty import (
    DISCRETE_PARAMETERS,
    LOGNORMAL_PARAMETERS,
    base_name,
    reduce_replicates,
    resample_values,
    resolve_spread,
)

K, C0 = 0.3, 10.0
T = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])


def group_curve(cv: float = 0.1, n: float | None = 12, label: str = "g") -> Timecourse:
    c = C0 * np.exp(-K * T)
    return Timecourse(
        time=T, value=c, sd=cv * c, n=n, time_unit="hr", unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS), substance="x", label=label,
    )


def test_base_name() -> None:
    assert base_name("auc_last_se") == "auc_last"
    assert base_name("auc_inf_obs_ci_high") == "auc_inf_obs"
    assert base_name("cl_f_geocv") == "cl_f"
    assert base_name("auc_last") is None
    assert base_name("flags") is None


def test_resolve_spread_from_sd_and_n() -> None:
    tcs = Timecourses.from_timecourses([group_curve(n=16)])
    se = resolve_spread(tcs, NCAOptions(bootstrap_spread=BootstrapSpread.SE))
    sd = resolve_spread(tcs, NCAOptions(bootstrap_spread=BootstrapSpread.SD))
    np.testing.assert_allclose(se * 4.0, sd)
    no_n = Timecourses.from_timecourses([group_curve(n=None)])
    assert no_n.se is None
    with pytest.raises(ValueError, match="se"):
        resolve_spread(no_n, NCAOptions(bootstrap_spread=BootstrapSpread.SE))
    np.testing.assert_allclose(resolve_spread(no_n, NCAOptions(bootstrap_spread=BootstrapSpread.SD)), 0.1 * C0 * np.exp(-K * T)[None, :])


def test_resample_values_shapes_and_distribution() -> None:
    rng = np.random.default_rng(0)
    c = np.array([[10.0, 5.0, np.nan, 1.0]])
    spread = np.array([[1.0, np.nan, 1.0, 0.5]])
    draws = resample_values(c, spread, 20000, rng, BootstrapDistribution.NORMAL)
    assert draws.shape == (1, 20000, 4)
    assert np.isnan(draws[0, :, 2]).all()
    np.testing.assert_allclose(draws[0, :, 1], 5.0)  # no spread: copied
    assert draws[0, :, 0].mean() == pytest.approx(10.0, abs=0.05)
    assert draws[0, :, 0].std() == pytest.approx(1.0, abs=0.05)
    assert (draws[0, :, 3] >= 0).all()  # clipped at 0
    logn = resample_values(c, spread, 20000, rng, BootstrapDistribution.LOGNORMAL)
    assert (logn[0, :, 3] > 0).all()
    assert logn[0, :, 0].mean() == pytest.approx(10.0, abs=0.05)
    assert logn[0, :, 0].std() == pytest.approx(1.0, abs=0.05)


def test_reduce_replicates_layout() -> None:
    rng = np.random.default_rng(1)
    reps = {"auc_last": np.exp(rng.normal(np.log(100.0), 0.1, size=(2, 5000))), "tmax": np.full((2, 5000), 2.0), "flags": np.zeros((2, 5000))}
    point = {"auc_last": np.array([100.0, 100.0]), "tmax": np.array([2.0, 2.0]), "flags": np.zeros(2)}
    out = reduce_replicates(reps, point, spread_kind=BootstrapSpread.SE, n_subjects=np.array([4.0, np.nan]), ci_level=0.95)
    assert set(out) == {"auc_last_sd", "auc_last_se", "auc_last_ci_low", "auc_last_ci_high", "auc_last_geomean", "auc_last_geocv"}
    assert out["auc_last_se"][0] == pytest.approx(reps["auc_last"][0].std(ddof=1))
    assert out["auc_last_sd"][0] == pytest.approx(out["auc_last_se"][0] * 2.0)
    assert np.isnan(out["auc_last_sd"][1])
    assert out["auc_last_ci_low"][0] < 100.0 < out["auc_last_ci_high"][0]
    assert out["auc_last_geomean"][0] == pytest.approx(100.0, rel=0.02)
    assert out["auc_last_geocv"][0] == pytest.approx(np.sqrt(np.expm1(0.1**2)), rel=0.1)


def test_bootstrap_default_for_group_data_and_reproducible() -> None:
    tc = group_curve()
    options = NCAOptions(seed=42, n_boot=500, auc_method=AUCMethod.LOG)
    a = nca_single(tc, options)
    b = nca_single(tc, options)
    assert a.has_uncertainty
    assert "auc_inf_obs_se" in a and "auc_inf_obs_ci_low" in a and "auc_inf_obs_geocv" in a
    assert "tmax_se" not in a  # discrete
    assert "auc_last" in a.parameters and "auc_last_se" not in a.parameters
    assert "auc_last_se" in a.derived_variables
    for name in a.derived_variables:
        np.testing.assert_allclose(a[name].values, b[name].values, equal_nan=True)
    q = a.to_quantities()
    assert q["auc_inf_obs"].magnitude == pytest.approx(nca_single(group_curve(), NCAOptions(uncertainty=UncertaintyMethod.NONE, auc_method=AUCMethod.LOG)).to_quantities()["auc_inf_obs"].magnitude)
    assert str(q["auc_inf_obs_se"].units) == str(q["auc_inf_obs"].units)
    assert str(q["auc_inf_obs_geocv"].units) == "dimensionless"
    assert q["n"].magnitude == 12
    assert q["auc_inf_obs_ci_low"].magnitude < q["auc_inf_obs"].magnitude < q["auc_inf_obs_ci_high"].magnitude


def test_bootstrap_se_scales_with_input_uncertainty() -> None:
    options = NCAOptions(seed=1, n_boot=2000, auc_method=AUCMethod.LINEAR)
    small = nca_single(group_curve(cv=0.05), options).to_quantities()["auc_last_se"].magnitude
    large = nca_single(group_curve(cv=0.10), options).to_quantities()["auc_last_se"].magnitude
    assert large / small == pytest.approx(2.0, rel=0.1)


def test_bootstrap_sd_spread_and_sd_se_relation() -> None:
    options = NCAOptions(seed=1, n_boot=2000, bootstrap_spread=BootstrapSpread.SD)
    q = nca_single(group_curve(n=16), options).to_quantities()
    assert q["auc_last_se"].magnitude == pytest.approx(q["auc_last_sd"].magnitude / 4.0)


def test_bootstrap_batch_shape_and_chunking() -> None:
    curves = [group_curve(label="a"), group_curve(cv=0.2, label="b"), group_curve(label="c")]
    tcs = Timecourses.from_timecourses(curves)
    options = NCAOptions(seed=3, n_boot=200)
    result = nca(tcs, options)
    assert result["auc_last_se"].dims == ("individual",)
    assert result["auc_last_se"].values[1] > result["auc_last_se"].values[0]
    chunked = nca(tcs, options.model_copy(update={"chunk_rows": 7}))
    for name in result.derived_variables:
        np.testing.assert_allclose(chunked[name].values, result[name].values, equal_nan=True)
    df = result.to_dataframe()
    assert "auc_last_se" in df.columns and "n" in df.columns and list(df.columns)[-1] == "flags"


def test_no_uncertainty_without_spread() -> None:
    tc = Timecourse(time=T, value=C0 * np.exp(-K * T), time_unit="hr", unit="mg/l")
    result = nca_single(tc)
    assert not result.has_uncertainty
    assert result.derived_variables == []
    assert np.isnan(result.to_quantities()["n"].magnitude)
    with pytest.raises(ValueError, match="sd"):
        nca_single(tc, NCAOptions(uncertainty=UncertaintyMethod.BOOTSTRAP))


def test_discrete_and_lognormal_sets() -> None:
    assert "tmax" in DISCRETE_PARAMETERS and "lambda_z_n_points" in DISCRETE_PARAMETERS
    assert "auc_last" in LOGNORMAL_PARAMETERS and "auc_extrap_fraction" not in LOGNORMAL_PARAMETERS
```

Append to `tests/nca/test_result.py`:
```python
def test_result_parameters_exclude_derived_variables() -> None:
    ds = xr.Dataset(
        {
            "auc_last": (("i",), np.array([1.0]), {"units": "hr*mg/l"}),
            "auc_last_se": (("i",), np.array([0.1]), {"units": "hr*mg/l"}),
            "auc_last_geocv": (("i",), np.array([0.1]), {"units": "dimensionless"}),
            "n": (("i",), np.array([5.0]), {"units": "dimensionless"}),
            "flags": (("i",), np.array([0]), {"units": "dimensionless"}),
        },
        coords={"i": ["a"]},
    )
    result = NCAResult(ds)
    assert result.parameters == ["auc_last"]
    assert result.derived_variables == ["auc_last_se", "auc_last_geocv"]
    assert result.has_uncertainty
    assert list(result.to_dataframe().columns) == ["i", "auc_last", "auc_last_se", "auc_last_geocv", "n", "flags"]
```

Run: `uv run pytest tests/nca/test_uncertainty.py tests/nca/test_result.py -n 0 -q` → FAIL with `ImportError`.

- [ ] **Step 2: Write `uncertainty.py`**

`src/pkpdutils/nca/uncertainty.py`:
```python
"""Uncertainty of the NCA parameters of group timecourses.

A group timecourse is the mean curve of several subjects with the standard
deviation (`sd`) or the standard error (`se`) per time point and the number of
subjects `n`. Two methods propagate this uncertainty to the parameters:

- the parametric **bootstrap** (Efron & Tibshirani 1993) draws every time
  point of every curve `n_boot` times from a normal (or log-normal)
  distribution with the observed mean and spread, analyses the replicates with
  the same vectorized code as the original curves and reduces them to the
  standard error (or standard deviation), the percentile confidence interval
  and, for log-normal parameters, the geometric mean and geometric CV;
- the **delta method** (`delta`) perturbs every time point once, forms the
  numerical Jacobian of every parameter with respect to the values and
  propagates the standard errors through it: `var(x) = sum_i (dx/dC_i)^2 se_i^2`.

The variables of a parameter `x` are `x_sd`, `x_se`, `x_ci_low`, `x_ci_high`
and, for log-normal parameters, `x_geomean`, `x_geocv`; `n` is the number of
subjects per sample. Discrete parameters (`tmax`, `tlast`, counts) carry no
uncertainty.
"""

import logging

import numpy as np

from pkpdutils.nca.options import BootstrapDistribution, BootstrapSpread, NCAOptions
from pkpdutils.timecourse import Timecourses

logger = logging.getLogger(__name__)

#: parameters reported with geometric statistics (positive, log-normal)
LOGNORMAL_PARAMETERS: frozenset[str] = frozenset(
    {
        "auc_last", "auc_inf_obs", "auc_inf_pred", "aumc_last", "aumc_inf", "auc_tau",
        "cmax", "cmax_ss", "cmin", "cmin_ss", "ctrough", "cavg", "clast", "c0",
        "cl", "cl_f", "cl_ss", "vz", "vz_f", "vss", "thalf", "lambda_z", "mrt",
        "auc_inf_dn", "cmax_dn",
    }
)

#: parameters without uncertainty (read from the observed points or counts)
DISCRETE_PARAMETERS: frozenset[str] = frozenset(
    {"tmax", "tmin", "tlast", "tmax_half", "temax", "lambda_z_n_points", "lambda_z_t_first", "flags", "n"}
)

#: suffixes of the uncertainty variables of a parameter
UNCERTAINTY_SUFFIXES: tuple[str, ...] = ("_sd", "_se", "_ci_low", "_ci_high", "_geomean", "_geocv")

#: suffixes of the summary variables of a parameter (`NCAResult.summarize`)
SUMMARY_SUFFIXES: tuple[str, ...] = ("_median", "_q25", "_q75", "_n")


def base_name(name: str) -> str | None:
    """The parameter a derived variable belongs to, `None` for a parameter itself."""
    for suffix in (*UNCERTAINTY_SUFFIXES, *SUMMARY_SUFFIXES):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def resolve_spread(timecourses: Timecourses, options: NCAOptions) -> np.ndarray:
    """The spread every time point is resampled with, `(n_samples, n_time)`.

    Args:
        timecourses: the batch
        options: `bootstrap_spread` selects `se` or `sd`; the missing one is
            derived from the other with `n`

    Returns:
        The spread per point (`NaN` where the batch has none).

    Raises:
        ValueError: if the requested spread is neither present nor derivable.
    """
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    se = None if timecourses.se is None else timecourses.se.reshape(n_rows, n_time)
    sd = None if timecourses.sd is None else timecourses.sd.reshape(n_rows, n_time)
    n = None if timecourses.n is None else np.asarray(timecourses.n, dtype=np.float64).reshape(n_rows)[:, None]
    if options.bootstrap_spread is BootstrapSpread.SE:
        if se is not None:
            return se
        if sd is not None and n is not None:
            return sd / np.sqrt(n)
        raise ValueError("The batch has no 'se' and it cannot be derived ('sd' and 'n' needed)")
    if sd is not None:
        return sd
    if se is not None and n is not None:
        return se * np.sqrt(n)
    raise ValueError("The batch has no 'sd' and it cannot be derived ('se' and 'n' needed)")


def resample_values(
    c: np.ndarray,
    spread: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    distribution: BootstrapDistribution,
) -> np.ndarray:
    """Draw bootstrap replicates of every time point of every row.

    Args:
        c: values `(N, n)`
        spread: spread per point `(N, n)`; a point without a finite positive spread is copied
        n_boot: number of replicates `B`
        rng: random generator
        distribution: normal (draws below 0 set to 0) or log-normal with the same mean and spread

    Returns:
        The replicates `(N, B, n)`.
    """
    n_rows, n_time = c.shape
    z = rng.standard_normal((n_rows, n_boot, n_time))
    mean = c[:, None, :]
    s = spread[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        usable = np.isfinite(s) & (s > 0) & np.isfinite(mean)
        if distribution is BootstrapDistribution.NORMAL:
            draws = np.where(usable, np.maximum(mean + s * z, 0.0), mean)
        else:
            positive = usable & (mean > 0)
            sigma2 = np.log1p((s / mean) ** 2)
            mu = np.log(mean) - sigma2 / 2.0
            draws = np.where(positive, np.exp(mu + np.sqrt(sigma2) * z), mean)
    return draws


def reduce_replicates(
    replicates: dict[str, np.ndarray],
    point: dict[str, np.ndarray],
    *,
    spread_kind: BootstrapSpread,
    n_subjects: np.ndarray | None,
    ci_level: float,
) -> dict[str, np.ndarray]:
    """Reduce the bootstrap replicates of every continuous parameter to its uncertainty variables.

    Args:
        replicates: parameter name to replicates `(N, B)`
        point: parameter name to the estimate of the original curve `(N,)`
        spread_kind: whether the replicates were drawn with `se` (their spread
            is the standard error of the parameter) or `sd` (their spread is
            the standard deviation over subjects)
        n_subjects: subjects per row, `None` or `NaN` when unknown
        ci_level: level of the percentile interval

    Returns:
        `x_sd`, `x_se`, `x_ci_low`, `x_ci_high` per parameter and `x_geomean`,
        `x_geocv` for log-normal parameters.
    """
    alpha = 1.0 - ci_level
    out: dict[str, np.ndarray] = {}
    sqrt_n = None if n_subjects is None else np.sqrt(np.asarray(n_subjects, dtype=np.float64))
    for name, reps in replicates.items():
        if name in DISCRETE_PARAMETERS or name not in point:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            finite = np.isfinite(reps)
            count = finite.sum(axis=1)
            filled = np.where(finite, reps, np.nan)
            std = np.where(count > 1, np.nanstd(filled, axis=1, ddof=1), np.nan)
            low, high = np.nanpercentile(filled, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
            if spread_kind is BootstrapSpread.SE:
                se = std
                sd = std * sqrt_n if sqrt_n is not None else np.full_like(std, np.nan)
            else:
                sd = std
                se = std / sqrt_n if sqrt_n is not None else np.full_like(std, np.nan)
            valid = np.isfinite(point[name]) & (count > 1)
            out[f"{name}_sd"] = np.where(valid, sd, np.nan)
            out[f"{name}_se"] = np.where(valid, se, np.nan)
            out[f"{name}_ci_low"] = np.where(valid, low, np.nan)
            out[f"{name}_ci_high"] = np.where(valid, high, np.nan)
            if name in LOGNORMAL_PARAMETERS:
                logs = np.where(finite & (reps > 0), np.log(np.where(reps > 0, reps, 1.0)), np.nan)
                n_pos = np.isfinite(logs).sum(axis=1)
                log_var = np.where(n_pos > 1, np.nanvar(logs, axis=1, ddof=1), np.nan)
                out[f"{name}_geomean"] = np.where(valid, np.exp(np.nanmean(logs, axis=1)), np.nan)
                out[f"{name}_geocv"] = np.where(valid, np.sqrt(np.expm1(log_var)), np.nan)
    return out


def bootstrap(
    timecourses: Timecourses,
    options: NCAOptions,
    point: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Bootstrap the parameters of a batch.

    Args:
        timecourses: the batch (group curves with `sd` or `se`)
        options: `n_boot`, `seed`, `bootstrap_spread`, `bootstrap_distribution`, `ci_level`
        point: the parameters of the original curves (`run_rows` output)

    Returns:
        The uncertainty variables per parameter (see `reduce_replicates`).
    """
    from pkpdutils.nca.nca import run_rows  # noqa: PLC0415  (nca imports this module)

    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    t = timecourses.times.reshape(n_rows, n_time)
    c = timecourses.values.reshape(n_rows, n_time)
    spread = resolve_spread(timecourses, options)
    rng = np.random.default_rng(options.seed)
    draws = resample_values(c, spread, options.n_boot, rng, options.bootstrap_distribution)
    b = options.n_boot

    def repeat(a: np.ndarray | None) -> np.ndarray | None:
        """Repeat a per-row array `B` times (rows stay grouped)."""
        return None if a is None else np.repeat(np.asarray(a, dtype=np.float64).reshape(n_rows), b)

    values = run_rows(
        np.repeat(t, b, axis=0),
        draws.reshape(n_rows * b, n_time),
        dose_amount=repeat(timecourses.dose_amount),
        dose_time=repeat(timecourses.dose_time),
        dose_duration=repeat(timecourses.dose_duration),
        route=timecourses.route,
        options=options,
    )
    replicates = {name: array.reshape(n_rows, b) for name, array in values.items()}
    n_subjects = None if timecourses.n is None else np.asarray(timecourses.n, dtype=np.float64).reshape(n_rows)
    logger.info("bootstrap: %d curves x %d replicates", n_rows, b)
    return reduce_replicates(
        replicates, point, spread_kind=options.bootstrap_spread, n_subjects=n_subjects, ci_level=options.ci_level
    )
```
If ruff does not know `PLC0415`, drop the `noqa`. `run_rows` must not itself trigger the bootstrap (it does not: only `nca()` resolves the method).

- [ ] **Step 3: Hook into `nca()`, `_to_result`, `NCAResult`**

`nca()` in `nca.py` after `values = run_rows(...)`:
```python
    method = options.resolve_uncertainty(timecourses.has_uncertainty)
    if method is UncertaintyMethod.BOOTSTRAP:
        from pkpdutils.nca.uncertainty import bootstrap  # noqa: PLC0415

        values.update(bootstrap(timecourses, options, values))
    elif method is UncertaintyMethod.DELTA:
        from pkpdutils.nca.uncertainty import delta  # noqa: PLC0415

        values.update(delta(timecourses, options, values))
    n_subjects = timecourses.n
    values["n"] = (
        np.full(n_rows, np.nan) if n_subjects is None else np.asarray(n_subjects, dtype=np.float64).reshape(n_rows)
    )
```
(`delta` is written in Task 5; until then the `DELTA` branch imports a missing name only when requested. To keep ty clean in this task, write the `DELTA` branch in Task 5 and leave only the `BOOTSTRAP` branch now.) Import `UncertaintyMethod` from `pkpdutils.nca.options`. Keep `flags` last: build `values` so that `flags` is popped and re-added after the update (`flags = values.pop("flags"); ...; values["flags"] = flags`).

`_to_result`: replace the direct `PARAMETER_UNITS[name]` lookup by
```python
def unit_expression(name: str) -> str:
    """Unit expression of a result variable, derived variables from their parameter."""
    if name in PARAMETER_UNITS:
        return PARAMETER_UNITS[name]
    if name == "n" or name.endswith("_geocv") or name.endswith("_n"):
        return "dimensionless"
    base = base_name(name)
    if base is None or base not in PARAMETER_UNITS:
        raise KeyError(f"No unit expression for '{name}'")
    return PARAMETER_UNITS[base]
```
(module level in `nca.py`, importing `base_name` from `uncertainty.py`; `uncertainty.py` imports `run_rows` lazily, so there is no cycle at import time).

`result.py`:
```python
    @property
    def parameters(self) -> list[str]:
        """Names of the parameters (the data variables except `flags`, `n` and the derived variables)."""
        return [str(name) for name in self.ds.data_vars if name not in ("flags", "n") and base_name(str(name)) is None]

    @property
    def derived_variables(self) -> list[str]:
        """Names of the uncertainty and summary variables (`x_se`, `x_ci_low`, ...)."""
        return [str(name) for name in self.ds.data_vars if base_name(str(name)) is not None]

    @property
    def has_uncertainty(self) -> bool:
        """Whether any parameter carries a standard error."""
        return any(name.endswith("_se") for name in self.derived_variables)
```
and `to_dataframe` uses `columns = [*self.sample_dims, *[str(v) for v in self.ds.data_vars if v != "flags"], "flags"]`. `to_quantities` iterates over every data variable except `flags` (so derived variables and `n` are included). Import `base_name` from `pkpdutils.nca.uncertainty` in `result.py` (no cycle: `uncertainty.py` imports `options` and `timecourse` only at module level).

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/nca -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. Notes: `np.nanpercentile` on an all-NaN row warns "All-NaN slice": guard with `count > 0` by filling such rows with 0 before the percentile and masking afterwards (the `valid` mask), or wrap in `warnings.catch_warnings()` filtering `RuntimeWarning` around the percentile call; both keep the output pristine. `nca_single` builds the batch with `from_timecourses`, which keeps `sd`, `se`, `n`.

```bash
git add src/pkpdutils/nca tests/nca
git commit -q -m "Add the bootstrap of the NCA parameters of group timecourses

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 5: Delta method

**Files:**
- Modify: `src/pkpdutils/nca/uncertainty.py` (add `delta`), `src/pkpdutils/nca/nca.py` (`DELTA` branch)
- Test: `tests/nca/test_uncertainty.py` (append)

**Interfaces:**
- Produces: `delta(timecourses, options, point) -> dict[str, np.ndarray]`: perturbs every time point `i` of every row by `h_i = options.delta_step * se_i` (points without finite positive `se` are skipped), runs `run_rows` on the `N * n_time` perturbed rows, forms `dx/dC_i = (x(C + h_i e_i) - x(C)) / h_i` and `var(x) = sum_i (dx/dC_i)^2 se_i^2`; returns `x_se = sqrt(var)`, `x_sd = x_se * sqrt(n)` (`NaN` without `n`), normal confidence interval `x ± z * x_se` (`z = norm.ppf(1 - alpha/2)`) for other parameters and, for log-normal parameters, the interval on the log scale `x * exp(±z * x_se / x)` with `x_geomean = x` and `x_geocv = sqrt(exp((x_se/x)^2) - 1)`; discrete parameters skipped. The delta method always uses `se` (the uncertainty of the mean curve); `bootstrap_spread` does not apply.

- [ ] **Step 1: Append the failing tests**

```python
def test_delta_linear_auc_matches_closed_form() -> None:
    tc = group_curve(cv=0.1, n=12)
    options = NCAOptions(uncertainty=UncertaintyMethod.DELTA, auc_method=AUCMethod.LINEAR, ci_level=0.95)
    q = nca_single(tc, options).to_quantities()
    # linear trapezoid to tlast: auc = sum_i w_i c_i with w_0 = (t_1 - t_0)/2, w_last = (t_last - t_last-1)/2, else (t_i+1 - t_i-1)/2;
    # plus the inserted (0, c0) segment for the bolus: w_0 gains t_0/2 and c0 depends on the first two points (ignored here: compare auc_last of a dose-less curve)
    tc_nd = tc.model_copy(update={"dose": None})
    q = nca_single(tc_nd, options).to_quantities()
    assert tc_nd.se is not None
    w = np.empty(T.size)
    w[0] = (T[1] - T[0]) / 2
    w[-1] = (T[-1] - T[-2]) / 2
    w[1:-1] = (T[2:] - T[:-2]) / 2
    expected_se = np.sqrt(np.sum((w * tc_nd.se) ** 2))
    assert q["auc_last_se"].magnitude == pytest.approx(expected_se, rel=1e-3)
    assert q["auc_last_sd"].magnitude == pytest.approx(expected_se * np.sqrt(12), rel=1e-3)
    z = 1.959963984540054
    assert q["auc_last_ci_high"].magnitude == pytest.approx(q["auc_last"].magnitude * np.exp(z * expected_se / q["auc_last"].magnitude), rel=1e-3)
    assert q["auc_last_geomean"].magnitude == pytest.approx(q["auc_last"].magnitude)
    assert np.isnan(q["tmax_se"].magnitude) if "tmax_se" in q else True


def test_delta_agrees_with_bootstrap() -> None:
    tc = group_curve(cv=0.05, n=12)
    delta = nca_single(tc, NCAOptions(uncertainty=UncertaintyMethod.DELTA, auc_method=AUCMethod.LINEAR)).to_quantities()
    boot = nca_single(tc, NCAOptions(uncertainty=UncertaintyMethod.BOOTSTRAP, auc_method=AUCMethod.LINEAR, n_boot=4000, seed=7)).to_quantities()
    for name in ("auc_last_se", "cmax_se", "lambda_z_se"):
        assert delta[name].magnitude == pytest.approx(boot[name].magnitude, rel=0.15), name


def test_delta_batch_and_no_se() -> None:
    tcs = Timecourses.from_timecourses([group_curve(label="a"), group_curve(cv=0.2, label="b")])
    result = nca(tcs, NCAOptions(uncertainty=UncertaintyMethod.DELTA))
    assert result.has_uncertainty
    assert result["auc_last_se"].values[1] > result["auc_last_se"].values[0]
    plain = Timecourse(time=T, value=C0 * np.exp(-K * T), time_unit="hr", unit="mg/l")
    with pytest.raises(ValueError, match="se"):
        nca_single(plain, NCAOptions(uncertainty=UncertaintyMethod.DELTA))
```

Run: `uv run pytest tests/nca/test_uncertainty.py -n 0 -q -k delta` → FAIL (`ImportError`/`ValueError`).

- [ ] **Step 2: Implement `delta`**

Append to `uncertainty.py`:
```python
def delta(
    timecourses: Timecourses,
    options: NCAOptions,
    point: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Delta method: propagate the standard errors of the points through the numerical Jacobian.

    `var(x) = sum_i (dx/dC_i)^2 se_i^2` with `dx/dC_i` from a forward difference
    of step `options.delta_step * se_i` (Efron & Tibshirani 1993, ch. 5).

    Args:
        timecourses: the batch (group curves with `se`, or `sd` and `n`)
        options: `delta_step`, `ci_level`
        point: the parameters of the original curves (`run_rows` output)

    Returns:
        The uncertainty variables per continuous parameter.

    Raises:
        ValueError: if the batch has no `se` and it cannot be derived.
    """
    from scipy.stats import norm  # noqa: PLC0415

    from pkpdutils.nca.nca import run_rows  # noqa: PLC0415

    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    t = timecourses.times.reshape(n_rows, n_time)
    c = timecourses.values.reshape(n_rows, n_time)
    se = resolve_spread(timecourses, options.model_copy(update={"bootstrap_spread": BootstrapSpread.SE}))
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(se) & (se > 0) & np.isfinite(c)
    h = np.where(usable, options.delta_step * se, 0.0)

    rows = np.arange(n_rows * n_time)
    cols = np.tile(np.arange(n_time), n_rows)
    c_pert = np.repeat(c, n_time, axis=0)
    c_pert[rows, cols] += np.repeat(h, n_time, axis=0)[rows, cols]

    def repeat(a: np.ndarray | None) -> np.ndarray | None:
        """Repeat a per-row array `n_time` times."""
        return None if a is None else np.repeat(np.asarray(a, dtype=np.float64).reshape(n_rows), n_time)

    perturbed = run_rows(
        np.repeat(t, n_time, axis=0),
        c_pert,
        dose_amount=repeat(timecourses.dose_amount),
        dose_time=repeat(timecourses.dose_time),
        dose_duration=repeat(timecourses.dose_duration),
        route=timecourses.route,
        options=options,
    )
    alpha = 1.0 - options.ci_level
    z = float(norm.ppf(1.0 - alpha / 2.0))
    n_subjects = None if timecourses.n is None else np.asarray(timecourses.n, dtype=np.float64).reshape(n_rows)
    out: dict[str, np.ndarray] = {}
    for name, base in point.items():
        if name in DISCRETE_PARAMETERS or name not in perturbed:
            continue
        pert = perturbed[name].reshape(n_rows, n_time)
        with np.errstate(invalid="ignore", divide="ignore"):
            derivative = np.where(h > 0, (pert - base[:, None]) / np.where(h > 0, h, 1.0), 0.0)
            var = np.sum((derivative * np.where(usable, se, 0.0)) ** 2, axis=1)
            valid = np.isfinite(base) & np.isfinite(var) & usable.any(axis=1)
            x_se = np.where(valid, np.sqrt(var), np.nan)
            x_sd = x_se * np.sqrt(n_subjects) if n_subjects is not None else np.full_like(x_se, np.nan)
            if name in LOGNORMAL_PARAMETERS:
                rel = x_se / base
                low = base * np.exp(-z * rel)
                high = base * np.exp(z * rel)
                out[f"{name}_geomean"] = np.where(valid, base, np.nan)
                out[f"{name}_geocv"] = np.where(valid, np.sqrt(np.expm1(rel * rel)), np.nan)
            else:
                low = base - z * x_se
                high = base + z * x_se
        out[f"{name}_sd"] = x_sd
        out[f"{name}_se"] = x_se
        out[f"{name}_ci_low"] = np.where(valid, low, np.nan)
        out[f"{name}_ci_high"] = np.where(valid, high, np.nan)
    logger.info("delta method: %d curves x %d perturbations", n_rows, n_time)
    return out
```
Add the `DELTA` branch to `nca()` (Task 4 Step 3 shows it). `scipy` is a runtime dependency; the local import keeps the module import light, a top-level `from scipy.stats import norm` is equally fine (choose the top-level import if ruff or ty prefers it).

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/nca -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. If `test_delta_agrees_with_bootstrap` fails for `lambda_z_se` beyond 15 %: the bootstrap of `lambda_z` with the `BEST_FIT` window can jump between windows; compare with `TerminalPhase(method=ALL_AFTER_TMAX)` in both options (adjust the test) and keep the tolerance.

```bash
git add src/pkpdutils/nca tests/nca
git commit -q -m "Add the delta method for the uncertainty of the NCA parameters

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 6: `NCAResult.summarize`

**Files:**
- Modify: `src/pkpdutils/nca/result.py`
- Test: `tests/nca/test_summarize.py`

**Interfaces:**
- Produces: `NCAResult.summarize(dim: str, ci_level: float = 0.95) -> NCAResult`: reduces the result over one sample dimension; for every parameter `x` (base names, finite values along `dim`): `x` = arithmetic mean, `x_sd` (ddof 1), `x_se = x_sd / sqrt(x_n)`, `x_ci_low`/`x_ci_high` = `x ± t_{x_n - 1, 1 - alpha/2} * x_se`, `x_median`, `x_q25`, `x_q75`, `x_n` (finite count) and, for log-normal parameters, `x_geomean`, `x_geocv` from the logarithms of the positive values; `n` = number of samples along `dim`; `flags` = bitwise OR along `dim`; units copied; existing derived variables of the input are dropped (a summary of individual results). `ValueError` when `dim` is not a sample dimension.

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_summarize.py`:
```python
import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca

K = 0.3
T = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])


def individuals(n: int = 6) -> Timecourses:
    rng = np.random.default_rng(0)
    curves = []
    for i in range(n):
        c0 = 10.0 * rng.lognormal(0, 0.2)
        curves.append(Timecourse(time=T, value=c0 * np.exp(-K * T), time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS), substance="x", label=f"s{i}"))
    return Timecourses.from_timecourses(curves)


def test_summarize_statistics() -> None:
    result = nca(individuals(), NCAOptions(auc_method=AUCMethod.LOG))
    summary = result.summarize("individual")
    assert summary.sample_dims == ()
    values = result["auc_inf_obs"].values
    q = summary.to_quantities()
    assert q["auc_inf_obs"].magnitude == pytest.approx(values.mean())
    assert q["auc_inf_obs_sd"].magnitude == pytest.approx(values.std(ddof=1))
    assert q["auc_inf_obs_se"].magnitude == pytest.approx(values.std(ddof=1) / np.sqrt(6))
    tq = student_t.ppf(0.975, 5)
    assert q["auc_inf_obs_ci_high"].magnitude == pytest.approx(values.mean() + tq * values.std(ddof=1) / np.sqrt(6))
    assert q["auc_inf_obs_median"].magnitude == pytest.approx(np.median(values))
    assert q["auc_inf_obs_q25"].magnitude == pytest.approx(np.percentile(values, 25))
    assert q["auc_inf_obs_geomean"].magnitude == pytest.approx(np.exp(np.log(values).mean()))
    assert q["auc_inf_obs_geocv"].magnitude == pytest.approx(np.sqrt(np.expm1(np.log(values).var(ddof=1))))
    assert q["auc_inf_obs_n"].magnitude == 6
    assert q["n"].magnitude == 6
    assert "tmax" in summary.parameters and "tmax_geomean" not in summary
    assert str(q["auc_inf_obs_sd"].units) == str(q["auc_inf_obs"].units)
    assert summary.flags() == []


def test_summarize_keeps_other_dims_and_ors_flags() -> None:
    tcs = individuals(4)
    ds = tcs.ds.assign_coords(individual=["a", "b", "c", "d"])
    result = nca(Timecourses(ds), NCAOptions())
    # a second sample dimension: stack two copies along "study"
    import xarray as xr

    two = Timecourses(xr.concat([tcs.ds, tcs.ds], dim="study").assign_coords(study=["s1", "s2"]))
    result2 = nca(two)
    summary = result2.summarize("individual")
    assert summary.sample_dims == ("study",)
    assert summary["auc_last"].shape == (2,)
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l", label="short")
    mixed = Timecourses.from_timecourses([*list(tcs), short])
    flags = nca(mixed).summarize("individual").flags()
    assert "TOO_FEW_POINTS" in flags
    assert nca(mixed).summarize("individual").to_quantities()["lambda_z_n"].magnitude == 4
    with pytest.raises(ValueError, match="dim"):
        result.summarize("study")
```

Run: `uv run pytest tests/nca/test_summarize.py -n 0 -q` → FAIL with `AttributeError: 'NCAResult' object has no attribute 'summarize'`.

- [ ] **Step 2: Implement**

In `result.py` add (imports: `from scipy.stats import t as student_t`, `from pkpdutils.nca.uncertainty import LOGNORMAL_PARAMETERS, base_name`):
```python
    def summarize(self, dim: str, ci_level: float = 0.95) -> "NCAResult":
        """Summarize the parameters of individual samples over one sample dimension.

        For every parameter `x` the summary carries the arithmetic mean `x`,
        `x_sd`, `x_se`, the t-based confidence interval `x_ci_low`/`x_ci_high`
        at `ci_level`, `x_median`, `x_q25`, `x_q75`, the count of finite values
        `x_n` and, for log-normal parameters, `x_geomean` and `x_geocv`;
        `n` is the number of samples along `dim` and `flags` the union of their
        flags. Derived variables of the input are dropped.

        Args:
            dim: the sample dimension to reduce
            ci_level: level of the confidence interval of the mean

        Returns:
            The summary over the remaining sample dimensions.

        Raises:
            ValueError: if `dim` is not a sample dimension of the result.
        """
        if dim not in self.sample_dims:
            raise ValueError(f"'{dim}' is not a sample dimension {self.sample_dims}")
        alpha = 1.0 - ci_level
        data_vars: dict[str, Any] = {}
        for name in self.parameters:
            da = self.ds[name].transpose(..., dim)
            values = da.to_numpy().astype(np.float64)
            dims = tuple(str(d) for d in da.dims if d != dim)
            units = self.units(name)
            with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                finite = np.isfinite(values)
                count = finite.sum(axis=-1).astype(np.float64)
                filled = np.where(finite, values, np.nan)
                mean = np.nanmean(filled, axis=-1)
                sd = np.where(count > 1, np.nanstd(filled, axis=-1, ddof=1), np.nan)
                se = sd / np.sqrt(count)
                tq = student_t.ppf(1.0 - alpha / 2.0, np.maximum(count - 1.0, 1.0))
                low, high = mean - tq * se, mean + tq * se
                median = np.nanmedian(filled, axis=-1)
                q25, q75 = np.nanpercentile(filled, [25, 75], axis=-1)
                mean = np.where(count > 0, mean, np.nan)
                median = np.where(count > 0, median, np.nan)
            data_vars[name] = (dims, mean, {"units": units})
            data_vars[f"{name}_sd"] = (dims, sd, {"units": units})
            data_vars[f"{name}_se"] = (dims, se, {"units": units})
            data_vars[f"{name}_ci_low"] = (dims, np.where(count > 1, low, np.nan), {"units": units})
            data_vars[f"{name}_ci_high"] = (dims, np.where(count > 1, high, np.nan), {"units": units})
            data_vars[f"{name}_median"] = (dims, median, {"units": units})
            data_vars[f"{name}_q25"] = (dims, np.where(count > 0, q25, np.nan), {"units": units})
            data_vars[f"{name}_q75"] = (dims, np.where(count > 0, q75, np.nan), {"units": units})
            data_vars[f"{name}_n"] = (dims, count, {"units": "dimensionless"})
            if name in LOGNORMAL_PARAMETERS:
                with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    logs = np.where(finite & (values > 0), np.log(np.where(values > 0, values, 1.0)), np.nan)
                    n_pos = np.isfinite(logs).sum(axis=-1)
                    geomean = np.where(n_pos > 0, np.exp(np.nanmean(logs, axis=-1)), np.nan)
                    geocv = np.where(n_pos > 1, np.sqrt(np.expm1(np.nanvar(logs, axis=-1, ddof=1))), np.nan)
                data_vars[f"{name}_geomean"] = (dims, geomean, {"units": units})
                data_vars[f"{name}_geocv"] = (dims, geocv, {"units": "dimensionless"})
        flags = self.ds["flags"].transpose(..., dim)
        remaining = tuple(str(d) for d in flags.dims if d != dim)
        data_vars["n"] = (remaining, np.full(flags.shape[:-1], float(self.ds.sizes[dim])), {"units": "dimensionless"})
        data_vars["flags"] = (remaining, np.bitwise_or.reduce(flags.to_numpy().astype(np.int64), axis=-1), {"units": "dimensionless"})
        coords = {d: self.ds[d] for d in remaining if d in self.ds.coords}
        return NCAResult(xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(self.ds.attrs)))
```
(`import warnings`, `import numpy as np` at the top of `result.py`.) The `np.bitwise_or.reduce` over an empty axis would fail only for a zero-length dim, which xarray does not produce here.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/nca -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. In `test_summarize_keeps_other_dims_and_ors_flags`, move the `import xarray as xr` to the top of the file (ruff `E402`); `xr.concat` of the two batch datasets needs matching coordinates (it does: the same dataset twice).

```bash
git add src/pkpdutils/nca/result.py tests/nca/test_summarize.py
git commit -q -m "Add NCAResult.summarize over a sample dimension

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 7: Example, documentation, exports

**Files:**
- Create: `examples/group_uncertainty.py`, `docs/uncertainty.md`, `docs/api/nca.uncertainty.md`
- Modify: `docs/nca.md` (API section: link), `docs/glossary.md` (rows), `docs/index.md` (feature bullet), `docs/api/index.md` (row), `zensical.toml` (nav), `CLAUDE.md` (architecture paragraph, example command), `examples/README.md` (row), `tests/examples/test_examples.py` (`SCRIPTS`), `src/pkpdutils/__init__.py` (export `partial_auc`)

- [ ] **Step 1: Example**

`examples/group_uncertainty.py`:
```python
"""Uncertainty of the NCA parameters of group timecourses.

A publication reports the mean concentration of a group with its standard
deviation and the number of subjects. The bootstrap and the delta method
propagate this uncertainty to the parameters; individual curves are summarized
over the individuals instead.

Run from the root of the repository with `python -m examples.group_uncertainty`.
Writes `group_uncertainty.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, NCAOptions, Route, Timecourse, Timecourses, nca, nca_single
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod, UncertaintyMethod, partial_auc
from pkpdutils.nca.options import BootstrapSpread
from pkpdutils.plot import plot_timecourse

t = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
mean = np.array([1.9, 2.6, 2.4, 1.8, 1.3, 0.95, 0.5, 0.12])
sd = np.array([0.4, 0.5, 0.4, 0.3, 0.25, 0.2, 0.12, 0.04])
group = Timecourse(
    time=t, value=mean, sd=sd, n=10, time_unit="hr", unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine", label="group mean",
)

if __name__ == "__main__":
    console.rule("Bootstrap (default for group data): uncertainty of the mean curve")
    boot = nca_single(group, NCAOptions(seed=1, n_boot=2000))
    df = boot.to_dataframe().T
    console.print(df.loc[[n for n in df.index if n.startswith(("auc_inf_obs", "cmax", "thalf", "cl_f"))]])

    console.rule("Bootstrap with the spread of individuals (sd) instead of the mean (se)")
    spread = nca_single(group, NCAOptions(seed=1, n_boot=2000, bootstrap_spread=BootstrapSpread.SD))
    console.print(spread.to_quantities()["auc_inf_obs_sd"], spread.to_quantities()["auc_inf_obs_geocv"])

    console.rule("Delta method")
    delta = nca_single(group, NCAOptions(uncertainty=UncertaintyMethod.DELTA))
    for name in ("auc_inf_obs", "cmax", "thalf"):
        console.print(f"{name:<12} {delta.to_quantities()[name]:~P}  se {delta.to_quantities()[name + '_se']:~P}")

    console.rule("Individuals: summarize over the sample dimension")
    rng = np.random.default_rng(2)
    curves = [
        Timecourse(time=t, value=mean * rng.lognormal(0, 0.15, size=t.size), time_unit="hr", unit="mg/l",
                   dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine", label=f"s{i}")
        for i in range(8)
    ]
    individuals = Timecourses.from_timecourses(curves)
    summary = nca(individuals, NCAOptions(auc_method=AUCMethod.LINEAR_LOG)).summarize("individual")
    console.print(summary.to_dataframe().T.loc[["auc_inf_obs", "auc_inf_obs_sd", "auc_inf_obs_ci_low", "auc_inf_obs_ci_high", "auc_inf_obs_geomean", "auc_inf_obs_geocv", "n"]])

    console.rule("Partial AUC 0-6 h of the individuals")
    console.print(partial_auc(individuals, 0.5, 6.0).values.round(3))

    fig = plot_timecourse(group)
    fig.savefig("group_uncertainty.png", dpi=120)
    console.print("written: group_uncertainty.png")
```
Register `"examples.group_uncertainty"` in `tests/examples/test_examples.py::SCRIPTS`; add the row `| \`examples/group_uncertainty.py\` | bootstrap and delta method for a group mean curve, summary over individuals, partial AUC |` to `examples/README.md`. Export `partial_auc` from `pkpdutils/__init__.py` (`__all__` sorted).

Run: `cd $(mktemp -d) && PYTHONPATH=/home/mkoenig/git/pkdb_analysis MPLBACKEND=Agg uv run --project /home/mkoenig/git/pkdb_analysis python -m examples.group_uncertainty; cd -` → exit 0, `group_uncertainty.png` written.

- [ ] **Step 2: `docs/uncertainty.md`**

```markdown
# Uncertainty

Published pharmacokinetic data are mostly group data: the mean concentration of a group at every sampling time with its standard deviation or standard error and the number of subjects. The parameters of the mean curve are point estimates; how uncertain they are depends on the uncertainty of the points and on how the parameters depend on them. `pkpdutils` propagates the uncertainty of a group timecourse to every parameter of the [non-compartmental analysis](nca.md) by a parametric bootstrap or by the delta method, and it summarizes the parameters of individual curves over the individuals with the same set of variables, so that the statistics of the next pages accept both.

## Concepts

**Spread of the mean and spread of the individuals.** The standard error \(\mathrm{se}_i = \mathrm{sd}_i / \sqrt{n}\) of a time point is the uncertainty of the group mean; the standard deviation \(\mathrm{sd}_i\) is the spread of the individual subjects. Which one to propagate depends on the question: the uncertainty of the parameters of the mean curve (`BootstrapSpread.SE`, the default) or the spread of the parameters over subjects (`BootstrapSpread.SD`). A `Timecourse` carries `sd`, `se` and `n` and derives the missing one; the result reports both `x_se` and `x_sd`, converting with \(\sqrt n\).

**Bootstrap.** The parametric bootstrap [^efron] draws every time point of the curve from a distribution with the observed mean and spread, analyses every replicate curve as if it were observed, and reads the uncertainty of a parameter from the spread of its replicates. Normal draws (the default) are set to 0 when they fall below 0; log-normal draws with the same mean and spread are positive by construction and suit concentrations with a large relative spread. `n_boot` replicates of every curve run through the same vectorized code as the curves themselves, so the bootstrap of a batch is one call, chunked by `NCAOptions.chunk_rows`.

**Delta method.** The delta method [^efron] linearizes a parameter around the observed curve: every time point is perturbed by a small step, the numerical derivative of every parameter with respect to every point is formed, and the variances of the points add through the squared derivatives. It costs one analysis per time point instead of one per replicate, gives symmetric normal intervals (log-normal parameters on the logarithmic scale) and is exact for the linear trapezoid area; it cannot follow a change of the terminal window.

**Discrete parameters.** `tmax`, `tlast`, `tmin`, `tmax_half`, `temax` and the counts of the terminal regression are read from the observed points; they carry no uncertainty variables.

**Individuals.** When every subject has its own curve the parameters of the subjects are a sample: `NCAResult.summarize(dim)` reduces the result over a sample dimension to the mean, standard deviation, standard error, a t-based confidence interval of the mean, median and quartiles, the number of values and, for log-normal parameters, the geometric mean and geometric CV. The variables have the same names as the bootstrap output, so a group result and a summary look alike.

## Math

Parametric bootstrap with \(B\) replicates of a point \(\bar C_i\) with spread \(s_i\):

\[
C_i^{(b)} \sim \mathcal N(\bar C_i, s_i^2) \quad\text{or}\quad C_i^{(b)} \sim \mathrm{LogNormal}\!\left(\ln \bar C_i - \tfrac{\sigma_i^2}{2},\ \sigma_i^2\right),\ \sigma_i^2 = \ln\!\left(1 + \frac{s_i^2}{\bar C_i^2}\right)
\]

\[
\mathrm{se}(x) = \sqrt{\frac{1}{B-1}\sum_b \left(x^{(b)} - \bar x^{(\cdot)}\right)^2}\ \ (s_i = \mathrm{se}_i), \qquad
\mathrm{CI} = \left[x^{(\alpha/2)},\ x^{(1-\alpha/2)}\right], \qquad
\mathrm{GM} = \exp\!\left(\overline{\ln x^{(b)}}\right), \quad
\mathrm{GCV} = \sqrt{e^{\mathrm{Var}(\ln x^{(b)})} - 1}
\]

Delta method with the step \(h_i = \delta\, \mathrm{se}_i\):

\[
\frac{\partial x}{\partial C_i} \approx \frac{x(C + h_i e_i) - x(C)}{h_i}, \qquad
\mathrm{Var}(x) = \sum_i \left(\frac{\partial x}{\partial C_i}\right)^2 \mathrm{se}_i^2, \qquad
\mathrm{CI} = x \pm z_{1-\alpha/2}\,\mathrm{se}(x) \ \text{ or } \ x\, e^{\pm z_{1-\alpha/2}\,\mathrm{se}(x)/x}
\]

For the linear trapezoid rule \(\mathrm{AUC} = \sum_i w_i C_i\) is linear in the points and the delta method is exact: \(\mathrm{Var}(\mathrm{AUC}) = \sum_i w_i^2 \mathrm{se}_i^2\).

Summary of \(n\) individual values \(x_j\): mean \(\bar x\), \(\mathrm{sd}\) with \(n-1\), \(\mathrm{se} = \mathrm{sd}/\sqrt n\), \(\mathrm{CI} = \bar x \pm t_{n-1,\,1-\alpha/2}\,\mathrm{se}\), geometric mean and CV from \(\ln x_j\).

## Variables

| variable | meaning | unit |
| --- | --- | --- |
| `x` | the parameter of the mean curve (bootstrap, delta) or the mean over the individuals (summary) | unit of `x` |
| `x_sd`, `x_se` | standard deviation over subjects and standard error of the mean | unit of `x` |
| `x_ci_low`, `x_ci_high` | confidence interval at `ci_level` (percentile, normal or t-based) | unit of `x` |
| `x_geomean`, `x_geocv` | geometric mean and geometric coefficient of variation (log-normal parameters) | unit of `x`, - |
| `x_median`, `x_q25`, `x_q75`, `x_n` | median, quartiles and count of finite values (summary only) | unit of `x`, - |
| `n` | number of subjects (group data) or of samples (summary) | - |

## API

Group data: the bootstrap is the default as soon as the timecourse carries `sd` or `se`:

```python
from pkpdutils import Dose, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.nca import UncertaintyMethod
from pkpdutils.nca.options import BootstrapDistribution, BootstrapSpread

group = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12, 24], value=[1.9, 2.6, 2.4, 1.8, 0.95, 0.5, 0.12],
    sd=[0.4, 0.5, 0.4, 0.3, 0.2, 0.12, 0.04], n=10,
    time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine",
)
result = nca_single(group, NCAOptions(seed=1, n_boot=2000))
q = result.to_quantities()
q["auc_inf_obs"], q["auc_inf_obs_se"], q["auc_inf_obs_ci_low"], q["auc_inf_obs_ci_high"]
result = nca_single(group, NCAOptions(bootstrap_spread=BootstrapSpread.SD, bootstrap_distribution=BootstrapDistribution.LOGNORMAL))
result = nca_single(group, NCAOptions(uncertainty=UncertaintyMethod.DELTA))
```

Individual curves:

```python
result = nca(individuals)             # individuals: Timecourses over "individual"
summary = result.summarize("individual")
summary.to_dataframe()                # auc_inf_obs, auc_inf_obs_sd, ..., auc_inf_obs_geocv, n
```

Partial areas, e.g. \(\mathrm{AUC}_{0\text{-}6}\) of every sample:

```python
from pkpdutils.nca import partial_auc

area = partial_auc(individuals, 0.5, 6.0)   # DataArray over the sample dims, attrs["units"]
```

The example is `examples/group_uncertainty.py`; the reference of the module is in [API: nca.uncertainty](api/nca.uncertainty.md).

## References

[^efron]: Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman & Hall/CRC; 1993, ch. 5 (delta method) and 6 (bootstrap). See [References](references.md#statistics).
```

`docs/api/nca.uncertainty.md`:
```markdown
# nca.uncertainty

::: pkpdutils.nca.uncertainty
```

- [ ] **Step 3: Wire the pages**

- `zensical.toml`: add `{ "Uncertainty" = "uncertainty.md" }` after the NCA entry in the user guide and `{ "uncertainty" = "api/nca.uncertainty.md" }` after `steady_state` in the `pkpdutils.nca` API list.
- `docs/api/index.md`: add the row `| [nca.uncertainty](nca.uncertainty.md) | bootstrap and delta method of the parameters of group timecourses |`.
- `docs/index.md`: add the feature bullet `- **[Uncertainty](uncertainty.md)** - bootstrap and delta method for group timecourses, summaries over individuals, partial areas.` after the NCA bullet.
- `docs/nca.md`: in the API section add the sentence "Group timecourses with `sd`/`se` get uncertainty variables per parameter, individual results are summarized with `NCAResult.summarize`, see [Uncertainty](uncertainty.md); partial areas come from `partial_auc`."
- `docs/glossary.md`: add the rows for `x_sd`/`x_se`, `x_ci_low`/`x_ci_high`, `x_geomean`/`x_geocv`, `x_median`/`x_q25`/`x_q75`/`x_n`, `n`, `auc_partial` pointing at [Uncertainty](uncertainty.md).
- `docs/timecourses.md` already links `uncertainty.md`.
- `CLAUDE.md`: add `python -m examples.group_uncertainty` to the commands; in the `nca/` architecture paragraph add: "`uncertainty.py` propagates `sd`/`se` of group curves: `bootstrap` resamples every point (`resample_values`), runs `run_rows` on the `N*B` replicate rows and reduces them (`reduce_replicates`); `delta` perturbs every point once and propagates `se` through the numerical Jacobian; `NCAOptions.resolve_uncertainty` picks the bootstrap by default for group data. Variables `x_sd`, `x_se`, `x_ci_low`, `x_ci_high`, `x_geomean`, `x_geocv`, `n`; `NCAResult.summarize(dim)` gives the same layout for individual results; `partial_auc` (`nca.py`) is the area between two times."

- [ ] **Step 4: Build and check**

Run: `uv run zensical build --clean && uv run python scripts/llms_txt.py && uv run pytest -q -W error && uv run ruff check && uv run ruff format --check && uv run ty check && uv run pre-commit run --all-files && grep -rn $'\xe2\x80\x94' docs README.md CLAUDE.md src examples | grep -v docs/superpowers | wc -l`
Expected: site builds (the `pd.md` link warning stays), all pass, `0` em dashes.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -q -m "Document the uncertainty of the NCA parameters and add the group example

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 8: Matrix and pull request

- [ ] **Step 1: Tox**

Run: `uv run tox run-parallel` → `py3.13`, `py3.14`, `ty` OK.

- [ ] **Step 2: Push and open the pull request**

```bash
git push -u origin uncertainty
gh pr create --base develop --title "Add the uncertainty of the NCA parameters" --body-file - <<'EOF'
## Summary

Phase 4 of the pkpdutils redesign (`docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`, section 2 "Uncertainty"): parametric bootstrap and delta method for group timecourses (`NCAOptions.uncertainty`, `n_boot`, `seed`, `ci_level`, `bootstrap_spread`, `bootstrap_distribution`, `delta_step`), result variables `x_sd`, `x_se`, `x_ci_low`, `x_ci_high`, `x_geomean`, `x_geocv`, `n`; `NCAResult.summarize(dim)` for individual results; `partial_auc`; documentation (`uncertainty.md`), the group example, and the em dashes replaced by plain dashes.

## Checklist

- [x] the pull request targets `develop`
- [x] tests were added or updated for the change
- [x] `tox run-parallel` passes locally (tests and `ty`)
- [x] `ruff check` and `ruff format` are clean, e.g., via `pre-commit run --all-files`
- [x] public functions and classes have type annotations and a docstring
- [x] user visible changes are in `docs/` (release notes are written at the release)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z
EOF
gh pr merge --squash --auto --delete-branch
```

- [ ] **Step 3: Wait for the checks and sync**

Run: `until [ "$(gh pr view --json state -q .state)" = "MERGED" ] || [ "$(gh pr checks --json bucket -q '[.[] | select(.bucket=="fail")] | length')" != "0" ]; do sleep 30; done; gh pr checks; git switch develop && git pull`

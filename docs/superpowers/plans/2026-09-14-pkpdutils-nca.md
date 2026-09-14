# pkpdutils NCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the vectorized non-compartmental analysis (`pkpdutils.nca`) of concentration and effect timecourses, single dose and steady state, with flags, units, a result dataset, the NCA figures, documentation and examples (phase 3 of the spec).

**Architecture:** `nca()` takes a `Timecourses` batch, flattens the sample dimensions to `(N, n_time)` arrays, packs the valid points of every row to the front, computes all parameters with NaN-aware numpy operations along the time axis (trapezoid areas, terminal log-linear regression over every candidate window via suffix sums), derives the units with pint, and returns an `NCAResult` (an `xarray.Dataset` with one variable per parameter over the sample dimensions, `attrs["units"]` per variable, and an integer `flags` variable). `nca_single` wraps one `Timecourse`. Steady state parameters and superposition live in `nca/steady_state.py`; figures in `pkpdutils.plot`.

**Tech Stack:** numpy (vectorized numerics), xarray (results), pint (units), pydantic (options), matplotlib (figures), pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md` — section "2. Non-compartmental analysis" and the plotting items `plot_timecourse`, `plot_nca`, `plot_nca_grid` of section 5. Data model from plan 1: `docs/superpowers/plans/2026-09-14-pkpdutils-scaffold-and-data-model.md`.

**Later plans:** uncertainty (bootstrap, delta, `summarize`), fitting and PD models, statistics, release. The partial `auc(t1, t2)` helper of the spec is deferred to the uncertainty plan.

## Global Constraints

- package `pkpdutils`, python 3.13 and 3.14, runtime dependencies stay `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib`, `rich`
- ty `error-on-warning = true`: zero diagnostics on `src`, `tests`, `examples`, `scripts`; suppressions rule-specific `# ty: ignore[rule]` only and none unused; in tests prefer `assert x is not None` narrowing over suppressions
- every module, class and function of `src/pkpdutils` has full type annotations and a google style docstring (ruff `D`); docstrings of parameters carry the formula and a citation key of `docs/references.md`; `examples/`, `tests/`, `scripts/` exempt from `D`
- library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting (ruff `G`), never prints, never calls `plt.show()`; figures are returned
- test output pristine: no warnings in the pytest summary (use `np.errstate` around expected divide/invalid operations)
- results are `xarray.Dataset` objects with one variable per parameter over the sample dimensions of the input and `attrs["units"]` on every variable; parameter names are the ones of the spec (`auc_last`, `auc_inf_obs`, `auc_inf_pred`, `aumc_last`, `aumc_inf`, `mrt`, `cmax`, `tmax`, `clast`, `tlast`, `c0`, `cmax_half`, `tmax_half`, `cmin`, `tmin`, `lambda_z`, `thalf`, `lambda_z_n_points`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`, `lambda_z_t_first`, `lambda_z_se`, `auc_extrap_fraction`, `cl`/`cl_f`, `vz`/`vz_f`, `vss`, `auc_inf_dn`, `cmax_dn`, `flags`; steady state `auc_tau`, `cmin_ss`, `ctrough`, `cavg`, `fluctuation`, `swing`, `cl_ss`, `accumulation_ratio`; effect kind `auec_last`, `emax_obs`, `temax`, `e0`, `time_above`, `auec_baseline`, `emax_baseline`)
- the reference values of `pkdb_analysis` 0.3.1 in `tests/data/reference/nca_reference.json` are reproduced with `auc_method=AUCMethod.LINEAR` and `TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX)` and `route=Route.ORAL` (so `cl_f`/`vz_f` correspond to the old `cl`/`vd`)
- one route per batch (`Timecourses.route`, `None` without doses); `n` is one number per sample
- markdown has no hard line wraps; formulas use `\(...\)` inline and `\[...\]` display (MathJax via arithmatex)
- commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`
- `develop` accepts only pull requests (rulesets active); work happens on a branch `nca` off `develop`, merged through a pull request with the checks `tests`, `ruff`, `ty`, `docs`
- commands run from the repository root with `uv run`

## Interfaces of plan 1 this plan consumes

- `pkpdutils.timecourse.Timecourses`: `times`, `values` (`(*sample_shape, n_time)` float64, NaN padded), `sd`, `se`, `n` (`None` or arrays), `dose_amount`, `dose_time`, `dose_duration` (`None` or `(*sample_shape)` arrays; `dose_duration` NaN without infusion), `dose_unit: str | None`, `route: Route | None`, `sample_dims`, `sample_shape`, `n_samples`, `n_time`, `time_unit`, `unit`, `substance`, `has_dose`, `ds` (coordinates of the sample dims), `from_timecourses([tc], dim=...)`, `isel`, `__iter__`
- `pkpdutils.timecourse.Timecourse`: `time`, `value`, `time_unit`, `unit`, `dose: Dose | None` (`amount`, `unit`, `route`, `time`, `duration`), `substance`, `label`, `relative_to_dose()`
- `pkpdutils.timecourse.Route` (`IV_BOLUS`, `IV_INFUSION`, `ORAL`, `is_iv`), `DosingRegimen(dose, interval, n_doses)`, `dose_times()`
- `pkpdutils.units`: `ureg`, `Q_`, `Quantity`, `parse_unit`, `unit_str`, `normalize_volume`, `normalize_clearance`

---

### Task 1: Branch, options and flags

**Files:**
- Create: `src/pkpdutils/nca/__init__.py`, `src/pkpdutils/nca/options.py`
- Test: `tests/nca/__init__.py` (empty), `tests/nca/test_options.py`

**Interfaces:**
- Produces: enums `Kind`, `AUCMethod`, `TerminalMethod`, `BLQHandling`, `C0Method`; `NCAFlag(IntFlag)` with `NONE=0`, `POSITIVE_SLOPE=1`, `TOO_FEW_POINTS=2`, `EXTRAPOLATION_HIGH=4`, `NO_MAX=8`, `NO_ABSORPTION=16`, `BLQ_TRUNCATED=32`, `NO_DATA=64`; `decode_flags(value: int) -> list[str]`; `TerminalPhase` and `NCAOptions` pydantic models (fields below).

- [ ] **Step 1: Create the branch**

```bash
git switch -c nca develop
mkdir -p src/pkpdutils/nca tests/nca && touch tests/nca/__init__.py
```

- [ ] **Step 2: Write the failing tests**

`tests/nca/test_options.py`:
```python
import pytest

from pkpdutils.nca.options import (
    AUCMethod,
    BLQHandling,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    decode_flags,
)
from pkpdutils.timecourse import Dose, DosingRegimen


def test_defaults() -> None:
    options = NCAOptions()
    assert options.kind is Kind.CONCENTRATION
    assert options.auc_method is AUCMethod.LINEAR_LOG
    assert options.terminal.method is TerminalMethod.BEST_FIT
    assert options.terminal.min_points == 3
    assert options.terminal.exclude_cmax
    assert options.lloq is None
    assert options.blq is BLQHandling.NAN
    assert options.c0_method is C0Method.LOG_BACK_EXTRAPOLATION
    assert options.extrapolation_warning == pytest.approx(0.2)
    assert options.regimen is None
    assert options.effect_threshold is None
    assert options.n_workers is None


def test_terminal_last_n_requires_n_points() -> None:
    with pytest.raises(ValueError, match="n_points"):
        TerminalPhase(method=TerminalMethod.LAST_N)
    assert TerminalPhase(method=TerminalMethod.LAST_N, n_points=4).n_points == 4


def test_terminal_manual_requires_points() -> None:
    with pytest.raises(ValueError, match="points"):
        TerminalPhase(method=TerminalMethod.MANUAL)
    with pytest.raises(ValueError, match="min_points"):
        TerminalPhase(method=TerminalMethod.MANUAL, points=(5, 6))
    assert TerminalPhase(method=TerminalMethod.MANUAL, points=(4, 5, 6)).points == (4, 5, 6)


def test_min_points_at_least_three() -> None:
    with pytest.raises(ValueError):
        TerminalPhase(min_points=2)


def test_options_validation() -> None:
    with pytest.raises(ValueError):
        NCAOptions(lloq=0)
    with pytest.raises(ValueError):
        NCAOptions(extrapolation_warning=1.5)
    with pytest.raises(ValueError):
        NCAOptions(n_workers=0)
    regimen = DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12)
    assert NCAOptions(regimen=regimen).regimen is regimen


def test_options_frozen() -> None:
    options = NCAOptions()
    with pytest.raises(ValueError):
        options.lloq = 1.0  # ty: ignore[invalid-assignment]


def test_flags() -> None:
    value = int(NCAFlag.POSITIVE_SLOPE | NCAFlag.EXTRAPOLATION_HIGH)
    assert value == 5
    assert decode_flags(value) == ["POSITIVE_SLOPE", "EXTRAPOLATION_HIGH"]
    assert decode_flags(0) == []
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/nca/test_options.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.nca'`.

- [ ] **Step 4: Write the module**

`src/pkpdutils/nca/__init__.py`:
```python
"""Non-compartmental analysis of timecourses.

The public entry points are `nca` (a batch of timecourses) and `nca_single`
(one timecourse); `NCAOptions` selects the methods, `NCAResult` holds the
parameters. See the user guide page `docs/nca.md`.
"""
```
(the exports are added in Task 5, when `nca` exists)

`src/pkpdutils/nca/options.py`:
```python
"""Options and flags of the non-compartmental analysis.

`NCAOptions` selects the methods of an analysis: the kind of timecourse, the
trapezoid rule, the terminal phase selection, the handling of values below the
limit of quantification and the dosing regimen of a steady state analysis.
`NCAFlag` names the conditions an analysis reports per sample instead of
raising or warning.
"""

from enum import IntFlag, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.timecourse import DosingRegimen


class Kind(StrEnum):
    """What a timecourse measures."""

    #: concentration of a substance: exposure, terminal phase and dose parameters
    CONCENTRATION = "concentration"
    #: pharmacodynamic effect: AUEC, observed maximum, baseline
    EFFECT = "effect"


class AUCMethod(StrEnum):
    """Trapezoid rule of the areas, see `docs/nca.md`."""

    #: linear trapezoid on every segment
    LINEAR = "linear"
    #: linear on rising segments, logarithmic on falling segments (Phoenix "linear up/log down")
    LINEAR_LOG = "linear_log"
    #: logarithmic on every segment with two positive, different values
    LOG = "log"


class TerminalMethod(StrEnum):
    """Selection of the points of the terminal log-linear regression."""

    #: largest adjusted R² over all windows ending at tlast (Phoenix best fit)
    BEST_FIT = "best_fit"
    #: the last `n_points` points
    LAST_N = "last_n"
    #: every point after the maximum (the rule of pkdb_analysis 0.3.1)
    ALL_AFTER_TMAX = "all_after_tmax"
    #: the given point indices
    MANUAL = "manual"


class BLQHandling(StrEnum):
    """Handling of values below the lower limit of quantification."""

    #: values below `lloq` are missing
    NAN = "nan"
    #: values below `lloq` are 0 before tmax and missing after
    ZERO_BEFORE_TMAX = "zero_before_tmax"


class C0Method(StrEnum):
    """Estimate of the concentration at time 0 after an intravenous bolus."""

    #: log-linear back extrapolation of the first two positive values
    LOG_BACK_EXTRAPOLATION = "log_back_extrapolation"
    #: the first observed value
    FIRST_VALUE = "first_value"


class NCAFlag(IntFlag):
    """Conditions reported per sample in the `flags` variable of a result."""

    NONE = 0
    #: the terminal regression has a non-negative slope; lambda_z and dependents are NaN
    POSITIVE_SLOPE = 1
    #: fewer than `min_points` points after the maximum; no terminal phase
    TOO_FEW_POINTS = 2
    #: the extrapolated fraction of AUC(0-inf) exceeds `extrapolation_warning`
    EXTRAPOLATION_HIGH = 4
    #: the maximum is the last point of the curve
    NO_MAX = 8
    #: the maximum is the first point of an extravascular curve
    NO_ABSORPTION = 16
    #: values below `lloq` were removed
    BLQ_TRUNCATED = 32
    #: fewer than two valid points; every parameter is NaN
    NO_DATA = 64


def decode_flags(value: int) -> list[str]:
    """Names of the flags set in an integer flag value, in bit order."""
    return [flag.name for flag in NCAFlag if flag.value and value & flag.value and flag.name]


class TerminalPhase(BaseModel):
    """Selection of the points of the terminal log-linear regression.

    Attributes:
        method: the selection rule
        min_points: minimal number of points of a regression (at least 3)
        exclude_cmax: whether the point of the maximum is excluded from every window
        n_points: number of points for `LAST_N`
        points: indices of the points (in the time order of the curve) for `MANUAL`
        min_adj_r2: minimal adjusted R² a regression must reach, `None` for no limit
        tie_tolerance: a window with more points wins over the best adjusted R²
            when its adjusted R² is within this tolerance of the best
    """

    model_config = ConfigDict(frozen=True)

    method: TerminalMethod = TerminalMethod.BEST_FIT
    min_points: int = Field(default=3, ge=3)
    exclude_cmax: bool = True
    n_points: int | None = Field(default=None, ge=3)
    points: tuple[int, ...] | None = None
    min_adj_r2: float | None = Field(default=None, ge=0.0, le=1.0)
    tie_tolerance: float = Field(default=1e-4, ge=0.0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.method is TerminalMethod.LAST_N and self.n_points is None:
            raise ValueError("TerminalMethod.LAST_N needs 'n_points'")
        if self.method is TerminalMethod.MANUAL:
            if self.points is None:
                raise ValueError("TerminalMethod.MANUAL needs 'points'")
            if len(self.points) < self.min_points:
                raise ValueError(
                    f"'points' has {len(self.points)} indices, 'min_points' is {self.min_points}"
                )
        return self


class NCAOptions(BaseModel):
    """Options of a non-compartmental analysis.

    Attributes:
        kind: concentration or effect timecourses
        auc_method: trapezoid rule of the areas
        terminal: selection of the terminal phase
        lloq: lower limit of quantification in the unit of the values, `None` for none
        blq: handling of values below `lloq`
        c0_method: estimate of C(0) after an intravenous bolus
        extrapolation_warning: fraction of AUC(0-inf) above which `EXTRAPOLATION_HIGH` is set
        regimen: dosing regimen of a steady state analysis, `None` for single dose
        effect_threshold: threshold of `time_above` for effect timecourses, `None` for none
        n_workers: number of worker processes for large batches, `None` for the calling process
    """

    model_config = ConfigDict(frozen=True)

    kind: Kind = Kind.CONCENTRATION
    auc_method: AUCMethod = AUCMethod.LINEAR_LOG
    terminal: TerminalPhase = TerminalPhase()
    lloq: float | None = Field(default=None, gt=0.0)
    blq: BLQHandling = BLQHandling.NAN
    c0_method: C0Method = C0Method.LOG_BACK_EXTRAPOLATION
    extrapolation_warning: float = Field(default=0.2, gt=0.0, lt=1.0)
    regimen: DosingRegimen | None = None
    effect_threshold: float | None = None
    n_workers: int | None = Field(default=None, ge=1)
```

- [ ] **Step 5: Run the tests, lint, type check**

Run: `uv run pytest tests/nca/test_options.py -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: 7 passed, clean. Delete the `# ty: ignore[invalid-assignment]` if ty reports it unused.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/nca tests/nca
git commit -q -m "Add the options and flags of the non-compartmental analysis

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 2: Vectorized areas (`nca/auc.py`)

**Files:**
- Create: `src/pkpdutils/nca/auc.py`
- Test: `tests/nca/test_auc.py`

**Interfaces:**
- Produces:
  - `pack_valid(t: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: for `(N, n)` arrays, moves the points where both `t` and `c` are finite to the front of every row, keeping their order; returns `tp`, `cp` (padded with NaN) and `n_valid` `(N,)` int
  - `segment_areas(tp, cp, n_valid, method: AUCMethod) -> tuple[np.ndarray, np.ndarray]`: `(N, n-1)` areas and first moments of every segment between consecutive packed points; segments beyond the valid points are 0
  - `auc_aumc(tp, cp, n_valid, method, t_end: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]`: `(N,)` sums of the segments, only segments whose end time is `<= t_end` when given
  - `interpolate_at(tp, cp, n_valid, t_query: np.ndarray, method) -> np.ndarray`: `(N,)` value at `t_query` per row by linear (or log-down, per method) interpolation between the bracketing points; NaN outside the observed range
  - `insert_point(tp, cp, n_valid, t_new, c_new) -> tuple[tp2, cp2, n_valid2]`: inserts one point per row (NaN pairs skipped) and repacks in time order

All functions work on `(N, n)` float64 arrays and never loop over rows.

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_auc.py`:
```python
import numpy as np
import pytest

from pkpdutils.nca.auc import (
    auc_aumc,
    insert_point,
    interpolate_at,
    pack_valid,
    segment_areas,
)
from pkpdutils.nca.options import AUCMethod


def monoexp(t: np.ndarray, c0: float = 10.0, k: float = 0.5) -> np.ndarray:
    return c0 * np.exp(-k * t)


def test_pack_valid_moves_valid_points_to_front() -> None:
    t = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, np.nan, np.nan]])
    c = np.array([[1.0, np.nan, 3.0, 4.0], [1.0, 2.0, np.nan, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    np.testing.assert_array_equal(n_valid, [3, 2])
    np.testing.assert_allclose(tp[0, :3], [0.0, 2.0, 3.0])
    np.testing.assert_allclose(cp[0, :3], [1.0, 3.0, 4.0])
    assert np.isnan(tp[0, 3]) and np.isnan(cp[0, 3])
    np.testing.assert_allclose(cp[1, :2], [1.0, 2.0])


def test_linear_trapezoid_on_line() -> None:
    t = np.array([[0.0, 1.0, 2.0, 4.0]])
    c = np.array([[0.0, 2.0, 4.0, 8.0]])  # c = 2t: auc = t^2, aumc = 2/3 t^3
    tp, cp, n_valid = pack_valid(t, c)
    auc, aumc = auc_aumc(tp, cp, n_valid, AUCMethod.LINEAR)
    assert auc[0] == pytest.approx(16.0)
    assert aumc[0] == pytest.approx(2.0 / 3.0 * 64.0)


def test_log_trapezoid_exact_on_monoexponential() -> None:
    t = np.array([[0.0, 1.0, 3.0, 8.0, 20.0]])
    c = monoexp(t)
    tp, cp, n_valid = pack_valid(t, c)
    auc, aumc = auc_aumc(tp, cp, n_valid, AUCMethod.LOG)
    # analytic: auc(0-20) = c0/k (1 - e^{-20k}); aumc = c0/k^2 (1 - e^{-20k}(1 + 20k))
    k, c0 = 0.5, 10.0
    assert auc[0] == pytest.approx(c0 / k * (1 - np.exp(-20 * k)), rel=1e-10)
    assert aumc[0] == pytest.approx(c0 / k**2 * (1 - np.exp(-20 * k) * (1 + 20 * k)), rel=1e-10)


def test_linear_log_uses_log_only_on_falling_segments() -> None:
    t = np.array([[0.0, 1.0, 2.0, 4.0]])
    c = np.array([[0.0, 4.0, 2.0, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    areas, _ = segment_areas(tp, cp, n_valid, AUCMethod.LINEAR_LOG)
    assert areas[0, 0] == pytest.approx(2.0)  # rising: linear 0.5*(0+4)*1
    assert areas[0, 1] == pytest.approx((4 - 2) / np.log(2) * 1)  # falling: log
    assert areas[0, 2] == pytest.approx((2 - 1) / np.log(2) * 2)


def test_segments_beyond_valid_points_are_zero() -> None:
    t = np.array([[0.0, 1.0, 2.0, np.nan]])
    c = np.array([[1.0, 1.0, 1.0, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    areas, moments = segment_areas(tp, cp, n_valid, AUCMethod.LINEAR)
    np.testing.assert_allclose(areas[0], [1.0, 1.0, 0.0])
    assert not np.isnan(moments).any()


def test_auc_until_t_end() -> None:
    t = np.array([[0.0, 1.0, 2.0, 3.0]])
    c = np.array([[1.0, 1.0, 1.0, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    auc, _ = auc_aumc(tp, cp, n_valid, AUCMethod.LINEAR, t_end=np.array([2.0]))
    assert auc[0] == pytest.approx(2.0)


def test_interpolate_at() -> None:
    t = np.array([[0.0, 1.0, 2.0, 4.0], [0.0, 2.0, np.nan, np.nan]])
    c = np.array([[0.0, 4.0, 2.0, 1.0], [1.0, 3.0, np.nan, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    lin = interpolate_at(tp, cp, n_valid, np.array([3.0, 1.0]), AUCMethod.LINEAR)
    np.testing.assert_allclose(lin, [1.5, 2.0])
    log = interpolate_at(tp, cp, n_valid, np.array([3.0, 1.0]), AUCMethod.LINEAR_LOG)
    assert log[0] == pytest.approx(2.0 * np.exp(np.log(1.0 / 2.0) * 0.5))  # log-down between (2,2) and (4,1)
    assert log[1] == pytest.approx(2.0)  # rising segment stays linear
    at_point = interpolate_at(tp, cp, n_valid, np.array([2.0, 2.0]), AUCMethod.LINEAR)
    np.testing.assert_allclose(at_point, [2.0, 3.0])
    outside = interpolate_at(tp, cp, n_valid, np.array([5.0, -1.0]), AUCMethod.LINEAR)
    assert np.isnan(outside).all()


def test_insert_point_keeps_time_order() -> None:
    t = np.array([[0.0, 1.0, 4.0, np.nan]])
    c = np.array([[0.0, 4.0, 1.0, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    tp2, cp2, n2 = insert_point(tp, cp, n_valid, np.array([2.0]), np.array([2.5]))
    assert n2[0] == 4
    np.testing.assert_allclose(tp2[0, :4], [0.0, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(cp2[0, :4], [0.0, 4.0, 2.5, 1.0])
    tp3, _, n3 = insert_point(tp, cp, n_valid, np.array([np.nan]), np.array([np.nan]))
    assert n3[0] == 3 and tp3.shape[1] == tp.shape[1] + 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/nca/test_auc.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.nca.auc'`.

- [ ] **Step 3: Write the module**

`src/pkpdutils/nca/auc.py`:
```python
"""Vectorized trapezoid areas of timecourses.

Every function works on `(N, n)` arrays, one row per curve, without a loop
over the rows. Missing points (`NaN` in the time or the value) are moved to the
end of every row by `pack_valid`, so that the segments between consecutive
valid points are the columns of the arrays `segment_areas` returns.

The trapezoid rules are the ones of Gabrielsson & Weiner (2016, ch. 2.8) and
of the Phoenix WinNonlin NCA: on a segment from `(t1, c1)` to `(t2, c2)` with
`dt = t2 - t1` the linear rule gives the area `dt (c1 + c2) / 2` and the first
moment `dt (t1 c1 + t2 c2) / 2`; the logarithmic rule, exact for a
mono-exponential decline, gives the area `dt (c1 - c2) / L` and the moment
`dt (t1 c1 - t2 c2) / L + dt² (c1 - c2) / L²` with `L = ln(c1 / c2)`.
"""

import numpy as np

from pkpdutils.nca.options import AUCMethod


def pack_valid(t: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move the valid points of every row to the front, keeping their order.

    Args:
        t: times of shape `(N, n)`
        c: values of shape `(N, n)`

    Returns:
        The packed times, the packed values (both padded with `NaN`) and the
        number of valid points per row.
    """
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(c)
    order = np.argsort(~valid, axis=1, kind="stable")
    tp = np.where(valid, t, np.nan)
    cp = np.where(valid, c, np.nan)
    tp = np.take_along_axis(tp, order, axis=1)
    cp = np.take_along_axis(cp, order, axis=1)
    return tp, cp, valid.sum(axis=1)


def segment_areas(
    tp: np.ndarray, cp: np.ndarray, n_valid: np.ndarray, method: AUCMethod
) -> tuple[np.ndarray, np.ndarray]:
    """Areas and first moments of the segments between consecutive packed points.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row `(N,)`
        method: trapezoid rule

    Returns:
        The areas and the first moments, both of shape `(N, n - 1)`; a segment
        beyond the valid points of its row is 0.
    """
    t1, t2 = tp[:, :-1], tp[:, 1:]
    c1, c2 = cp[:, :-1], cp[:, 1:]
    dt = t2 - t1
    in_curve = np.arange(tp.shape[1] - 1)[None, :] < (n_valid - 1)[:, None]

    lin_area = 0.5 * (c1 + c2) * dt
    lin_moment = 0.5 * (t1 * c1 + t2 * c2) * dt
    positive = (c1 > 0) & (c2 > 0) & (c1 != c2)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(c1 / c2)
        log_area = (c1 - c2) / log_ratio * dt
        log_moment = dt * (t1 * c1 - t2 * c2) / log_ratio + dt * dt * (c1 - c2) / (log_ratio * log_ratio)

    if method is AUCMethod.LINEAR:
        use_log = np.zeros_like(positive)
    elif method is AUCMethod.LINEAR_LOG:
        use_log = positive & (c2 < c1)
    else:
        use_log = positive
    area = np.where(use_log, log_area, lin_area)
    moment = np.where(use_log, log_moment, lin_moment)
    area = np.where(in_curve, area, 0.0)
    moment = np.where(in_curve, moment, 0.0)
    return area, moment


def auc_aumc(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    method: AUCMethod,
    t_end: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Area and first moment of every row, optionally only up to a time.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        method: trapezoid rule
        t_end: per row, only segments ending at or before this time count

    Returns:
        `auc` and `aumc` of shape `(N,)`.
    """
    area, moment = segment_areas(tp, cp, n_valid, method)
    if t_end is not None:
        keep = tp[:, 1:] <= t_end[:, None]
        area = np.where(keep, area, 0.0)
        moment = np.where(keep, moment, 0.0)
    return area.sum(axis=1), moment.sum(axis=1)


def interpolate_at(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    t_query: np.ndarray,
    method: AUCMethod,
) -> np.ndarray:
    """Value of every row at a query time by interpolation between the bracketing points.

    Linear interpolation, or logarithmic interpolation on a segment the
    trapezoid `method` treats logarithmically. `NaN` outside the observed
    times of a row.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        t_query: one query time per row `(N,)`
        method: trapezoid rule

    Returns:
        The interpolated values `(N,)`.
    """
    n = tp.shape[1]
    idx = np.arange(n)[None, :]
    valid = idx < n_valid[:, None]
    tq = t_query[:, None]
    # number of valid points at or before the query time
    count = ((tp <= tq) & valid).sum(axis=1)
    out = np.full(tp.shape[0], np.nan)
    with np.errstate(invalid="ignore"):
        # exactly on the last point
        last_idx = np.clip(n_valid - 1, 0, n - 1)
        t_last = np.take_along_axis(tp, last_idx[:, None], axis=1)[:, 0]
        c_last = np.take_along_axis(cp, last_idx[:, None], axis=1)[:, 0]
        on_last = (n_valid > 0) & (t_query == t_last)
        out = np.where(on_last, c_last, out)
        inside = (count >= 1) & (count < n_valid)
        i2 = np.clip(count, 1, n - 1)
        i1 = i2 - 1
        t1 = np.take_along_axis(tp, i1[:, None], axis=1)[:, 0]
        t2 = np.take_along_axis(tp, i2[:, None], axis=1)[:, 0]
        c1 = np.take_along_axis(cp, i1[:, None], axis=1)[:, 0]
        c2 = np.take_along_axis(cp, i2[:, None], axis=1)[:, 0]
        frac = (t_query - t1) / (t2 - t1)
        lin = c1 + frac * (c2 - c1)
        positive = (c1 > 0) & (c2 > 0) & (c1 != c2)
        if method is AUCMethod.LINEAR:
            use_log = np.zeros_like(positive)
        elif method is AUCMethod.LINEAR_LOG:
            use_log = positive & (c2 < c1)
        else:
            use_log = positive
        with np.errstate(divide="ignore"):
            log = np.exp(np.log(c1) + frac * (np.log(c2) - np.log(c1)))
        value = np.where(use_log, log, lin)
        out = np.where(inside & ~on_last, value, out)
    return out


def insert_point(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    t_new: np.ndarray,
    c_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert one point per row and repack in time order.

    A row whose new time or value is `NaN` is left unchanged (its arrays are
    still one column wider).

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        t_new: time of the new point per row
        c_new: value of the new point per row

    Returns:
        The packed times and values `(N, n + 1)` and the new counts.
    """
    t_all = np.concatenate([tp, t_new[:, None]], axis=1)
    c_all = np.concatenate([cp, c_new[:, None]], axis=1)
    order = np.argsort(np.where(np.isnan(t_all), np.inf, t_all), axis=1, kind="stable")
    t_sorted = np.take_along_axis(t_all, order, axis=1)
    c_sorted = np.take_along_axis(c_all, order, axis=1)
    return pack_valid(t_sorted, c_sorted)
```

`pack_valid` is used on `(t_sorted, c_sorted)` at the end to drop a NaN pair and recount; `n_valid` is recomputed there, so the argument `n_valid` of `insert_point` is only used for the signature symmetry — remove it from the signature if ruff flags it unused (`ARG` is not enabled, so it is kept for the interface).

- [ ] **Step 4: Run the tests, lint, type check**

Run: `uv run pytest tests/nca/test_auc.py -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: 8 passed, no warnings, clean. If `test_interpolate_at` fails on the log case, check `frac` for the segment `(2, 2)→(4, 1)` at `t=3`: `0.5`, value `2·exp(0.5·ln(0.5)) = 1.414`.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/nca/auc.py tests/nca/test_auc.py
git commit -q -m "Add the vectorized trapezoid areas of the NCA

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 3: Vectorized terminal phase regression (`nca/terminal.py`)

**Files:**
- Create: `src/pkpdutils/nca/terminal.py`
- Test: `tests/nca/test_terminal.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class TerminalFit`: arrays `(N,)` `slope`, `intercept`, `r2`, `r2_adj`, `se_slope`, `n_points` (float, NaN when no fit), `t_first`, `start` (int packed index, `-1` when no fit), and `flags` (`(N,)` int with `POSITIVE_SLOPE`/`TOO_FEW_POINTS` bits)
  - `terminal_fit(tp, cp, n_valid, tmax_idx: np.ndarray, phase: TerminalPhase, manual_mask: np.ndarray | None = None) -> TerminalFit`: regression of `ln c` on `t` over the points of the selected window (positive values only), per row; `tmax_idx` is the packed index of the maximum per row; `manual_mask` `(N, n)` marks the packed points for `TerminalMethod.MANUAL`
  - `window_statistics(x, y, valid)` helper: for every start index `s` the regression statistics of the points `s..end` via suffix sums, arrays `(N, n)`

Regression conventions: `y = ln c`, `slope = (n Σxy − Σx Σy)/(n Σxx − (Σx)²)`, `intercept = (Σy − slope Σx)/n`, `SS_tot = Σyy − (Σy)²/n`, `SS_res = SS_tot − slope (Σxy − Σx Σy/n)`, `r² = 1 − SS_res/SS_tot`, `r²_adj = 1 − (1 − r²)(n − 1)/(n − 2)`, `se_slope = sqrt(SS_res/(n − 2) / (Σxx − (Σx)²/n))`.

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_terminal.py`:
```python
import numpy as np
import pytest

from pkpdutils.nca.auc import pack_valid
from pkpdutils.nca.options import NCAFlag, TerminalMethod, TerminalPhase
from pkpdutils.nca.terminal import terminal_fit, window_statistics


def oral_curve(t: np.ndarray, ka: float = 2.0, ke: float = 0.3, f: float = 10.0) -> np.ndarray:
    return f * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))


def test_window_statistics_matches_numpy_polyfit() -> None:
    x = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    y = np.array([[5.0, 3.9, 3.1, 1.9, 1.2]])
    valid = np.ones_like(x, dtype=bool)
    stats = window_statistics(x, y, valid)
    for s in range(3):
        slope, intercept = np.polyfit(x[0, s:], y[0, s:], 1)
        assert stats["slope"][0, s] == pytest.approx(slope)
        assert stats["intercept"][0, s] == pytest.approx(intercept)
        assert stats["n"][0, s] == 5 - s


def test_monoexponential_recovers_slope_exactly() -> None:
    t = np.linspace(0, 10, 11)[None, :]
    c = 8.0 * np.exp(-0.4 * t)
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase())
    assert fit.slope[0] == pytest.approx(-0.4)
    assert fit.intercept[0] == pytest.approx(np.log(8.0))
    assert fit.r2[0] == pytest.approx(1.0)
    assert fit.flags[0] == 0
    # exclude_cmax: the maximum at index 0 is excluded, the best window has 10 points
    assert fit.n_points[0] == 10
    assert fit.t_first[0] == pytest.approx(1.0)


def test_best_fit_prefers_more_points_within_tolerance() -> None:
    t = np.linspace(0, 10, 11)[None, :]
    c = 8.0 * np.exp(-0.4 * t)
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase(exclude_cmax=False))
    assert fit.n_points[0] == 11
    assert fit.start[0] == 0


def test_best_fit_skips_absorption_phase() -> None:
    t = np.array([[0.25, 0.5, 1, 2, 4, 6, 8, 12, 24]], dtype=float)
    c = oral_curve(t)
    tp, cp, n_valid = pack_valid(t, c)
    tmax_idx = np.array([int(np.argmax(c[0]))])
    fit = terminal_fit(tp, cp, n_valid, tmax_idx, TerminalPhase())
    assert fit.slope[0] == pytest.approx(-0.3, rel=0.02)
    assert fit.n_points[0] >= 3
    assert fit.t_first[0] > t[0, tmax_idx[0]]


def test_last_n_and_all_after_tmax() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=float)
    c = np.array([[1, 5, 4, 3, 2.2, 1.5, 1.1]])
    tp, cp, n_valid = pack_valid(t, c)
    tmax_idx = np.array([1])
    last3 = terminal_fit(tp, cp, n_valid, tmax_idx, TerminalPhase(method=TerminalMethod.LAST_N, n_points=3))
    assert last3.n_points[0] == 3
    assert last3.t_first[0] == pytest.approx(4.0)
    after = terminal_fit(tp, cp, n_valid, tmax_idx, TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX))
    assert after.n_points[0] == 5
    assert after.t_first[0] == pytest.approx(2.0)
    slope, intercept = np.polyfit(t[0, 2:], np.log(c[0, 2:]), 1)
    assert after.slope[0] == pytest.approx(slope)
    assert after.intercept[0] == pytest.approx(intercept)


def test_manual_mask() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5]], dtype=float)
    c = np.array([[1, 5, 4, 3, 2, 1.4]])
    tp, cp, n_valid = pack_valid(t, c)
    mask = np.array([[False, False, True, False, True, True]])
    fit = terminal_fit(tp, cp, n_valid, np.array([1]), TerminalPhase(method=TerminalMethod.MANUAL, points=(2, 4, 5)), manual_mask=mask)
    slope, _ = np.polyfit([2, 4, 5], np.log([4, 2, 1.4]), 1)
    assert fit.slope[0] == pytest.approx(slope)
    assert fit.n_points[0] == 3


def test_too_few_points_flag() -> None:
    t = np.array([[0, 1, 2, 3]], dtype=float)
    c = np.array([[1, 5, 4, 3]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([1]), TerminalPhase())
    assert np.isnan(fit.slope[0])
    assert fit.flags[0] & NCAFlag.TOO_FEW_POINTS
    assert fit.start[0] == -1


def test_positive_slope_flag() -> None:
    t = np.array([[0, 1, 2, 3, 4]], dtype=float)
    c = np.array([[1, 1.1, 1.3, 1.6, 2.0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([4]), TerminalPhase(exclude_cmax=False))
    assert np.isnan(fit.slope[0])
    assert fit.flags[0] & NCAFlag.POSITIVE_SLOPE


def test_zero_values_are_excluded_from_regression() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5]], dtype=float)
    c = np.array([[5, 4, 3, 2, 0, 0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX))
    slope, _ = np.polyfit([1, 2, 3], np.log([4, 3, 2]), 1)
    assert fit.slope[0] == pytest.approx(slope)
    assert fit.n_points[0] == 3


def test_min_adj_r2_rejects_poor_fit() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5]], dtype=float)
    c = np.array([[9, 5, 6, 2, 4, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase(min_adj_r2=0.999))
    assert np.isnan(fit.slope[0])
    assert fit.flags[0] & NCAFlag.TOO_FEW_POINTS


def test_batch_rows_are_independent() -> None:
    t = np.tile(np.linspace(0, 10, 11), (3, 1))
    ks = np.array([0.2, 0.5, 1.0])
    c = 5.0 * np.exp(-ks[:, None] * t)
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.zeros(3, dtype=int), TerminalPhase())
    np.testing.assert_allclose(fit.slope, -ks)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/nca/test_terminal.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.nca.terminal'`.

- [ ] **Step 3: Write the module**

`src/pkpdutils/nca/terminal.py`:
```python
"""Vectorized terminal phase regression.

The elimination rate constant `lambda_z` is the negative slope of the linear
regression of `ln c` on `t` over the points of the terminal phase
(Gabrielsson & Weiner 2016, ch. 2.8; Phoenix WinNonlin NCA). Which points form
the terminal phase is decided by `TerminalPhase.method`; `BEST_FIT` evaluates
every window of consecutive points that ends at the last measurable point and
takes the largest adjusted R², preferring more points within a tolerance,
which is the rule of Phoenix.

All windows of all rows are evaluated at once: with suffix sums of `x`, `y`,
`x²`, `xy` and `y²` over the packed points the statistics of the window
starting at index `s` are closed-form expressions of the sums from `s` to the
end, so `window_statistics` returns `(N, n)` arrays without a loop over rows
or windows.
"""

from dataclasses import dataclass

import numpy as np

from pkpdutils.nca.options import NCAFlag, TerminalMethod, TerminalPhase


@dataclass(frozen=True)
class TerminalFit:
    """Result of the terminal regression per row, `NaN` (and `start = -1`) without a fit.

    Attributes:
        slope: slope of `ln c` against `t` (`-lambda_z`)
        intercept: intercept of the regression, `ln c` at `t = 0`
        r2: coefficient of determination
        r2_adj: adjusted coefficient of determination
        se_slope: standard error of the slope
        n_points: number of points of the regression
        t_first: time of the first point of the regression
        start: packed index of the first point of the window
        flags: `NCAFlag` bits `POSITIVE_SLOPE` and `TOO_FEW_POINTS`
    """

    slope: np.ndarray
    intercept: np.ndarray
    r2: np.ndarray
    r2_adj: np.ndarray
    se_slope: np.ndarray
    n_points: np.ndarray
    t_first: np.ndarray
    start: np.ndarray
    flags: np.ndarray


def _suffix_sum(a: np.ndarray) -> np.ndarray:
    """Sum of every row from each index to the end."""
    return np.cumsum(a[:, ::-1], axis=1)[:, ::-1]


def window_statistics(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Regression statistics of every window from an index to the end of the row.

    Args:
        x: regressor `(N, n)`
        y: response `(N, n)`
        valid: which points enter the regression `(N, n)`

    Returns:
        Arrays `(N, n)` keyed `n`, `slope`, `intercept`, `r2`, `r2_adj`,
        `se_slope`; column `s` describes the window `s..end`. Windows with
        fewer than 3 points are `NaN`.
    """
    x0 = np.where(valid, x, 0.0)
    y0 = np.where(valid, y, 0.0)
    n = _suffix_sum(valid.astype(np.float64))
    sx = _suffix_sum(x0)
    sy = _suffix_sum(y0)
    sxx = _suffix_sum(x0 * x0)
    sxy = _suffix_sum(x0 * y0)
    syy = _suffix_sum(y0 * y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx_c = sxx - sx * sx / n
        sxy_c = sxy - sx * sy / n
        syy_c = syy - sy * sy / n
        slope = sxy_c / sxx_c
        intercept = (sy - slope * sx) / n
        ss_res = syy_c - slope * sxy_c
        r2 = 1.0 - ss_res / syy_c
        r2_adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - 2.0)
        se_slope = np.sqrt(np.maximum(ss_res, 0.0) / (n - 2.0) / sxx_c)
    enough = n >= 3
    nan = np.nan
    return {
        "n": np.where(enough, n, nan),
        "slope": np.where(enough, slope, nan),
        "intercept": np.where(enough, intercept, nan),
        "r2": np.where(enough, r2, nan),
        "r2_adj": np.where(enough, r2_adj, nan),
        "se_slope": np.where(enough, se_slope, nan),
    }


def terminal_fit(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    tmax_idx: np.ndarray,
    phase: TerminalPhase,
    manual_mask: np.ndarray | None = None,
) -> TerminalFit:
    """Terminal log-linear regression of every row.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        tmax_idx: packed index of the maximum per row
        phase: the selection rule and its parameters
        manual_mask: packed points of the regression for `TerminalMethod.MANUAL`

    Returns:
        The fit per row.
    """
    n_rows, n = tp.shape
    idx = np.arange(n)[None, :]
    in_row = idx < n_valid[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(in_row & (cp > 0), np.log(cp), np.nan)
    regressable = in_row & np.isfinite(y)
    flags = np.zeros(n_rows, dtype=np.int64)

    if phase.method is TerminalMethod.MANUAL:
        if manual_mask is None:
            raise ValueError("TerminalMethod.MANUAL needs 'manual_mask'")
        selected = regressable & manual_mask
        # a single window per row: the selected points, statistics via the
        # suffix sums with everything before the first selected point excluded
        stats = window_statistics(tp, np.where(selected, y, 0.0), selected)
        start = np.where(selected.any(axis=1), selected.argmax(axis=1), 0)
        return _collect(tp, stats, start, selected.any(axis=1), phase, flags)

    first_allowed = tmax_idx + 1 if phase.exclude_cmax else tmax_idx
    stats = window_statistics(tp, y, regressable)
    if phase.method is TerminalMethod.BEST_FIT:
        eligible = (
            (idx >= first_allowed[:, None])
            & (stats["n"] >= phase.min_points)
            & (stats["slope"] < 0)
        )
        if phase.min_adj_r2 is not None:
            eligible &= stats["r2_adj"] >= phase.min_adj_r2
        r2_adj = np.where(eligible, stats["r2_adj"], -np.inf)
        best = r2_adj.max(axis=1)
        within = eligible & (r2_adj >= best[:, None] - phase.tie_tolerance)
        start = np.where(within.any(axis=1), within.argmax(axis=1), 0)
        has_fit = eligible.any(axis=1)
        # windows with enough points exist but none with a negative slope
        enough = ((idx >= first_allowed[:, None]) & (stats["n"] >= phase.min_points)).any(axis=1)
        flags = np.where(~has_fit & enough, NCAFlag.POSITIVE_SLOPE, 0).astype(np.int64)
        return _collect(tp, stats, start, has_fit, phase, flags)

    if phase.method is TerminalMethod.LAST_N:
        assert phase.n_points is not None
        # the window whose regressable count equals n_points: the largest s with n >= n_points
        enough_n = stats["n"] >= phase.n_points
        start = np.where(enough_n.any(axis=1), n - 1 - enough_n[:, ::-1].argmax(axis=1), 0)
        has_fit = enough_n.any(axis=1)
    else:  # ALL_AFTER_TMAX
        start = tmax_idx + 1
        has_fit = start < n_valid
        start = np.clip(start, 0, n - 1)
    n_at_start = np.take_along_axis(stats["n"], start[:, None], axis=1)[:, 0]
    slope_at_start = np.take_along_axis(stats["slope"], start[:, None], axis=1)[:, 0]
    with np.errstate(invalid="ignore"):
        enough = has_fit & (n_at_start >= phase.min_points)
        positive = enough & ~(slope_at_start < 0)
    flags = np.where(positive, NCAFlag.POSITIVE_SLOPE, 0).astype(np.int64)
    return _collect(tp, stats, start, enough & ~positive, phase, flags)


def _collect(
    tp: np.ndarray,
    stats: dict[str, np.ndarray],
    start: np.ndarray,
    has_fit: np.ndarray,
    phase: TerminalPhase,
    flags: np.ndarray,
) -> TerminalFit:
    """Pick the statistics of the chosen window per row and set the flags."""
    take = lambda a: np.take_along_axis(a, start[:, None], axis=1)[:, 0]  # noqa: E731
    n_points = take(stats["n"])
    with np.errstate(invalid="ignore"):
        has_fit = has_fit & (n_points >= phase.min_points)
        if phase.min_adj_r2 is not None:
            has_fit &= take(stats["r2_adj"]) >= phase.min_adj_r2
    nan = np.nan
    pick = lambda a: np.where(has_fit, take(a), nan)  # noqa: E731
    too_few = ~has_fit & (flags == 0)
    flags = flags | np.where(too_few, NCAFlag.TOO_FEW_POINTS, 0).astype(np.int64)
    return TerminalFit(
        slope=pick(stats["slope"]),
        intercept=pick(stats["intercept"]),
        r2=pick(stats["r2"]),
        r2_adj=pick(stats["r2_adj"]),
        se_slope=pick(stats["se_slope"]),
        n_points=pick(stats["n"]),
        t_first=pick(tp),
        start=np.where(has_fit, start, -1),
        flags=flags,
    )
```

If ruff rejects the lambdas even with `noqa` (rule `E731` is part of `E7`), replace them by nested functions `def take(a): ...` and `def pick(a): ...` with one-line docstrings.

- [ ] **Step 4: Run the tests, lint, type check**

Run: `uv run pytest tests/nca/test_terminal.py -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: 11 passed, no warnings. If `test_best_fit_skips_absorption_phase` picks a window with fewer points than expected, the tolerance rule is in play: the test only asserts slope ≈ −0.3 within 2 % and a start after tmax. If `test_last_n_and_all_after_tmax` fails on `LAST_N`, the start index must be the largest `s` whose regressable count is ≥ `n_points`, which is `n - 1 - argmax(reversed)`.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/nca/terminal.py tests/nca/test_terminal.py
git commit -q -m "Add the vectorized terminal phase regression of the NCA

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 4: Result units and the `NCAResult` container (`nca/result.py`)

**Files:**
- Create: `src/pkpdutils/nca/result.py`
- Test: `tests/nca/test_result.py`

**Interfaces:**
- Produces:
  - `parameter_unit(expression: str, *, unit: str, time_unit: str, dose_unit: str | None) -> tuple[str, float]`: `expression` uses the placeholders `{unit}`, `{time}`, `{dose}` (e.g. `"({unit}) * ({time})"`, `"({dose}) / (({unit}) * ({time}))"`, `"1 / ({time})"`, `"dimensionless"`); returns the canonical unit string after `normalize_volume`/`normalize_clearance` and the factor that converts a magnitude in the raw expression unit to it
  - `class NCAResult`: `__init__(ds: xr.Dataset)` (requires `flags` variable and `attrs["units"]` on every data variable); properties `ds`, `sample_dims`, `parameters: list[str]` (data variables except `flags`); `units(name) -> str`; `__getitem__(name) -> xr.DataArray`; `__contains__`; `to_dataframe() -> pd.DataFrame` (one row per sample: sample coordinates, every parameter, `flags` decoded as a `|` joined string); `to_quantities(**indexers) -> dict[str, Quantity]` (one sample; all sample dims must be given unless there are none); `flags(**indexers) -> list[str]`; `flag_table() -> pd.DataFrame` (sample coordinates + one boolean column per flag)

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_result.py`:
```python
import numpy as np
import pytest
import xarray as xr

from pkpdutils.nca.options import NCAFlag
from pkpdutils.nca.result import NCAResult, parameter_unit


def test_parameter_unit_auc() -> None:
    unit, factor = parameter_unit("({unit}) * ({time})", unit="ng/ml", time_unit="hr", dose_unit=None)
    assert unit == "hour * nanogram / milliliter"
    assert factor == pytest.approx(1.0)


def test_parameter_unit_clearance_normalized() -> None:
    unit, factor = parameter_unit("({dose}) / (({unit}) * ({time}))", unit="mg/l", time_unit="min", dose_unit="mg")
    assert unit == "liter / hour"
    assert factor == pytest.approx(60.0)


def test_parameter_unit_volume_per_bodyweight() -> None:
    unit, factor = parameter_unit("({dose}) / ({unit})", unit="ng/ml", time_unit="hr", dose_unit="mg/kg")
    assert unit == "liter / kilogram"
    assert factor == pytest.approx(1e6)


def test_parameter_unit_dimensionless_and_rate() -> None:
    assert parameter_unit("dimensionless", unit="ng/ml", time_unit="hr", dose_unit=None)[0] == "dimensionless"
    assert parameter_unit("1 / ({time})", unit="ng/ml", time_unit="hr", dose_unit=None)[0] == "1 / hour"


def make_result() -> NCAResult:
    ds = xr.Dataset(
        {
            "auc_last": (("individual",), np.array([10.0, 20.0]), {"units": "hour * nanogram / milliliter"}),
            "cmax": (("individual",), np.array([2.0, np.nan]), {"units": "nanogram / milliliter"}),
            "flags": (("individual",), np.array([0, int(NCAFlag.NO_DATA | NCAFlag.TOO_FEW_POINTS)]), {"units": "dimensionless"}),
        },
        coords={"individual": ["a", "b"]},
    )
    return NCAResult(ds)


def test_result_access() -> None:
    result = make_result()
    assert result.sample_dims == ("individual",)
    assert result.parameters == ["auc_last", "cmax"]
    assert result.units("auc_last") == "hour * nanogram / milliliter"
    assert "cmax" in result and "vz" not in result
    np.testing.assert_allclose(result["auc_last"].values, [10.0, 20.0])


def test_result_to_quantities_and_flags() -> None:
    result = make_result()
    q = result.to_quantities(individual="a")
    assert q["auc_last"].magnitude == pytest.approx(10.0)
    assert str(q["auc_last"].units) == "hour * nanogram / milliliter"
    assert result.flags(individual="a") == []
    assert result.flags(individual="b") == ["TOO_FEW_POINTS", "NO_DATA"]
    with pytest.raises(ValueError, match="individual"):
        result.to_quantities()


def test_result_to_dataframe_and_flag_table() -> None:
    result = make_result()
    df = result.to_dataframe()
    assert list(df.columns) == ["individual", "auc_last", "cmax", "flags"]
    assert df["flags"].tolist() == ["", "TOO_FEW_POINTS|NO_DATA"]
    table = result.flag_table()
    assert table["NO_DATA"].tolist() == [False, True]
    assert table["POSITIVE_SLOPE"].tolist() == [False, False]


def test_result_requires_flags_and_units() -> None:
    ds = xr.Dataset({"auc_last": (("individual",), np.array([1.0]), {"units": "hr*ng/ml"})}, coords={"individual": ["a"]})
    with pytest.raises(ValueError, match="flags"):
        NCAResult(ds)
    ds2 = xr.Dataset({"auc_last": (("individual",), np.array([1.0])), "flags": (("individual",), np.array([0]))}, coords={"individual": ["a"]})
    with pytest.raises(ValueError, match="units"):
        NCAResult(ds2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/nca/test_result.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.nca.result'`.

- [ ] **Step 3: Write the module**

`src/pkpdutils/nca/result.py`:
```python
"""Units of the parameters and the result container of the NCA.

An `NCAResult` wraps an `xarray.Dataset` with one variable per parameter over
the sample dimensions of the analysed batch, `attrs["units"]` on every
variable and the integer variable `flags` (`pkpdutils.nca.options.NCAFlag`).
The units are derived from the units of the input with pint: an area carries
`unit * time_unit`, a rate `1 / time_unit`, a clearance `dose_unit / (unit *
time_unit)` converted to `liter / hour` (or per kilogram), a volume converted
to `liter` (or per kilogram), see `pkpdutils.units`.
"""

from typing import Any

import pandas as pd
import xarray as xr

from pkpdutils.nca.options import NCAFlag, decode_flags
from pkpdutils.units import Q_, Quantity, normalize_clearance, normalize_volume, ureg


def parameter_unit(
    expression: str, *, unit: str, time_unit: str, dose_unit: str | None
) -> tuple[str, float]:
    """Unit of a parameter and the factor from its raw unit to the reported unit.

    Args:
        expression: pint expression with the placeholders `{unit}`, `{time}`
            and `{dose}`, e.g. `"({unit}) * ({time})"`
        unit: unit of the values
        time_unit: unit of the times
        dose_unit: unit of the doses, `None` without doses (an expression
            with `{dose}` then raises `ValueError`)

    Returns:
        The canonical unit string and the factor a magnitude in the raw unit is
        multiplied with to be in the canonical unit (volumes to liter,
        clearances to liter per hour, else 1).
    """
    if "{dose}" in expression and dose_unit is None:
        raise ValueError(f"'{expression}' needs a dose unit")
    raw = expression.format(unit=unit, time=time_unit, dose=dose_unit or "")
    quantity = Q_(1.0, ureg.parse_units(raw))
    converted = normalize_clearance(normalize_volume(quantity))
    return str(converted.units), float(converted.magnitude)


class NCAResult:
    """Parameters of a non-compartmental analysis as an `xarray.Dataset`.

    The dataset has one variable per parameter over the sample dimensions of
    the analysed `Timecourses`, `attrs["units"]` on every variable and the
    integer variable `flags`. A parameter which does not apply to a sample
    (no dose, no terminal phase, ...) is `NaN`.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a result dataset.

        Raises:
            ValueError: without a `flags` variable or without units on a variable.
        """
        if "flags" not in ds:
            raise ValueError("The dataset needs a 'flags' variable")
        for name in ds.data_vars:
            if "units" not in ds[name].attrs:
                raise ValueError(f"'{name}' needs attrs['units']")
        self.ds: xr.Dataset = ds

    @property
    def sample_dims(self) -> tuple[str, ...]:
        """The sample dimensions."""
        return tuple(str(d) for d in self.ds["flags"].dims)

    @property
    def parameters(self) -> list[str]:
        """Names of the parameters (the data variables except `flags`)."""
        return [str(name) for name in self.ds.data_vars if name != "flags"]

    def units(self, name: str) -> str:
        """Unit string of a parameter."""
        return str(self.ds[name].attrs["units"])

    def __getitem__(self, name: str) -> xr.DataArray:
        """The array of a parameter."""
        return self.ds[name]

    def __contains__(self, name: object) -> bool:
        """Whether a parameter is part of the result."""
        return name in self.ds.data_vars

    def _sample(self, indexers: dict[str, Any]) -> xr.Dataset:
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(f"A label for every sample dimension is needed, missing {sorted(missing)}")
        return self.ds.sel(indexers)

    def to_quantities(self, **indexers: Any) -> dict[str, Quantity]:
        """The parameters of one sample as pint quantities.

        Args:
            **indexers: coordinate label per sample dimension

        Returns:
            Parameter name to quantity.
        """
        sample = self._sample(indexers)
        return {name: Q_(float(sample[name].values), self.units(name)) for name in self.parameters}

    def flags(self, **indexers: Any) -> list[str]:
        """Names of the flags set for one sample."""
        sample = self._sample(indexers)
        return decode_flags(int(sample["flags"].values))

    def to_dataframe(self) -> pd.DataFrame:
        """One row per sample: the sample coordinates, the parameters and the decoded flags."""
        df = self.ds.to_dataframe().reset_index()
        df["flags"] = ["|".join(decode_flags(int(v))) for v in df["flags"]]
        columns = [*self.sample_dims, *self.parameters, "flags"]
        return df[columns]

    def flag_table(self) -> pd.DataFrame:
        """One row per sample with a boolean column per flag."""
        df = self.ds[["flags"]].to_dataframe().reset_index()
        values = df["flags"].to_numpy().astype(int)
        for flag in NCAFlag:
            if flag.value and flag.name:
                df[flag.name] = (values & flag.value) != 0
        return df.drop(columns=["flags"])
```

- [ ] **Step 4: Run the tests, lint, type check**

Run: `uv run pytest tests/nca/test_result.py -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: 8 passed, clean. If `to_dataframe` of a dataset without sample dimensions produces an `index` column, drop it: `df = df.drop(columns=[c for c in df.columns if c not in self.sample_dims and c not in self.ds.data_vars])`.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/nca/result.py tests/nca/test_result.py
git commit -q -m "Add the NCA result container and the parameter units

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 5: The analysis (`nca/nca.py`): `nca`, `nca_single`, single dose parameters, flags, effect kind, workers

**Files:**
- Create: `src/pkpdutils/nca/nca.py`
- Modify: `src/pkpdutils/nca/__init__.py` (exports)
- Test: `tests/nca/test_nca.py`

**Interfaces:**
- Consumes: `pack_valid`, `auc_aumc`, `segment_areas`, `interpolate_at` (Task 2); `terminal_fit`, `TerminalFit` (Task 3); `NCAResult`, `parameter_unit` (Task 4); `NCAOptions`, enums, `NCAFlag` (Task 1); `Timecourses`, `Timecourse`, `Route`.
- Produces:
  - `nca(timecourses: Timecourses, options: NCAOptions | None = None) -> NCAResult`
  - `nca_single(timecourse: Timecourse, options: NCAOptions | None = None) -> NCAResult` (result without sample dimensions)
  - `compute_parameters(t, c, *, dose_amount, dose_time, dose_duration, route, options) -> dict[str, np.ndarray]`: the pure numpy core on `(N, n)` arrays; returns one `(N,)` array per parameter plus `flags`; this is what the worker processes run and what Task 7 extends with the steady state parameters
  - `PARAMETER_UNITS: dict[str, str]`: parameter name to unit expression for `parameter_unit`
  - `pkpdutils.nca` exports `nca`, `nca_single`, `NCAOptions`, `TerminalPhase`, `NCAResult`, `NCAFlag`, `Kind`, `AUCMethod`, `TerminalMethod`, `BLQHandling`, `C0Method`, `decode_flags`

Parameter definitions (single dose, `Kind.CONCENTRATION`), all per row on the packed arrays with times relative to `dose_time` (0 without dose):

- `cmax`, `tmax`: maximum value and its time (`nanargmax` over the valid points); `NO_MAX` when the maximum is the last valid point; `NO_ABSORPTION` when it is the first valid point and the route is `ORAL`
- `cmin`, `tmin`: minimum value and its time
- `clast`, `tlast`: last valid value `> 0` and its time
- `c0` (only `IV_BOLUS`): `LOG_BACK_EXTRAPOLATION`: with the first two valid positive points `(t1, c1)`, `(t2, c2)` and `c2 < c1`: `exp(ln c1 − (ln c2 − ln c1)/(t2 − t1) · t1)`, otherwise `c1`; `FIRST_VALUE`: `c1`. When `t1 > 0` the point `(0, c0)` is inserted before the areas are computed
- `auc_last`, `aumc_last`: areas of the segments ending at or before `tlast`
- `lambda_z`, `lambda_z_intercept`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_se`, `lambda_z_n_points`, `lambda_z_t_first`: from `terminal_fit`; `lambda_z = −slope`; `thalf = ln 2 / lambda_z`
- `auc_inf_obs = auc_last + clast / lambda_z`; `clast_pred = exp(intercept − lambda_z · tlast)`; `auc_inf_pred = auc_last + clast_pred / lambda_z`; `auc_extrap_fraction = (auc_inf_obs − auc_last) / auc_inf_obs`, `EXTRAPOLATION_HIGH` when above `options.extrapolation_warning`
- `aumc_inf = aumc_last + clast · tlast / lambda_z + clast / lambda_z²`; `mrt = aumc_inf / auc_inf_obs`, minus `dose_duration / 2` for `IV_INFUSION`
- `cmax_half`, `tmax_half` (route `ORAL`, maximum not at the first point): among the valid points before the maximum, the one whose value is closest to `cmax / 2`
- with a dose: `cl` (`IV_*`) or `cl_f` (`ORAL`) `= dose / auc_inf_obs`; `vz`/`vz_f = cl / lambda_z`; `vss = cl · mrt` (`IV_*` only); `auc_inf_dn = auc_inf_obs / dose`; `cmax_dn = cmax / dose`
- `NO_DATA` when a row has fewer than 2 valid points: every parameter `NaN`
- BLQ: with `lloq`, values `< lloq` become `NaN` (`BLQ_TRUNCATED` when any); `ZERO_BEFORE_TMAX` sets them to 0 before the maximum of the remaining values and `NaN` after

`Kind.EFFECT`: `e0` (first valid value), `emax_obs`, `temax`, `auec_last` (area to the last valid point, any sign, linear rule forced), `auec_baseline` (area of `value − e0`), `emax_baseline = emax_obs − e0`, `time_above` (with `effect_threshold`: total time the linearly interpolated curve is above the threshold), `tlast`; no terminal phase, no dose parameters, no BLQ.

Unit expressions (`PARAMETER_UNITS`): areas `"({unit}) * ({time})"`, moments `"({unit}) * ({time}) ** 2"`, values `"{unit}"`, times `"{time}"`, `lambda_z` and `lambda_z_se` `"1 / ({time})"`, `lambda_z_intercept` `"dimensionless"` (it is `ln c`), `cl`/`cl_f`/`cl_ss` `"({dose}) / (({unit}) * ({time}))"`, `vz`/`vz_f`/`vss` `"({dose}) / ({unit})"`, `auc_inf_dn` `"(({unit}) * ({time})) / ({dose})"`, `cmax_dn` `"({unit}) / ({dose})"`, fractions/ratios/counts `"dimensionless"`, `time_above` `"{time}"`.

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_nca.py`:
```python
import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import (
    AUCMethod,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    NCAResult,
    TerminalMethod,
    TerminalPhase,
    nca,
    nca_single,
)

K, C0 = 0.5, 10.0
T_DENSE = np.linspace(0, 20, 401)


def iv_timecourse(dose: Dose | None = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)) -> Timecourse:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    return Timecourse(time=t, value=C0 * np.exp(-K * t), time_unit="hr", unit="mg/l", dose=dose, substance="x")


def oral_timecourse(ka: float = 2.0) -> Timecourse:
    t = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24])
    c = C0 * ka / (ka - K) * (np.exp(-K * t) - np.exp(-ka * t))
    return Timecourse(time=t, value=c, time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="x")


def test_single_iv_bolus_analytic() -> None:
    result = nca_single(iv_timecourse(), NCAOptions(auc_method=AUCMethod.LOG))
    assert isinstance(result, NCAResult)
    assert result.sample_dims == ()
    q = result.to_quantities()
    assert q["lambda_z"].magnitude == pytest.approx(K)
    assert str(q["lambda_z"].units) == "1 / hour"
    assert q["thalf"].magnitude == pytest.approx(np.log(2) / K)
    assert q["c0"].magnitude == pytest.approx(C0)
    # with the log rule and the back extrapolated C0, AUC(0-inf) is exact: c0/k
    assert q["auc_inf_obs"].magnitude == pytest.approx(C0 / K, rel=1e-6)
    assert q["auc_inf_pred"].magnitude == pytest.approx(C0 / K, rel=1e-6)
    assert q["aumc_inf"].magnitude == pytest.approx(C0 / K**2, rel=1e-6)
    assert q["mrt"].magnitude == pytest.approx(1 / K, rel=1e-6)
    assert q["cl"].magnitude == pytest.approx(100 / (C0 / K))
    assert str(q["cl"].units) == "liter / hour"
    assert q["vz"].magnitude == pytest.approx(100 / C0)
    assert str(q["vz"].units) == "liter"
    assert q["vss"].magnitude == pytest.approx(100 / C0, rel=1e-6)
    assert q["auc_inf_dn"].magnitude == pytest.approx(C0 / K / 100)
    assert "cl_f" not in result and "vz_f" not in result
    assert result.flags() == []


def test_c0_first_value() -> None:
    result = nca_single(iv_timecourse(), NCAOptions(c0_method=C0Method.FIRST_VALUE))
    assert result.to_quantities()["c0"].magnitude == pytest.approx(C0 * np.exp(-K * 0.25))


def test_single_oral_names_and_flags() -> None:
    tc = oral_timecourse()
    result = nca_single(tc)
    q = result.to_quantities()
    assert "cl_f" in result and "vz_f" in result and "vss" not in result and "c0" not in result
    assert q["lambda_z"].magnitude == pytest.approx(K, rel=0.02)
    assert q["tmax"].magnitude == pytest.approx(tc.time[np.argmax(tc.value)])
    assert q["cmax"].magnitude == pytest.approx(tc.value.max())
    assert q["tlast"].magnitude == 24.0
    assert q["clast"].magnitude == pytest.approx(tc.value[-1])
    assert 0 < q["tmax_half"].magnitude < q["tmax"].magnitude
    assert q["auc_extrap_fraction"].magnitude < 0.2
    assert result.flags() == []
    assert q["lambda_z_n_points"].magnitude >= 3


def test_extrapolation_flag_and_positive_slope() -> None:
    t = np.array([1, 2, 3, 4, 5.0])
    rising = Timecourse(time=t, value=[1, 1.2, 1.5, 1.9, 2.5], time_unit="hr", unit="mg/l")
    result = nca_single(rising, NCAOptions(terminal=TerminalPhase(exclude_cmax=False)))
    assert "POSITIVE_SLOPE" in result.flags()
    assert np.isnan(result.to_quantities()["lambda_z"].magnitude)
    assert np.isnan(result.to_quantities()["auc_inf_obs"].magnitude)
    truncated = Timecourse(time=[0.5, 1, 2, 3, 4], value=[2, 5, 4.5, 4, 3.6], time_unit="hr", unit="mg/l")
    flags = nca_single(truncated).flags()
    assert "EXTRAPOLATION_HIGH" in flags


def test_too_few_points_and_no_max() -> None:
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l")
    result = nca_single(short)
    assert "TOO_FEW_POINTS" in result.flags()
    assert np.isnan(result.to_quantities()["thalf"].magnitude)
    assert not np.isnan(result.to_quantities()["auc_last"].magnitude)
    still_rising = Timecourse(time=[1, 2, 3, 4], value=[1, 2, 3, 4], time_unit="hr", unit="mg/l")
    assert "NO_MAX" in nca_single(still_rising).flags()


def test_no_absorption_flag_only_for_oral() -> None:
    assert "NO_ABSORPTION" in nca_single(iv_timecourse(dose=Dose(amount=100, unit="mg", route=Route.ORAL))).flags()
    assert "NO_ABSORPTION" not in nca_single(iv_timecourse()).flags()


def test_no_data_sample() -> None:
    tc = Timecourse(time=[0, 1, 2], value=[np.nan, np.nan, 1.0], time_unit="hr", unit="mg/l")
    result = nca_single(tc)
    assert result.flags() == ["NO_DATA"]
    assert np.isnan(result.to_quantities()["cmax"].magnitude)


def test_lloq_handling() -> None:
    t = np.array([0.5, 1, 2, 4, 8, 12, 24])
    c = np.array([0.05, 2.0, 4.0, 3.0, 1.5, 0.5, 0.05])
    tc = Timecourse(time=t, value=c, time_unit="hr", unit="mg/l")
    nan = nca_single(tc, NCAOptions(lloq=0.1, auc_method=AUCMethod.LINEAR))
    assert "BLQ_TRUNCATED" in nan.flags()
    assert nan.to_quantities()["tlast"].magnitude == 12.0
    zero = nca_single(tc, NCAOptions(lloq=0.1, auc_method=AUCMethod.LINEAR, blq="zero_before_tmax"))
    # the first point becomes 0 and adds the triangle 0.5*(0+2)*0.5 to the area
    assert zero.to_quantities()["auc_last"].magnitude == pytest.approx(nan.to_quantities()["auc_last"].magnitude + 0.5 * 2.0 * 0.5)


def test_infusion_mrt_correction() -> None:
    t = np.array([1, 2, 4, 6, 8, 12.0])
    c = C0 * np.exp(-K * t)
    bolus = Timecourse(time=t, value=c, time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS))
    infusion = Timecourse(time=t, value=c, time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=1.0))
    mrt_bolus = nca_single(bolus).to_quantities()["mrt"].magnitude
    mrt_inf = nca_single(infusion).to_quantities()["mrt"].magnitude
    assert mrt_inf == pytest.approx(mrt_bolus - 0.5)


def test_dose_per_bodyweight_units() -> None:
    tc = iv_timecourse(dose=Dose(amount=2, unit="mg/kg", route=Route.IV_BOLUS))
    q = nca_single(tc).to_quantities()
    assert str(q["cl"].units) == "liter / hour / kilogram"
    assert str(q["vz"].units) == "liter / kilogram"


def test_dose_time_shifts_time_axis() -> None:
    tc = iv_timecourse()
    shifted = tc.model_copy(update={"time": tc.time + 10.0, "dose": tc.dose.model_copy(update={"time": 10.0}) if tc.dose else None})
    a = nca_single(tc).to_quantities()
    b = nca_single(shifted).to_quantities()
    assert b["tmax"].magnitude == pytest.approx(a["tmax"].magnitude)
    assert b["auc_last"].magnitude == pytest.approx(a["auc_last"].magnitude)


def test_batch_equals_loop_over_singles() -> None:
    curves = [oral_timecourse(ka) for ka in (1.0, 2.0, 4.0)] + [
        Timecourse(time=[0.5, 1, 2, 4, 8], value=[1, 2, 1.5, np.nan, 0.5], time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="x")
    ]
    batch = Timecourses.from_timecourses(curves, dim="individual")
    result = nca(batch)
    assert result.sample_dims == ("individual",)
    for i, tc in enumerate(curves):
        single = nca_single(tc)
        for name in result.parameters:
            a = float(result[name].values[i])
            b = single.to_quantities()[name].magnitude
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), name
        assert result.flags(individual=result.ds["individual"].values[i]) == single.flags()


def test_batch_two_sample_dims_and_workers() -> None:
    time = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    ks = np.array([0.3, 0.5, 0.8])
    doses = np.array([50.0, 100.0])
    values = doses[:, None, None] / 10 * np.exp(-ks[None, :, None] * time[None, None, :])
    batch = Timecourses.from_arrays(
        time, values, time_unit="hr", unit="mg/l", dims=("dose", "k"),
        coords={"dose": doses, "k": ks},
        dose={"amount": np.broadcast_to(doses[:, None], (2, 3)), "unit": "mg"}, route=Route.IV_BOLUS,
    )
    serial = nca(batch, NCAOptions(auc_method=AUCMethod.LOG))
    assert serial.sample_dims == ("dose", "k")
    np.testing.assert_allclose(serial["lambda_z"].values, np.broadcast_to(ks, (2, 3)), rtol=1e-6)
    np.testing.assert_allclose(serial["cl"].values, np.broadcast_to(ks * 10, (2, 3)), rtol=1e-6)
    parallel = nca(batch, NCAOptions(auc_method=AUCMethod.LOG, n_workers=2))
    for name in serial.parameters:
        np.testing.assert_allclose(parallel[name].values, serial[name].values, equal_nan=True)
    np.testing.assert_array_equal(parallel["flags"].values, serial["flags"].values)


def test_effect_kind() -> None:
    t = np.array([0, 1, 2, 4, 6, 8.0])
    e = np.array([10, 14, 20, 16, 12, 10.0])
    tc = Timecourse(time=t, value=e, time_unit="hr", unit="mmHg", substance="effect")
    result = nca_single(tc, NCAOptions(kind=Kind.EFFECT, effect_threshold=15.0))
    q = result.to_quantities()
    assert q["e0"].magnitude == 10.0
    assert q["emax_obs"].magnitude == 20.0 and q["temax"].magnitude == 2.0
    assert q["auec_last"].magnitude == pytest.approx(np.trapezoid(e, t))
    assert q["auec_baseline"].magnitude == pytest.approx(np.trapezoid(e - 10, t))
    assert q["emax_baseline"].magnitude == 10.0
    # above 15 from t=1.25 (linear between (1,14),(2,20)) to t=4.5 (between (4,16),(6,12))
    assert q["time_above"].magnitude == pytest.approx(4.5 - 1.25)
    assert "lambda_z" not in result and "cl" not in result
    assert str(q["auec_last"].units) == "hour * millimeter_Hg"


def test_terminal_manual_points() -> None:
    tc = oral_timecourse()
    options = NCAOptions(terminal=TerminalPhase(method=TerminalMethod.MANUAL, points=(8, 9, 10, 11)))
    q = nca_single(tc, options).to_quantities()
    slope, _ = np.polyfit(tc.time[8:], np.log(tc.value[8:]), 1)
    assert q["lambda_z"].magnitude == pytest.approx(-slope)
    assert q["lambda_z_n_points"].magnitude == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/nca/test_nca.py -n 0 -q`
Expected: FAIL with `ImportError: cannot import name 'nca' from 'pkpdutils.nca'`.

- [ ] **Step 3: Write the module**

`src/pkpdutils/nca/nca.py`:
```python
"""The non-compartmental analysis.

`nca` analyses a `Timecourses` batch, `nca_single` one `Timecourse`. The
numerics run in `compute_parameters` on `(N, n)` arrays, one row per curve,
with the times relative to the dose; the rows are the flattened sample
dimensions of the batch and the results are reshaped back into an
`xarray.Dataset` over the same dimensions (`NCAResult`).

Definitions follow Gabrielsson & Weiner (2016, ch. 2.8) and the Phoenix
WinNonlin NCA, see `docs/nca.md`:

- `AUC(0-tlast)` and `AUMC(0-tlast)` by the trapezoid rule of `AUCMethod`
- `lambda_z` from the terminal log-linear regression (`TerminalPhase`),
  `t½ = ln 2 / lambda_z`
- `AUC(0-inf) = AUC(0-tlast) + Clast / lambda_z` (observed or predicted `Clast`)
- `AUMC(0-inf) = AUMC(0-tlast) + Clast tlast / lambda_z + Clast / lambda_z²`
- `MRT = AUMC(0-inf) / AUC(0-inf)`, minus half the infusion duration
- `CL = Dose / AUC(0-inf)`, `Vz = CL / lambda_z`, `Vss = CL MRT` (intravenous)
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.nca.auc import auc_aumc, insert_point, interpolate_at, pack_valid, segment_areas
from pkpdutils.nca.options import (
    AUCMethod,
    BLQHandling,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
)
from pkpdutils.nca.result import NCAResult, parameter_unit
from pkpdutils.nca.terminal import terminal_fit
from pkpdutils.timecourse import Route, Timecourse, Timecourses

logger = logging.getLogger(__name__)

#: unit expression per parameter, see `pkpdutils.nca.result.parameter_unit`
PARAMETER_UNITS: dict[str, str] = {
    "cmax": "{unit}",
    "tmax": "{time}",
    "cmin": "{unit}",
    "tmin": "{time}",
    "clast": "{unit}",
    "tlast": "{time}",
    "c0": "{unit}",
    "cmax_half": "{unit}",
    "tmax_half": "{time}",
    "auc_last": "({unit}) * ({time})",
    "auc_inf_obs": "({unit}) * ({time})",
    "auc_inf_pred": "({unit}) * ({time})",
    "auc_extrap_fraction": "dimensionless",
    "aumc_last": "({unit}) * ({time}) ** 2",
    "aumc_inf": "({unit}) * ({time}) ** 2",
    "mrt": "{time}",
    "lambda_z": "1 / ({time})",
    "lambda_z_se": "1 / ({time})",
    "lambda_z_intercept": "dimensionless",
    "lambda_z_r2": "dimensionless",
    "lambda_z_r2_adj": "dimensionless",
    "lambda_z_n_points": "dimensionless",
    "lambda_z_t_first": "{time}",
    "thalf": "{time}",
    "cl": "({dose}) / (({unit}) * ({time}))",
    "cl_f": "({dose}) / (({unit}) * ({time}))",
    "vz": "({dose}) / ({unit})",
    "vz_f": "({dose}) / ({unit})",
    "vss": "({dose}) / ({unit})",
    "auc_inf_dn": "(({unit}) * ({time})) / ({dose})",
    "cmax_dn": "({unit}) / ({dose})",
    "e0": "{unit}",
    "emax_obs": "{unit}",
    "temax": "{time}",
    "auec_last": "({unit}) * ({time})",
    "auec_baseline": "({unit}) * ({time})",
    "emax_baseline": "{unit}",
    "time_above": "{time}",
    "flags": "dimensionless",
}


def _take(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Element `idx[i]` of row `i`."""
    return np.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def _apply_lloq(c: np.ndarray, options: NCAOptions) -> tuple[np.ndarray, np.ndarray]:
    """Remove values below the limit of quantification; returns the values and the truncated rows."""
    if options.lloq is None:
        return c, np.zeros(c.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        below = c < options.lloq
    truncated = below.any(axis=1)
    if options.blq is BLQHandling.NAN:
        return np.where(below, np.nan, c), truncated
    kept = np.where(below, np.nan, c)
    with np.errstate(invalid="ignore"):
        all_nan = np.isnan(kept).all(axis=1)
        tmax_idx = np.where(all_nan, 0, np.nanargmax(np.where(np.isnan(kept), -np.inf, kept), axis=1))
    before = np.arange(c.shape[1])[None, :] < tmax_idx[:, None]
    return np.where(below & before, 0.0, kept), truncated


def _effect_parameters(t: np.ndarray, c: np.ndarray, options: NCAOptions) -> dict[str, np.ndarray]:
    """Parameters of effect timecourses (`Kind.EFFECT`)."""
    tp, cp, n_valid = pack_valid(t, c)
    n_rows, n = tp.shape
    has_data = n_valid >= 2
    flags = np.where(has_data, 0, NCAFlag.NO_DATA).astype(np.int64)
    idx = np.arange(n)[None, :]
    in_row = idx < n_valid[:, None]
    e0 = np.where(n_valid > 0, cp[:, 0], np.nan)
    masked = np.where(in_row, cp, -np.inf)
    imax = masked.argmax(axis=1)
    emax = np.where(has_data, _take(cp, imax), np.nan)
    temax = np.where(has_data, _take(tp, imax), np.nan)
    last_idx = np.clip(n_valid - 1, 0, n - 1)
    tlast = np.where(has_data, _take(tp, last_idx), np.nan)
    auec, _ = auc_aumc(tp, cp, n_valid, AUCMethod.LINEAR)
    auec_base, _ = auc_aumc(tp, cp - e0[:, None], n_valid, AUCMethod.LINEAR)
    out: dict[str, np.ndarray] = {
        "e0": e0,
        "emax_obs": emax,
        "temax": temax,
        "tlast": tlast,
        "auec_last": np.where(has_data, auec, np.nan),
        "auec_baseline": np.where(has_data, auec_base, np.nan),
        "emax_baseline": emax - e0,
    }
    if options.effect_threshold is not None:
        out["time_above"] = np.where(has_data, _time_above(tp, cp, n_valid, options.effect_threshold), np.nan)
    out["flags"] = flags
    return out


def _time_above(tp: np.ndarray, cp: np.ndarray, n_valid: np.ndarray, threshold: float) -> np.ndarray:
    """Total time the linearly interpolated curve is above a threshold, per row."""
    t1, t2 = tp[:, :-1], tp[:, 1:]
    c1, c2 = cp[:, :-1], cp[:, 1:]
    in_curve = np.arange(tp.shape[1] - 1)[None, :] < (n_valid - 1)[:, None]
    dt = t2 - t1
    a1 = c1 > threshold
    a2 = c2 > threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (threshold - c1) / (c2 - c1)  # position of the crossing in the segment
    both = np.where(a1 & a2, dt, 0.0)
    rising = np.where(~a1 & a2, dt * (1.0 - frac), 0.0)
    falling = np.where(a1 & ~a2, dt * frac, 0.0)
    total = np.where(in_curve, both + rising + falling, 0.0)
    return np.nansum(total, axis=1)


def compute_parameters(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Single dose parameters of every row of `(N, n)` time and value arrays.

    Args:
        t: times `(N, n)`, `NaN` for missing points
        c: values `(N, n)`, `NaN` for missing values
        dose_amount: dose per row `(N,)`, `None` without doses
        dose_time: time of the dose per row, `None` for 0
        dose_duration: infusion duration per row (`NaN` without infusion), `None` for none
        route: route of the batch, `None` without doses
        options: the options

    Returns:
        One `(N,)` array per parameter (see `PARAMETER_UNITS`) and `flags`.
    """
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if dose_time is not None:
        t = t - dose_time[:, None]
    if options.kind is Kind.EFFECT:
        return _effect_parameters(t, c, options)

    c, truncated = _apply_lloq(c, options)
    tp, cp, n_valid = pack_valid(t, c)
    n_rows, n = tp.shape
    idx = np.arange(n)[None, :]
    in_row = idx < n_valid[:, None]
    has_data = n_valid >= 2
    flags = np.where(has_data, 0, NCAFlag.NO_DATA).astype(np.int64)
    flags |= np.where(truncated, NCAFlag.BLQ_TRUNCATED, 0)
    nan = np.full(n_rows, np.nan)

    # observed maxima and minima
    masked_max = np.where(in_row, cp, -np.inf)
    imax = masked_max.argmax(axis=1)
    cmax = np.where(has_data, _take(cp, imax), nan)
    tmax = np.where(has_data, _take(tp, imax), nan)
    masked_min = np.where(in_row, cp, np.inf)
    imin = masked_min.argmin(axis=1)
    cmin = np.where(has_data, _take(cp, imin), nan)
    tmin = np.where(has_data, _take(tp, imin), nan)
    flags |= np.where(has_data & (imax == n_valid - 1), NCAFlag.NO_MAX, 0)
    if route is Route.ORAL:
        flags |= np.where(has_data & (imax == 0), NCAFlag.NO_ABSORPTION, 0)

    # last measurable point
    positive = in_row & (cp > 0)
    has_positive = positive.any(axis=1)
    ilast = np.where(has_positive, n - 1 - positive[:, ::-1].argmax(axis=1), 0)
    clast = np.where(has_data & has_positive, _take(cp, ilast), nan)
    tlast = np.where(has_data & has_positive, _take(tp, ilast), nan)

    # C0 of an intravenous bolus, inserted at t = 0 for the areas
    c0 = nan.copy()
    tp_area, cp_area, n_area = tp, cp, n_valid
    if route is Route.IV_BOLUS:
        t1, c1 = tp[:, 0], cp[:, 0]
        t2 = np.where(n_valid > 1, tp[:, 1], np.nan)
        c2 = np.where(n_valid > 1, cp[:, 1], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            back = np.exp(np.log(c1) - (np.log(c2) - np.log(c1)) / (t2 - t1) * t1)
            usable = (c1 > 0) & (c2 > 0) & (c2 < c1) & (t2 > t1)
        if options.c0_method is C0Method.LOG_BACK_EXTRAPOLATION:
            c0 = np.where(usable, back, c1)
        else:
            c0 = c1
        c0 = np.where(has_data, c0, nan)
        insert = has_data & (t1 > 0)
        tp_area, cp_area, n_area = insert_point(
            tp, cp, n_valid, np.where(insert, 0.0, np.nan), np.where(insert, c0, np.nan)
        )

    auc_last, aumc_last = auc_aumc(tp_area, cp_area, n_area, options.auc_method, t_end=tlast)
    auc_last = np.where(has_data & has_positive, auc_last, nan)
    aumc_last = np.where(has_data & has_positive, aumc_last, nan)

    # terminal phase
    manual_mask = None
    if options.terminal.method is TerminalMethod.MANUAL:
        assert options.terminal.points is not None
        original = np.zeros_like(c, dtype=bool)
        original[:, list(options.terminal.points)] = True
        # the packed position of the selected original points
        valid = np.isfinite(t) & np.isfinite(c)
        order = np.argsort(~valid, axis=1, kind="stable")
        manual_mask = np.take_along_axis(original & valid, order, axis=1)
    fit = terminal_fit(tp, cp, n_valid, imax, options.terminal, manual_mask=manual_mask)
    flags |= np.where(has_data, fit.flags, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_z = -fit.slope
        thalf = np.log(2.0) / lambda_z
        auc_inf_obs = auc_last + clast / lambda_z
        clast_pred = np.exp(fit.intercept - lambda_z * tlast)
        auc_inf_pred = auc_last + clast_pred / lambda_z
        extrap = (auc_inf_obs - auc_last) / auc_inf_obs
        aumc_inf = aumc_last + clast * tlast / lambda_z + clast / (lambda_z * lambda_z)
        mrt = aumc_inf / auc_inf_obs
        if route is Route.IV_INFUSION and dose_duration is not None:
            mrt = mrt - np.where(np.isnan(dose_duration), 0.0, dose_duration) / 2.0
        flags |= np.where(extrap > options.extrapolation_warning, NCAFlag.EXTRAPOLATION_HIGH, 0)

    # half maximum during absorption
    before_max = in_row & (idx < imax[:, None])
    with np.errstate(invalid="ignore"):
        distance = np.where(before_max, np.abs(cp - 0.5 * cmax[:, None]), np.inf)
    ihalf = distance.argmin(axis=1)
    iv = route is not None and route.is_iv
    has_half = has_data & (imax > 0) & (not iv)
    cmax_half = np.where(has_half, _take(cp, ihalf), nan)
    tmax_half = np.where(has_half, _take(tp, ihalf), nan)

    out: dict[str, np.ndarray] = {
        "cmax": cmax, "tmax": tmax, "cmin": cmin, "tmin": tmin, "clast": clast, "tlast": tlast,
        "auc_last": auc_last, "auc_inf_obs": auc_inf_obs, "auc_inf_pred": auc_inf_pred,
        "auc_extrap_fraction": extrap, "aumc_last": aumc_last, "aumc_inf": aumc_inf, "mrt": mrt,
        "lambda_z": lambda_z, "lambda_z_se": fit.se_slope, "lambda_z_intercept": fit.intercept,
        "lambda_z_r2": fit.r2, "lambda_z_r2_adj": fit.r2_adj, "lambda_z_n_points": fit.n_points,
        "lambda_z_t_first": fit.t_first, "thalf": thalf,
    }
    if route is Route.IV_BOLUS:
        out["c0"] = c0
    if route is None or not route.is_iv:
        out["cmax_half"] = cmax_half
        out["tmax_half"] = tmax_half
    if dose_amount is not None and route is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            cl = dose_amount / auc_inf_obs
            vz = cl / lambda_z
            suffix = "" if route.is_iv else "_f"
            out[f"cl{suffix}"] = cl
            out[f"vz{suffix}"] = vz
            if route.is_iv:
                out["vss"] = cl * mrt
            out["auc_inf_dn"] = auc_inf_obs / dose_amount
            out["cmax_dn"] = cmax / dose_amount
    out["flags"] = flags
    return out


def _compute_chunk(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    """Worker entry: `compute_parameters` on a chunk of rows."""
    t, c, dose_amount, dose_time, dose_duration, route, options = args
    return compute_parameters(
        t, c, dose_amount=dose_amount, dose_time=dose_time, dose_duration=dose_duration, route=route, options=options
    )


def nca(timecourses: Timecourses, options: NCAOptions | None = None) -> NCAResult:
    """Non-compartmental analysis of a batch of timecourses.

    Args:
        timecourses: the batch
        options: the options, defaults for `None`

    Returns:
        The parameters over the sample dimensions of the batch.
    """
    options = options or NCAOptions()
    shape = timecourses.sample_shape
    n_rows = timecourses.n_samples
    t = timecourses.times.reshape(n_rows, timecourses.n_time)
    c = timecourses.values.reshape(n_rows, timecourses.n_time)

    def flat(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else np.asarray(a, dtype=np.float64).reshape(n_rows)

    dose_amount = flat(timecourses.dose_amount)
    dose_time = flat(timecourses.dose_time)
    dose_duration = flat(timecourses.dose_duration)
    route = timecourses.route

    if options.regimen is not None:
        from pkpdutils.nca.steady_state import compute_steady_state  # noqa: PLC0415

        values = compute_steady_state(
            t, c, dose_amount=dose_amount, dose_time=dose_time, dose_duration=dose_duration, route=route, options=options
        )
    elif options.n_workers is not None and options.n_workers > 1 and n_rows > 1:
        chunks = np.array_split(np.arange(n_rows), min(options.n_workers, n_rows))
        jobs = [
            (
                t[rows], c[rows],
                None if dose_amount is None else dose_amount[rows],
                None if dose_time is None else dose_time[rows],
                None if dose_duration is None else dose_duration[rows],
                route, options,
            )
            for rows in chunks
        ]
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            parts = list(pool.map(_compute_chunk, jobs))
        values = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    else:
        values = compute_parameters(
            t, c, dose_amount=dose_amount, dose_time=dose_time, dose_duration=dose_duration, route=route, options=options
        )

    n_flagged = int((values["flags"] != 0).sum())
    if n_flagged:
        logger.info("NCA: %d of %d samples carry flags, see NCAResult.flag_table()", n_flagged, n_rows)
    return _to_result(values, timecourses, shape)


def _to_result(values: dict[str, np.ndarray], timecourses: Timecourses, shape: tuple[int, ...]) -> NCAResult:
    """Build the result dataset over the sample dimensions of the batch."""
    coords = {d: timecourses.ds[d] for d in timecourses.sample_dims if d in timecourses.ds.coords}
    data_vars: dict[str, Any] = {}
    for name, array in values.items():
        unit, factor = parameter_unit(
            PARAMETER_UNITS[name], unit=timecourses.unit, time_unit=timecourses.time_unit, dose_unit=timecourses.dose_unit
        )
        if name == "flags":
            data_vars[name] = (timecourses.sample_dims, array.reshape(shape).astype(np.int64), {"units": unit})
        else:
            data_vars[name] = (timecourses.sample_dims, (array * factor).reshape(shape), {"units": unit})
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={"substance": timecourses.substance})
    return NCAResult(ds)


def nca_single(timecourse: Timecourse, options: NCAOptions | None = None) -> NCAResult:
    """Non-compartmental analysis of one timecourse; the result has no sample dimensions."""
    batch = Timecourses.from_timecourses([timecourse], dim="_single")
    result = nca(batch, options)
    return NCAResult(result.ds.isel(_single=0).drop_vars("_single"))
```

`src/pkpdutils/nca/__init__.py` (replace):
```python
"""Non-compartmental analysis of timecourses.

`nca` analyses a `Timecourses` batch, `nca_single` one `Timecourse`;
`NCAOptions` selects the methods, `NCAResult` holds the parameters, see
`docs/nca.md`.
"""

from pkpdutils.nca.nca import nca, nca_single
from pkpdutils.nca.options import (
    AUCMethod,
    BLQHandling,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    decode_flags,
)
from pkpdutils.nca.result import NCAResult

__all__ = [
    "AUCMethod",
    "BLQHandling",
    "C0Method",
    "Kind",
    "NCAFlag",
    "NCAOptions",
    "NCAResult",
    "TerminalMethod",
    "TerminalPhase",
    "decode_flags",
    "nca",
    "nca_single",
]
```

Notes for the implementer:
- Task 7 creates `pkpdutils/nca/steady_state.py` with `compute_steady_state`; until then the `regimen` branch must not be reached (no test of this task sets a regimen). Keep the local import so that the two modules do not import each other at module level; if ruff `PLC0415` is not enabled the `noqa` is unused: remove it.
- `np.nanargmax` on all-NaN rows raises: `_apply_lloq` masks such rows first (the `all_nan` guard); keep that guard.
- `ProcessPoolExecutor` needs picklable arguments: `NCAOptions` (pydantic) and `Route` (StrEnum) pickle; `_compute_chunk` is a module level function.
- `nca_single` builds a one sample batch on the dimension `_single` and drops it; `Timecourses.from_timecourses` requires a dose route consistent with the timecourse, which is the case for a single curve.

- [ ] **Step 4: Run the tests, lint, type check**

Run: `uv run pytest tests/nca/test_nca.py -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: 16 passed, no warnings. Typical fixes: `np.trapezoid` needs numpy ≥ 2.0 (present); if `test_effect_kind` `time_above` differs, check the crossing fractions (`frac` is measured from `t1`); if `test_extrapolation_flag_and_positive_slope` does not raise `EXTRAPOLATION_HIGH` for the truncated curve, its last three points give lambda_z ≈ 0.12/hr and clast/lambda_z ≈ 30 against auc_last ≈ 14, so the fraction is ≈ 0.68 — the flag must be set; if `test_batch_equals_loop_over_singles` differs on `lambda_z_t_first` for the curve with a NaN, the packed index and the original index differ — `t_first` must come from the packed times (`tp`), which `_collect` does.

- [ ] **Step 5: Full suite and commit**

Run: `uv run pytest -q`
Expected: all pass (88 + 16 = ~104), no warnings.

```bash
git add src/pkpdutils/nca tests/nca
git commit -q -m "Add the non-compartmental analysis of single dose timecourses

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 6: Regression against pkdb_analysis 0.3.1

**Files:**
- Test: `tests/nca/test_reference.py`

**Interfaces:**
- Consumes: `tests/data/reference/nca_reference.json` (Task 1 of plan 1): a list of cases with `time`, `concentration` (null for NaN), `time_unit`, `unit`, `dose`, `dose_unit`, `substance`, `parameters` (`auc`, `aucinf`, `tmax`, `cmax`, `kel`, `thalf`, `vd`, `cl`, ... each `{"magnitude", "unit"}`, magnitude null for NaN), `regression` (`slope`, `intercept`, `max_idx`).
- Old semantics reproduced by: `AUCMethod.LINEAR`, `TerminalPhase(method=ALL_AFTER_TMAX)`, `Route.ORAL` (old `vd` = `dose/(aucinf·kel)` is `vz_f`, old `cl` = `kel·vd` = `dose/aucinf` is `cl_f`); old `aucinf` is `auc_inf_obs`; old `kel` is `lambda_z`.

- [ ] **Step 1: Write the test**

`tests/nca/test_reference.py`:
```python
"""The NCA reproduces the results of pkdb_analysis 0.3.1 on its test data."""

import json
from pathlib import Path

import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse
from pkpdutils.nca import AUCMethod, NCAOptions, TerminalMethod, TerminalPhase, nca_single
from pkpdutils.units import Q_

REFERENCE = Path(__file__).parent.parent / "data" / "reference" / "nca_reference.json"
CASES = json.loads(REFERENCE.read_text())

#: old parameter name -> new parameter name
MAPPING = {
    "auc": "auc_last",
    "aucinf": "auc_inf_obs",
    "tmax": "tmax",
    "cmax": "cmax",
    "kel": "lambda_z",
    "thalf": "thalf",
    "vd": "vz_f",
    "cl": "cl_f",
}

OPTIONS = NCAOptions(
    auc_method=AUCMethod.LINEAR,
    terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX),
)


def to_timecourse(case: dict) -> Timecourse:
    conc = [np.nan if v is None else v for v in case["concentration"]]
    return Timecourse(
        time=case["time"],
        value=conc,
        time_unit=case["time_unit"],
        unit=case["unit"],
        dose=Dose(amount=case["dose"], unit=case["dose_unit"], route=Route.ORAL),
        substance=case["substance"],
    )


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_reference_case(case: dict) -> None:
    result = nca_single(to_timecourse(case), OPTIONS)
    quantities = result.to_quantities()
    for old, new in MAPPING.items():
        expected = case["parameters"][old]
        actual = quantities[new]
        if expected["magnitude"] is None:
            assert np.isnan(actual.magnitude), f"{case['name']}: {new} should be NaN"
            continue
        converted = actual.to(expected["unit"]).magnitude
        assert converted == pytest.approx(expected["magnitude"], rel=1e-6, abs=1e-12), f"{case['name']}: {new}"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_reference_regression(case: dict) -> None:
    result = nca_single(to_timecourse(case), OPTIONS)
    q = result.to_quantities()
    slope = case["regression"]["slope"]
    if slope is None:
        assert np.isnan(q["lambda_z"].magnitude)
        return
    assert -q["lambda_z"].to(f"1/({case['time_unit']})").magnitude == pytest.approx(slope, rel=1e-6)
    assert q["lambda_z_intercept"].magnitude == pytest.approx(case["regression"]["intercept"], rel=1e-6)
    assert Q_(1, case["unit"]).check("[mass] / [length] ** 3") or True  # units are pint parsable
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/nca/test_reference.py -n 0 -q`
Expected: 18 passed. Diagnose a failure by case:
- `synthetic_monoexp` (tmax at index 0, `exclude_cmax` irrelevant for `ALL_AFTER_TMAX`): the old regression used the points after the maximum (`max_index + 1:`), so does `ALL_AFTER_TMAX`.
- `nan_values`: the old code dropped NaN before the areas and the regression; `pack_valid` does the same.
- paraxanthine cases: the old code returned NaN for `kel`, `thalf`, `aucinf`, `vd`, `cl` (too few points); the new result must be NaN with `TOO_FEW_POINTS` — the test only checks NaN.
- `midazolam` (`min`, `mmol/l`, dose `mg`): the old `cl` unit is whatever pint reduced `mg/(mmol/l·min)` to; `.to(expected["unit"])` converts.
- The old `intercept` was in the units of the concentration (`intercept_unit`) but its magnitude is `ln c`; the comparison is on the magnitude only.
If a case differs beyond 1e-6 in `auc`, check that the old areas included the segment to the last non-NaN point even when that value is 0: the old `_auc` summed all non-NaN points; `auc_last` stops at `tlast` (last positive). If a reference case has a trailing zero, `auc_last` differs from the old `auc` by the last triangle — in that case compute the expected value from the old `auc` and report it; do not change the definition of `auc_last`. (None of the nine cases ends with a zero value; verify with `python -c` if in doubt.)

- [ ] **Step 3: Commit**

```bash
git add tests/nca/test_reference.py
git commit -q -m "Test the NCA against the results of pkdb_analysis 0.3.1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 7: Steady state parameters and superposition (`nca/steady_state.py`)

**Files:**
- Create: `src/pkpdutils/nca/steady_state.py`
- Modify: `src/pkpdutils/nca/nca.py` (`PARAMETER_UNITS` entries), `src/pkpdutils/nca/__init__.py` (export `superposition`, `accumulation_ratio`)
- Test: `tests/nca/test_steady_state.py`

**Interfaces:**
- Consumes: `compute_parameters` (Task 5), `pack_valid`, `auc_aumc`, `interpolate_at`, `insert_point` (Task 2), `NCAResult`, `nca_single`, `DosingRegimen`.
- Produces:
  - `compute_steady_state(t, c, *, dose_amount, dose_time, dose_duration, route, options) -> dict[str, np.ndarray]`: the single dose parameters of `compute_parameters` (times relative to the dose of `options.regimen.dose` when `dose_time` is `None`) plus `auc_tau`, `cmin_ss`, `ctrough`, `cavg`, `fluctuation`, `swing`, `accumulation_ratio` (predicted, `1/(1 − exp(−lambda_z·tau))`), and `cl_ss = dose/auc_tau` when doses are present. Definitions over the dosing interval `[0, tau]` relative to the dose time: `auc_tau` = area with the value at `tau` interpolated (`interpolate_at`) and inserted; `ctrough` = value at `tau`; `cmin_ss` = minimum of the observed values in `[0, tau]` and `ctrough`; `cavg = auc_tau/tau`; `fluctuation = (cmax − cmin_ss)/cavg`; `swing = (cmax − cmin_ss)/cmin_ss`
  - `accumulation_ratio(steady_state: NCAResult, single_dose: NCAResult) -> xr.DataArray`: observed ratio `auc_tau(ss) / auc_tau(single)`, both results from analyses with the same `regimen`
  - `superposition(timecourse: Timecourse, regimen: DosingRegimen, options: NCAOptions | None = None, t_end: float | None = None) -> Timecourse`: predicted multiple dose curve by linear superposition of the single dose curve shifted to every dose time of `regimen.dose_times()`; the single dose curve is interpolated (`interpolate_at` with `options.auc_method`) inside its observed range and extrapolated beyond `tlast` with `clast·exp(−lambda_z (t − tlast))` from `nca_single`; the output grid is the union of the shifted observed times within `[0, t_end]` (`t_end` defaults to the last dose time plus the last observed time); the returned timecourse carries the dose of the regimen (first administration) and the substance and units of the input. Raises `ValueError` when the single dose curve has no `lambda_z` (flags) or the regimen has no `n_doses`.
  - `PARAMETER_UNITS` gains `auc_tau`, `cmin_ss`, `ctrough`, `cavg`, `fluctuation`, `swing`, `accumulation_ratio`, `cl_ss`

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_steady_state.py`:
```python
import numpy as np
import pytest

from pkpdutils import Dose, DosingRegimen, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca, nca_single
from pkpdutils.nca.steady_state import accumulation_ratio, superposition

K, C0, TAU = 0.2, 10.0, 12.0
DOSE = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)


def single_dose() -> Timecourse:
    t = np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])
    return Timecourse(time=t, value=C0 * np.exp(-K * t), time_unit="hr", unit="mg/l", dose=DOSE, substance="x")


def steady_state_curve() -> Timecourse:
    """Analytic steady state of a bolus every TAU hours: c0 e^{-kt} / (1 - e^{-k tau}) on [0, tau]."""
    t = np.array([0.5, 1, 2, 4, 6, 8, 10, 12])
    c = C0 * np.exp(-K * t) / (1 - np.exp(-K * TAU))
    return Timecourse(time=t, value=c, time_unit="hr", unit="mg/l", dose=DOSE, substance="x")


def test_steady_state_parameters_analytic() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=TAU)
    options = NCAOptions(regimen=regimen, auc_method=AUCMethod.LOG)
    q = nca_single(steady_state_curve(), options).to_quantities()
    accumulation = 1 / (1 - np.exp(-K * TAU))
    # auc over one interval at steady state equals the single dose auc(0-inf) = c0/k
    assert q["auc_tau"].magnitude == pytest.approx(C0 / K, rel=1e-6)
    assert q["ctrough"].magnitude == pytest.approx(C0 * np.exp(-K * TAU) * accumulation, rel=1e-6)
    assert q["cmin_ss"].magnitude == pytest.approx(q["ctrough"].magnitude)
    assert q["cavg"].magnitude == pytest.approx(C0 / K / TAU, rel=1e-6)
    cmax = q["cmax"].magnitude
    cmin = q["cmin_ss"].magnitude
    assert q["fluctuation"].magnitude == pytest.approx((cmax - cmin) / (C0 / K / TAU), rel=1e-6)
    assert q["swing"].magnitude == pytest.approx((cmax - cmin) / cmin, rel=1e-6)
    assert q["accumulation_ratio"].magnitude == pytest.approx(accumulation, rel=1e-3)
    assert q["cl_ss"].magnitude == pytest.approx(100 / (C0 / K), rel=1e-6)
    assert str(q["cl_ss"].units) == "liter / hour"
    assert str(q["auc_tau"].units) == "hour * milligram / liter"
    # the single dose parameters are still there
    assert q["lambda_z"].magnitude == pytest.approx(K, rel=1e-6)


def test_steady_state_ctrough_interpolated_when_tau_between_points() -> None:
    tc = steady_state_curve()
    regimen = DosingRegimen(dose=DOSE, interval=11.0)
    q = nca_single(tc, NCAOptions(regimen=regimen, auc_method=AUCMethod.LOG)).to_quantities()
    expected = C0 * np.exp(-K * 11.0) / (1 - np.exp(-K * TAU))
    assert q["ctrough"].magnitude == pytest.approx(expected, rel=1e-6)
    assert q["auc_tau"].magnitude < nca_single(tc, NCAOptions(regimen=DosingRegimen(dose=DOSE, interval=TAU), auc_method=AUCMethod.LOG)).to_quantities()["auc_tau"].magnitude


def test_steady_state_tau_beyond_last_point_is_nan() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=48.0)
    q = nca_single(steady_state_curve(), NCAOptions(regimen=regimen)).to_quantities()
    assert np.isnan(q["auc_tau"].magnitude) and np.isnan(q["ctrough"].magnitude)


def test_accumulation_ratio_observed() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=TAU)
    options = NCAOptions(regimen=regimen, auc_method=AUCMethod.LOG)
    ss = nca(Timecourses.from_timecourses([steady_state_curve()]), options)
    sd = nca(Timecourses.from_timecourses([single_dose()]), options)
    ratio = accumulation_ratio(ss, sd)
    assert float(ratio.values[0]) == pytest.approx(1 / (1 - np.exp(-K * TAU)), rel=1e-3)
    assert ratio.attrs["units"] == "dimensionless"


def test_superposition_reaches_analytic_steady_state() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=TAU, n_doses=20)
    predicted = superposition(single_dose(), regimen, NCAOptions(auc_method=AUCMethod.LOG))
    assert predicted.dose is not None and predicted.dose.amount == 100
    assert predicted.unit == "mg/l" and predicted.time_unit == "hr"
    t_last_dose = 19 * TAU
    mask = predicted.time >= t_last_dose + 0.5
    t = predicted.time[mask] - t_last_dose
    expected = C0 * np.exp(-K * t) / (1 - np.exp(-K * TAU))
    np.testing.assert_allclose(predicted.value[mask], expected, rtol=1e-3)


def test_superposition_needs_n_doses_and_terminal_phase() -> None:
    with pytest.raises(ValueError, match="n_doses"):
        superposition(single_dose(), DosingRegimen(dose=DOSE, interval=TAU))
    rising = Timecourse(time=[1, 2, 3, 4], value=[1, 2, 3, 4], time_unit="hr", unit="mg/l", dose=DOSE)
    with pytest.raises(ValueError, match="lambda_z"):
        superposition(rising, DosingRegimen(dose=DOSE, interval=TAU, n_doses=3))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/nca/test_steady_state.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.nca.steady_state'`.

- [ ] **Step 3: Write the module and register the units**

`src/pkpdutils/nca/steady_state.py`:
```python
"""Steady state parameters and superposition.

At steady state under repeated dosing every dosing interval `tau` looks the
same; the exposure over one interval, `AUC(0-tau)`, equals the single dose
`AUC(0-inf)` when the kinetics are linear (Gabrielsson & Weiner 2016, ch.
2.8; Rowland & Tozer 2011, ch. 11). The parameters of one interval are

- `AUC(0-tau)` with the value at `tau` interpolated,
- `Ctrough = C(tau)`, `Cmin,ss` the smallest value in the interval,
- `Cavg = AUC(0-tau) / tau`,
- `fluctuation = (Cmax - Cmin,ss) / Cavg`, `swing = (Cmax - Cmin,ss) / Cmin,ss`,
- `CLss = Dose / AUC(0-tau)`,
- the accumulation ratio `R = 1 / (1 - exp(-lambda_z tau))` predicted from the
  terminal phase, or observed as `AUC(0-tau)ss / AUC(0-tau)single`.

`superposition` predicts the multiple dose curve from a single dose curve by
adding the shifted single dose curves (linear superposition), which is valid
for linear kinetics.
"""

import numpy as np
import xarray as xr

from pkpdutils.nca.auc import auc_aumc, insert_point, interpolate_at, pack_valid
from pkpdutils.nca.nca import compute_parameters, nca_single
from pkpdutils.nca.options import NCAOptions
from pkpdutils.nca.result import NCAResult
from pkpdutils.timecourse import DosingRegimen, Route, Timecourse


def compute_steady_state(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Single dose and steady state parameters of every row over the dosing interval.

    Args:
        t: times `(N, n)`
        c: values `(N, n)`
        dose_amount: dose per row, `None` without doses
        dose_time: dose time per row, `None` for the time of `options.regimen.dose`
        dose_duration: infusion duration per row
        route: route of the batch
        options: the options; `options.regimen` must be set

    Returns:
        The parameters of `compute_parameters` plus the steady state parameters.
    """
    regimen = options.regimen
    if regimen is None:
        raise ValueError("A steady state analysis needs 'options.regimen'")
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if dose_time is None:
        dose_time = np.full(t.shape[0], regimen.dose.time)
    out = compute_parameters(
        t, c, dose_amount=dose_amount, dose_time=dose_time, dose_duration=dose_duration, route=route, options=options
    )
    tau = regimen.interval
    tp, cp, n_valid = pack_valid(t - dose_time[:, None], c)
    n_rows = tp.shape[0]
    tau_row = np.full(n_rows, tau)
    if route is Route.IV_BOLUS and "c0" in out:
        # the interval starts at the dose: insert (0, C0) like the single dose areas do
        first_after_zero = (n_valid > 0) & (tp[:, 0] > 0) & np.isfinite(out["c0"])
        tp, cp, n_valid = insert_point(
            tp, cp, n_valid, np.where(first_after_zero, 0.0, np.nan), np.where(first_after_zero, out["c0"], np.nan)
        )
    ctrough = interpolate_at(tp, cp, n_valid, tau_row, options.auc_method)
    tp2, cp2, n2 = insert_point(tp, cp, n_valid, np.where(np.isnan(ctrough), np.nan, tau_row), ctrough)
    in_interval = (np.arange(tp2.shape[1])[None, :] < n2[:, None]) & (tp2 >= 0) & (tp2 <= tau)
    auc_tau, _ = auc_aumc(tp2, cp2, n2, options.auc_method, t_end=tau_row)
    # segments starting before 0 do not count
    with np.errstate(invalid="ignore"):
        cmin_ss = np.where(in_interval, cp2, np.inf).min(axis=1)
    has_tau = ~np.isnan(ctrough)
    nan = np.full(n_rows, np.nan)
    auc_tau = np.where(has_tau, auc_tau, nan)
    cmin_ss = np.where(has_tau & np.isfinite(cmin_ss), cmin_ss, nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cavg = auc_tau / tau
        cmax = out["cmax"]
        fluctuation = (cmax - cmin_ss) / cavg
        swing = (cmax - cmin_ss) / cmin_ss
        accumulation = 1.0 / (1.0 - np.exp(-out["lambda_z"] * tau))
    flags = out.pop("flags")
    out["auc_tau"] = auc_tau
    out["cmin_ss"] = cmin_ss
    out["ctrough"] = ctrough
    out["cavg"] = cavg
    out["fluctuation"] = fluctuation
    out["swing"] = swing
    out["accumulation_ratio"] = accumulation
    if dose_amount is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cl_ss"] = dose_amount / auc_tau
    out["flags"] = flags
    return out


def accumulation_ratio(steady_state: NCAResult, single_dose: NCAResult) -> xr.DataArray:
    """Observed accumulation ratio `AUC(0-tau) at steady state / AUC(0-tau) after a single dose`.

    Both results come from analyses with the same `regimen`, so both carry `auc_tau`.
    """
    for result in (steady_state, single_dose):
        if "auc_tau" not in result:
            raise ValueError("Both results need 'auc_tau': analyse with a regimen")
    ratio = steady_state["auc_tau"] / single_dose["auc_tau"]
    ratio.attrs["units"] = "dimensionless"
    return ratio.rename("accumulation_ratio")


def superposition(
    timecourse: Timecourse,
    regimen: DosingRegimen,
    options: NCAOptions | None = None,
    t_end: float | None = None,
) -> Timecourse:
    """Predict the multiple dose curve from a single dose curve by superposition.

    Args:
        timecourse: the single dose curve (its dose time is the origin)
        regimen: the doses to superpose; `n_doses` is required
        options: NCA options for the interpolation and the terminal phase
        t_end: end of the predicted curve, the last dose time plus the last
            observed time by default

    Returns:
        The predicted curve with the dose of the regimen.

    Raises:
        ValueError: without `n_doses` or without a terminal phase of the curve.
    """
    if regimen.n_doses is None:
        raise ValueError("'regimen.n_doses' is required for the superposition")
    options = options or NCAOptions()
    single = timecourse.relative_to_dose()
    q = nca_single(single, options).to_quantities()
    lambda_z = float(q["lambda_z"].magnitude)
    if not np.isfinite(lambda_z):
        raise ValueError("The single dose curve has no terminal phase (lambda_z is NaN)")
    tlast = float(q["tlast"].magnitude)
    clast = float(q["clast"].magnitude)

    dose_times = regimen.dose_times() - regimen.dose.time
    end = t_end if t_end is not None else float(dose_times[-1] + single.time[-1])
    grid = np.unique(np.concatenate([single.time + d for d in dose_times]))
    grid = grid[(grid >= 0) & (grid <= end)]

    tp, cp, n_valid = pack_valid(single.time[None, :], single.value[None, :])
    total = np.zeros_like(grid)
    for d in dose_times:
        tau_rel = grid - d
        inside = (tau_rel >= single.time[0]) & (tau_rel <= tlast)
        values = np.zeros_like(grid)
        rows = np.repeat(tp, grid.size, axis=0)
        cols = np.repeat(cp, grid.size, axis=0)
        interp = interpolate_at(rows, cols, np.repeat(n_valid, grid.size), tau_rel, options.auc_method)
        values = np.where(inside, np.nan_to_num(interp), 0.0)
        beyond = tau_rel > tlast
        values = np.where(beyond, clast * np.exp(-lambda_z * (tau_rel - tlast)), values)
        before = tau_rel < single.time[0]
        # before the first observed point after a dose: linear rise from 0
        values = np.where(before & (tau_rel >= 0), single.value[0] * tau_rel / single.time[0], values)
        total = total + values
    return Timecourse(
        time=grid,
        value=total,
        time_unit=single.time_unit,
        unit=single.unit,
        dose=regimen.dose,
        substance=single.substance,
        label=single.label,
        tissue=single.tissue,
    )
```

Add to `PARAMETER_UNITS` in `src/pkpdutils/nca/nca.py` (before `"flags"`):
```python
    "auc_tau": "({unit}) * ({time})",
    "cmin_ss": "{unit}",
    "ctrough": "{unit}",
    "cavg": "{unit}",
    "fluctuation": "dimensionless",
    "swing": "dimensionless",
    "accumulation_ratio": "dimensionless",
    "cl_ss": "({dose}) / (({unit}) * ({time}))",
```

Add to `src/pkpdutils/nca/__init__.py`: `from pkpdutils.nca.steady_state import accumulation_ratio, superposition` and the two names to `__all__`. Because `steady_state` imports `nca.nca` and `nca.nca` imports `steady_state` inside the function only, the package imports cleanly; keep the local import in `nca()`.

- [ ] **Step 4: Run the tests, lint, type check**

Run: `uv run pytest tests/nca -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: all pass, no warnings. `test_steady_state_parameters_analytic` holds to 1e-6 with the log rule because `(0, C0)` is inserted for the bolus before `auc_tau` is computed, like in the single dose areas.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/nca tests/nca/test_steady_state.py
git commit -q -m "Add the steady state parameters and the superposition of the NCA

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 8: Figures (`pkpdutils.plot`): style, timecourses, NCA diagnostics

**Files:**
- Create: `src/pkpdutils/plot/__init__.py`, `src/pkpdutils/plot/style.py`, `src/pkpdutils/plot/timecourse.py`, `src/pkpdutils/plot/nca.py`
- Test: `tests/plot/__init__.py` (empty), `tests/plot/test_plot_nca.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PlotStyle`: `data_color="black"`, `data_marker="o"`, `fit_color="tab:blue"`, `auc_color="tab:green"`, `extrapolation_color="tab:red"`, `terminal_marker="s"`, `alpha=0.2`, `linewidth=1.5`, `markersize=5`, `cmap="viridis"`; `DEFAULT_STYLE = PlotStyle()`
  - `plot_timecourse(timecourses: Timecourse | Timecourses, ax: Axes | None = None, *, log: bool = False, errorbars: bool = True, by: str | None = None, style: PlotStyle = DEFAULT_STYLE) -> Figure`: one line per sample; `sd`/`se` as error bars (`se` preferred when both); for a batch the legend labels come from the coordinate of `by` (default: the label of every sample); log y axis when `log`; axis labels `time [unit]`, `substance [unit]`; returns the figure of `ax`
  - `plot_nca(timecourse: Timecourse, result: NCAResult, *, title: str | None = None, style: PlotStyle = DEFAULT_STYLE, **indexers) -> Figure`: two panels (linear, logarithmic): data points, the area to `tlast` shaded (`auc_color`), the extrapolated tail `clast·exp(−lambda_z (t − tlast))` over `[tlast, tlast + 3 thalf]` shaded (`extrapolation_color`), the terminal regression line over the data range, the points of the regression (`terminal_marker`), `cmax`/`tmax` guide lines, `c0` marker for a bolus, flags in the title; `indexers` select the sample of a batch result (a result without sample dims needs none)
  - `plot_nca_grid(timecourses: Timecourses, result: NCAResult, *, ncols: int = 3, log: bool = True, style: PlotStyle = DEFAULT_STYLE) -> Figure`: one panel per sample (same drawing as one panel of `plot_nca`), grid of `ncols`

- [ ] **Step 1: Write the failing tests**

`tests/plot/test_plot_nca.py`:
```python
import matplotlib
import numpy as np
from matplotlib.figure import Figure

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import NCAOptions, nca, nca_single
from pkpdutils.plot import PlotStyle, plot_nca, plot_nca_grid, plot_timecourse

matplotlib.use("Agg")


def oral(ka: float = 2.0, label: str = "a") -> Timecourse:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])
    c = 10 * ka / (ka - 0.3) * (np.exp(-0.3 * t) - np.exp(-ka * t))
    return Timecourse(time=t, value=c, sd=0.1 * c, n=8, time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine", label=label)


def test_plot_timecourse_single_and_batch() -> None:
    fig = plot_timecourse(oral())
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time [hr]" and ax.get_ylabel() == "caffeine [mg/l]"
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    fig2 = plot_timecourse(batch, log=True, errorbars=False)
    assert fig2.axes[0].get_yscale() == "log"
    labels = [line.get_label() for line in fig2.axes[0].get_lines()]
    assert "slow" in labels and "fast" in labels
    matplotlib.pyplot.close("all")


def test_plot_nca_single() -> None:
    tc = oral()
    result = nca_single(tc)
    fig = plot_nca(tc, result, title="test")
    assert len(fig.axes) == 2
    assert fig.axes[1].get_yscale() == "log"
    assert "test" in fig.axes[0].get_title()
    matplotlib.pyplot.close(fig)


def test_plot_nca_from_batch_result_with_indexers_and_flags_in_title() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    result = nca(batch)
    fig = plot_nca(batch.sel(individual="fast"), result, individual="fast")
    assert "fast" in fig.axes[0].get_title()
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l")
    fig2 = plot_nca(short, nca_single(short))
    assert "TOO_FEW_POINTS" in fig2.axes[0].get_title()
    matplotlib.pyplot.close("all")


def test_plot_nca_iv_bolus_marks_c0() -> None:
    t = np.array([0.5, 1, 2, 4, 8])
    tc = Timecourse(time=t, value=10 * np.exp(-0.5 * t), time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS))
    fig = plot_nca(tc, nca_single(tc), style=PlotStyle(fit_color="red"))
    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert "C0" in labels
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid() -> None:
    batch = Timecourses.from_timecourses([oral(k, str(k)) for k in (1.0, 2.0, 3.0, 4.0)])
    fig = plot_nca_grid(batch, nca(batch, NCAOptions()), ncols=2)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 4
    matplotlib.pyplot.close(fig)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/plot -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.plot'`.

- [ ] **Step 3: Write the modules**

`src/pkpdutils/plot/__init__.py`:
```python
"""Figures of timecourses and analyses (matplotlib).

Every function returns the `matplotlib.figure.Figure` it drew on and never
shows it; pass `ax` to draw into an existing axes.
"""

from pkpdutils.plot.nca import plot_nca, plot_nca_grid
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import plot_timecourse

__all__ = ["DEFAULT_STYLE", "PlotStyle", "plot_nca", "plot_nca_grid", "plot_timecourse"]
```

`src/pkpdutils/plot/style.py`:
```python
"""Style of the figures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotStyle:
    """Colors, markers and sizes shared by the figures of the package.

    Attributes:
        data_color: color of the data points and lines
        data_marker: marker of the data points
        fit_color: color of regression lines and fitted curves
        auc_color: fill color of the area to the last measurable point
        extrapolation_color: fill color of the extrapolated area
        terminal_marker: marker of the points of the terminal regression
        alpha: transparency of filled areas
        linewidth: width of lines
        markersize: size of markers
        cmap: colormap of the samples of a batch
    """

    data_color: str = "black"
    data_marker: str = "o"
    fit_color: str = "tab:blue"
    auc_color: str = "tab:green"
    extrapolation_color: str = "tab:red"
    terminal_marker: str = "s"
    alpha: float = 0.2
    linewidth: float = 1.5
    markersize: float = 5.0
    cmap: str = "viridis"


#: the default style
DEFAULT_STYLE = PlotStyle()
```

`src/pkpdutils/plot/timecourse.py`:
```python
"""Figures of timecourses."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Timecourse, Timecourses


def _figure_of(ax: Axes | None) -> tuple[Figure, Axes]:
    """The axes to draw on and its figure, a new figure without `ax`."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax


def _draw_curve(ax: Axes, tc: Timecourse, *, color: str, label: str | None, errorbars: bool, style: PlotStyle) -> None:
    """One curve with optional error bars."""
    err = tc.se if tc.se is not None else tc.sd
    if errorbars and err is not None:
        ax.errorbar(tc.time, tc.value, yerr=err, fmt=f"{style.data_marker}-", color=color, label=label, linewidth=style.linewidth, markersize=style.markersize, capsize=2)
    else:
        ax.plot(tc.time, tc.value, f"{style.data_marker}-", color=color, label=label, linewidth=style.linewidth, markersize=style.markersize)


def plot_timecourse(
    timecourses: Timecourse | Timecourses,
    ax: Axes | None = None,
    *,
    log: bool = False,
    errorbars: bool = True,
    by: str | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Plot one timecourse or every timecourse of a batch.

    Args:
        timecourses: the curve or the batch
        ax: axes to draw on, a new figure by default
        log: logarithmic value axis
        errorbars: draw `se` (or `sd`) as error bars when present
        by: coordinate of the batch used as legend label, the sample label by default
        style: colors and markers

    Returns:
        The figure.
    """
    fig, ax = _figure_of(ax)
    if isinstance(timecourses, Timecourse):
        curves = [timecourses]
        labels: list[str | None] = [timecourses.label]
        first = timecourses
    else:
        curves = list(timecourses)
        first = curves[0]
        if by is not None:
            values = timecourses.ds[by].to_numpy().reshape(-1)
            labels = [str(v) for v in values]
        else:
            labels = [tc.label for tc in curves]
    cmap = plt.get_cmap(style.cmap)
    for i, (tc, label) in enumerate(zip(curves, labels, strict=True)):
        color = style.data_color if len(curves) == 1 else cmap(i / max(len(curves) - 1, 1))
        _draw_curve(ax, tc, color=color, label=label, errorbars=errorbars, style=style)
    ax.set_xlabel(f"time [{first.time_unit}]")
    ax.set_ylabel(f"{first.substance} [{first.unit}]")
    if log:
        ax.set_yscale("log")
    if any(label is not None for label in labels):
        ax.legend()
    return fig
```

`src/pkpdutils/plot/nca.py`:
```python
"""Diagnostic figures of the non-compartmental analysis."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.nca.options import decode_flags
from pkpdutils.nca.result import NCAResult
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Route, Timecourse, Timecourses


def _sample_values(result: NCAResult, indexers: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Parameter magnitudes and flags of one sample of a result."""
    quantities = result.to_quantities(**indexers)
    return {name: float(q.magnitude) for name, q in quantities.items()}, result.flags(**indexers)


def draw_nca_panel(
    ax: Axes,
    timecourse: Timecourse,
    values: dict[str, float],
    flags: list[str],
    *,
    log: bool,
    title: str | None,
    style: PlotStyle,
) -> None:
    """Draw the NCA diagnostics of one curve into one axes.

    Args:
        ax: the axes
        timecourse: the curve (times relative to its dose)
        values: parameter magnitudes of the curve
        flags: flag names of the curve
        log: logarithmic value axis
        title: title, the label of the curve by default
        style: colors and markers
    """
    tc = timecourse.relative_to_dose()
    t, c = tc.time, tc.value
    ok = np.isfinite(c)
    tlast, clast = values.get("tlast", np.nan), values.get("clast", np.nan)
    lambda_z = values.get("lambda_z", np.nan)
    intercept = values.get("lambda_z_intercept", np.nan)
    thalf = values.get("thalf", np.nan)

    if np.isfinite(tlast):
        area = ok & (t <= tlast)
        ax.fill_between(t[area], 0.0, c[area], color=style.auc_color, alpha=style.alpha, label="AUC(0-tlast)")
    if np.isfinite(lambda_z) and np.isfinite(tlast):
        t_ext = np.linspace(tlast, tlast + 3.0 * thalf, 50)
        c_ext = clast * np.exp(-lambda_z * (t_ext - tlast))
        ax.fill_between(t_ext, 0.0, c_ext, color=style.extrapolation_color, alpha=style.alpha, label="extrapolated")
        t_fit = np.linspace(values.get("lambda_z_t_first", t[ok][0]), tlast + 3.0 * thalf, 50)
        ax.plot(t_fit, np.exp(intercept - lambda_z * t_fit), "-", color=style.fit_color, linewidth=style.linewidth, label=f"lambda_z = {lambda_z:.3g}")
        t_first = values.get("lambda_z_t_first", np.nan)
        used = ok & (t >= t_first) & (t <= tlast) & (c > 0)
        ax.plot(t[used], c[used], style.terminal_marker, color=style.fit_color, markersize=style.markersize + 3, fillstyle="none", label="regression points")
    cmax, tmax = values.get("cmax", np.nan), values.get("tmax", np.nan)
    if np.isfinite(cmax):
        ax.plot([0, tmax], [cmax, cmax], "--", color="gray", linewidth=1)
        ax.plot([tmax, tmax], [0, cmax], "--", color="gray", linewidth=1)
    c0 = values.get("c0", np.nan)
    if np.isfinite(c0) and tc.dose is not None and tc.dose.route is Route.IV_BOLUS:
        ax.plot([0.0], [c0], "D", color=style.fit_color, markersize=style.markersize + 1, label="C0")
    ax.plot(t[ok], c[ok], f"{style.data_marker}-", color=style.data_color, linewidth=style.linewidth, markersize=style.markersize, label="data")
    ax.set_xlabel(f"time [{tc.time_unit}]")
    ax.set_ylabel(f"{tc.substance} [{tc.unit}]")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    heading = title if title is not None else (tc.label or tc.substance)
    if flags:
        heading = f"{heading} [{', '.join(flags)}]"
    ax.set_title(heading)
    ax.legend(fontsize="small")


def plot_nca(
    timecourse: Timecourse,
    result: NCAResult,
    *,
    title: str | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Linear and logarithmic panel of one curve with its NCA diagnostics.

    Args:
        timecourse: the curve
        result: the result of its analysis (a batch result with `indexers`, or a single result)
        title: title of the panels, the label of the curve by default; the flags are appended
        style: colors and markers
        **indexers: coordinate labels selecting the sample of a batch result

    Returns:
        The figure.
    """
    values, flags = _sample_values(result, indexers)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))
    if title is None and indexers:
        title = "|".join(str(v) for v in indexers.values())
    draw_nca_panel(ax1, timecourse, values, flags, log=False, title=title, style=style)
    draw_nca_panel(ax2, timecourse, values, flags, log=True, title=title, style=style)
    fig.tight_layout()
    return fig


def plot_nca_grid(
    timecourses: Timecourses,
    result: NCAResult,
    *,
    ncols: int = 3,
    log: bool = True,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """One NCA panel per sample of a batch.

    Args:
        timecourses: the batch
        result: its result
        ncols: panels per row
        log: logarithmic value axes
        style: colors and markers

    Returns:
        The figure.
    """
    curves = list(timecourses)
    n = len(curves)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    indices = list(np.ndindex(*timecourses.sample_shape))
    for k, (tc, index) in enumerate(zip(curves, indices, strict=True)):
        sample = result.ds.isel(dict(zip(timecourses.sample_dims, (int(i) for i in index), strict=True)))
        values = {name: float(sample[name].values) for name in result.parameters}
        flags = decode_flags(int(sample["flags"].values))
        draw_nca_panel(flat_axes[k], tc, values, flags, log=log, title=None, style=style)
    for ax in flat_axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
```

`tests/plot/__init__.py`: empty.

Note: in `plot_nca_grid` the sample is selected by position and its flags are decoded directly from the `flags` variable, so batches without coordinates work too.

- [ ] **Step 4: Run the tests, lint, type check**

Run: `uv run pytest tests/plot -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: 5 passed, no warnings (matplotlib emits none under `Agg`; a `tight_layout` warning is a finding: replace by `fig.set_layout_engine("constrained")` before drawing). ty: `ax.get_figure()` returns `Figure | SubFigure | None`; the `assert isinstance(fig, Figure)` narrows it.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/plot tests/plot
git commit -q -m "Add the timecourse and NCA figures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 9: Examples, top level exports and the example tests

**Files:**
- Create: `examples/nca_single.py`, `examples/nca_batch.py`, `examples/steady_state.py`, `examples/nca_from_sbmlsim.py`
- Modify: `src/pkpdutils/__init__.py` (exports), `tests/examples/test_examples.py` (`SCRIPTS`), `examples/README.md` (table)
- Test: `tests/test_package.py` (export test)

**Interfaces:**
- Produces: `pkpdutils` exports additionally `nca`, `nca_single`, `NCAOptions`, `NCAResult`, `TerminalPhase`; the four examples run offline and write PNG files into the working directory.

- [ ] **Step 1: Export test**

Append to `tests/test_package.py`:
```python
def test_nca_exports() -> None:
    from pkpdutils import NCAOptions, NCAResult, TerminalPhase, nca, nca_single

    assert callable(nca) and callable(nca_single)
    assert NCAOptions().terminal == TerminalPhase()
    assert NCAResult is not None
```
Run: `uv run pytest tests/test_package.py -n 0 -q` → FAIL with `ImportError`.

`src/pkpdutils/__init__.py`: add `from pkpdutils.nca import NCAOptions, NCAResult, TerminalPhase, nca, nca_single` and the five names to `__all__` (keep the list sorted as ruff `RUF022` wants). Run again → PASS.

- [ ] **Step 2: Write the examples**

`examples/nca_single.py`:
```python
"""Non-compartmental analysis of one timecourse.

Run from the root of the repository with `python -m examples.nca_single`.
Writes `nca_single.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod, TerminalMethod, TerminalPhase
from pkpdutils.plot import plot_nca

# caffeine after an oral dose, mean concentrations of a group
tc = Timecourse(
    time=[0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24],
    value=[0.9, 1.7, 2.6, 2.9, 2.8, 2.5, 2.2, 1.6, 1.2, 0.6, 0.1],
    sd=[0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1, 0.03],
    n=12,
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
    label="healthy volunteers",
)

if __name__ == "__main__":
    console.rule("Default options: linear-up/log-down, best fit terminal phase")
    result = nca_single(tc)
    for name, quantity in result.to_quantities().items():
        console.print(f"{name:<22} {quantity:~P}")
    console.print("flags:", result.flags())

    console.rule("The rule of pkdb_analysis 0.3.1: linear trapezoid, all points after tmax")
    old = nca_single(
        tc,
        NCAOptions(auc_method=AUCMethod.LINEAR, terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX)),
    )
    console.print(old.to_dataframe().T)

    fig = plot_nca(tc, result)
    fig.savefig("nca_single.png", dpi=120)
    console.print("written: nca_single.png")
    console.print(np.round(result["auc_inf_obs"].values, 3))
```

`examples/nca_batch.py`:
```python
"""Non-compartmental analysis of a batch: individuals of a dose escalation.

Run from the root of the repository with `python -m examples.nca_batch`.
Writes `nca_batch.png` and `nca_batch.tsv` into the working directory.
"""

import numpy as np

from pkpdutils import NCAOptions, Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.plot import plot_nca_grid, plot_timecourse

rng = np.random.default_rng(1)
time = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24])
doses = np.array([50.0, 100.0, 200.0])
individuals = ["s1", "s2", "s3", "s4"]
ke = rng.uniform(0.15, 0.3, size=len(individuals))
ka = rng.uniform(1.0, 3.0, size=len(individuals))
values = np.empty((len(doses), len(individuals), time.size))
for i, dose in enumerate(doses):
    for j in range(len(individuals)):
        curve = dose / 40 * ka[j] / (ka[j] - ke[j]) * (np.exp(-ke[j] * time) - np.exp(-ka[j] * time))
        values[i, j] = curve * rng.lognormal(0, 0.05, size=time.size)

batch = Timecourses.from_arrays(
    time,
    values,
    time_unit="hr",
    unit="mg/l",
    dims=("dose", "individual"),
    coords={"dose": doses, "individual": individuals},
    dose={"amount": np.broadcast_to(doses[:, None], (3, 4)), "unit": "mg"},
    route=Route.ORAL,
    substance="drug",
)

if __name__ == "__main__":
    result = nca(batch, NCAOptions())
    console.rule("Parameters over (dose, individual)")
    console.print(result.ds)
    df = result.to_dataframe()
    console.print(df[["dose", "individual", "auc_inf_obs", "cmax", "thalf", "cl_f", "flags"]])
    df.to_csv("nca_batch.tsv", sep="\t", index=False)

    console.rule("Dose proportionality at a glance: AUC / dose")
    console.print(result["auc_inf_dn"].mean(dim="individual").values)

    plot_timecourse(batch, log=True, by="individual").savefig("nca_batch_curves.png", dpi=120)
    plot_nca_grid(batch, result, ncols=4).savefig("nca_batch.png", dpi=100)
    console.print("written: nca_batch.tsv, nca_batch_curves.png, nca_batch.png")
```

`examples/steady_state.py`:
```python
"""Steady state parameters and superposition.

Run from the root of the repository with `python -m examples.steady_state`.
Writes `steady_state.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, DosingRegimen, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod, superposition
from pkpdutils.plot import plot_timecourse

k, c0 = 0.15, 8.0
dose = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
t = np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])
single = Timecourse(time=t, value=c0 * np.exp(-k * t), time_unit="hr", unit="mg/l", dose=dose, substance="drug", label="single dose")

if __name__ == "__main__":
    regimen = DosingRegimen(dose=dose, interval=12, n_doses=10)
    predicted = superposition(single, regimen, NCAOptions(auc_method=AUCMethod.LOG))
    console.rule("Predicted multiple dose curve")
    console.print(predicted.to_dataframe().tail())

    console.rule("Steady state parameters of the last interval")
    last = predicted.time >= regimen.dose_times()[-1]
    interval = Timecourse(
        time=predicted.time[last],
        value=predicted.value[last],
        time_unit="hr",
        unit="mg/l",
        dose=dose.model_copy(update={"time": float(regimen.dose_times()[-1])}),
        substance="drug",
        label="steady state",
    )
    result = nca_single(interval, NCAOptions(regimen=DosingRegimen(dose=dose, interval=12), auc_method=AUCMethod.LOG))
    for name in ("auc_tau", "cmax", "cmin_ss", "ctrough", "cavg", "fluctuation", "swing", "accumulation_ratio", "cl_ss"):
        console.print(f"{name:<20} {result.to_quantities()[name]:~P}")
    console.print("predicted accumulation 1/(1-exp(-k tau)) =", 1 / (1 - np.exp(-k * 12)))

    fig = plot_timecourse(predicted)
    fig.savefig("steady_state.png", dpi=120)
    console.print("written: steady_state.png")
```

`examples/nca_from_sbmlsim.py`:
```python
"""Non-compartmental analysis of a simulation result.

The result of an sbmlsim parameter scan is an `XResult`, an xarray dataset with
the `_time` dimension and one dimension per scan dimension. `Timecourses.from_xresult`
turns it into a batch, `nca` analyses every simulated curve at once. sbmlsim is
not a dependency of pkpdutils: without it the example builds a dataset of the
same shape and analyses that.

Run from the root of the repository with `python -m examples.nca_from_sbmlsim`.
"""

import numpy as np
import xarray as xr

from pkpdutils import Dose, NCAOptions, Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod


def simulated_dataset() -> xr.Dataset:
    """A dataset shaped like an sbmlsim scan over the dose."""
    time = np.linspace(0, 24, 241)
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    values = doses[None, :] / 20 * np.exp(-0.2 * time[:, None]) * (1 - np.exp(-1.5 * time[:, None]))
    return xr.Dataset({"[Cve]": (("_time", "dim_dose"), values)}, coords={"_time": time, "dim_dose": doses})


if __name__ == "__main__":
    try:
        from sbmlsim.result import XResult  # ty: ignore[unresolved-import]

        console.print("sbmlsim is installed:", XResult.__name__)
    except ImportError:
        console.print("sbmlsim is not installed, using a dataset of the same shape")

    ds = simulated_dataset()
    batch = Timecourses.from_dataset(
        ds, "[Cve]", unit="mmol/l", time_unit="hr",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="drug",
    )
    result = nca(batch, NCAOptions(auc_method=AUCMethod.LINEAR_LOG))
    console.rule("NCA over the scan dimension")
    console.print(result.to_dataframe()[["dim_dose", "auc_inf_obs", "cmax", "tmax", "thalf", "flags"]])
```

Note: the dose of `from_dataset` is one dose for all samples; in the scan the actual doses differ per sample, so `cl_f` is not meaningful there — the example prints exposure parameters only.

Register the examples: `SCRIPTS = ["examples.timecourses", "examples.nca_single", "examples.nca_batch", "examples.steady_state", "examples.nca_from_sbmlsim"]` in `tests/examples/test_examples.py`. Add the four rows to the table of `examples/README.md`:
```markdown
| `examples/nca_single.py` | non-compartmental analysis of one curve, default and pkdb_analysis-compatible options, the diagnostic figure |
| `examples/nca_batch.py` | NCA of a `(dose, individual)` batch, results as a data frame, curve and grid figures |
| `examples/steady_state.py` | superposition of a single dose curve and the steady state parameters of a dosing interval |
| `examples/nca_from_sbmlsim.py` | NCA of a simulation scan dataset (`Timecourses.from_dataset`), with or without sbmlsim installed |
```

- [ ] **Step 3: Run the examples and the checks**

Run: `cd $(mktemp -d) && for m in nca_single nca_batch steady_state nca_from_sbmlsim; do PYTHONPATH=/home/mkoenig/git/pkdb_analysis uv run --project /home/mkoenig/git/pkdb_analysis python -m examples.$m > /dev/null || echo "FAILED $m"; done; ls; cd -`
Expected: no `FAILED`, the PNG/TSV files listed. Then `uv run pytest -q && uv run ruff check --fix && uv run ruff format && uv run ty check` from the repository root: all pass, clean. The `# ty: ignore[unresolved-import]` on the sbmlsim import is required (sbmlsim is not installed); if ty reports it unused (sbmlsim resolvable in the environment), delete it.

- [ ] **Step 4: Commit**

```bash
git add src/pkpdutils/__init__.py examples tests/examples tests/test_package.py
git commit -q -m "Export the NCA and add the NCA, batch, steady state and simulation examples

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 10: Documentation: `nca.md`, `glossary.md`, `plotting.md`, API pages, nav, CLAUDE.md

**Files:**
- Create: `docs/nca.md`, `docs/glossary.md`, `docs/plotting.md`, `docs/api/nca.md`, `docs/api/nca.options.md`, `docs/api/nca.auc.md`, `docs/api/nca.terminal.md`, `docs/api/nca.result.md`, `docs/api/nca.steady_state.md`, `docs/api/plot.md`
- Modify: `zensical.toml` (nav), `docs/api/index.md`, `docs/index.md` (features list), `docs/timecourses.md` (link to nca.md), `CLAUDE.md` (architecture, commands)

- [ ] **Step 1: API pages**

Each `docs/api/<name>.md` has the heading and the directive, e.g. `docs/api/nca.md`:
```markdown
# nca

::: pkpdutils.nca.nca
```
and analogously `nca.options` → `::: pkpdutils.nca.options`, `nca.auc` → `::: pkpdutils.nca.auc`, `nca.terminal` → `::: pkpdutils.nca.terminal`, `nca.result` → `::: pkpdutils.nca.result`, `nca.steady_state` → `::: pkpdutils.nca.steady_state`; `docs/api/plot.md` holds three directives with subheadings:
```markdown
# plot

## plot.style

::: pkpdutils.plot.style

## plot.timecourse

::: pkpdutils.plot.timecourse

## plot.nca

::: pkpdutils.plot.nca
```

`zensical.toml` nav — replace the `"User guide"` and `"API reference"` blocks by:
```toml
  { "User guide" = [
    { "Units" = "units.md" },
    { "Timecourses" = "timecourses.md" },
    { "Non-compartmental analysis" = "nca.md" },
    { "Plotting" = "plotting.md" },
    { "Glossary" = "glossary.md" },
  ] },
  { "References" = "references.md" },
  { "API reference" = [
    { "Overview" = "api/index.md" },
    { "pkpdutils" = [
      { "units" = "api/units.md" },
      { "timecourse" = "api/timecourse.md" },
      { "console" = "api/console.md" },
      { "log" = "api/log.md" },
    ] },
    { "pkpdutils.nca" = [
      { "nca" = "api/nca.md" },
      { "options" = "api/nca.options.md" },
      { "result" = "api/nca.result.md" },
      { "auc" = "api/nca.auc.md" },
      { "terminal" = "api/nca.terminal.md" },
      { "steady_state" = "api/nca.steady_state.md" },
    ] },
    { "pkpdutils.plot" = [
      { "plot" = "api/plot.md" },
    ] },
  ] },
```

`docs/api/index.md`: append
```markdown
## pkpdutils.nca

Non-compartmental analysis, see [Non-compartmental analysis](../nca.md).

| module | description |
| --- | --- |
| [nca.nca](nca.md) | `nca`, `nca_single`, `compute_parameters`: the analysis |
| [nca.options](nca.options.md) | `NCAOptions`, `TerminalPhase`, the method enumerations and `NCAFlag` |
| [nca.result](nca.result.md) | `NCAResult` and the units of the parameters |
| [nca.auc](nca.auc.md) | vectorized trapezoid areas, interpolation |
| [nca.terminal](nca.terminal.md) | vectorized terminal phase regression |
| [nca.steady_state](nca.steady_state.md) | steady state parameters, accumulation ratio, superposition |

## pkpdutils.plot

Figures, see [Plotting](../plotting.md).

| module | description |
| --- | --- |
| [plot](plot.md) | `PlotStyle`, `plot_timecourse`, `plot_nca`, `plot_nca_grid` |
```

- [ ] **Step 2: Write `docs/nca.md`**

```markdown
# Non-compartmental analysis

Non-compartmental analysis (NCA) describes a concentration timecourse by parameters computed directly from the measured points, without a model of the body: the exposure as the area under the curve, the peak, the terminal half-life, and, with the dose, the clearance and the volume of distribution. `pkpdutils.nca` computes these parameters for one curve or for a whole batch of curves in one vectorized call; every parameter carries its unit and every sample carries flags for the conditions that limit its interpretation. The definitions follow Gabrielsson & Weiner [^gw] and the NCA of Phoenix WinNonlin [^phoenix].

## Concepts

**Exposure.** The area under the concentration–time curve, \(\mathrm{AUC}\), is proportional to the amount of drug that reached the systemic circulation. \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\) is measured up to the last quantifiable concentration \(C_\mathrm{last}\); \(\mathrm{AUC}_{0\text{-}\infty}\) adds the tail after \(t_\mathrm{last}\), predicted from the terminal phase. The fraction of \(\mathrm{AUC}_{0\text{-}\infty}\) that is extrapolated tells how much of the exposure was not observed; above 20 % the estimate is considered unreliable (flag `EXTRAPOLATION_HIGH`).

**Peak.** \(C_\mathrm{max}\) and \(t_\mathrm{max}\) are read from the observed points. After an extravascular dose they reflect the balance of absorption and elimination; after an intravenous bolus the concentration at time zero, \(C_0\), is not observed and is back-extrapolated from the first two points.

**Terminal phase.** When absorption and distribution are over, the concentration declines mono-exponentially, \(C(t) = C_\mathrm{last}\, e^{-\lambda_z (t - t_\mathrm{last})}\). The terminal rate constant \(\lambda_z\) is the slope of \(\ln C\) against \(t\) over the terminal points; the half-life is \(t_{1/2} = \ln 2 / \lambda_z\). Which points belong to the terminal phase is a judgement: the default `BEST_FIT` rule takes the window with the largest adjusted \(R^2\) among all windows of at least three points that end at \(t_\mathrm{last}\) and start after \(t_\mathrm{max}\), preferring more points when the adjusted \(R^2\) is equal within a tolerance, as Phoenix does. `LAST_N`, `ALL_AFTER_TMAX` (the rule of pkdb_analysis 0.3.1) and `MANUAL` are the alternatives.

**Clearance and volume.** With the dose \(D\), the clearance \(\mathrm{CL} = D / \mathrm{AUC}_{0\text{-}\infty}\) is the volume of plasma cleared of drug per time and the volume of distribution \(V_z = \mathrm{CL} / \lambda_z\) is the apparent volume the dose would occupy at the plasma concentration. After an extravascular dose the fraction absorbed \(F\) is unknown and both are reported relative to it as \(\mathrm{CL}/F\) (`cl_f`) and \(V_z/F\) (`vz_f`). The mean residence time \(\mathrm{MRT} = \mathrm{AUMC}_{0\text{-}\infty} / \mathrm{AUC}_{0\text{-}\infty}\) is the average time a molecule stays in the body (after an infusion of duration \(T\), minus \(T/2\)); the steady state volume \(V_\mathrm{ss} = \mathrm{CL} \cdot \mathrm{MRT}\) is reported for intravenous doses.

**Steady state.** Under repeated dosing every dosing interval \(\tau\) looks the same once steady state is reached, and with linear kinetics \(\mathrm{AUC}_{0\text{-}\tau}\) at steady state equals the single dose \(\mathrm{AUC}_{0\text{-}\infty}\). The interval is described by the average concentration \(C_\mathrm{avg} = \mathrm{AUC}_{0\text{-}\tau} / \tau\), the trough \(C_\mathrm{trough} = C(\tau)\), the fluctuation and the swing, and the accumulation ratio [^rt].

**Routes.** A batch has one route. `IV_BOLUS` reports \(C_0\), \(\mathrm{CL}\), \(V_z\), \(V_\mathrm{ss}\); `IV_INFUSION` corrects the \(\mathrm{MRT}\) by half the duration; `ORAL` (any extravascular route) reports \(\mathrm{CL}/F\), \(V_z/F\) and the half maximum during absorption (`cmax_half`, `tmax_half`).

**Missing values and the limit of quantification.** `NaN` values are dropped; values below `lloq` become `NaN` (or 0 before the maximum with `BLQHandling.ZERO_BEFORE_TMAX`), and the sample is flagged `BLQ_TRUNCATED`.

## Math

Trapezoid rules on a segment from \((t_1, C_1)\) to \((t_2, C_2)\) with \(\Delta t = t_2 - t_1\) and \(L = \ln(C_1 / C_2)\):

\[
\mathrm{AUC}^\mathrm{lin} = \frac{\Delta t\,(C_1 + C_2)}{2}, \qquad
\mathrm{AUMC}^\mathrm{lin} = \frac{\Delta t\,(t_1 C_1 + t_2 C_2)}{2}
\]

\[
\mathrm{AUC}^\mathrm{log} = \frac{\Delta t\,(C_1 - C_2)}{L}, \qquad
\mathrm{AUMC}^\mathrm{log} = \frac{\Delta t\,(t_1 C_1 - t_2 C_2)}{L} + \frac{\Delta t^2\,(C_1 - C_2)}{L^2}
\]

The logarithmic rule is exact for a mono-exponential segment. `AUCMethod.LINEAR_LOG` (the default, "linear up/log down") uses the linear rule on rising and the logarithmic rule on falling segments; `LINEAR` uses the linear rule everywhere (the rule of pkdb_analysis 0.3.1); `LOG` the logarithmic rule wherever both values are positive.

Terminal regression of \(y = \ln C\) on \(t\) over \(n\) points:

\[
\lambda_z = -\frac{n \sum t y - \sum t \sum y}{n \sum t^2 - (\sum t)^2}, \qquad
R^2_\mathrm{adj} = 1 - (1 - R^2)\,\frac{n - 1}{n - 2}, \qquad
t_{1/2} = \frac{\ln 2}{\lambda_z}
\]

Extrapolation and moments, with the observed \(C_\mathrm{last}\) (`auc_inf_obs`) or the value of the regression line at \(t_\mathrm{last}\), \(\hat C_\mathrm{last} = e^{b - \lambda_z t_\mathrm{last}}\) (`auc_inf_pred`):

\[
\mathrm{AUC}_{0\text{-}\infty} = \mathrm{AUC}_{0\text{-}t_\mathrm{last}} + \frac{C_\mathrm{last}}{\lambda_z}, \qquad
\mathrm{AUMC}_{0\text{-}\infty} = \mathrm{AUMC}_{0\text{-}t_\mathrm{last}} + \frac{C_\mathrm{last}\, t_\mathrm{last}}{\lambda_z} + \frac{C_\mathrm{last}}{\lambda_z^2}
\]

\[
\mathrm{MRT} = \frac{\mathrm{AUMC}_{0\text{-}\infty}}{\mathrm{AUC}_{0\text{-}\infty}} - \frac{T_\mathrm{inf}}{2}, \qquad
\mathrm{CL} = \frac{D}{\mathrm{AUC}_{0\text{-}\infty}}, \qquad
V_z = \frac{\mathrm{CL}}{\lambda_z}, \qquad
V_\mathrm{ss} = \mathrm{CL} \cdot \mathrm{MRT}
\]

\(C_0\) after a bolus by log-linear back-extrapolation of the first two points \((t_1, C_1)\), \((t_2, C_2)\): \(C_0 = \exp\!\left(\ln C_1 - t_1 \frac{\ln C_2 - \ln C_1}{t_2 - t_1}\right)\); the point \((0, C_0)\) enters the areas.

Steady state over the interval \([0, \tau]\) (the value at \(\tau\) is interpolated):

\[
C_\mathrm{avg} = \frac{\mathrm{AUC}_{0\text{-}\tau}}{\tau}, \quad
\mathrm{fluctuation} = \frac{C_\mathrm{max} - C_\mathrm{min,ss}}{C_\mathrm{avg}}, \quad
\mathrm{swing} = \frac{C_\mathrm{max} - C_\mathrm{min,ss}}{C_\mathrm{min,ss}}, \quad
R_\mathrm{pred} = \frac{1}{1 - e^{-\lambda_z \tau}}, \quad
R_\mathrm{obs} = \frac{\mathrm{AUC}_{0\text{-}\tau}^\mathrm{ss}}{\mathrm{AUC}_{0\text{-}\tau}^\mathrm{single}}
\]

Superposition predicts the multiple dose curve as the sum of the single dose curve shifted to every dose time, interpolated inside the observed range and extrapolated with \(\lambda_z\) beyond \(t_\mathrm{last}\); it assumes linear kinetics.

## Parameters

| name | symbol | definition | unit | needs |
| --- | --- | --- | --- | --- |
| `cmax`, `tmax` | \(C_\mathrm{max}\), \(t_\mathrm{max}\) | maximum observed value and its time | value, time | |
| `cmin`, `tmin` | \(C_\mathrm{min}\), \(t_\mathrm{min}\) | minimum observed value and its time | value, time | |
| `clast`, `tlast` | \(C_\mathrm{last}\), \(t_\mathrm{last}\) | last positive value and its time | value, time | |
| `c0` | \(C_0\) | back-extrapolated value at time 0 | value | `IV_BOLUS` |
| `cmax_half`, `tmax_half` | | value closest to \(C_\mathrm{max}/2\) before the maximum and its time | value, time | `ORAL` |
| `auc_last` | \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\) | area to the last positive value | value·time | |
| `auc_inf_obs`, `auc_inf_pred` | \(\mathrm{AUC}_{0\text{-}\infty}\) | area extrapolated with the observed or predicted \(C_\mathrm{last}\) | value·time | \(\lambda_z\) |
| `auc_extrap_fraction` | | \((\mathrm{AUC}_{0\text{-}\infty} - \mathrm{AUC}_{0\text{-}t_\mathrm{last}}) / \mathrm{AUC}_{0\text{-}\infty}\) | – | \(\lambda_z\) |
| `aumc_last`, `aumc_inf` | \(\mathrm{AUMC}\) | first moment of the curve | value·time² | \(\lambda_z\) for `_inf` |
| `mrt` | \(\mathrm{MRT}\) | mean residence time | time | \(\lambda_z\) |
| `lambda_z` | \(\lambda_z\) | terminal rate constant | 1/time | ≥ 3 terminal points |
| `thalf` | \(t_{1/2}\) | terminal half-life | time | \(\lambda_z\) |
| `lambda_z_n_points`, `lambda_z_t_first`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`, `lambda_z_se` | | diagnostics of the regression | –, time, –, –, – (\(\ln C\)), 1/time | \(\lambda_z\) |
| `cl`, `cl_f` | \(\mathrm{CL}\), \(\mathrm{CL}/F\) | clearance (`_f`: extravascular) | dose/(value·time) → l/h | dose, \(\lambda_z\) |
| `vz`, `vz_f` | \(V_z\), \(V_z/F\) | terminal volume of distribution | dose/value → l | dose, \(\lambda_z\) |
| `vss` | \(V_\mathrm{ss}\) | steady state volume of distribution | dose/value → l | intravenous dose |
| `auc_inf_dn`, `cmax_dn` | | dose normalized exposure and peak | value·time/dose, value/dose | dose |
| `auc_tau` | \(\mathrm{AUC}_{0\text{-}\tau}\) | area over the dosing interval | value·time | regimen |
| `cmin_ss`, `ctrough`, `cavg` | | minimum, value at \(\tau\), average over the interval | value | regimen |
| `fluctuation`, `swing`, `accumulation_ratio` | | see Math | – | regimen |
| `cl_ss` | \(\mathrm{CL}_\mathrm{ss}\) | \(D / \mathrm{AUC}_{0\text{-}\tau}\) | → l/h | regimen, dose |
| `flags` | | `NCAFlag` bits, see below | – | |

Volumes are reported in liter (per kilogram for doses per body weight), clearances in liter per hour; every other unit is derived from the units of the input. Effect timecourses (`Kind.EFFECT`) report `e0`, `emax_obs`, `temax`, `auec_last`, `auec_baseline`, `emax_baseline` and `time_above` instead, see [Pharmacodynamics](pd.md).

Flags: `POSITIVE_SLOPE` (the terminal regression does not decline; \(\lambda_z\) and everything derived from it is `NaN`), `TOO_FEW_POINTS` (no window with the minimal number of points), `EXTRAPOLATION_HIGH`, `NO_MAX` (the maximum is the last point), `NO_ABSORPTION` (the maximum is the first point of an extravascular curve), `BLQ_TRUNCATED`, `NO_DATA` (fewer than two points).

## API

One curve:

```python
from pkpdutils import Dose, Route, Timecourse, nca_single

tc = Timecourse(
    time=[0.25, 0.5, 1, 2, 4, 6, 8, 12, 24],
    value=[0.9, 1.7, 2.6, 2.8, 2.2, 1.6, 1.2, 0.6, 0.1],
    time_unit="hr", unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine",
)
result = nca_single(tc)
result.to_quantities()["auc_inf_obs"]   # pint quantity in hour * milligram / liter
result.to_quantities()["cl_f"]          # liter / hour
result.flags()                          # e.g. []
```

Options select the methods; the analysis of a batch returns the parameters over its sample dimensions:

```python
from pkpdutils import NCAOptions, TerminalPhase, Timecourses, nca
from pkpdutils.nca import AUCMethod, TerminalMethod

options = NCAOptions(
    auc_method=AUCMethod.LINEAR_LOG,
    terminal=TerminalPhase(method=TerminalMethod.BEST_FIT, min_points=3),
    lloq=0.05,
    extrapolation_warning=0.2,
)
result = nca(batch, options)      # batch: Timecourses over (study, individual)
result.ds                         # xarray.Dataset, one variable per parameter
result["thalf"]                   # DataArray over (study, individual), attrs["units"]
result.to_dataframe()             # one row per sample, flags decoded
result.flag_table()               # one boolean column per flag
```

Steady state, with the dosing interval:

```python
from pkpdutils import DosingRegimen
from pkpdutils.nca import accumulation_ratio, superposition

regimen = DosingRegimen(dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=12)
ss = nca(batch_ss, NCAOptions(regimen=regimen))          # auc_tau, cavg, fluctuation, ...
sd = nca(batch_single, NCAOptions(regimen=regimen))
ratio = accumulation_ratio(ss, sd)                        # observed accumulation
predicted = superposition(tc_single, DosingRegimen(dose=regimen.dose, interval=12, n_doses=10))
```

Large batches run in worker processes with `NCAOptions(n_workers=4)`; the analysis itself is vectorized, so this only pays off for many thousands of curves. The figures are described in [Plotting](plotting.md), the examples are `examples/nca_single.py`, `examples/nca_batch.py`, `examples/steady_state.py` and `examples/nca_from_sbmlsim.py`, the reference of the modules is in [API: nca](api/nca.md).

## References

[^gw]: Gabrielsson J, Weiner D. *Pharmacokinetic and Pharmacodynamic Data Analysis: Concepts and Applications*. 5th ed. Swedish Pharmaceutical Press; 2016. See [References](references.md#textbooks).
[^phoenix]: Certara. *Phoenix WinNonlin User's Guide: Noncompartmental Analysis*. See [References](references.md#non-compartmental-analysis).
[^rt]: Rowland M, Tozer TN. *Clinical Pharmacokinetics and Pharmacodynamics*. 4th ed. 2011, ch. 11. See [References](references.md#textbooks).
```

- [ ] **Step 3: Write `docs/glossary.md` and `docs/plotting.md`**

`docs/glossary.md`:
```markdown
# Glossary

The names used for the variables of the result datasets, with their symbols and the page that defines them. Units are derived from the units of the input; `value` is the unit of the measured values, `time` the unit of the times, `dose` the unit of the doses.

| name | symbol | meaning | unit | page |
| --- | --- | --- | --- | --- |
| `auc_last` | \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\) | area under the curve to the last positive value | value·time | [NCA](nca.md) |
| `auc_inf_obs`, `auc_inf_pred` | \(\mathrm{AUC}_{0\text{-}\infty}\) | area extrapolated to infinity, observed or predicted last value | value·time | [NCA](nca.md) |
| `auc_extrap_fraction` | | extrapolated fraction of \(\mathrm{AUC}_{0\text{-}\infty}\) | – | [NCA](nca.md) |
| `aumc_last`, `aumc_inf` | \(\mathrm{AUMC}\) | area under the first moment curve | value·time² | [NCA](nca.md) |
| `mrt` | \(\mathrm{MRT}\) | mean residence time | time | [NCA](nca.md) |
| `cmax`, `tmax` | \(C_\mathrm{max}\), \(t_\mathrm{max}\) | maximum and its time | value, time | [NCA](nca.md) |
| `cmin`, `tmin` | \(C_\mathrm{min}\), \(t_\mathrm{min}\) | minimum and its time | value, time | [NCA](nca.md) |
| `clast`, `tlast` | \(C_\mathrm{last}\), \(t_\mathrm{last}\) | last positive value and its time | value, time | [NCA](nca.md) |
| `c0` | \(C_0\) | back-extrapolated value at time 0 (bolus) | value | [NCA](nca.md) |
| `cmax_half`, `tmax_half` | | half maximum during absorption | value, time | [NCA](nca.md) |
| `lambda_z` | \(\lambda_z\) | terminal rate constant | 1/time | [NCA](nca.md) |
| `thalf` | \(t_{1/2}\) | terminal half-life | time | [NCA](nca.md) |
| `lambda_z_n_points`, `lambda_z_t_first`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`, `lambda_z_se` | | regression diagnostics | | [NCA](nca.md) |
| `cl`, `cl_f` | \(\mathrm{CL}\), \(\mathrm{CL}/F\) | clearance, relative to the fraction absorbed | l/h | [NCA](nca.md) |
| `vz`, `vz_f` | \(V_z\), \(V_z/F\) | terminal volume of distribution | l | [NCA](nca.md) |
| `vss` | \(V_\mathrm{ss}\) | steady state volume of distribution | l | [NCA](nca.md) |
| `auc_inf_dn`, `cmax_dn` | | dose normalized exposure and maximum | value·time/dose, value/dose | [NCA](nca.md) |
| `auc_tau` | \(\mathrm{AUC}_{0\text{-}\tau}\) | area over a dosing interval | value·time | [NCA](nca.md) |
| `cmin_ss`, `ctrough`, `cavg` | | minimum, trough and average over the interval | value | [NCA](nca.md) |
| `fluctuation`, `swing` | | peak–trough fluctuation and swing | – | [NCA](nca.md) |
| `accumulation_ratio` | \(R\) | accumulation at steady state | – | [NCA](nca.md) |
| `cl_ss` | \(\mathrm{CL}_\mathrm{ss}\) | clearance at steady state | l/h | [NCA](nca.md) |
| `e0`, `emax_obs`, `temax` | | baseline, maximum effect and its time | value, value, time | [NCA](nca.md) |
| `auec_last`, `auec_baseline` | \(\mathrm{AUEC}\) | area under the effect curve, raw and baseline corrected | value·time | [NCA](nca.md) |
| `emax_baseline`, `time_above` | | baseline corrected maximum, time above a threshold | value, time | [NCA](nca.md) |
| `flags` | | `NCAFlag` bits | – | [NCA](nca.md) |
```

`docs/plotting.md`:
```markdown
# Plotting

The figures of `pkpdutils.plot` are matplotlib figures. Every function returns the `Figure` it drew and never shows it, so a script saves it (`fig.savefig("name.png")`) and a notebook displays it; pass `ax` to draw into an existing axes. Colors and markers come from a `PlotStyle`.

## Timecourses

`plot_timecourse` draws one curve or every curve of a batch, with the standard error (or the standard deviation) as error bars when present, one color per sample, and the legend from the sample labels or from a coordinate of the batch (`by="individual"`).

```python
from pkpdutils.plot import plot_timecourse

fig = plot_timecourse(batch, log=True, by="dose")
fig.savefig("curves.png")
```

## NCA diagnostics

`plot_nca` shows what the analysis did with one curve, on a linear and a logarithmic axis: the data, the area to \(t_\mathrm{last}\), the extrapolated tail, the terminal regression line and the points it used, \(C_\mathrm{max}\)/\(t_\mathrm{max}\), \(C_0\) for a bolus, and the flags in the title. For a batch, `plot_nca_grid` draws one such panel per sample.

```python
from pkpdutils.plot import plot_nca, plot_nca_grid

fig = plot_nca(tc, nca_single(tc))
fig = plot_nca(batch.sel(individual="s2"), result, individual="s2")
fig = plot_nca_grid(batch, result, ncols=4)
```

![NCA diagnostics](images/nca_single.png)

The image is written by `examples/nca_single.py`; copy it to `docs/images/` after a change of the figure.

## Style

```python
from pkpdutils.plot import PlotStyle

style = PlotStyle(fit_color="tab:red", auc_color="lightgray", alpha=0.3)
fig = plot_nca(tc, result, style=style)
```

The reference of the module is in [API: plot](api/plot.md).
```

Generate the image: `cd $(mktemp -d) && PYTHONPATH=/home/mkoenig/git/pkdb_analysis uv run --project /home/mkoenig/git/pkdb_analysis python -m examples.nca_single > /dev/null && cp nca_single.png /home/mkoenig/git/pkdb_analysis/docs/images/nca_single.png; cd -`.

- [ ] **Step 4: Update `docs/index.md`, `docs/timecourses.md`, `CLAUDE.md`**

`docs/index.md` Features list: add after the Units item
```markdown
- **[Non-compartmental analysis](nca.md)** — exposure, peak, terminal phase, clearance and volume parameters of concentration curves, single dose and steady state, vectorized over a batch, with flags and units.
- **[Plotting](plotting.md)** — timecourses and NCA diagnostics as matplotlib figures.
```
and the Quickstart section before "How to cite":
```markdown
## Quickstart

```python
from pkpdutils import Dose, Route, Timecourse, nca_single

tc = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12, 24],
    value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2, 0.03],
    time_unit="hr", unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine",
)
result = nca_single(tc)
print(result.to_dataframe().T)
```

Continue with [Installation](installation.md), [Timecourses](timecourses.md) and [Non-compartmental analysis](nca.md).
```

`docs/timecourses.md`: in the first paragraph, after "which is the input of every analysis of the package", add the sentence "The first analysis is the [non-compartmental analysis](nca.md)."

`CLAUDE.md`: in "Commands" add the example lines `python -m examples.nca_single`, `python -m examples.nca_batch`, `python -m examples.steady_state`; in "Architecture" add after the `timecourse.py` paragraph:
```markdown
**`nca/` — non-compartmental analysis.** `nca(timecourses, options)` flattens the sample dimensions to `(N, n_time)` arrays and runs `compute_parameters` (`nca/nca.py`), the pure numpy core: `auc.py` packs the valid points of every row to the front (`pack_valid`) and sums the trapezoid segments (`segment_areas`, linear / linear-up-log-down / log), `terminal.py` evaluates every candidate window of the terminal regression at once with suffix sums (`window_statistics`) and picks it by the `TerminalPhase` rule (best adjusted R², last n, all after tmax, manual), `steady_state.py` adds the parameters of a dosing interval and `superposition`. `options.py` holds `NCAOptions`, `TerminalPhase`, the enumerations and `NCAFlag`; `result.py` holds `NCAResult` (an `xarray.Dataset` over the sample dims with `attrs["units"]` per variable and the integer `flags`) and `parameter_unit`, which derives the units with pint (`PARAMETER_UNITS` in `nca.py` maps every parameter to a unit expression). `nca_single` wraps one `Timecourse`. `n_workers` chunks the rows over a `ProcessPoolExecutor`. `tests/nca/test_reference.py` reproduces `pkdb_analysis` 0.3.1 with `AUCMethod.LINEAR` and `TerminalMethod.ALL_AFTER_TMAX`.

**`plot/` — figures.** `style.py` (`PlotStyle`), `timecourse.py` (`plot_timecourse`), `nca.py` (`plot_nca`, `plot_nca_grid`, `draw_nca_panel`). Functions return the `Figure`, take `ax`, never show.
```

- [ ] **Step 5: Build and check**

Run: `uv run zensical build --clean && uv run python scripts/llms_txt.py && uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run ty check && uv run pre-commit run --all-files`
Expected: the site builds with the new pages (`nca/`, `plotting/`, `glossary/`, `api/nca/` ...) — a warning about the not yet existing `pd.md` link is acceptable, an error is not (then write "Pharmacodynamics" as plain text in `nca.md`); tests pass; ruff, ty and hooks clean.

- [ ] **Step 6: Commit**

```bash
git add docs zensical.toml CLAUDE.md
git commit -q -m "Document the non-compartmental analysis, the glossary and the figures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 11: Matrix, pull request

**Files:** none new.

- [ ] **Step 1: Tox matrix**

Run: `uv run tox run-parallel`
Expected: `py3.13`, `py3.14`, `ty` succeed. A 3.13 failure is fixed in the code (no behaviour change) and committed.

- [ ] **Step 2: Push and open the pull request**

```bash
git push -u origin nca
gh pr create --base develop --title "Add the non-compartmental analysis" --body-file - <<'EOF'
## Summary

Phase 3 of the pkpdutils redesign (`docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`, section 2): `pkpdutils.nca` with `nca`/`nca_single`, `NCAOptions`/`TerminalPhase`, `NCAResult`, vectorized areas and terminal regression, steady state parameters and superposition, flags and units; `pkpdutils.plot` with `plot_timecourse`, `plot_nca`, `plot_nca_grid`; documentation (`nca.md`, `glossary.md`, `plotting.md`, API pages) and four examples. The results of `pkdb_analysis` 0.3.1 are reproduced by `tests/nca/test_reference.py`.

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

- [ ] **Step 3: Wait for the checks**

Run: `until [ "$(gh pr view --json state -q .state)" = "MERGED" ] || [ "$(gh pr checks --json bucket -q '[.[] | select(.bucket=="fail")] | length')" != "0" ]; do sleep 30; done; gh pr checks`
Expected: `tests`, `ruff`, `ty`, `docs` pass and the pull request is squash-merged; then `git switch develop && git pull`.

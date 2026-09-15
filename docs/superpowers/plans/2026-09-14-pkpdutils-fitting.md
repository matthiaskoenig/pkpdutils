# pkpdutils Fitting and Pharmacodynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the curve fitting engine (`pkpdutils.fit`) with the exponential, Bateman, Emax family, power and covariate models, standard errors, confidence intervals, bootstrap, model comparison and dose proportionality, plus the fit figures and the fitting/PD documentation (phase 5 of the spec).

**Architecture:** A `Model` is a small object with named `ModelParameter`s, `predict(x, p)`, `derived(p)` and `initial_guess(x, y)`. The engine (`fit/engine.py`) fits one model to one or many `(x, y)` rows with `scipy.optimize.least_squares` in a scaled parameter space (log10 for positive parameters), from one or several Latin hypercube start points, serially or in a process pool; it derives standard errors from the Jacobian, transforms t-based intervals back to the linear scale, computes the goodness-of-fit statistics and, on request, a residual bootstrap. Results are a `FitResult`, an `xarray.Dataset` over the sample dimensions built on the new shared `ParameterResult` base class that `NCAResult` also uses (`pkpdutils/result.py`): one variable per parameter plus `_se`, `_ci_low`, `_ci_high`, `_cv`, the derived parameters, the statistics, the data and the prediction per point, and `flags`. Front ends: `fit(model, x, y, ...)` for arrays, `fit_timecourses(model, timecourses, ...)` for a batch, `fit_table(model, ds, x, y, dim=...)` for a parameter against a covariate or dose across a sample dimension (dose proportionality, allometry). `compare_models` ranks models by AICc, `proportionality_test` applies the Smith criterion. Figures in `pkpdutils.plot.fit`.

**Tech Stack:** numpy, scipy (`optimize.least_squares`, `stats.t`, `stats.qmc.LatinHypercube`), xarray, pint, pydantic, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md` section "3. Curve fitting" and the plot items `plot_fit`, `plot_dose_proportionality`, `plot_goodness_of_fit` of section 5 (`plot_bland_altman`, `plot_parameters`, `plot_forest`, `plot_ratio` belong to the statistics plan). Plans 1-3 for the interfaces this plan consumes.

**Later plans:** statistics (6), release (7).

## Global Constraints

- package `pkpdutils`, python 3.13 and 3.14, runtime dependencies stay `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib`, `rich`
- ty `error-on-warning = true`: zero diagnostics on `src`, `tests`, `examples`, `scripts`; rule-specific `# ty: ignore[rule]` only, none unused; narrowing asserts in tests
- every module, class and function of `src/pkpdutils` has full type annotations and a google style docstring with the formula and a citation key of `docs/references.md` where a formula exists; `examples/`, `tests/`, `scripts/` exempt from `D`
- library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, never prints, never calls `plt.show()`
- test output pristine: `uv run pytest -q -W error` passes
- results are `xarray.Dataset` objects over the sample dimensions with `attrs["units"]` on every variable; parameter variables `p`, `p_se`, `p_ci_low`, `p_ci_high` (unit of `p`), `p_cv` (dimensionless, percent); derived parameters likewise; statistics `cost`, `r2`, `rmse`, `aic`, `aicc`, `bic`, `n_points`, `n_parameters`, `n_starts_converged`; data variables `x_data`, `y_data`, `y_pred`, `residuals` over `(*sample_dims, "point")`; `correlation` over `(*sample_dims, "parameter", "parameter_")`; `flags` (`FitFlag` bits `NOT_CONVERGED = 1`, `AT_BOUND = 2`, `TOO_FEW_POINTS = 4`, `FLIP_FLOP = 8`, `SINGULAR = 16`, `NO_DATA = 32`)
- weighting is the variance model of the residuals: `NONE` (constant), `INV_Y` (variance proportional to `y`), `INV_Y2` (proportional to `y²`), `INV_SD` (variance `sd²`); the weighted residual is `(y - f) / sqrt(var)`
- parameter scales: `LOG10` (default) and `LOG` apply to parameters with `positive=True`; other parameters stay linear; `LINEAR` applies to all
- text rules: never write the em dash character (U+2014), use "-"; commit messages carry NO `Co-Authored-By` line and end with the single line `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`; markdown has no hard line wraps; formulas `\(...\)`/`\[...\]`
- `develop` accepts only pull requests; work happens on the branch `fitting` off `develop`, merged with the checks `tests`, `ruff`, `ty`, `docs`
- commands run from the repository root with `uv run`

## Interfaces of plans 1-3 this plan consumes

- `pkpdutils.timecourse.Timecourses`: `times`, `values`, `sd`, `se`, `n`, `dose_amount`, `dose_time`, `dose_duration`, `dose_unit`, `route`, `sample_dims`, `sample_shape`, `n_samples`, `n_time`, `time_unit`, `unit`, `substance`, `ds`; `Timecourse` with `time`, `value`, `sd`, `se`, `time_unit`, `unit`, `dose`, `relative_to_dose()`
- `pkpdutils.nca.result.NCAResult(ds)`: `sample_dims`, `parameters`, `derived_variables`, `has_uncertainty`, `units`, `__getitem__`, `__contains__`, `to_quantities`, `flags`, `to_dataframe`, `flag_table`, `summarize`; `parameter_unit(expression, *, unit, time_unit, dose_unit)`
- `pkpdutils.nca.uncertainty`: `LOGNORMAL_PARAMETERS`, `DISCRETE_PARAMETERS`, `base_name`, `UNCERTAINTY_SUFFIXES`, `SUMMARY_SUFFIXES`
- `pkpdutils.nca.options.NCAFlag`, `decode_flags`
- `pkpdutils.units`: `ureg`, `Q_`, `Quantity`, `parse_unit`, `normalize_volume`, `normalize_clearance`
- `pkpdutils.plot`: `PlotStyle`, `DEFAULT_STYLE`, `_figure_of(ax)` in `plot/timecourse.py`

---

### Task 1: Branch, the shared `ParameterResult` base class, and the citation link artifact

**Files:**
- Create: `src/pkpdutils/result.py`
- Modify: `src/pkpdutils/nca/result.py` (`NCAResult` subclasses `ParameterResult`), `src/pkpdutils/nca/uncertainty.py` (no change of names; `base_name` stays there), `README.md`, `docs/index.md`
- Test: `tests/test_result.py`

**Interfaces:**
- Produces `pkpdutils.result.ParameterResult`: generic container of a dataset over sample dimensions with `attrs["units"]` per variable and an integer `flags` variable. Class attributes `flag_type: type[IntFlag]` (decoder of `flags`), `lognormal_parameters: frozenset[str]` (geometric statistics in `summarize`), `discrete_parameters: frozenset[str]`; `__init__(ds)` validates `flags` and units; properties `ds`, `sample_dims`, `parameters` (data variables that are not `flags`, `n`, derived variables (`base_name` is not `None`) or data-point variables (variables with a dimension outside the sample dims)), `derived_variables`, `has_uncertainty`, `point_variables` (variables with extra dimensions); methods `units(name)`, `__getitem__`, `__contains__`, `to_quantities(**indexers)` (scalar variables only), `flags(**indexers) -> list[str]`, `to_dataframe()` (scalar variables), `flag_table()`, `summarize(dim, ci_level=0.95)` (as in plan 3, using the class attributes; point variables dropped). `decode_flags(value)` becomes a method using `flag_type`.
- `NCAResult(ParameterResult)` with `flag_type = NCAFlag`, `lognormal_parameters = LOGNORMAL_PARAMETERS`, `discrete_parameters = DISCRETE_PARAMETERS`; behaviour unchanged (all existing tests pass without modification, except imports).

- [ ] **Step 1: Branch and the link artifact**

```bash
git switch fitting   # the branch exists and holds this plan
```
In `README.md:30` and `docs/index.md:47` the citation line contains `[Computer software]`, which markdown reads as a link reference. Replace `[Computer software]` by `\[Computer software\]` in both files (`sed -i 's/ \[Computer software\]/ \\[Computer software\\]/' README.md docs/index.md`). Run `uv run zensical build --clean 2>&1 | grep -c "Computer software"` → `0`.

```bash
git add README.md docs/index.md
git commit -q -m "Escape the citation brackets that markdown read as a link reference

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

- [ ] **Step 2: Write the failing test**

`tests/test_result.py`:
```python
from enum import IntFlag

import numpy as np
import pytest
import xarray as xr

from pkpdutils.nca import NCAResult
from pkpdutils.result import ParameterResult


class MyFlag(IntFlag):
    NONE = 0
    BAD = 1
    WORSE = 2


class MyResult(ParameterResult):
    flag_type = MyFlag
    lognormal_parameters = frozenset({"a"})
    discrete_parameters = frozenset({"k"})


def make() -> MyResult:
    ds = xr.Dataset(
        {
            "a": (("s",), np.array([1.0, 2.0, 4.0]), {"units": "mg"}),
            "a_se": (("s",), np.array([0.1, 0.2, 0.4]), {"units": "mg"}),
            "k": (("s",), np.array([3.0, 3.0, 4.0]), {"units": "dimensionless"}),
            "y_pred": (("s", "point"), np.ones((3, 2)), {"units": "mg"}),
            "n": (("s",), np.array([5.0, 5.0, 5.0]), {"units": "dimensionless"}),
            "flags": (("s",), np.array([0, 1, 3]), {"units": "dimensionless"}),
        },
        coords={"s": ["x", "y", "z"]},
    )
    return MyResult(ds)


def test_generic_result_classification() -> None:
    r = make()
    assert r.sample_dims == ("s",)
    assert r.parameters == ["a", "k"]
    assert r.derived_variables == ["a_se"]
    assert r.point_variables == ["y_pred"]
    assert r.has_uncertainty
    assert r.flags(s="z") == ["BAD", "WORSE"]
    assert set(r.to_quantities(s="x")) == {"a", "a_se", "k", "n"}
    assert list(r.to_dataframe().columns) == ["s", "a", "a_se", "k", "n", "flags"]
    assert r.flag_table()["WORSE"].tolist() == [False, False, True]


def test_generic_summarize_uses_class_sets() -> None:
    s = make().summarize("s")
    q = s.to_quantities()
    assert q["a"].magnitude == pytest.approx(7.0 / 3.0)
    assert q["a_geomean"].magnitude == pytest.approx(2.0)
    assert "k_geomean" not in s
    assert "y_pred" not in s
    assert s.flags() == ["BAD", "WORSE"]


def test_nca_result_is_parameter_result() -> None:
    assert issubclass(NCAResult, ParameterResult)
    assert NCAResult.flag_type.__name__ == "NCAFlag"
```

Run: `uv run pytest tests/test_result.py -n 0 -q` → FAIL with `ModuleNotFoundError: No module named 'pkpdutils.result'`.

- [ ] **Step 3: Extract the base class**

Create `src/pkpdutils/result.py` by moving the body of `NCAResult` from `src/pkpdutils/nca/result.py` into `class ParameterResult` with these changes:
- module docstring: "Shared container of parameter results (`NCAResult`, `FitResult`): an `xarray.Dataset` over sample dimensions, units per variable, an integer `flags` variable, quantities, data frames and summaries."
- class attributes with docstring comments:
```python
    #: the IntFlag type that decodes the `flags` variable
    flag_type: ClassVar[type[IntFlag]] = IntFlag
    #: parameters reported with geometric statistics in `summarize`
    lognormal_parameters: ClassVar[frozenset[str]] = frozenset()
    #: parameters without uncertainty variables
    discrete_parameters: ClassVar[frozenset[str]] = frozenset()
```
- `decode_flags(self, value: int) -> list[str]`: `[f.name for f in self.flag_type if f.value and value & f.value and f.name]` (write the comprehension so that ty does not report a redundant condition: `f.name is not None`).
- `point_variables` property: variables with any dimension not in `sample_dims`; `parameters` excludes them; `_variables` (scalar variables except `flags`) excludes them; `to_dataframe` uses `_variables`; `summarize` skips them and uses `self.lognormal_parameters`; the returned object is `type(self)(...)`.
- `summarize` builds the dataset and returns `self._new(ds)`; `_new(self, ds) -> Self` returns `type(self)(ds)` and is the hook subclasses with extra constructor arguments override; annotate with `typing.Self`.
- Import `base_name` from `pkpdutils.nca.uncertainty` stays (no cycle: `uncertainty` imports `options` and `timecourse`).

`src/pkpdutils/nca/result.py` keeps `parameter_unit` and becomes:
```python
class NCAResult(ParameterResult):
    """Parameters of a non-compartmental analysis as an `xarray.Dataset`.

    One variable per parameter over the sample dimensions of the analysed
    `Timecourses`, `attrs["units"]` on every variable, the uncertainty
    variables of `pkpdutils.nca.uncertainty` and the integer variable `flags`
    (`NCAFlag`). See `pkpdutils.result.ParameterResult` for the interface.
    """

    flag_type = NCAFlag
    lognormal_parameters = LOGNORMAL_PARAMETERS
    discrete_parameters = DISCRETE_PARAMETERS
```
Keep the `decode_flags` import out of `nca/result.py` if unused. `pkpdutils/nca/__init__.py` still exports `NCAResult`.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest -q -W error && uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: all existing tests pass unchanged (NCA behaviour identical) plus the three new ones. ty: `ClassVar[type[IntFlag]]` assigned `NCAFlag` is fine; if ty complains about `IntFlag` iteration typing, annotate `f` explicitly.

```bash
git add src/pkpdutils/result.py src/pkpdutils/nca/result.py tests/test_result.py
git commit -q -m "Extract the shared ParameterResult base class of the result datasets

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 2: Model protocol, parameters, units and fit options

**Files:**
- Create: `src/pkpdutils/fit/__init__.py`, `src/pkpdutils/fit/model.py`, `src/pkpdutils/fit/options.py`
- Test: `tests/fit/__init__.py` (empty), `tests/fit/test_model.py`, `tests/fit/test_fit_options.py`

**Interfaces:**
- `fit/model.py`:
  - `@dataclass(frozen=True) class ModelParameter`: `name: str`, `unit_expr: str` (`"[y]"`, `"[x]"`, `"1/[x]"`, `"[y]/[x]"`, `"dimensionless"`, ...), `lower: float = -inf`, `upper: float = inf`, `positive: bool = True` (log scale applies), `description: str = ""`
  - `class Model(ABC)`: `name: str` (class attribute), `parameters: tuple[ModelParameter, ...]` (property or class attribute), `parameter_names` property, `n_parameters`, abstract `predict(x: np.ndarray, p: np.ndarray) -> np.ndarray` (vectorized over `x`, `p` a 1-D array in the order of `parameters`), `derived(p: np.ndarray) -> dict[str, float]` (default `{}`), `derived_units: dict[str, str]` (unit expressions of the derived parameters, default `{}`), abstract `initial_guess(x: np.ndarray, y: np.ndarray) -> np.ndarray`, `bounds() -> tuple[np.ndarray, np.ndarray]`, `parameter(name) -> ModelParameter`, `__repr__`
  - `parameter_unit_expression(unit_expr: str, *, x_unit: str, y_unit: str) -> tuple[str, float]`: replaces `[x]`/`[y]` by the units and returns the canonical unit string and the conversion factor (via `pkpdutils.units`, like `parameter_unit` of the NCA; volumes/clearances normalized the same way)
- `fit/options.py`: `ParameterScale(StrEnum)`: `LOG10`, `LOG`, `LINEAR`; `Weighting(StrEnum)`: `NONE`, `INV_Y`, `INV_Y2`, `INV_SD`; `FitFlag(IntFlag)` with the bits of the Global Constraints; `decode_fit_flags(value) -> list[str]`; `FitOptions` (frozen pydantic): `parameter_scale = LOG10`, `weighting = NONE`, `loss: str = "linear"` (scipy names), `n_starts: int = 1 (ge 1)`, `seed: int | None = None`, `n_workers: int | None = None (ge 1)`, `ci_level = 0.95`, `bootstrap: int = 0 (ge 0)`, `max_nfev: int | None = None`, `ftol = 1e-10`, `xtol = 1e-10`, `gtol = 1e-10`, `fixed: dict[str, float] = {}` (parameters held at a value), `bounds: dict[str, tuple[float, float]] = {}` (override), `initial: dict[str, float] = {}` (override), `start_spread: float = 100.0 (gt 1)` (multi-start box: `initial/spread .. initial*spread` for positive parameters, `initial ± spread*|initial|` (or `±spread` when `initial == 0`) for linear ones), `at_bound_tolerance: float = 1e-3 (relative)`; `FitOptions.scale_of(parameter: ModelParameter) -> ParameterScale` (LINEAR for non-positive parameters).
- `pkpdutils.fit` exports (extended by later tasks): `Model`, `ModelParameter`, `FitOptions`, `ParameterScale`, `Weighting`, `FitFlag`, `decode_fit_flags`.

- [ ] **Step 1: Write the failing tests**

`tests/fit/test_model.py`:
```python
import numpy as np
import pytest

from pkpdutils.fit.model import Model, ModelParameter, parameter_unit_expression


class Line(Model):
    name = "line"
    parameters = (
        ModelParameter("a", "[y]", positive=False, description="intercept"),
        ModelParameter("b", "[y]/[x]", lower=0.0, description="slope"),
    )
    derived_units = {"x_zero": "[x]"}

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x

    def derived(self, p: np.ndarray) -> dict[str, float]:
        return {"x_zero": -p[0] / p[1]}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([y[0], (y[-1] - y[0]) / (x[-1] - x[0])])


def test_model_metadata() -> None:
    m = Line()
    assert m.parameter_names == ("a", "b")
    assert m.n_parameters == 2
    assert m.parameter("b").lower == 0.0
    lower, upper = m.bounds()
    np.testing.assert_array_equal(lower, [-np.inf, 0.0])
    np.testing.assert_array_equal(upper, [np.inf, np.inf])
    with pytest.raises(KeyError):
        m.parameter("c")
    assert "line" in repr(m)


def test_model_predict_and_derived() -> None:
    m = Line()
    p = np.array([1.0, 2.0])
    np.testing.assert_allclose(m.predict(np.array([0.0, 1.0]), p), [1.0, 3.0])
    assert m.derived(p) == {"x_zero": -0.5}
    np.testing.assert_allclose(m.initial_guess(np.array([0.0, 2.0]), np.array([1.0, 5.0])), [1.0, 2.0])


def test_parameter_unit_expression() -> None:
    unit, factor = parameter_unit_expression("[y]/[x]", x_unit="hr", y_unit="mg/l")
    assert unit == "milligram / hour / liter"
    assert factor == pytest.approx(1.0)
    unit, factor = parameter_unit_expression("1/[x]", x_unit="min", y_unit="mg/l")
    assert unit == "1 / minute"
    assert parameter_unit_expression("dimensionless", x_unit="hr", y_unit="mg/l")[0] == "dimensionless"
    unit, factor = parameter_unit_expression("[y]*[x]", x_unit="hr", y_unit="ng/ml")
    assert unit == "hour * nanogram / milliliter"
```

`tests/fit/test_fit_options.py`:
```python
import pytest

from pkpdutils.fit.model import ModelParameter
from pkpdutils.fit.options import FitFlag, FitOptions, ParameterScale, Weighting, decode_fit_flags


def test_defaults() -> None:
    o = FitOptions()
    assert o.parameter_scale is ParameterScale.LOG10
    assert o.weighting is Weighting.NONE
    assert o.loss == "linear"
    assert o.n_starts == 1 and o.seed is None and o.n_workers is None
    assert o.ci_level == pytest.approx(0.95) and o.bootstrap == 0
    assert o.fixed == {} and o.bounds == {} and o.initial == {}
    assert o.start_spread == pytest.approx(100.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        FitOptions(n_starts=0)
    with pytest.raises(ValueError):
        FitOptions(bootstrap=-1)
    with pytest.raises(ValueError):
        FitOptions(start_spread=1.0)
    with pytest.raises(ValueError):
        FitOptions(loss="quadratic")
    with pytest.raises(ValueError):
        FitOptions(bounds={"a": (2.0, 1.0)})


def test_scale_of() -> None:
    pos = ModelParameter("k", "1/[x]")
    lin = ModelParameter("e0", "[y]", positive=False)
    assert FitOptions().scale_of(pos) is ParameterScale.LOG10
    assert FitOptions().scale_of(lin) is ParameterScale.LINEAR
    assert FitOptions(parameter_scale=ParameterScale.LINEAR).scale_of(pos) is ParameterScale.LINEAR
    assert FitOptions(parameter_scale=ParameterScale.LOG).scale_of(pos) is ParameterScale.LOG


def test_flags() -> None:
    assert decode_fit_flags(int(FitFlag.NOT_CONVERGED | FitFlag.FLIP_FLOP)) == ["NOT_CONVERGED", "FLIP_FLOP"]
    assert FitFlag.NO_DATA == 32
```

Run: `uv run pytest tests/fit -n 0 -q` → FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Write the modules**

`src/pkpdutils/fit/__init__.py`:
```python
"""Curve fitting of timecourses and parameters.

`fit`, `fit_timecourses` and `fit_table` fit a `Model` from
`pkpdutils.fit.models` to data with `scipy.optimize.least_squares`, see
`docs/fitting.md`; `FitOptions` selects the scale, the weighting and the
multi-start and bootstrap settings, `FitResult` holds the parameters.
"""

from pkpdutils.fit.model import Model, ModelParameter, parameter_unit_expression
from pkpdutils.fit.options import FitFlag, FitOptions, ParameterScale, Weighting, decode_fit_flags

__all__ = [
    "FitFlag",
    "FitOptions",
    "Model",
    "ModelParameter",
    "ParameterScale",
    "Weighting",
    "decode_fit_flags",
    "parameter_unit_expression",
]
```

`src/pkpdutils/fit/model.py`:
```python
"""Models of the curve fitting.

A `Model` names its parameters (`ModelParameter`: name, unit expression,
bounds, whether it is positive and therefore fitted on the log scale) and
computes the curve `predict(x, p)`, the derived parameters `derived(p)` and an
`initial_guess(x, y)`. The unit of a parameter is derived from the units of `x`
and `y` with `parameter_unit_expression`, e.g. `"[y]/[x]"` for a slope.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from pkpdutils.units import Q_, normalize_clearance, normalize_volume, ureg


@dataclass(frozen=True)
class ModelParameter:
    """A parameter of a model.

    Attributes:
        name: name of the parameter, the variable name in the result
        unit_expr: unit expression with `[x]` and `[y]` for the units of the data,
            e.g. `"[y]"`, `"1/[x]"`, `"[y]/[x]"`, `"dimensionless"`
        lower: lower bound (linear scale)
        upper: upper bound (linear scale)
        positive: whether the parameter is positive and fitted on the log scale
        description: one line for the documentation
    """

    name: str
    unit_expr: str
    lower: float = -math.inf
    upper: float = math.inf
    positive: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        """Check the bounds and the positivity."""
        if self.upper <= self.lower:
            raise ValueError(f"'{self.name}': upper bound {self.upper} <= lower bound {self.lower}")
        if self.positive and self.lower < 0:
            object.__setattr__(self, "lower", 0.0)


class Model(ABC):
    """A curve `y = f(x; p)` with named parameters.

    Subclasses set `name`, `parameters` (and optionally `derived_units`) and
    implement `predict`, `initial_guess` and, when they report derived
    parameters, `derived`. Parameters are passed as a 1-D array in the order
    of `parameters`.
    """

    #: name of the model in results and tables
    name: ClassVar[str] = "model"
    #: the parameters, in the order of the parameter vector
    parameters: ClassVar[tuple[ModelParameter, ...]] = ()
    #: unit expressions of the derived parameters
    derived_units: ClassVar[dict[str, str]] = {}

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the parameters in order."""
        return tuple(p.name for p in self.parameters)

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def parameter(self, name: str) -> ModelParameter:
        """The parameter of a name.

        Raises:
            KeyError: for an unknown name.
        """
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"{self.name} has no parameter '{name}'")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds as arrays."""
        return (
            np.array([p.lower for p in self.parameters], dtype=np.float64),
            np.array([p.upper for p in self.parameters], dtype=np.float64),
        )

    @abstractmethod
    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve at `x` for the parameters `p`."""

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """Derived parameters of `p`, empty by default."""
        return {}

    @abstractmethod
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """A start vector from the data (finite points only)."""

    def __repr__(self) -> str:
        """`Model(name, parameters)`."""
        return f"{type(self).__name__}({self.name}, {', '.join(self.parameter_names)})"


def parameter_unit_expression(unit_expr: str, *, x_unit: str, y_unit: str) -> tuple[str, float]:
    """Unit of a parameter from its expression and the units of the data.

    Args:
        unit_expr: expression with `[x]` and `[y]`, e.g. `"[y]/[x]"`
        x_unit: unit of the independent variable
        y_unit: unit of the dependent variable

    Returns:
        The canonical unit string (volumes in liter, clearances in liter per
        hour, like the NCA) and the factor from the raw to the canonical unit.
    """
    raw = unit_expr.replace("[x]", f"({x_unit})").replace("[y]", f"({y_unit})")
    quantity = Q_(1.0, ureg.parse_units(raw))
    converted = normalize_clearance(normalize_volume(quantity))
    return str(converted.units), float(converted.magnitude)
```
(`field` unused: drop the import.)

`src/pkpdutils/fit/options.py`:
```python
"""Options and flags of the curve fitting."""

from enum import IntFlag, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.fit.model import ModelParameter

#: loss functions of `scipy.optimize.least_squares`
LOSSES: tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy", "arctan")


class ParameterScale(StrEnum):
    """Space the optimizer searches; bounds, start values and results stay linear."""

    #: `log10(p)` for positive parameters
    LOG10 = "log10"
    #: `ln(p)` for positive parameters
    LOG = "log"
    #: no transformation
    LINEAR = "linear"


class Weighting(StrEnum):
    """Variance model of the residuals; the weighted residual is `(y - f) / sqrt(var)`."""

    #: constant variance
    NONE = "none"
    #: variance proportional to `y`
    INV_Y = "inv_y"
    #: variance proportional to `y²` (constant CV)
    INV_Y2 = "inv_y2"
    #: variance `sd²` from the data
    INV_SD = "inv_sd"


class FitFlag(IntFlag):
    """Conditions reported per sample in the `flags` variable of a fit result."""

    NONE = 0
    #: the optimizer did not report convergence from any start
    NOT_CONVERGED = 1
    #: a parameter ended within `at_bound_tolerance` of a bound
    AT_BOUND = 2
    #: fewer points than parameters + 1
    TOO_FEW_POINTS = 4
    #: the absorption rate is smaller than the elimination rate (Bateman)
    FLIP_FLOP = 8
    #: the Jacobian is singular, no standard errors
    SINGULAR = 16
    #: fewer than two finite points
    NO_DATA = 32


def decode_fit_flags(value: int) -> list[str]:
    """Names of the flags set in an integer value, in bit order."""
    return [f.name for f in FitFlag if f.value and value & f.value and f.name is not None]


class FitOptions(BaseModel):
    """Options of a fit.

    Attributes:
        parameter_scale: space of the search for positive parameters
        weighting: variance model of the residuals
        loss: loss function of `scipy.optimize.least_squares`
        n_starts: number of start points (Latin hypercube in the start box)
        seed: seed of the start point sampling and the bootstrap
        n_workers: worker processes for many samples or starts, `None` for the calling process
        ci_level: level of the confidence intervals
        bootstrap: number of residual bootstrap replicates, 0 for none
        max_nfev: maximal function evaluations per start, `None` for the scipy default
        ftol: scipy `ftol`
        xtol: scipy `xtol`
        gtol: scipy `gtol`
        fixed: parameters held at a value (not fitted)
        bounds: bounds overriding the model's, per parameter
        initial: start values overriding the model's guess, per parameter
        start_spread: half width of the start box around the initial guess, as a
            factor for log scale parameters and as a multiple of the guess for linear ones
        at_bound_tolerance: relative distance to a bound that sets `AT_BOUND`
    """

    model_config = ConfigDict(frozen=True)

    parameter_scale: ParameterScale = ParameterScale.LOG10
    weighting: Weighting = Weighting.NONE
    loss: str = "linear"
    n_starts: int = Field(default=1, ge=1)
    seed: int | None = None
    n_workers: int | None = Field(default=None, ge=1)
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap: int = Field(default=0, ge=0)
    max_nfev: int | None = Field(default=None, ge=1)
    ftol: float = Field(default=1e-10, gt=0.0)
    xtol: float = Field(default=1e-10, gt=0.0)
    gtol: float = Field(default=1e-10, gt=0.0)
    fixed: dict[str, float] = Field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    initial: dict[str, float] = Field(default_factory=dict)
    start_spread: float = Field(default=100.0, gt=1.0)
    at_bound_tolerance: float = Field(default=1e-3, gt=0.0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.loss not in LOSSES:
            raise ValueError(f"'loss' must be one of {LOSSES}, not '{self.loss}'")
        for name, (lower, upper) in self.bounds.items():
            if upper <= lower:
                raise ValueError(f"bounds of '{name}': upper {upper} <= lower {lower}")
        return self

    def scale_of(self, parameter: ModelParameter) -> ParameterScale:
        """The scale a parameter is searched on: linear unless it is positive."""
        if not parameter.positive:
            return ParameterScale.LINEAR
        return self.parameter_scale
```

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/fit -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. ty may want `parameters: ClassVar[tuple[ModelParameter, ...]]` assignments in subclasses without annotation; the test's `Line` assigns plain tuples, which is fine.

```bash
git add src/pkpdutils/fit tests/fit
git commit -q -m "Add the model protocol and the options of the curve fitting

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 3: Exponential models (`fit/models_exponential.py`)

**Files:**
- Create: `src/pkpdutils/fit/models_exponential.py`, `src/pkpdutils/fit/models.py` (re-export module, extended in Task 4)
- Test: `tests/fit/test_models_exponential.py`

**Interfaces:**
- `MonoExp`: `a` (`[y]`), `k` (`1/[x]`); `y = a·exp(-k x)`; derived `thalf = ln2/k`, `auc = a/k`; guess: `k` from the log-linear regression of the positive points (slope), `a` from its intercept (fallback: `a = max y`, `k = ln2 / (range/3)`)
- `BiExp`: `a1`, `k1`, `a2`, `k2` (`k1 > k2`: fast and slow phase); `y = a1·exp(-k1 x) + a2·exp(-k2 x)`; derived `lambda_z = k2`, `thalf_1`, `thalf_2`, `auc = a1/k1 + a2/k2`; guess by curve stripping (Gibaldi & Perrier 1982, ch. 2): terminal regression on the last half of the positive points → `a2`, `k2`; residuals `y - a2·exp(-k2 x)` of the first half (positive ones) → `a1`, `k1`; fallback `k1 = 5·k2`, `a1 = a2`. After a fit the phases are ordered (`sort_parameters(p)` swaps so that `k1 >= k2`); the engine calls `sort_parameters` when the model defines it.
- `TriExp`: `a1..a3`, `k1..k3` ordered, derived `lambda_z = k3`, `thalf_1..3`, `auc`; guess: strip twice.
- `Bateman(lag: bool = False)`: `a` (`[y]`), `ka` (`1/[x]`), `ke` (`1/[x]`) and `tlag` (`[x]`, `positive=False`, `lower=0`) when `lag`; `y = a·ka/(ka - ke)·(exp(-ke τ) - exp(-ka τ))`, `τ = max(x - tlag, 0)`; for `|ka - ke| < 1e-9` the limit `a·ka·τ·exp(-ke τ)`; derived `tmax = ln(ka/ke)/(ka - ke) + tlag`, `cmax = y(tmax)`, `thalf = ln2/ke`, `auc = a/ke`, `flip_flop` (1.0 when `ka < ke` else 0.0, dimensionless) - the engine sets `FitFlag.FLIP_FLOP` when a derived `flip_flop` is 1; guess: `ke` from the terminal regression after the maximum, `ka = 5·ke` (or from the rising points: slope of `ln(y_max - y)` when it exists), `a = y_max·(ka - ke)/ka / (exp(-ke tmax) - exp(-ka tmax))`, `tlag = 0`.
- `models.py`: `from pkpdutils.fit.models_exponential import Bateman, BiExp, MonoExp, TriExp` (+ Task 4 models), `__all__`.
- helper in `models_exponential.py`: `log_linear_regression(x, y) -> tuple[float, float]` (slope, intercept of `ln y` on `x` over positive finite points; `NaN`s when fewer than 2) and `terminal_guess(x, y) -> tuple[float, float]` (`a`, `k` from the last half of the points after the maximum), reused by Task 4's guesses.

- [ ] **Step 1: Write the failing tests**

`tests/fit/test_models_exponential.py`:
```python
import numpy as np
import pytest

from pkpdutils.fit.models import Bateman, BiExp, MonoExp, TriExp
from pkpdutils.fit.models_exponential import log_linear_regression, terminal_guess

T = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])


def test_log_linear_regression() -> None:
    slope, intercept = log_linear_regression(T, 5.0 * np.exp(-0.3 * T))
    assert slope == pytest.approx(-0.3)
    assert intercept == pytest.approx(np.log(5.0))
    s, i = log_linear_regression(np.array([1.0]), np.array([2.0]))
    assert np.isnan(s) and np.isnan(i)
    s, _ = log_linear_regression(T, np.where(T > 4, 0.0, 5.0 * np.exp(-0.3 * T)))
    assert s == pytest.approx(-0.3)


def test_monoexp() -> None:
    m = MonoExp()
    assert m.name == "monoexp" and m.parameter_names == ("a", "k")
    p = np.array([5.0, 0.3])
    np.testing.assert_allclose(m.predict(T, p), 5.0 * np.exp(-0.3 * T))
    d = m.derived(p)
    assert d["thalf"] == pytest.approx(np.log(2) / 0.3) and d["auc"] == pytest.approx(5.0 / 0.3)
    assert m.derived_units == {"thalf": "[x]", "auc": "[y]*[x]"}
    guess = m.initial_guess(T, 5.0 * np.exp(-0.3 * T))
    np.testing.assert_allclose(guess, p, rtol=1e-6)


def test_biexp_predict_strip_and_sort() -> None:
    m = BiExp()
    assert m.parameter_names == ("a1", "k1", "a2", "k2")
    p = np.array([8.0, 2.0, 2.0, 0.2])
    y = m.predict(T, p)
    np.testing.assert_allclose(y, 8 * np.exp(-2 * T) + 2 * np.exp(-0.2 * T))
    guess = m.initial_guess(T, y)
    assert guess[3] == pytest.approx(0.2, rel=0.3)
    assert guess[1] > guess[3]
    d = m.derived(p)
    assert d["lambda_z"] == pytest.approx(0.2) and d["thalf_2"] == pytest.approx(np.log(2) / 0.2)
    assert d["auc"] == pytest.approx(8 / 2 + 2 / 0.2)
    swapped = m.sort_parameters(np.array([2.0, 0.2, 8.0, 2.0]))
    np.testing.assert_allclose(swapped, p)


def test_triexp() -> None:
    m = TriExp()
    assert m.parameter_names == ("a1", "k1", "a2", "k2", "a3", "k3")
    p = np.array([10.0, 5.0, 4.0, 1.0, 1.0, 0.1])
    y = m.predict(T, p)
    guess = m.initial_guess(T, y)
    assert guess.shape == (6,) and np.all(np.isfinite(guess)) and guess[5] < guess[3] < guess[1]
    assert m.derived(p)["lambda_z"] == pytest.approx(0.1)
    np.testing.assert_allclose(m.sort_parameters(np.array([1.0, 0.1, 10.0, 5.0, 4.0, 1.0])), p)


def test_bateman_with_and_without_lag() -> None:
    m = Bateman()
    assert m.parameter_names == ("a", "ka", "ke")
    p = np.array([10.0, 2.0, 0.3])
    y = m.predict(T, p)
    expected = 10 * 2 / (2 - 0.3) * (np.exp(-0.3 * T) - np.exp(-2 * T))
    np.testing.assert_allclose(y, expected)
    d = m.derived(p)
    tmax = np.log(2 / 0.3) / (2 - 0.3)
    assert d["tmax"] == pytest.approx(tmax)
    assert d["cmax"] == pytest.approx(10 * 2 / 1.7 * (np.exp(-0.3 * tmax) - np.exp(-2 * tmax)))
    assert d["thalf"] == pytest.approx(np.log(2) / 0.3) and d["auc"] == pytest.approx(10 / 0.3)
    assert d["flip_flop"] == 0.0
    assert m.derived(np.array([10.0, 0.2, 0.5]))["flip_flop"] == 1.0
    guess = m.initial_guess(T, y)
    assert guess[2] == pytest.approx(0.3, rel=0.2)
    assert guess[1] > guess[2]
    lag = Bateman(lag=True)
    assert lag.parameter_names == ("a", "ka", "ke", "tlag")
    assert not lag.parameter("tlag").positive and lag.parameter("tlag").lower == 0.0
    y_lag = lag.predict(T, np.array([10.0, 2.0, 0.3, 1.0]))
    assert y_lag[T <= 1.0].max() == 0.0
    np.testing.assert_allclose(y_lag[T > 1.0], 10 * 2 / 1.7 * (np.exp(-0.3 * (T[T > 1] - 1)) - np.exp(-2 * (T[T > 1] - 1))))
    assert lag.derived(np.array([10.0, 2.0, 0.3, 1.0]))["tmax"] == pytest.approx(tmax + 1.0)


def test_bateman_ka_equals_ke_limit() -> None:
    m = Bateman()
    y = m.predict(np.array([1.0, 2.0]), np.array([10.0, 0.5, 0.5]))
    np.testing.assert_allclose(y, 10 * 0.5 * np.array([1.0, 2.0]) * np.exp(-0.5 * np.array([1.0, 2.0])))


def test_terminal_guess() -> None:
    a, k = terminal_guess(T, 5.0 * np.exp(-0.3 * T))
    assert k == pytest.approx(0.3) and a == pytest.approx(5.0)
```

Run: `uv run pytest tests/fit/test_models_exponential.py -n 0 -q` → FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Write the module**

`src/pkpdutils/fit/models_exponential.py`:
```python
"""Exponential models of concentration timecourses.

Sums of exponentials describe the decline of a concentration after an
intravenous dose and, with an absorption term, after an extravascular dose
(Gibaldi & Perrier 1982, ch. 1-2; Gabrielsson & Weiner 2016, ch. 3). The
models here are descriptive: the coefficients `a_i` and rate constants `k_i`
carry no compartmental interpretation; `lambda_z` is the smallest rate
constant, `t½ = ln 2 / k`, and the area is `sum a_i / k_i`.

- `MonoExp`: `y = a e^{-k x}`
- `BiExp`: `y = a1 e^{-k1 x} + a2 e^{-k2 x}` with `k1 > k2`
- `TriExp`: three terms with `k1 > k2 > k3`
- `Bateman`: `y = a ka / (ka - ke) (e^{-ke t} - e^{-ka t})`, the one
  compartment curve with first order absorption, optionally with a lag time;
  `ka < ke` is a flip-flop (the terminal phase reflects absorption)

The initial guesses use the method of residuals (curve stripping): the
terminal phase is regressed on the last points, its contribution is subtracted
and the residuals give the faster phase.
"""

import math
from typing import ClassVar

import numpy as np

from pkpdutils.fit.model import Model, ModelParameter

LN2 = math.log(2.0)


def log_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of `ln y` on `x` over the finite positive points.

    Returns `(nan, nan)` with fewer than two usable points.
    """
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return math.nan, math.nan
    slope, intercept = np.polyfit(x[ok], np.log(y[ok]), 1)
    return float(slope), float(intercept)


def terminal_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """`a` and `k` of the terminal phase from the last half of the points after the maximum.

    Falls back to the maximum and `ln 2 / (range / 3)` when no regression is possible.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return math.nan, math.nan
    imax = int(np.argmax(y))
    tail = np.arange(x.size) >= max(imax, imax + (x.size - imax) // 2)
    if tail.sum() < 3:
        tail = np.arange(x.size) >= imax
    slope, intercept = log_linear_regression(x[tail], y[tail])
    if not np.isfinite(slope) or slope >= 0:
        k = LN2 / max((x.max() - x.min()) / 3.0, 1e-12)
        return float(y.max()), float(k)
    return float(math.exp(intercept)), float(-slope)


def _strip(x: np.ndarray, y: np.ndarray, phases: int) -> np.ndarray:
    """Curve stripping into `phases` exponential phases, slowest last.

    Returns `[a_fast, k_fast, ..., a_slow, k_slow]`.
    """
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    coefficients: list[tuple[float, float]] = []
    residual = y.copy()
    for _ in range(phases - 1):
        a, k = terminal_guess(x, residual)
        coefficients.append((a, k))
        residual = residual - a * np.exp(-k * x)
        keep = residual > 0
        if keep.sum() < 2:
            break
        # the remaining phase lives in the early points
        x, residual = x[keep], residual[keep]
    a, k = terminal_guess(x, residual) if residual.size >= 2 else (math.nan, math.nan)
    coefficients.append((a, k))
    # fill failed phases from the slowest one
    slow_a, slow_k = coefficients[0]
    fixed: list[tuple[float, float]] = []
    for i, (a, k) in enumerate(reversed(coefficients)):
        if not (np.isfinite(a) and np.isfinite(k)) or k <= 0:
            factor = 5.0 ** (len(coefficients) - 1 - i)
            a, k = slow_a, slow_k * factor
        fixed.append((a, k))
    # fixed is fast..slow; enforce strict ordering of the rates
    out = []
    for a, k in fixed:
        out.extend([a, k])
    p = np.array(out, dtype=np.float64)
    return _sort_phases(p)


def _sort_phases(p: np.ndarray) -> np.ndarray:
    """Order `[a1, k1, a2, k2, ...]` by decreasing rate; equal rates are spread by 1 %."""
    pairs = p.reshape(-1, 2)
    order = np.argsort(-pairs[:, 1], kind="stable")
    pairs = pairs[order]
    for i in range(1, pairs.shape[0]):
        if pairs[i, 1] >= pairs[i - 1, 1]:
            pairs[i, 1] = pairs[i - 1, 1] * 0.99
    return pairs.reshape(-1)


class MonoExp(Model):
    """`y = a exp(-k x)`."""

    name = "monoexp"
    parameters = (
        ModelParameter("a", "[y]", description="value at x = 0"),
        ModelParameter("k", "1/[x]", description="rate constant"),
    )
    derived_units = {"thalf": "[x]", "auc": "[y]*[x]"}

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        return p[0] * np.exp(-p[1] * x)

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """Half-life and area."""
        return {"thalf": LN2 / p[1], "auc": p[0] / p[1]}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """From the log-linear regression of the positive points."""
        slope, intercept = log_linear_regression(x, y)
        if np.isfinite(slope) and slope < 0:
            return np.array([math.exp(intercept), -slope])
        a, k = terminal_guess(x, y)
        return np.array([a, k])


class _SumOfExponentials(Model):
    """Shared code of `BiExp` and `TriExp`."""

    #: number of phases
    phases: ClassVar[int] = 2

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The sum of the phases."""
        pairs = p.reshape(-1, 2)
        return np.sum(pairs[:, 0][:, None] * np.exp(-pairs[:, 1][:, None] * x[None, :]), axis=0)

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`lambda_z`, the half-lives of the phases and the area."""
        pairs = p.reshape(-1, 2)
        out: dict[str, float] = {"lambda_z": float(pairs[-1, 1])}
        for i, (_, k) in enumerate(pairs, start=1):
            out[f"thalf_{i}"] = LN2 / k
        out["auc"] = float(np.sum(pairs[:, 0] / pairs[:, 1]))
        return out

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Curve stripping."""
        return _strip(x, y, self.phases)

    def sort_parameters(self, p: np.ndarray) -> np.ndarray:
        """Order the phases by decreasing rate constant (the engine calls this after a fit)."""
        pairs = p.reshape(-1, 2)
        return pairs[np.argsort(-pairs[:, 1], kind="stable")].reshape(-1)


class BiExp(_SumOfExponentials):
    """`y = a1 exp(-k1 x) + a2 exp(-k2 x)` with `k1 > k2`."""

    name = "biexp"
    phases = 2
    parameters = (
        ModelParameter("a1", "[y]", description="coefficient of the fast phase"),
        ModelParameter("k1", "1/[x]", description="rate constant of the fast phase"),
        ModelParameter("a2", "[y]", description="coefficient of the slow phase"),
        ModelParameter("k2", "1/[x]", description="rate constant of the slow phase"),
    )
    derived_units = {"lambda_z": "1/[x]", "thalf_1": "[x]", "thalf_2": "[x]", "auc": "[y]*[x]"}


class TriExp(_SumOfExponentials):
    """`y = a1 exp(-k1 x) + a2 exp(-k2 x) + a3 exp(-k3 x)` with `k1 > k2 > k3`."""

    name = "triexp"
    phases = 3
    parameters = (
        ModelParameter("a1", "[y]", description="coefficient of the fastest phase"),
        ModelParameter("k1", "1/[x]", description="rate constant of the fastest phase"),
        ModelParameter("a2", "[y]", description="coefficient of the middle phase"),
        ModelParameter("k2", "1/[x]", description="rate constant of the middle phase"),
        ModelParameter("a3", "[y]", description="coefficient of the slowest phase"),
        ModelParameter("k3", "1/[x]", description="rate constant of the slowest phase"),
    )
    derived_units = {
        "lambda_z": "1/[x]", "thalf_1": "[x]", "thalf_2": "[x]", "thalf_3": "[x]", "auc": "[y]*[x]",
    }


class Bateman(Model):
    """One compartment with first order absorption: `y = a ka/(ka - ke) (e^{-ke t} - e^{-ka t})`.

    `t = max(x - tlag, 0)` with the optional lag time. `a` is the dose over the
    apparent volume (`D F / V`), so `auc = a / ke`.

    Args:
        lag: whether a lag time `tlag` is fitted
    """

    name = "bateman"
    derived_units = {"tmax": "[x]", "cmax": "[y]", "thalf": "[x]", "auc": "[y]*[x]", "flip_flop": "dimensionless"}

    def __init__(self, lag: bool = False) -> None:
        """Create the model with or without a lag time."""
        self.lag = lag
        base = (
            ModelParameter("a", "[y]", description="dose over the apparent volume"),
            ModelParameter("ka", "1/[x]", description="absorption rate constant"),
            ModelParameter("ke", "1/[x]", description="elimination rate constant"),
        )
        self.parameters = (*base, ModelParameter("tlag", "[x]", lower=0.0, positive=False, description="lag time")) if lag else base  # ty: ignore[invalid-assignment]

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve, with the `ka == ke` limit `a ka t exp(-ke t)`."""
        a, ka, ke = p[0], p[1], p[2]
        t = np.maximum(x - (p[3] if self.lag else 0.0), 0.0)
        if abs(ka - ke) < 1e-9:
            return a * ka * t * np.exp(-ke * t)
        return a * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """Time and value of the maximum, half-life, area and the flip-flop indicator."""
        a, ka, ke = float(p[0]), float(p[1]), float(p[2])
        tlag = float(p[3]) if self.lag else 0.0
        tmax = (math.log(ka / ke) / (ka - ke) if abs(ka - ke) >= 1e-9 else 1.0 / ke) + tlag
        cmax = float(self.predict(np.array([tmax]), p)[0])
        return {"tmax": tmax, "cmax": cmax, "thalf": LN2 / ke, "auc": a / ke, "flip_flop": 1.0 if ka < ke else 0.0}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`ke` from the terminal phase, `ka` from the rise (or 5 ke), `a` from the maximum."""
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[ok], y[ok]
        _, ke = terminal_guess(xs, ys)
        imax = int(np.argmax(ys))
        ka = 5.0 * ke
        if imax >= 2:
            rise = ys[: imax + 1]
            slope, _ = log_linear_regression(xs[: imax + 1], np.maximum(rise.max() - rise, 1e-12)[: imax + 1])
            if np.isfinite(slope) and slope < 0 and -slope > ke:
                ka = -slope
        tmax = xs[imax]
        denominator = math.exp(-ke * tmax) - math.exp(-ka * tmax)
        a = ys[imax] * (ka - ke) / ka / denominator if denominator > 0 else ys[imax]
        guess = [a, ka, ke]
        if self.lag:
            guess.append(0.0)
        return np.array(guess, dtype=np.float64)
```
Notes: `parameters` is a class variable on `Model`; `Bateman` sets it per instance (the `ty: ignore` may be unnecessary - remove if ty accepts the assignment, otherwise keep the rule ty names). `_sort_phases` after stripping guarantees strictly decreasing rates for the start vector.

`src/pkpdutils/fit/models.py`:
```python
"""The model library of the curve fitting (see `docs/fitting.md`)."""

from pkpdutils.fit.models_exponential import Bateman, BiExp, MonoExp, TriExp

__all__ = ["Bateman", "BiExp", "MonoExp", "TriExp"]
```

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/fit -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. If `test_biexp_predict_strip_and_sort` fails on `guess[3]` within 30 % of 0.2: the terminal half of the points of `T` (`8, 12, 24`) may still contain the fast phase at `t = 8`: `8·exp(-16) ≈ 0`, so it does not; check `terminal_guess`'s tail selection (`max(imax, imax + (n - imax)//2)`). If `TriExp` stripping yields a non-finite guess, the fallback fills from the slowest phase; the test only requires finite ordered rates.

```bash
git add src/pkpdutils/fit tests/fit
git commit -q -m "Add the exponential and Bateman models

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 4: Response, power and covariate models (`fit/models_response.py`, `fit/models_linear.py`)

**Files:**
- Create: `src/pkpdutils/fit/models_response.py`, `src/pkpdutils/fit/models_linear.py`
- Modify: `src/pkpdutils/fit/models.py`
- Test: `tests/fit/test_models_response.py`

**Interfaces:**
- `models_response.py`: `Emax`: `e0` (`[y]`, `positive=False`), `emax` (`[y]`, `positive=False`), `ec50` (`[x]`); `y = e0 + emax·x/(ec50 + x)`; derived `ec90 = 9·ec50` (`[x]`). `SigmoidEmax`: + `hill` (dimensionless, positive, `lower=0.1`, `upper=20`); `y = e0 + emax·xⁿ/(ec50ⁿ + xⁿ)`; `ec90 = 9^(1/n)·ec50`. `Imax`: `e0` (`[y]`, positive), `imax` (dimensionless, `lower=0`, `upper=1`, `positive=False`), `ic50` (`[x]`); `y = e0·(1 - imax·x/(ic50 + x))`; derived `ic90 = 9·ic50`. `SigmoidImax` with `hill`. Guesses: `e0` = value at the smallest `x`, `emax` = value at the largest `x` minus `e0` (or `y.max() - e0`), `ec50` = `x` at which `y` crosses `e0 + emax/2` (interpolated) or the median of `x`; `imax` = `1 - y.min()/e0` clipped to `(0.05, 0.95)`, `ic50` analogous; `hill = 1`.
- `models_linear.py`: `Linear`: `intercept` (`[y]`, `positive=False`), `slope` (`[y]/[x]`, `positive=False`); `LogLinear`: `intercept` (`[y]`, `positive=False`), `slope` (`[y]`, `positive=False`), `y = intercept + slope·ln x` (`x > 0`); `Power`: `a` (`[y]`, positive), `b` (dimensionless, `positive=False`), `y = a·x^b` (`x > 0`); `Allometric(exponent: float | None = None)`: `a` (`[y]`) and, when `exponent is None`, `b` (dimensionless, `positive=False`, guess 0.75); `y = a·x^b` with `b` fixed to `exponent` otherwise; derived none. Guesses by least squares on the (log) transformed data.
- `models.py` re-exports all: `Allometric`, `Bateman`, `BiExp`, `Emax`, `Imax`, `Linear`, `LogLinear`, `MonoExp`, `Power`, `SigmoidEmax`, `SigmoidImax`, `TriExp`.

- [ ] **Step 1: Write the failing tests**

`tests/fit/test_models_response.py`:
```python
import numpy as np
import pytest

from pkpdutils.fit.models import Allometric, Emax, Imax, Linear, LogLinear, Power, SigmoidEmax, SigmoidImax

C = np.array([0.0, 0.5, 1, 2, 5, 10, 20, 50, 100])


def test_emax_family() -> None:
    m = Emax()
    assert m.parameter_names == ("e0", "emax", "ec50")
    assert not m.parameter("e0").positive and not m.parameter("emax").positive and m.parameter("ec50").positive
    p = np.array([2.0, 10.0, 5.0])
    y = m.predict(C, p)
    np.testing.assert_allclose(y, 2 + 10 * C / (5 + C))
    assert m.derived(p) == {"ec90": 45.0}
    guess = m.initial_guess(C, y)
    assert guess[0] == pytest.approx(2.0, abs=0.5)
    assert guess[1] == pytest.approx(10.0, rel=0.3)
    assert guess[2] == pytest.approx(5.0, rel=0.5)

    s = SigmoidEmax()
    assert s.parameter_names == ("e0", "emax", "ec50", "hill")
    p = np.array([0.0, 1.0, 10.0, 2.0])
    np.testing.assert_allclose(s.predict(C, p), C**2 / (100 + C**2))
    assert s.derived(p)["ec90"] == pytest.approx(3.0 * 10.0)
    assert s.initial_guess(C, s.predict(C, p))[3] == 1.0


def test_imax_family() -> None:
    m = Imax()
    assert m.parameter_names == ("e0", "imax", "ic50")
    p = np.array([10.0, 0.8, 4.0])
    y = m.predict(C, p)
    np.testing.assert_allclose(y, 10 * (1 - 0.8 * C / (4 + C)))
    assert m.derived(p) == {"ic90": 36.0}
    guess = m.initial_guess(C, y)
    assert guess[0] == pytest.approx(10.0, rel=0.1)
    assert 0.05 <= guess[1] <= 0.95
    assert guess[2] == pytest.approx(4.0, rel=0.6)
    assert m.parameter("imax").upper == 1.0 and m.parameter("imax").lower == 0.0
    s = SigmoidImax()
    assert s.parameter_names == ("e0", "imax", "ic50", "hill")
    np.testing.assert_allclose(s.predict(C, np.array([10.0, 1.0, 4.0, 1.0])), 10 * (1 - C / (4 + C)))


def test_linear_models() -> None:
    lin = Linear()
    np.testing.assert_allclose(lin.predict(C, np.array([1.0, 2.0])), 1 + 2 * C)
    np.testing.assert_allclose(lin.initial_guess(C, 1 + 2 * C), [1.0, 2.0])
    log = LogLinear()
    x = C[1:]
    np.testing.assert_allclose(log.predict(x, np.array([1.0, 2.0])), 1 + 2 * np.log(x))
    np.testing.assert_allclose(log.initial_guess(x, 1 + 2 * np.log(x)), [1.0, 2.0])
    assert lin.parameter("slope").unit_expr == "[y]/[x]" and log.parameter("slope").unit_expr == "[y]"


def test_power_and_allometric() -> None:
    pw = Power()
    assert pw.parameter_names == ("a", "b") and not pw.parameter("b").positive
    x = C[1:]
    np.testing.assert_allclose(pw.predict(x, np.array([3.0, 0.8])), 3 * x**0.8)
    np.testing.assert_allclose(pw.initial_guess(x, 3 * x**0.8), [3.0, 0.8])
    free = Allometric()
    assert free.parameter_names == ("a", "b")
    assert free.initial_guess(x, 3 * x**0.75)[1] == pytest.approx(0.75)
    fixed = Allometric(exponent=0.75)
    assert fixed.parameter_names == ("a",)
    np.testing.assert_allclose(fixed.predict(x, np.array([3.0])), 3 * x**0.75)
    assert fixed.initial_guess(x, 3 * x**0.75)[0] == pytest.approx(3.0)
    assert fixed.exponent == 0.75
    assert fixed.name == "allometric_0.75" and free.name == "allometric"
```

Run: `uv run pytest tests/fit/test_models_response.py -n 0 -q` → FAIL with `ImportError`.

- [ ] **Step 2: Write the modules**

`src/pkpdutils/fit/models_response.py`:
```python
"""Concentration-effect models (Emax family).

The Emax model `E = E0 + Emax C / (EC50 + C)` and its sigmoid form with the
Hill coefficient `n` describe a saturable response (Gabrielsson & Weiner 2016,
ch. 4); `Imax` is the inhibitory form `E = E0 (1 - Imax C / (IC50 + C))`.
`EC90 = 9^{1/n} EC50` is the concentration of 90 % of the maximal effect. The
same models describe a pharmacokinetic parameter against an inhibitor dose.
"""

import numpy as np

from pkpdutils.fit.model import Model, ModelParameter


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """First `x` at which the sorted curve crosses `level` (linear interpolation), else the median of `x`."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    above = ys >= level if ys[-1] >= ys[0] else ys <= level
    idx = np.argmax(above) if above.any() else 0
    if idx == 0 or not above.any():
        return float(np.median(xs[xs > 0])) if (xs > 0).any() else float(np.median(xs))
    x1, x2, y1, y2 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
    if y2 == y1:
        return float(x2)
    return float(x1 + (level - y1) / (y2 - y1) * (x2 - x1))


class Emax(Model):
    """`E = e0 + emax x / (ec50 + x)`."""

    name = "emax"
    parameters = (
        ModelParameter("e0", "[y]", positive=False, description="baseline effect"),
        ModelParameter("emax", "[y]", positive=False, description="maximal effect above baseline"),
        ModelParameter("ec50", "[x]", description="concentration of half-maximal effect"),
    )
    derived_units = {"ec90": "[x]"}

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        return p[0] + p[1] * x / (p[2] + x)

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ec90 = 9 ec50`."""
        return {"ec90": 9.0 * float(p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Baseline at the smallest `x`, plateau at the largest, `ec50` at the half-way crossing."""
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[ok], y[ok]
        order = np.argsort(xs)
        e0, top = float(ys[order[0]]), float(ys[order[-1]])
        emax = top - e0 if top != e0 else float(ys.max() - e0) or 1.0
        ec50 = max(_crossing(xs, ys, e0 + emax / 2.0), 1e-12)
        return np.array([e0, emax, ec50])


class SigmoidEmax(Emax):
    """`E = e0 + emax x^n / (ec50^n + x^n)` with the Hill coefficient `n`."""

    name = "sigmoid_emax"
    parameters = (
        *Emax.parameters,
        ModelParameter("hill", "dimensionless", lower=0.1, upper=20.0, description="Hill coefficient"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        xn = np.power(np.maximum(x, 0.0), p[3])
        return p[0] + p[1] * xn / (p[2] ** p[3] + xn)

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ec90 = 9^(1/n) ec50`."""
        return {"ec90": float(9.0 ** (1.0 / p[3]) * p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The Emax guess with `hill = 1`."""
        return np.append(super().initial_guess(x, y), 1.0)


class Imax(Model):
    """`E = e0 (1 - imax x / (ic50 + x))`."""

    name = "imax"
    parameters = (
        ModelParameter("e0", "[y]", description="baseline effect"),
        ModelParameter("imax", "dimensionless", lower=0.0, upper=1.0, positive=False, description="maximal fractional inhibition"),
        ModelParameter("ic50", "[x]", description="concentration of half-maximal inhibition"),
    )
    derived_units = {"ic90": "[x]"}

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        return p[0] * (1.0 - p[1] * x / (p[2] + x))

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ic90 = 9 ic50`."""
        return {"ic90": 9.0 * float(p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Baseline at the smallest `x`, inhibition from the minimum, `ic50` at the half-way crossing."""
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[ok], y[ok]
        e0 = max(float(ys[np.argmin(xs)]), 1e-12)
        imax = float(np.clip(1.0 - ys.min() / e0, 0.05, 0.95))
        ic50 = max(_crossing(xs, ys, e0 * (1.0 - imax / 2.0)), 1e-12)
        return np.array([e0, imax, ic50])


class SigmoidImax(Imax):
    """`E = e0 (1 - imax x^n / (ic50^n + x^n))`."""

    name = "sigmoid_imax"
    parameters = (
        *Imax.parameters,
        ModelParameter("hill", "dimensionless", lower=0.1, upper=20.0, description="Hill coefficient"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        xn = np.power(np.maximum(x, 0.0), p[3])
        return p[0] * (1.0 - p[1] * xn / (p[2] ** p[3] + xn))

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ic90 = 9^(1/n) ic50`."""
        return {"ic90": float(9.0 ** (1.0 / p[3]) * p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The Imax guess with `hill = 1`."""
        return np.append(super().initial_guess(x, y), 1.0)
```

`src/pkpdutils/fit/models_linear.py`:
```python
"""Linear, log-linear, power and allometric models.

`Power` (`y = a x^b`) is the model of dose proportionality (Smith et al.
2000): `b = 1` is proportional; `proportionality_test` in
`pkpdutils.fit.proportionality` applies the confidence interval criterion.
`Allometric` is the same model for a parameter against body weight with the
exponent free or fixed (0.75 for clearances, 1 for volumes; Rowland & Tozer
2011, ch. 12). `Linear` and `LogLinear` describe an effect or a parameter
against a concentration or covariate.
"""

import numpy as np

from pkpdutils.fit.model import Model, ModelParameter


def _finite(x: np.ndarray, y: np.ndarray, positive_x: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The finite points, optionally with positive `x`."""
    ok = np.isfinite(x) & np.isfinite(y)
    if positive_x:
        ok &= x > 0
    return x[ok], y[ok]


class Linear(Model):
    """`y = intercept + slope x`."""

    name = "linear"
    parameters = (
        ModelParameter("intercept", "[y]", positive=False, description="value at x = 0"),
        ModelParameter("slope", "[y]/[x]", positive=False, description="slope"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The line."""
        return p[0] + p[1] * x

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ordinary least squares."""
        xs, ys = _finite(x, y)
        slope, intercept = np.polyfit(xs, ys, 1) if xs.size >= 2 else (0.0, float(ys.mean()))
        return np.array([intercept, slope], dtype=np.float64)


class LogLinear(Model):
    """`y = intercept + slope ln x` for `x > 0`."""

    name = "loglinear"
    parameters = (
        ModelParameter("intercept", "[y]", positive=False, description="value at x = 1"),
        ModelParameter("slope", "[y]", positive=False, description="change per e-fold of x"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve (`NaN` for `x <= 0`)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[0] + p[1] * np.log(x)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least squares on `ln x`."""
        xs, ys = _finite(x, y, positive_x=True)
        slope, intercept = np.polyfit(np.log(xs), ys, 1) if xs.size >= 2 else (0.0, float(ys.mean()))
        return np.array([intercept, slope], dtype=np.float64)


class Power(Model):
    """`y = a x^b` for `x > 0`."""

    name = "power"
    parameters = (
        ModelParameter("a", "[y]", description="value at x = 1"),
        ModelParameter("b", "dimensionless", positive=False, description="exponent"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[0] * np.power(x, p[1])

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least squares on the log-log data."""
        xs, ys = _finite(x, y, positive_x=True)
        ok = ys > 0
        if ok.sum() >= 2:
            b, log_a = np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)
            return np.array([np.exp(log_a), b], dtype=np.float64)
        return np.array([max(float(ys.mean()), 1e-12), 1.0])


class Allometric(Model):
    """`y = a x^b` against body weight with the exponent free or fixed.

    Args:
        exponent: fixed exponent (`0.75` for clearances, `1` for volumes), `None` to fit it
    """

    name = "allometric"

    def __init__(self, exponent: float | None = None) -> None:
        """Create the model with a free or a fixed exponent."""
        self.exponent = exponent
        self.name = "allometric" if exponent is None else f"allometric_{exponent:g}"
        a = ModelParameter("a", "[y]", description="value at unit weight")
        if exponent is None:
            self.parameters = (a, ModelParameter("b", "dimensionless", positive=False, description="allometric exponent"))  # ty: ignore[invalid-assignment]
        else:
            self.parameters = (a,)  # ty: ignore[invalid-assignment]

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        b = p[1] if self.exponent is None else self.exponent
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[0] * np.power(x, b)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log-log least squares (`a` only when the exponent is fixed)."""
        xs, ys = _finite(x, y, positive_x=True)
        ok = ys > 0
        if self.exponent is None:
            if ok.sum() >= 2:
                b, log_a = np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)
                return np.array([np.exp(log_a), b], dtype=np.float64)
            return np.array([max(float(ys.mean()), 1e-12), 0.75])
        if ok.any():
            log_a = np.mean(np.log(ys[ok]) - self.exponent * np.log(xs[ok]))
            return np.array([np.exp(log_a)], dtype=np.float64)
        return np.array([1.0])
```
The `# ty: ignore[invalid-assignment]` comments on instance assignments to a `ClassVar` are removed if ty does not complain (unused suppressions fail the check); if ty insists, an alternative is to declare `parameters` on `Model` as a plain annotated attribute `parameters: tuple[ModelParameter, ...] = ()` (not `ClassVar`) - choose whichever passes cleanly and apply it consistently in `model.py` (`Bateman` too).

`src/pkpdutils/fit/models.py` re-exports all twelve models with a sorted `__all__`.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/fit -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass (the guess tolerances are loose on purpose).

```bash
git add src/pkpdutils/fit tests/fit
git commit -q -m "Add the Emax, linear, power and allometric models

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 5: The engine: single-start fits, statistics and `FitResult` (`fit/engine.py`, `fit/result.py`)

**Files:**
- Create: `src/pkpdutils/fit/engine.py`, `src/pkpdutils/fit/result.py`
- Modify: `src/pkpdutils/fit/models_exponential.py` (add `parameter_order`), `src/pkpdutils/fit/__init__.py` (exports)
- Test: `tests/fit/test_engine.py`

**Interfaces:**
- `fit/result.py`: `class FitResult(ParameterResult)` with `flag_type = FitFlag`, `discrete_parameters = frozenset({"n_points", "n_parameters", "n_starts_converged", "flip_flop"})`; `__init__(ds, model: Model)`; attribute `model`; `attrs`: `model` (name), `parameter_names` (list), `x_unit`, `y_unit`; methods `predict(x: np.ndarray, **indexers) -> np.ndarray` (one sample; all sample dims given, none for a 0-D result), `predict_all(x) -> xr.DataArray` over `(*sample_dims, "x")`, `parameter_vector(**indexers) -> np.ndarray` (fitted `p` of one sample in model order, fixed parameters included), `correlation(**indexers) -> pd.DataFrame`; `summarize` inherited.
- `fit/engine.py`:
  - `fit(model, x, y, *, sd=None, options=None, x_unit="dimensionless", y_unit="dimensionless", dims=None, coords=None) -> FitResult`: `x` 1-D `(n,)` shared or 2-D `(N, n)`; `y` 1-D → one sample (no sample dims) or 2-D `(N, n)` with `dims` (default `("sample",)`) and `coords`; `sd` like `y` (required for `Weighting.INV_SD`); NaN padding allowed
  - `fit_rows(model, x, y, sd, options) -> list[RowFit]` (the per-row loop; Task 6 parallelizes it), `fit_row(model, x, y, sd, options, rng) -> RowFit` (one row, one or more starts), `@dataclass RowFit` (`p`, `q`, `se_p`, `ci_low`, `ci_high`, `cov_q`, `correlation`, `derived`, `derived_se`, `derived_ci_low`, `derived_ci_high`, `cost`, `r2`, `rmse`, `aic`, `aicc`, `bic`, `n_points`, `n_starts_converged`, `y_pred` `(n,)`, `residuals` `(n,)`, `flags: int`, `nfev`)
  - helpers: `variance_of(y, sd, weighting) -> np.ndarray`, `to_scale(p, scales)`, `from_scale(q, scales)`, `scale_derivative(p, scales)` (`dp/dq`), `bounds_in_scale(lower, upper, scales)`, `covariance(jac, cost, n, k) -> (cov, singular: bool)`
  - `build_result(model, rows, *, x, y, sd, x_unit, y_unit, dims, coords, options) -> FitResult`: assembles the dataset (variables of the Global Constraints; units from `parameter_unit_expression`; fixed parameters reported with `NaN` uncertainties; derived parameters as variables with their own `_se`, `_ci_low`, `_ci_high`, `_cv`)
- `_SumOfExponentials.parameter_order(p) -> np.ndarray`: the permutation that orders the phases by decreasing rate (the engine permutes `p`, `q` and the Jacobian columns with it after every fit when the model defines it)

Definitions (Seber & Wild 1989, ch. 2; Gabrielsson & Weiner 2016, ch. 6; the scipy documentation of `least_squares`): the weighted residual is `r_i = (y_i - f(x_i; p)) / sqrt(var_i)`; scipy minimizes `0.5 Σ ρ(r_i²)`; with the linear loss the residual variance is `s² = Σ r_i² / (n - k)`, the covariance of the scaled parameters `cov(q) = s² (JᵀJ)⁻¹` with `J` the Jacobian in the scaled space, `se(p_j) = se(q_j) |dp_j/dq_j|` (`p ln 10` for log10, `p` for ln, 1 for linear), the interval `q_j ± t_{n-k, 1-α/2} se(q_j)` transformed back (asymmetric for log parameters), `CV% = 100 se(p)/|p|`, `corr = cov / (se seᵀ)`; derived parameters `d(p)` get `se(d) = sqrt(g cov(q) gᵀ)` with `g = ∂d/∂q` by central differences and `d ± t se(d)`; goodness of fit on the unweighted residuals `e = y - f`: `R² = 1 - Σe²/Σ(y - ȳ)²`, `RMSE = sqrt(Σe²/n)`; on the weighted ones `AIC = n ln(Σr²/n) + 2k`, `AICc = AIC + 2k(k+1)/(n-k-1)`, `BIC = n ln(Σr²/n) + k ln n`.

- [ ] **Step 1: Write the failing tests**

`tests/fit/test_engine.py`:
```python
import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils.fit import FitOptions, FitResult, ParameterScale, Weighting, fit
from pkpdutils.fit.engine import fit_row, from_scale, scale_derivative, to_scale, variance_of
from pkpdutils.fit.models import Bateman, BiExp, Emax, Linear, MonoExp

T = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24])
RNG = np.random.default_rng(0)


def noisy_monoexp(a: float = 10.0, k: float = 0.3, cv: float = 0.05, rng: np.random.Generator = RNG) -> tuple[np.ndarray, np.ndarray]:
    y = a * np.exp(-k * T)
    return y, y * rng.lognormal(0.0, cv, size=T.size)


def test_scale_helpers() -> None:
    p = np.array([10.0, -2.0, 0.5])
    scales = [ParameterScale.LOG10, ParameterScale.LINEAR, ParameterScale.LOG]
    q = to_scale(p, scales)
    np.testing.assert_allclose(q, [1.0, -2.0, np.log(0.5)])
    np.testing.assert_allclose(from_scale(q, scales), p)
    np.testing.assert_allclose(scale_derivative(p, scales), [10.0 * np.log(10.0), 1.0, 0.5])


def test_variance_of() -> None:
    y = np.array([1.0, 4.0, 0.0])
    np.testing.assert_allclose(variance_of(y, None, Weighting.NONE), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(variance_of(y, None, Weighting.INV_Y2), [1.0, 16.0, 1.0])  # y = 0 -> smallest positive
    np.testing.assert_allclose(variance_of(y, None, Weighting.INV_Y), [1.0, 4.0, 1.0])
    np.testing.assert_allclose(variance_of(y, np.array([0.1, 0.2, 0.3]), Weighting.INV_SD), [0.01, 0.04, 0.09])
    with pytest.raises(ValueError, match="sd"):
        variance_of(y, None, Weighting.INV_SD)


def test_monoexp_recovery_and_statistics() -> None:
    truth, y = noisy_monoexp()
    result = fit(MonoExp(), T, y, x_unit="hr", y_unit="mg/l")
    assert isinstance(result, FitResult)
    assert result.sample_dims == ()
    q = result.to_quantities()
    assert q["a"].magnitude == pytest.approx(10.0, rel=0.1)
    assert q["k"].magnitude == pytest.approx(0.3, rel=0.1)
    assert str(q["k"].units) == "1 / hour" and str(q["a"].units) == "milligram / liter"
    assert q["k_se"].magnitude > 0 and q["k_cv"].magnitude > 0
    assert q["k_ci_low"].magnitude < q["k"].magnitude < q["k_ci_high"].magnitude
    assert q["thalf"].magnitude == pytest.approx(np.log(2) / q["k"].magnitude)
    assert q["thalf_se"].magnitude > 0 and str(q["thalf"].units) == "hour"
    assert str(q["auc"].units) == "hour * milligram / liter"
    assert q["r2"].magnitude > 0.95 and q["rmse"].magnitude > 0
    assert q["n_points"].magnitude == T.size and q["n_parameters"].magnitude == 2
    assert np.isfinite(q["aic"].magnitude) and q["aicc"].magnitude > q["aic"].magnitude and np.isfinite(q["bic"].magnitude)
    assert result.flags() == []
    assert result.point_variables == ["x_data", "y_data", "y_pred", "residuals"]
    np.testing.assert_allclose(result["y_pred"].values, result.predict(T), rtol=1e-12)
    corr = result.correlation()
    assert corr.shape == (2, 2) and corr.loc["a", "a"] == pytest.approx(1.0) and abs(corr.loc["a", "k"]) < 1.0
    np.testing.assert_allclose(result.parameter_vector(), [q["a"].magnitude, q["k"].magnitude])


def test_exact_data_gives_tiny_uncertainty_and_t_interval() -> None:
    truth, _ = noisy_monoexp()
    result = fit(MonoExp(), T, truth * (1 + 1e-6 * np.sin(T)), options=FitOptions(parameter_scale=ParameterScale.LINEAR))
    q = result.to_quantities()
    assert q["k_se"].magnitude < 1e-4
    tq = student_t.ppf(0.975, T.size - 2)
    assert q["k_ci_high"].magnitude == pytest.approx(q["k"].magnitude + tq * q["k_se"].magnitude, rel=1e-6)


def test_log_scale_interval_is_asymmetric() -> None:
    _, y = noisy_monoexp(cv=0.2)
    q = fit(MonoExp(), T, y).to_quantities()
    k, low, high = q["k"].magnitude, q["k_ci_low"].magnitude, q["k_ci_high"].magnitude
    assert np.log(k / low) == pytest.approx(np.log(high / k), rel=1e-6)


def test_batch_rows_dims_and_nan_padding() -> None:
    ys = np.stack([noisy_monoexp(k=k, rng=np.random.default_rng(i))[1] for i, k in enumerate((0.2, 0.3, 0.5))])
    ys[2, -2:] = np.nan
    result = fit(MonoExp(), T, ys, dims=("individual",), coords={"individual": ["a", "b", "c"]}, x_unit="hr", y_unit="mg/l")
    assert result.sample_dims == ("individual",)
    np.testing.assert_allclose(result["k"].values, [0.2, 0.3, 0.5], rtol=0.15)
    assert result["n_points"].values[2] == T.size - 2
    assert result["y_pred"].dims == ("individual", "point")
    assert np.isnan(result["residuals"].values[2, -2:]).all()
    assert result["correlation"].dims == ("individual", "parameter", "parameter_")
    df = result.to_dataframe()
    assert "k_se" in df.columns and "y_pred" not in df.columns and list(df.columns)[-1] == "flags"
    single = fit(MonoExp(), T, ys[1], x_unit="hr", y_unit="mg/l")
    assert single.to_quantities()["k"].magnitude == pytest.approx(result["k"].values[1])
    assert result.predict(T, individual="b").shape == T.shape
    assert result.predict_all(T).dims == ("individual", "x")


def test_weighting_inv_sd_and_fixed_and_bounds() -> None:
    _, y = noisy_monoexp()
    sd = 0.05 * y
    weighted = fit(MonoExp(), T, y, sd=sd, options=FitOptions(weighting=Weighting.INV_SD))
    assert weighted.flags() == []
    fixed = fit(MonoExp(), T, y, options=FitOptions(fixed={"k": 0.3}))
    q = fixed.to_quantities()
    assert q["k"].magnitude == 0.3 and np.isnan(q["k_se"].magnitude) and q["n_parameters"].magnitude == 1
    bounded = fit(MonoExp(), T, y, options=FitOptions(bounds={"k": (0.5, 1.0)}))
    assert bounded.to_quantities()["k"].magnitude == pytest.approx(0.5, rel=1e-3)
    assert "AT_BOUND" in bounded.flags()


def test_too_few_points_and_no_data() -> None:
    r = fit(Emax(), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert "TOO_FEW_POINTS" in r.flags() and np.isnan(r.to_quantities()["ec50"].magnitude)
    r = fit(MonoExp(), T, np.full(T.size, np.nan))
    assert "NO_DATA" in r.flags()


def test_bateman_flip_flop_flag_and_derived() -> None:
    y = Bateman().predict(T, np.array([10.0, 0.2, 0.6])) * np.random.default_rng(3).lognormal(0, 0.02, T.size)
    result = fit(Bateman(), T, y, options=FitOptions(initial={"ka": 0.2, "ke": 0.6}))
    q = result.to_quantities()
    assert "FLIP_FLOP" in result.flags() or q["ka"].magnitude > q["ke"].magnitude
    assert "tmax" in result and "cmax" in result and "flip_flop" in result


def test_biexp_phase_ordering() -> None:
    y = BiExp().predict(T, np.array([8.0, 2.0, 2.0, 0.2])) * np.random.default_rng(4).lognormal(0, 0.02, T.size)
    result = fit(BiExp(), T, y, options=FitOptions(initial={"a1": 2.0, "k1": 0.2, "a2": 8.0, "k2": 2.0}))
    q = result.to_quantities()
    assert q["k1"].magnitude > q["k2"].magnitude
    assert q["lambda_z"].magnitude == pytest.approx(q["k2"].magnitude)


def test_linear_scale_for_non_positive_parameters() -> None:
    x = np.linspace(0, 10, 11)
    y = -3.0 + 0.5 * x + np.random.default_rng(5).normal(0, 0.1, x.size)
    q = fit(Linear(), x, y).to_quantities()
    assert q["intercept"].magnitude == pytest.approx(-3.0, abs=0.2)
    assert q["slope"].magnitude == pytest.approx(0.5, abs=0.05)


def test_fit_row_returns_rowfit() -> None:
    _, y = noisy_monoexp()
    row = fit_row(MonoExp(), T, y, None, FitOptions(), np.random.default_rng(0))
    assert row.p.shape == (2,) and row.cov_q.shape == (2, 2) and row.flags == 0
    assert row.y_pred.shape == T.shape and row.n_starts_converged == 1
```

Run: `uv run pytest tests/fit/test_engine.py -n 0 -q` → FAIL with `ImportError`.

- [ ] **Step 2: Add `parameter_order` to the sums of exponentials**

In `models_exponential.py`, `_SumOfExponentials`:
```python
    def parameter_order(self, p: np.ndarray) -> np.ndarray:
        """Permutation of the parameter vector that orders the phases by decreasing rate."""
        pairs = p.reshape(-1, 2)
        order = np.argsort(-pairs[:, 1], kind="stable")
        return np.concatenate([[2 * i, 2 * i + 1] for i in order]).astype(np.intp)
```
and implement `sort_parameters` as `return p[self.parameter_order(p)]`.

- [ ] **Step 3: Write `fit/result.py`**

```python
"""Result of a fit: parameters, uncertainties, statistics, data and predictions."""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pkpdutils.fit.model import Model
from pkpdutils.fit.options import FitFlag
from pkpdutils.result import ParameterResult


class FitResult(ParameterResult):
    """Parameters of a fit as an `xarray.Dataset` over the sample dimensions.

    Variables: every parameter `p` with `p_se`, `p_ci_low`, `p_ci_high`,
    `p_cv`; the derived parameters likewise; the statistics `cost`, `r2`,
    `rmse`, `aic`, `aicc`, `bic`, `n_points`, `n_parameters`,
    `n_starts_converged`; the data and the prediction per point (`x_data`,
    `y_data`, `y_pred`, `residuals` over `point`); the correlation matrix over
    `(parameter, parameter_)`; and the integer `flags` (`FitFlag`). The model
    object is kept for `predict`.
    """

    flag_type = FitFlag
    discrete_parameters = frozenset({"n_points", "n_parameters", "n_starts_converged", "flip_flop"})

    def __init__(self, ds: xr.Dataset, model: Model) -> None:
        """Wrap the dataset of a fit of `model`."""
        super().__init__(ds)
        self.model = model

    def parameter_vector(self, **indexers: Any) -> np.ndarray:
        """The fitted parameters of one sample in the order of the model."""
        sample = self._sample(indexers)
        return np.array([float(sample[name].values) for name in self.model.parameter_names])

    def predict(self, x: np.ndarray, **indexers: Any) -> np.ndarray:
        """The fitted curve of one sample at `x`."""
        return self.model.predict(np.asarray(x, dtype=np.float64), self.parameter_vector(**indexers))

    def predict_all(self, x: np.ndarray) -> xr.DataArray:
        """The fitted curves of every sample at `x`, over `(*sample_dims, "x")`."""
        x = np.asarray(x, dtype=np.float64)
        names = self.model.parameter_names
        stacked = np.stack([self.ds[name].to_numpy() for name in names], axis=-1)  # (*sample_shape, k)
        flat = stacked.reshape(-1, len(names))
        curves = np.stack([self.model.predict(x, p) for p in flat]).reshape(*stacked.shape[:-1], x.size)
        coords = {d: self.ds[d] for d in self.sample_dims if d in self.ds.coords}
        coords["x"] = x
        return xr.DataArray(curves, dims=(*self.sample_dims, "x"), coords=coords, name="y_pred", attrs={"units": self.units("y_pred")})

    def correlation(self, **indexers: Any) -> pd.DataFrame:
        """The correlation matrix of the fitted parameters of one sample."""
        sample = self._sample(indexers)
        names = list(sample["parameter"].values)
        return pd.DataFrame(sample["correlation"].values, index=names, columns=names)

    def _new(self, ds: xr.Dataset) -> "FitResult":
        """A result of the same kind (used by `summarize`), keeping the model."""
        return FitResult(ds, self.model)
```
`ParameterResult.summarize` builds the summary dataset and returns `self._new(ds)` (Task 1), so the summary of a `FitResult` keeps its model.

- [ ] **Step 4: Write `fit/engine.py`**

```python
"""The fitting engine.

`fit` fits a `Model` to one or many `(x, y)` rows with
`scipy.optimize.least_squares` (trust region reflective, bounds) in a scaled
parameter space, derives the standard errors from the Jacobian, transforms the
t-based intervals back to the linear scale, and computes the goodness-of-fit
statistics (Seber & Wild 1989, ch. 2; Gabrielsson & Weiner 2016, ch. 6):

- weighted residuals `r = (y - f) / sqrt(var)` with the variance model of `Weighting`
- `cov(q) = s² (JᵀJ)⁻¹`, `s² = Σ r² / (n - k)`, in the scaled space `q`
- `se(p) = se(q) |dp/dq|`, interval `q ± t_{n-k} se(q)` transformed back
- derived parameters by the delta method with a central difference gradient
- `R²`, `RMSE` on the unweighted residuals; `AIC`, `AICc`, `BIC` on the weighted ones
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from pkpdutils.fit.model import Model, parameter_unit_expression
from pkpdutils.fit.options import FitFlag, FitOptions, ParameterScale, Weighting
from pkpdutils.fit.result import FitResult

logger = logging.getLogger(__name__)

LN10 = math.log(10.0)


@dataclass
class RowFit:
    """The fit of one row; arrays are in model parameter order (fixed parameters included)."""

    p: np.ndarray
    q: np.ndarray
    se_p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    cov_q: np.ndarray
    correlation: np.ndarray
    derived: dict[str, float]
    derived_se: dict[str, float]
    derived_ci_low: dict[str, float]
    derived_ci_high: dict[str, float]
    cost: float
    r2: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    n_points: int
    n_starts_converged: int
    y_pred: np.ndarray
    residuals: np.ndarray
    flags: int
    nfev: int


def variance_of(y: np.ndarray, sd: np.ndarray | None, weighting: Weighting) -> np.ndarray:
    """Variance of every point under the weighting (`y <= 0` uses the smallest positive `|y|`)."""
    if weighting is Weighting.NONE:
        return np.ones_like(y)
    if weighting is Weighting.INV_SD:
        if sd is None:
            raise ValueError("Weighting.INV_SD needs 'sd'")
        return np.asarray(sd, dtype=np.float64) ** 2
    positive = np.abs(y[np.isfinite(y) & (y != 0)])
    floor = positive.min() if positive.size else 1.0
    base = np.where(np.isfinite(y) & (np.abs(y) > 0), np.abs(y), floor)
    return base if weighting is Weighting.INV_Y else base**2


def to_scale(p: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """Linear parameters to the scaled space."""
    q = np.array(p, dtype=np.float64)
    for i, scale in enumerate(scales):
        if scale is ParameterScale.LOG10:
            q[i] = np.log10(p[i])
        elif scale is ParameterScale.LOG:
            q[i] = np.log(p[i])
    return q


def from_scale(q: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """Scaled parameters back to the linear space."""
    p = np.array(q, dtype=np.float64)
    for i, scale in enumerate(scales):
        if scale is ParameterScale.LOG10:
            p[i] = 10.0 ** q[i]
        elif scale is ParameterScale.LOG:
            p[i] = np.exp(q[i])
    return p


def scale_derivative(p: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """`dp/dq` per parameter."""
    d = np.ones_like(p, dtype=np.float64)
    for i, scale in enumerate(scales):
        if scale is ParameterScale.LOG10:
            d[i] = p[i] * LN10
        elif scale is ParameterScale.LOG:
            d[i] = p[i]
    return d


def bounds_in_scale(lower: np.ndarray, upper: np.ndarray, scales: Sequence[ParameterScale]) -> tuple[np.ndarray, np.ndarray]:
    """Bounds in the scaled space (a non-positive lower bound of a log parameter becomes -inf)."""
    lq, uq = np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)
    with np.errstate(divide="ignore"):
        for i, scale in enumerate(scales):
            if scale is ParameterScale.LINEAR:
                continue
            log = np.log10 if scale is ParameterScale.LOG10 else np.log
            lq[i] = log(lower[i]) if lower[i] > 0 else -np.inf
            uq[i] = log(upper[i]) if np.isfinite(upper[i]) else np.inf
    return lq, uq


def covariance(jac: np.ndarray, cost: float, n: int, k: int) -> tuple[np.ndarray, bool]:
    """`s² (JᵀJ)⁻¹` with `s² = 2 cost / (n - k)`; the flag says whether `JᵀJ` was singular."""
    if n <= k:
        return np.full((k, k), np.nan), True
    jtj = jac.T @ jac
    try:
        inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return np.full((k, k), np.nan), True
    if not np.all(np.isfinite(inv)):
        return np.full((k, k), np.nan), True
    return inv * (2.0 * cost / (n - k)), False


def _problem(model: Model, options: FitOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ParameterScale], np.ndarray]:
    """Free-parameter mask, bounds and scales of a model under the options."""
    names = model.parameter_names
    unknown = set(options.fixed) | set(options.bounds) | set(options.initial)
    unknown -= set(names)
    if unknown:
        raise ValueError(f"{model.name} has no parameters {sorted(unknown)}")
    free = np.array([name not in options.fixed for name in names])
    lower, upper = model.bounds()
    for name, (lo, hi) in options.bounds.items():
        i = names.index(name)
        lower[i], upper[i] = lo, hi
    scales = [options.scale_of(parameter) for parameter in model.parameters]
    fixed_values = np.array([options.fixed.get(name, np.nan) for name in names])
    return free, lower, upper, scales, fixed_values


def _start_vector(model: Model, x: np.ndarray, y: np.ndarray, options: FitOptions, lower: np.ndarray, upper: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """Initial guess inside the bounds, overridden by `options.initial`."""
    p0 = np.array(model.initial_guess(x, y), dtype=np.float64)
    for name, value in options.initial.items():
        p0[model.parameter_names.index(name)] = value
    for i, scale in enumerate(scales):
        if scale is not ParameterScale.LINEAR:
            p0[i] = max(p0[i], 1e-12) if not np.isfinite(p0[i]) or p0[i] <= 0 else p0[i]
    inside = np.clip(p0, lower, upper)
    # strictly inside for the optimizer
    span = np.where(np.isfinite(upper - lower), (upper - lower) * 1e-6, 0.0)
    return np.clip(inside, lower + span, upper - span)


def _nan_rowfit(k: int, n: int, flags: int, names: Sequence[str], model: Model) -> RowFit:
    """A row without a fit."""
    nan_k = np.full(k, np.nan)
    derived_names = list(model.derived_units)
    nan_d = dict.fromkeys(derived_names, math.nan)
    return RowFit(
        p=nan_k, q=nan_k.copy(), se_p=nan_k.copy(), ci_low=nan_k.copy(), ci_high=nan_k.copy(),
        cov_q=np.full((k, k), np.nan), correlation=np.full((k, k), np.nan),
        derived=dict(nan_d), derived_se=dict(nan_d), derived_ci_low=dict(nan_d), derived_ci_high=dict(nan_d),
        cost=math.nan, r2=math.nan, rmse=math.nan, aic=math.nan, aicc=math.nan, bic=math.nan,
        n_points=n, n_starts_converged=0, y_pred=np.full(n, np.nan), residuals=np.full(n, np.nan), flags=flags, nfev=0,
    )


def _starts(q0: np.ndarray, lq: np.ndarray, uq: np.ndarray, scales: Sequence[ParameterScale], options: FitOptions, rng: np.random.Generator) -> np.ndarray:
    """Start points in the scaled space: the guess plus Latin hypercube samples in the start box."""
    if options.n_starts == 1:
        return q0[None, :]
    from scipy.stats.qmc import LatinHypercube  # noqa: PLC0415

    half = np.array([
        math.log10(options.start_spread) if s is ParameterScale.LOG10 else math.log(options.start_spread) if s is ParameterScale.LOG else options.start_spread * max(abs(v), 1.0)
        for s, v in zip(scales, q0, strict=True)
    ])
    lo = np.maximum(q0 - half, lq)
    hi = np.minimum(q0 + half, uq)
    unit = LatinHypercube(d=q0.size, seed=rng).random(options.n_starts - 1)
    return np.vstack([q0[None, :], lo + unit * (hi - lo)])


def fit_row(model: Model, x: np.ndarray, y: np.ndarray, sd: np.ndarray | None, options: FitOptions, rng: np.random.Generator) -> RowFit:
    """Fit one row from one or several start points and compute its statistics.

    Args:
        model: the model
        x: independent variable `(n,)`
        y: dependent variable `(n,)`, `NaN` for missing points
        sd: standard deviation per point for `Weighting.INV_SD`, else `None`
        options: the options
        rng: random generator of the start points

    Returns:
        The fit of the row.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if sd is not None:
        sd = np.asarray(sd, dtype=np.float64)
        ok &= np.isfinite(sd) if options.weighting is Weighting.INV_SD else True
    names = model.parameter_names
    k_all = len(names)
    free, lower, upper, scales, fixed_values = _problem(model, options)
    k = int(free.sum())
    n = int(ok.sum())
    if n < 2:
        return _nan_rowfit(k_all, x.size, int(FitFlag.NO_DATA), names, model)
    if n < k + 1:
        return _nan_rowfit(k_all, x.size, int(FitFlag.TOO_FEW_POINTS), names, model)
    xs, ys = x[ok], y[ok]
    var = variance_of(ys, None if sd is None else sd[ok], options.weighting)
    sqrt_var = np.sqrt(var)
    p0 = _start_vector(model, xs, ys, options, lower, upper, scales)
    p0 = np.where(free, p0, fixed_values)
    lq_all, uq_all = bounds_in_scale(lower, upper, scales)
    q0_all = to_scale(p0, scales)

    def full_p(q_free: np.ndarray) -> np.ndarray:
        q = q0_all.copy()
        q[free] = q_free
        return from_scale(q, scales)

    def residuals(q_free: np.ndarray) -> np.ndarray:
        return (ys - model.predict(xs, full_p(q_free))) / sqrt_var

    starts = _starts(q0_all[free], lq_all[free], uq_all[free], [s for s, f in zip(scales, free, strict=True) if f], options, rng)
    best = None
    n_converged = 0
    nfev = 0
    for q_start in starts:
        try:
            solution = least_squares(
                residuals, q_start, bounds=(lq_all[free], uq_all[free]), method="trf", loss=options.loss,
                max_nfev=options.max_nfev, ftol=options.ftol, xtol=options.xtol, gtol=options.gtol,
            )
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.debug("start failed: %s", err)
            continue
        nfev += int(solution.nfev)
        converged = solution.status > 0
        n_converged += int(converged)
        if best is None or (converged and not best[1]) or (converged == best[1] and solution.cost < best[0].cost):
            best = (solution, converged)
    if best is None:
        return _nan_rowfit(k_all, x.size, int(FitFlag.NOT_CONVERGED), names, model)
    solution, converged = best
    flags = 0 if converged else int(FitFlag.NOT_CONVERGED)

    q_free = np.array(solution.x, dtype=np.float64)
    jac = np.asarray(solution.jac, dtype=np.float64)
    p = full_p(q_free)
    # reorder phases of sums of exponentials
    order_fn = getattr(model, "parameter_order", None)
    if order_fn is not None:
        order = np.asarray(order_fn(p))
        if not np.array_equal(order, np.arange(k_all)):
            p = p[order]
            free_order = [int(np.flatnonzero(np.flatnonzero(free) == i)[0]) for i in order if free[i]]
            jac = jac[:, free_order]
            q_free = to_scale(p, scales)[free]
    cov_free, singular = covariance(jac, float(solution.cost), n, k)
    if singular:
        flags |= int(FitFlag.SINGULAR)
    q_all = to_scale(p, scales)
    dpdq = scale_derivative(p, scales)
    se_q = np.sqrt(np.clip(np.diag(cov_free), 0.0, None))
    tq = float(student_t.ppf(1.0 - (1.0 - options.ci_level) / 2.0, max(n - k, 1)))
    se_p = np.full(k_all, np.nan)
    se_p[free] = se_q * np.abs(dpdq[free])
    lo_q, hi_q = q_all.copy(), q_all.copy()
    lo_q[free] = q_all[free] - tq * se_q
    hi_q[free] = q_all[free] + tq * se_q
    ci_low = np.where(free, from_scale(lo_q, scales), np.nan)
    ci_high = np.where(free, from_scale(hi_q, scales), np.nan)
    corr = np.full((k_all, k_all), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_free = cov_free / np.outer(se_q, se_q)
    corr[np.ix_(free, free)] = corr_free
    # at bound
    for i in np.flatnonzero(free):
        for bound in (lower[i], upper[i]):
            if np.isfinite(bound) and np.isclose(p[i], bound, rtol=options.at_bound_tolerance, atol=options.at_bound_tolerance * 1e-12):
                flags |= int(FitFlag.AT_BOUND)
    # derived parameters with the delta method
    derived = model.derived(p)
    derived_se: dict[str, float] = {}
    derived_lo: dict[str, float] = {}
    derived_hi: dict[str, float] = {}
    for name, value in derived.items():
        grad = np.zeros(k)
        for j, i in enumerate(np.flatnonzero(free)):
            h = 1e-6 * max(abs(q_all[i]), 1.0)
            q_plus, q_minus = q_all.copy(), q_all.copy()
            q_plus[i] += h
            q_minus[i] -= h
            grad[j] = (model.derived(from_scale(q_plus, scales))[name] - model.derived(from_scale(q_minus, scales))[name]) / (2 * h)
        var_d = float(grad @ cov_free @ grad) if not singular else math.nan
        se_d = math.sqrt(var_d) if var_d >= 0 else math.nan
        derived_se[name] = se_d
        derived_lo[name] = value - tq * se_d
        derived_hi[name] = value + tq * se_d
    if derived.get("flip_flop", 0.0) >= 1.0:
        flags |= int(FitFlag.FLIP_FLOP)
    # statistics
    y_pred = model.predict(xs, p)
    e = ys - y_pred
    r = e / sqrt_var
    rss_w = float(np.sum(r**2))
    tss = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(e**2)) / tss if tss > 0 else math.nan
    rmse = math.sqrt(float(np.sum(e**2)) / n)
    with np.errstate(divide="ignore"):
        ln_term = n * math.log(rss_w / n) if rss_w > 0 else -math.inf
    aic = ln_term + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.nan
    bic = ln_term + k * math.log(n)
    full_pred = np.full(x.size, np.nan)
    full_res = np.full(x.size, np.nan)
    full_pred[ok] = y_pred
    full_res[ok] = r
    return RowFit(
        p=p, q=q_all, se_p=se_p, ci_low=ci_low, ci_high=ci_high, cov_q=_embed(cov_free, free, k_all), correlation=corr,
        derived=derived, derived_se=derived_se, derived_ci_low=derived_lo, derived_ci_high=derived_hi,
        cost=float(solution.cost), r2=r2, rmse=rmse, aic=aic, aicc=aicc, bic=bic, n_points=n,
        n_starts_converged=n_converged, y_pred=full_pred, residuals=full_res, flags=flags, nfev=nfev,
    )
```
Add the helper:
```python
def _embed(cov_free: np.ndarray, free: np.ndarray, k_all: int) -> np.ndarray:
    """The covariance of the free parameters in the full parameter grid (NaN for fixed ones)."""
    cov = np.full((k_all, k_all), np.nan)
    cov[np.ix_(free, free)] = cov_free
    return cov
```
and the batch functions:
```python
def fit_rows(model: Model, x: np.ndarray, y: np.ndarray, sd: np.ndarray | None, options: FitOptions) -> list[RowFit]:
    """Fit every row of `(N, n)` arrays serially (Task 6 adds the worker pool)."""
    rng = np.random.default_rng(options.seed)
    seeds = rng.integers(0, 2**32 - 1, size=y.shape[0])
    return [
        fit_row(model, x[i], y[i], None if sd is None else sd[i], options, np.random.default_rng(int(seeds[i])))
        for i in range(y.shape[0])
    ]


def _as_rows(x: Any, y: Any, sd: Any | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """`(N, n)` arrays from 1-D or 2-D inputs; the flag says whether the input was a single row."""
    y_arr = np.asarray(y, dtype=np.float64)
    single = y_arr.ndim == 1
    if single:
        y_arr = y_arr[None, :]
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = np.broadcast_to(x_arr, y_arr.shape).copy()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"'x' has shape {x_arr.shape}, 'y' has shape {y_arr.shape}")
    sd_arr = None
    if sd is not None:
        sd_arr = np.asarray(sd, dtype=np.float64)
        if sd_arr.ndim == 1:
            sd_arr = sd_arr[None, :] if single else np.broadcast_to(sd_arr, y_arr.shape).copy()
        if sd_arr.shape != y_arr.shape:
            raise ValueError(f"'sd' has shape {sd_arr.shape}, 'y' has shape {y_arr.shape}")
    return x_arr, y_arr, sd_arr, single


def fit(
    model: Model,
    x: Any,
    y: Any,
    *,
    sd: Any | None = None,
    options: FitOptions | None = None,
    x_unit: str = "dimensionless",
    y_unit: str = "dimensionless",
    dims: Sequence[str] | None = None,
    coords: dict[str, Any] | None = None,
) -> FitResult:
    """Fit a model to one or many rows of data.

    Args:
        model: the model
        x: independent variable, `(n,)` shared by all rows or `(N, n)`
        y: dependent variable, `(n,)` for one sample or `(N, n)`; `NaN` for missing points
        sd: standard deviation per point (needed for `Weighting.INV_SD`), like `y`
        options: the options, defaults for `None`
        x_unit: unit of `x`
        y_unit: unit of `y`
        dims: sample dimension names for a 2-D `y`, `("sample",)` by default
        coords: coordinate labels of the sample dimensions

    Returns:
        The result over the sample dimensions (none for a 1-D `y`).
    """
    options = options or FitOptions()
    x_arr, y_arr, sd_arr, single = _as_rows(x, y, sd)
    rows = fit_rows(model, x_arr, y_arr, sd_arr, options)
    sample_dims: tuple[str, ...] = () if single else tuple(dims or ("sample",))
    if not single and len(sample_dims) != 1:
        raise ValueError("2-D data has exactly one sample dimension; use fit_timecourses or fit_table for more")
    return build_result(model, rows, x=x_arr, y=y_arr, sd=sd_arr, x_unit=x_unit, y_unit=y_unit, dims=sample_dims, coords=coords or {}, options=options)


def build_result(model: Model, rows: list[RowFit], *, x: np.ndarray, y: np.ndarray, sd: np.ndarray | None, x_unit: str, y_unit: str, dims: tuple[str, ...], coords: dict[str, Any], options: FitOptions, shape: tuple[int, ...] | None = None) -> FitResult:
    """Assemble the result dataset of the fitted rows.

    Args:
        model: the model
        rows: one `RowFit` per row
        x: `(N, n)` independent variable
        y: `(N, n)` dependent variable
        sd: `(N, n)` standard deviations or `None`
        x_unit: unit of `x`
        y_unit: unit of `y`
        dims: sample dimension names (`()` for a single row)
        coords: coordinates of the sample dimensions
        options: the options (stored in `attrs`)
        shape: sample shape for several sample dimensions (`N = prod(shape)`), `(N,)` by default

    Returns:
        The `FitResult`.
    """
    n_rows, n_points = y.shape
    sample_shape: tuple[int, ...] = () if not dims else (shape if shape is not None else (n_rows,))
    names = model.parameter_names
    k = len(names)

    def units_of(expr: str) -> tuple[str, float]:
        return parameter_unit_expression(expr, x_unit=x_unit, y_unit=y_unit)

    data_vars: dict[str, Any] = {}

    def scalar(name: str, values: list[float] | np.ndarray, unit: str) -> None:
        data_vars[name] = (dims, np.asarray(values, dtype=np.float64).reshape(sample_shape), {"units": unit})

    for j, parameter in enumerate(model.parameters):
        unit, factor = units_of(parameter.unit_expr)
        scalar(parameter.name, [r.p[j] * factor for r in rows], unit)
        scalar(f"{parameter.name}_se", [r.se_p[j] * factor for r in rows], unit)
        scalar(f"{parameter.name}_ci_low", [r.ci_low[j] * factor for r in rows], unit)
        scalar(f"{parameter.name}_ci_high", [r.ci_high[j] * factor for r in rows], unit)
        with np.errstate(divide="ignore", invalid="ignore"):
            scalar(f"{parameter.name}_cv", [100.0 * r.se_p[j] / abs(r.p[j]) for r in rows], "dimensionless")
    for name, expr in model.derived_units.items():
        unit, factor = units_of(expr)
        scalar(name, [r.derived.get(name, math.nan) * factor for r in rows], unit)
        scalar(f"{name}_se", [r.derived_se.get(name, math.nan) * factor for r in rows], unit)
        scalar(f"{name}_ci_low", [r.derived_ci_low.get(name, math.nan) * factor for r in rows], unit)
        scalar(f"{name}_ci_high", [r.derived_ci_high.get(name, math.nan) * factor for r in rows], unit)
        with np.errstate(divide="ignore", invalid="ignore"):
            scalar(f"{name}_cv", [100.0 * r.derived_se.get(name, math.nan) / abs(r.derived.get(name, math.nan)) for r in rows], "dimensionless")
    for stat in ("cost", "r2", "rmse", "aic", "aicc", "bic"):
        scalar(stat, [getattr(r, stat) for r in rows], "dimensionless")
    scalar("n_points", [r.n_points for r in rows], "dimensionless")
    scalar("n_parameters", [int(np.sum([n not in options.fixed for n in names])) for _ in rows], "dimensionless")
    scalar("n_starts_converged", [r.n_starts_converged for r in rows], "dimensionless")
    point_dims = (*dims, "point")
    point_shape = (*sample_shape, n_points)
    data_vars["x_data"] = (point_dims, x.reshape(point_shape), {"units": x_unit})
    data_vars["y_data"] = (point_dims, y.reshape(point_shape), {"units": y_unit})
    data_vars["y_pred"] = (point_dims, np.stack([r.y_pred for r in rows]).reshape(point_shape), {"units": y_unit})
    data_vars["residuals"] = (point_dims, np.stack([r.residuals for r in rows]).reshape(point_shape), {"units": "dimensionless"})
    data_vars["correlation"] = ((*dims, "parameter", "parameter_"), np.stack([r.correlation for r in rows]).reshape(*sample_shape, k, k), {"units": "dimensionless"})
    data_vars["flags"] = (dims, np.array([r.flags for r in rows], dtype=np.int64).reshape(sample_shape), {"units": "dimensionless"})
    all_coords: dict[str, Any] = {**coords, "parameter": list(names), "parameter_": list(names)}
    attrs = {"model": model.name, "parameter_names": list(names), "x_unit": x_unit, "y_unit": y_unit, "weighting": str(options.weighting), "parameter_scale": str(options.parameter_scale)}
    return FitResult(xr.Dataset(data_vars=data_vars, coords=all_coords, attrs=attrs), model)
```
Notes for the implementer:
- `ParameterResult.parameters` must treat `correlation`, `x_data`, `y_data`, `y_pred`, `residuals` as point variables (they carry extra dims) - already the case from Task 1.
- The `_cv` of a fixed parameter is NaN (se NaN).
- `to_quantities` includes the statistics (`cost`, `r2`, ...) as dimensionless quantities; `n_points` etc. are discrete.
- Exports: add `FitResult`, `fit`, `fit_row`, `fit_rows`, `RowFit`, `build_result` to `pkpdutils/fit/__init__.py` (`__all__` sorted).
- `Bateman` with `initial={"ka": 0.2, "ke": 0.6}`: the optimizer may or may not swap into a non-flip-flop optimum; the test accepts either the flag or `ka > ke`.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/fit -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. Common issues: scipy emits no warnings with the defaults; `np.polyfit` on degenerate data can warn (`RankWarning`) inside `initial_guess` - wrap `np.polyfit` calls in `warnings.catch_warnings()` if a test triggers it; `student_t.ppf` with `df >= 1` is fine.

```bash
git add src/pkpdutils/fit tests/fit
git commit -q -m "Add the fitting engine and the fit result

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 6: Multi-start, worker pool, residual bootstrap and model comparison

**Files:**
- Modify: `src/pkpdutils/fit/engine.py` (`fit_rows` pool, bootstrap), `src/pkpdutils/fit/__init__.py`
- Create: `src/pkpdutils/fit/compare.py`
- Test: `tests/fit/test_engine_advanced.py`

**Interfaces:**
- `fit_rows(model, x, y, sd, options) -> list[RowFit]`: with `options.n_workers > 1` and more than one row the rows are mapped in order over a `ProcessPoolExecutor(max_workers=n_workers)` with the module-level worker `_fit_row_job(args)`; per-row seeds come from `np.random.default_rng(options.seed)` so results do not depend on the number of workers
- residual bootstrap in `fit_row` when `options.bootstrap > 0` (Efron & Tibshirani 1993, ch. 9): the weighted residuals `r` of the fit are resampled with replacement, `y* = f + r* sqrt(var)`, the model is refitted from the fitted `p` (single start, same bounds and scale), `B` times; then `se_p` = standard deviation of the replicate parameters (ddof 1), `ci_low`/`ci_high` = percentiles at `ci_level`, the derived parameters likewise (`derived_se`, `derived_ci_*` from the replicate derived values), the correlation matrix from the replicates in the scaled space; replicates whose refit fails are skipped; `RowFit.n_bootstrap` (new field, 0 without bootstrap) counts the successful replicates; the result dataset gets `attrs["bootstrap"] = options.bootstrap`
- `fit/compare.py`: `@dataclass ModelComparison`: `results: dict[str, FitResult]`, `table: pd.DataFrame` (columns: the sample dims, `model`, `n_parameters`, `aicc`, `delta_aicc`, `akaike_weight`, `best` (bool)), `best: xr.DataArray` (model name per sample over the sample dims, `""` when no model fitted); `compare_models(models: Sequence[Model], x, y, *, sd=None, options=None, x_unit=..., y_unit=..., dims=None, coords=None) -> ModelComparison`: fits every model with `fit`, ranks by AICc per sample (Burnham & Anderson 2002: `Δ_i = AICc_i - min AICc`, `w_i = exp(-Δ_i/2) / Σ exp(-Δ_j/2)`); models with NaN AICc get weight 0
- exports `compare_models`, `ModelComparison`

- [ ] **Step 1: Write the failing tests**

`tests/fit/test_engine_advanced.py`:
```python
import numpy as np
import pytest

from pkpdutils.fit import FitOptions, compare_models, fit
from pkpdutils.fit.models import BiExp, Emax, MonoExp, SigmoidEmax

T = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24])


def data(seed: int = 0, cv: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 10.0 * np.exp(-0.3 * T) * rng.lognormal(0.0, cv, size=T.size)


def test_multistart_recovers_from_a_bad_start() -> None:
    y = BiExp().predict(T, np.array([8.0, 2.0, 2.0, 0.2])) * np.random.default_rng(1).lognormal(0, 0.02, T.size)
    bad = FitOptions(initial={"a1": 0.01, "k1": 30.0, "a2": 0.01, "k2": 10.0}, n_starts=1, max_nfev=50)
    good = FitOptions(initial={"a1": 0.01, "k1": 30.0, "a2": 0.01, "k2": 10.0}, n_starts=20, seed=0)
    single = fit(BiExp(), T, y, options=bad).to_quantities()
    multi = fit(BiExp(), T, y, options=good)
    q = multi.to_quantities()
    assert q["cost"].magnitude <= single["cost"].magnitude
    assert q["k2"].magnitude == pytest.approx(0.2, rel=0.1)
    assert q["n_starts_converged"].magnitude >= 1


def test_workers_equal_serial() -> None:
    ys = np.stack([data(seed=i) for i in range(4)])
    serial = fit(MonoExp(), T, ys, options=FitOptions(n_starts=3, seed=5))
    parallel = fit(MonoExp(), T, ys, options=FitOptions(n_starts=3, seed=5, n_workers=2))
    for name in serial.parameters:
        np.testing.assert_allclose(parallel[name].values, serial[name].values, equal_nan=True)
    np.testing.assert_array_equal(parallel["flags"].values, serial["flags"].values)


def test_bootstrap_agrees_with_jacobian() -> None:
    y = data(seed=2)
    jac = fit(MonoExp(), T, y).to_quantities()
    boot = fit(MonoExp(), T, y, options=FitOptions(bootstrap=400, seed=3))
    q = boot.to_quantities()
    assert boot.ds.attrs["bootstrap"] == 400
    assert q["n_bootstrap"].magnitude > 350
    assert q["k_se"].magnitude == pytest.approx(jac["k_se"].magnitude, rel=0.5)
    assert q["k_ci_low"].magnitude < q["k"].magnitude < q["k_ci_high"].magnitude
    assert q["thalf_se"].magnitude > 0
    corr = boot.correlation()
    assert abs(corr.loc["a", "k"]) < 1.0


def test_bootstrap_reproducible_with_seed() -> None:
    y = data(seed=4)
    a = fit(MonoExp(), T, y, options=FitOptions(bootstrap=100, seed=7)).to_quantities()
    b = fit(MonoExp(), T, y, options=FitOptions(bootstrap=100, seed=7)).to_quantities()
    assert a["k_se"].magnitude == b["k_se"].magnitude


def test_ci_coverage_of_the_jacobian_interval() -> None:
    hits = 0
    for seed in range(40):
        q = fit(MonoExp(), T, data(seed=100 + seed, cv=0.1)).to_quantities()
        hits += q["k_ci_low"].magnitude <= 0.3 <= q["k_ci_high"].magnitude
    assert hits >= 32  # 95 % nominal, allow 80 %


def test_compare_models_prefers_the_true_model() -> None:
    y = data(seed=6)
    comparison = compare_models([MonoExp(), BiExp()], T, y, x_unit="hr", y_unit="mg/l")
    assert set(comparison.results) == {"monoexp", "biexp"}
    table = comparison.table
    assert list(table.columns) == ["model", "n_parameters", "aicc", "delta_aicc", "akaike_weight", "best"]
    assert table["akaike_weight"].sum() == pytest.approx(1.0)
    assert str(comparison.best.values) == "monoexp"
    assert table.loc[table["best"], "model"].item() == "monoexp"
    assert table.loc[table["model"] == "monoexp", "delta_aicc"].item() == 0.0


def test_compare_models_batch() -> None:
    ys = np.stack([data(seed=i) for i in range(3)])
    comparison = compare_models([MonoExp(), BiExp()], T, ys, dims=("individual",), coords={"individual": ["a", "b", "c"]})
    assert comparison.best.dims == ("individual",)
    assert list(comparison.table.columns)[0] == "individual"
    assert len(comparison.table) == 6
    weights = comparison.table.groupby("individual")["akaike_weight"].sum()
    np.testing.assert_allclose(weights.values, 1.0)


def test_compare_models_with_a_failed_model() -> None:
    c = np.array([0.5, 1, 2, 5, 10, 20, 50])
    e = Emax().predict(c, np.array([1.0, 8.0, 5.0]))
    comparison = compare_models([Emax(), SigmoidEmax()], c, e)
    assert set(comparison.results) == {"emax", "sigmoid_emax"}
    assert comparison.table["akaike_weight"].sum() == pytest.approx(1.0)
```

Run: `uv run pytest tests/fit/test_engine_advanced.py -n 0 -q` → FAIL with `ImportError: cannot import name 'compare_models'`.

- [ ] **Step 2: Implement**

In `engine.py`:
- add `n_bootstrap: int = 0` to `RowFit` (last field), and `scalar("n_bootstrap", [r.n_bootstrap for r in rows], "dimensionless")` in `build_result`, plus `attrs["bootstrap"] = options.bootstrap`; add `"n_bootstrap"` to `FitResult.discrete_parameters`.
- residual bootstrap at the end of `fit_row` (after the statistics, before returning), when `options.bootstrap > 0`:
```python
    if options.bootstrap > 0:
        rows_p, rows_q, rows_d = [], [], []
        for _ in range(options.bootstrap):
            r_star = rng.choice(r, size=r.size, replace=True)
            y_star = y_pred + r_star * sqrt_var

            def residuals_star(qf: np.ndarray, y_star: np.ndarray = y_star) -> np.ndarray:
                return (y_star - model.predict(xs, full_p(qf))) / sqrt_var

            try:
                sol = least_squares(residuals_star, q_free, bounds=(lq_all[free], uq_all[free]), method="trf", loss=options.loss, max_nfev=options.max_nfev, ftol=options.ftol, xtol=options.xtol, gtol=options.gtol)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if sol.status <= 0:
                continue
            p_star = full_p(np.asarray(sol.x))
            if order_fn is not None:
                p_star = p_star[np.asarray(order_fn(p_star))]
            rows_p.append(p_star)
            rows_q.append(to_scale(p_star, scales))
            rows_d.append(model.derived(p_star))
        n_boot = len(rows_p)
        if n_boot >= 2:
            arr_p = np.stack(rows_p)
            arr_q = np.stack(rows_q)
            alpha = 1.0 - options.ci_level
            se_p = np.where(free, arr_p.std(axis=0, ddof=1), np.nan)
            ci_low = np.where(free, np.percentile(arr_p, 100 * alpha / 2, axis=0), np.nan)
            ci_high = np.where(free, np.percentile(arr_p, 100 * (1 - alpha / 2), axis=0), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_b = np.corrcoef(arr_q[:, free], rowvar=False)
            corr = np.full((k_all, k_all), np.nan)
            corr[np.ix_(free, free)] = np.atleast_2d(corr_b)
            for name in derived:
                values = np.array([d[name] for d in rows_d], dtype=np.float64)
                derived_se[name] = float(values.std(ddof=1))
                derived_lo[name] = float(np.percentile(values, 100 * alpha / 2))
                derived_hi[name] = float(np.percentile(values, 100 * (1 - alpha / 2)))
    else:
        n_boot = 0
```
(`r` and `y_pred` are the weighted residuals and the prediction of the statistics block, which therefore runs before the bootstrap). Pass `n_bootstrap=n_boot` into the returned `RowFit`.
- worker pool:
```python
def _fit_row_job(args: tuple[Any, ...]) -> RowFit:
    """Worker entry: one row."""
    model, x, y, sd, options, seed = args
    return fit_row(model, x, y, sd, options, np.random.default_rng(int(seed)))


def fit_rows(model: Model, x: np.ndarray, y: np.ndarray, sd: np.ndarray | None, options: FitOptions) -> list[RowFit]:
    """Fit every row of `(N, n)` arrays, serially or in a process pool; row seeds come from `options.seed`."""
    rng = np.random.default_rng(options.seed)
    seeds = rng.integers(0, 2**32 - 1, size=y.shape[0])
    jobs = [(model, x[i], y[i], None if sd is None else sd[i], options, seeds[i]) for i in range(y.shape[0])]
    if options.n_workers is not None and options.n_workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=options.n_workers) as pool:
            return list(pool.map(_fit_row_job, jobs))
    return [_fit_row_job(job) for job in jobs]
```
Models must be picklable: they are plain objects (`Bateman(lag=...)`, `Allometric(exponent=...)` store attributes) - fine.

`src/pkpdutils/fit/compare.py`:
```python
"""Comparison of models by the corrected Akaike information criterion.

`compare_models` fits every model to the same data and ranks them per sample
by AICc (Burnham & Anderson 2002): `Δ_i = AICc_i - min_j AICc_j` and the
Akaike weight `w_i = exp(-Δ_i / 2) / Σ_j exp(-Δ_j / 2)`, the probability that
model `i` is the best of the set.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pkpdutils.fit.engine import fit
from pkpdutils.fit.model import Model
from pkpdutils.fit.options import FitOptions
from pkpdutils.fit.result import FitResult


@dataclass
class ModelComparison:
    """The fits of several models and their ranking.

    Attributes:
        results: model name to its `FitResult`
        table: one row per sample and model with `aicc`, `delta_aicc`, `akaike_weight`, `best`
        best: name of the best model per sample (empty string when no model fitted)
    """

    results: dict[str, FitResult]
    table: pd.DataFrame
    best: xr.DataArray


def compare_models(
    models: Sequence[Model],
    x: Any,
    y: Any,
    *,
    sd: Any | None = None,
    options: FitOptions | None = None,
    x_unit: str = "dimensionless",
    y_unit: str = "dimensionless",
    dims: Sequence[str] | None = None,
    coords: dict[str, Any] | None = None,
) -> ModelComparison:
    """Fit every model and rank them by AICc per sample.

    Args:
        models: the models (distinct names)
        x: independent variable, as for `fit`
        y: dependent variable, as for `fit`
        sd: standard deviations, as for `fit`
        options: fit options shared by every model
        x_unit: unit of `x`
        y_unit: unit of `y`
        dims: sample dimension names for a 2-D `y`
        coords: coordinates of the sample dimensions

    Returns:
        The comparison.
    """
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"Model names must be distinct: {names}")
    results = {
        m.name: fit(m, x, y, sd=sd, options=options, x_unit=x_unit, y_unit=y_unit, dims=dims, coords=coords)
        for m in models
    }
    first = next(iter(results.values()))
    sample_dims = first.sample_dims
    aicc = np.stack([results[name]["aicc"].to_numpy() for name in names], axis=-1)  # (*sample_shape, n_models)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(aicc)
        best_value = np.nanmin(np.where(finite, aicc, np.inf), axis=-1)
        delta = np.where(finite, aicc - best_value[..., None], np.nan)
        weight = np.where(finite, np.exp(-0.5 * np.where(finite, delta, 0.0)), 0.0)
        total = weight.sum(axis=-1, keepdims=True)
        weight = np.where(total > 0, weight / np.where(total > 0, total, 1.0), 0.0)
    best_index = np.where(finite.any(axis=-1), np.argmax(weight, axis=-1), -1)
    best_names = np.array([*names, ""], dtype=object)[best_index]
    coords_sample = {d: first.ds[d] for d in sample_dims if d in first.ds.coords}
    best = xr.DataArray(best_names, dims=sample_dims, coords=coords_sample, name="best_model")
    rows = []
    for index in np.ndindex(*aicc.shape[:-1]):
        labels = {d: (first.ds[d].values[i] if d in first.ds.coords else i) for d, i in zip(sample_dims, index, strict=True)}
        for j, name in enumerate(names):
            rows.append(
                {
                    **labels,
                    "model": name,
                    "n_parameters": int(results[name]["n_parameters"].to_numpy()[index]),
                    "aicc": float(aicc[(*index, j)]),
                    "delta_aicc": float(delta[(*index, j)]),
                    "akaike_weight": float(weight[(*index, j)]),
                    "best": bool(best_index[index] == j),
                }
            )
    table = pd.DataFrame(rows, columns=[*sample_dims, "model", "n_parameters", "aicc", "delta_aicc", "akaike_weight", "best"])
    return ModelComparison(results=results, table=table, best=best)
```
Export `compare_models`, `ModelComparison` from `pkpdutils.fit`.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/fit -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. `test_multistart_recovers_from_a_bad_start`: with `max_nfev=50` the single bad start does not reach the optimum; if it does on this platform, tighten to `max_nfev=20`. `test_compare_models_with_a_failed_model`: SigmoidEmax on 7 points with 4 parameters still fits; the test only checks the weights sum to 1.

```bash
git add src/pkpdutils/fit tests/fit
git commit -q -m "Add multi-start, the worker pool, the residual bootstrap and the model comparison

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 7: Front ends: `fit_timecourses`, `fit_table`, `proportionality_test`

**Files:**
- Create: `src/pkpdutils/fit/frontends.py`, `src/pkpdutils/fit/proportionality.py`
- Modify: `src/pkpdutils/fit/__init__.py`
- Test: `tests/fit/test_frontends.py`

**Interfaces:**
- `fit_timecourses(model, timecourses: Timecourses, options=None) -> FitResult`: `x` = times relative to the dose (`dose_time` subtracted), `y` = values, `sd` = `timecourses.sd` (needed for `INV_SD`), units from the batch, sample dims and coordinates from the batch (any number of sample dims; the rows are the flattened samples and `build_result` gets `shape=timecourses.sample_shape`); `x_unit = time_unit`, `y_unit = unit`
- `fit_table(model, ds: xr.Dataset, x: str, y: str, *, dim: str, sd: str | None = None, options=None) -> FitResult`: fits `y` against `x` along `dim` for every combination of the other dimensions of `y`; `x` may be a coordinate or a variable of `ds` (broadcast); units from `attrs["units"]` of `x` (`"dimensionless"` when absent) and `y`; the result's sample dims are the remaining dims of `y`, 0-D when `y` has only `dim`. Works directly on `NCAResult.ds` (e.g. `fit_table(Power(), result.ds, "dose", "auc_inf_obs", dim="dose")` when `dose` is a coordinate)
- `proportionality.py`: `proportionality_test(result: FitResult, *, dose_range: tuple[float, float], criterion: tuple[float, float] = (0.8, 1.25)) -> xr.Dataset`: for a `Power` (or `Allometric` with a free exponent) result, per sample: `b`, `b_ci_low`, `b_ci_high` (from the result), `bound_low = 1 + ln(θ_L)/ln(r)`, `bound_high = 1 + ln(θ_H)/ln(r)` with `r = high/low` of `dose_range`, `proportional` (bool: the CI lies inside the bounds), `inconclusive` (bool: the CI overlaps but is not inside); Smith et al. 2000. `ValueError` when the result has no `b`.
- exports `fit_timecourses`, `fit_table`, `proportionality_test`

- [ ] **Step 1: Write the failing tests**

`tests/fit/test_frontends.py`:
```python
import numpy as np
import pytest
import xarray as xr

from pkpdutils import Dose, Route, Timecourse, Timecourses, nca
from pkpdutils.fit import FitOptions, Weighting, fit_table, fit_timecourses, proportionality_test
from pkpdutils.fit.models import Allometric, MonoExp, Power

T = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])


def curves(n: int = 3) -> Timecourses:
    rng = np.random.default_rng(0)
    tcs = []
    for i in range(n):
        k = 0.2 + 0.1 * i
        y = 10 * np.exp(-k * T) * rng.lognormal(0, 0.03, T.size)
        tcs.append(Timecourse(time=T + 5.0, value=y, sd=0.05 * y, time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS, time=5.0), substance="x", label=f"s{i}"))
    return Timecourses.from_timecourses(tcs)


def test_fit_timecourses_relative_to_dose_with_units() -> None:
    result = fit_timecourses(MonoExp(), curves(), FitOptions(weighting=Weighting.INV_SD))
    assert result.sample_dims == ("individual",)
    np.testing.assert_allclose(result["k"].values, [0.2, 0.3, 0.4], rtol=0.1)
    assert result.units("k") == "1 / hour" and result.units("a") == "milligram / liter"
    assert result.ds.attrs["x_unit"] == "hr"
    np.testing.assert_allclose(result["x_data"].values[0], T)
    assert list(result.ds["individual"].values) == ["s0", "s1", "s2"]


def test_fit_timecourses_two_sample_dims() -> None:
    time = T
    values = np.stack([np.stack([10 * np.exp(-k * time) * d for k in (0.2, 0.4)]) for d in (1.0, 2.0)])  # (dose, k, time)
    tcs = Timecourses.from_arrays(time, values, time_unit="hr", unit="mg/l", dims=("dose", "k"), coords={"dose": [1.0, 2.0], "k": [0.2, 0.4]})
    result = fit_timecourses(MonoExp(), tcs)
    assert result.sample_dims == ("dose", "k")
    np.testing.assert_allclose(result["k"].values, [[0.2, 0.4], [0.2, 0.4]], rtol=1e-4)
    np.testing.assert_allclose(result["a"].values, [[10, 10], [20, 20]], rtol=1e-4)


def test_fit_table_dose_proportionality() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    individuals = ["a", "b", "c"]
    rng = np.random.default_rng(1)
    auc = 2.0 * doses[:, None] ** 1.05 * rng.lognormal(0, 0.05, (4, 3))
    ds = xr.Dataset({"auc_inf_obs": (("dose", "individual"), auc, {"units": "hour * milligram / liter"})}, coords={"dose": ("dose", doses, {"units": "milligram"}), "individual": individuals})
    result = fit_table(Power(), ds, "dose", "auc_inf_obs", dim="dose")
    assert result.sample_dims == ("individual",)
    np.testing.assert_allclose(result["b"].values, 1.05, atol=0.15)
    assert result.units("a") == "hour * milligram / liter"
    pooled = fit_table(Power(), ds.mean("individual"), "dose", "auc_inf_obs", dim="dose")
    assert pooled.sample_dims == ()
    test = proportionality_test(pooled, dose_range=(25.0, 200.0))
    assert set(test.data_vars) >= {"b", "b_ci_low", "b_ci_high", "bound_low", "bound_high", "proportional", "inconclusive"}
    r = 200.0 / 25.0
    assert float(test["bound_low"].values) == pytest.approx(1 + np.log(0.8) / np.log(r))
    assert float(test["bound_high"].values) == pytest.approx(1 + np.log(1.25) / np.log(r))
    assert bool(test["proportional"].values) or bool(test["inconclusive"].values)


def test_proportionality_test_detects_nonproportional() -> None:
    doses = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 400.0])
    auc = 3.0 * doses**1.6
    result = fit_table(Power(), xr.Dataset({"auc": (("dose",), auc)}, coords={"dose": doses}), "dose", "auc", dim="dose")
    test = proportionality_test(result, dose_range=(10.0, 400.0))
    assert not bool(test["proportional"].values) and not bool(test["inconclusive"].values)
    with pytest.raises(ValueError, match="b"):
        proportionality_test(fit_table(Allometric(exponent=0.75), xr.Dataset({"cl": (("w",), 3.0 * doses**0.75)}, coords={"w": doses}), "w", "cl", dim="w"), dose_range=(10.0, 400.0))


def test_fit_table_on_nca_result() -> None:
    doses = np.array([50.0, 100.0, 200.0])
    time = T
    values = np.stack([d / 10 * np.exp(-0.3 * time) for d in doses])
    tcs = Timecourses.from_arrays(time, values, time_unit="hr", unit="mg/l", dims=("dose",), coords={"dose": doses}, dose={"amount": doses, "unit": "mg"}, route=Route.IV_BOLUS)
    result = nca(tcs)
    prop = fit_table(Power(), result.ds, "dose", "auc_inf_obs", dim="dose")
    assert prop.sample_dims == ()
    assert prop.to_quantities()["b"].magnitude == pytest.approx(1.0, abs=0.01)
```

Run: `uv run pytest tests/fit/test_frontends.py -n 0 -q` → FAIL with `ImportError`.

- [ ] **Step 2: Implement**

`src/pkpdutils/fit/frontends.py`:
```python
"""Front ends of the engine for batches of timecourses and for tables of parameters."""

from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.fit.engine import build_result, fit_rows
from pkpdutils.fit.model import Model
from pkpdutils.fit.options import FitOptions
from pkpdutils.fit.result import FitResult
from pkpdutils.timecourse import Timecourses


def fit_timecourses(model: Model, timecourses: Timecourses, options: FitOptions | None = None) -> FitResult:
    """Fit a model to every curve of a batch, with the times relative to the dose.

    Args:
        model: the model (`x` is the time, `y` the value)
        timecourses: the batch; `sd` is used for `Weighting.INV_SD`
        options: the options

    Returns:
        The result over the sample dimensions of the batch.
    """
    options = options or FitOptions()
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    x = timecourses.times.reshape(n_rows, n_time)
    if timecourses.dose_time is not None:
        x = x - np.asarray(timecourses.dose_time, dtype=np.float64).reshape(n_rows)[:, None]
    y = timecourses.values.reshape(n_rows, n_time)
    sd = None if timecourses.sd is None else timecourses.sd.reshape(n_rows, n_time)
    rows = fit_rows(model, x, y, sd, options)
    coords = {d: timecourses.ds[d] for d in timecourses.sample_dims if d in timecourses.ds.coords}
    return build_result(
        model, rows, x=x, y=y, sd=sd, x_unit=timecourses.time_unit, y_unit=timecourses.unit,
        dims=timecourses.sample_dims, coords=coords, options=options, shape=timecourses.sample_shape,
    )


def fit_table(
    model: Model,
    ds: xr.Dataset,
    x: str,
    y: str,
    *,
    dim: str,
    sd: str | None = None,
    options: FitOptions | None = None,
) -> FitResult:
    """Fit `y` against `x` along one dimension of a dataset, for every combination of the other dimensions.

    Args:
        model: the model
        ds: dataset with the variable `y` and the coordinate or variable `x` (e.g. `NCAResult.ds`)
        x: name of the independent variable
        y: name of the dependent variable
        dim: the dimension along which the points of one fit lie (e.g. `"dose"`)
        sd: name of the standard deviation variable of `y`, for `Weighting.INV_SD`
        options: the options

    Returns:
        The result over the remaining dimensions of `y` (none when `y` has only `dim`).
    """
    options = options or FitOptions()
    if dim not in ds[y].dims:
        raise ValueError(f"'{y}' has no dimension '{dim}'")
    y_da = ds[y].transpose(..., dim)
    x_da = xr.broadcast(ds[x], y_da)[0].transpose(..., dim)
    sample_dims = tuple(str(d) for d in y_da.dims if d != dim)
    shape = tuple(int(y_da.sizes[d]) for d in sample_dims)
    n = int(y_da.sizes[dim])
    n_rows = int(np.prod(shape, dtype=int)) if shape else 1
    x_arr = x_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    y_arr = y_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    sd_arr = None if sd is None else xr.broadcast(ds[sd], y_da)[0].transpose(..., dim).to_numpy().astype(np.float64).reshape(n_rows, n)
    rows = fit_rows(model, x_arr, y_arr, sd_arr, options)
    coords: dict[str, Any] = {d: ds[d] for d in sample_dims if d in ds.coords}
    return build_result(
        model, rows, x=x_arr, y=y_arr, sd=sd_arr,
        x_unit=str(ds[x].attrs.get("units", "dimensionless")), y_unit=str(ds[y].attrs.get("units", "dimensionless")),
        dims=sample_dims, coords=coords, options=options, shape=shape,
    )
```
`build_result` with `shape` for several sample dims: `dims` may have length > 1 there; the check "2-D data has exactly one sample dimension" lives in `fit()` only.

`src/pkpdutils/fit/proportionality.py`:
```python
"""Dose proportionality by the power model and the confidence interval criterion.

With `AUC = a D^b` the exposure is dose proportional when `b = 1`. Smith et
al. (2000) accept proportionality over a dose range `r = D_high / D_low` when
the confidence interval of `b` lies within
`[1 + ln(θ_L) / ln(r), 1 + ln(θ_H) / ln(r)]` with the acceptance limits
`(θ_L, θ_H) = (0.8, 1.25)` of the dose-normalized exposure ratio.
"""

import math

import numpy as np
import xarray as xr

from pkpdutils.fit.result import FitResult


def proportionality_test(
    result: FitResult,
    *,
    dose_range: tuple[float, float],
    criterion: tuple[float, float] = (0.8, 1.25),
) -> xr.Dataset:
    """Apply the confidence interval criterion to the exponent of a power model fit.

    Args:
        result: fit of `Power` (or `Allometric` with a free exponent), with `b`, `b_ci_low`, `b_ci_high`
        dose_range: lowest and highest dose of the range the criterion refers to
        criterion: acceptance limits of the dose-normalized ratio

    Returns:
        Dataset over the sample dimensions with `b`, `b_ci_low`, `b_ci_high`,
        `bound_low`, `bound_high`, `proportional` (the interval is inside the
        bounds) and `inconclusive` (the interval overlaps the bounds but is not inside).

    Raises:
        ValueError: if the result has no exponent `b` or the dose range is invalid.
    """
    if "b" not in result or "b_ci_low" not in result:
        raise ValueError("The result has no exponent 'b' with a confidence interval")
    low, high = dose_range
    if not (high > low > 0):
        raise ValueError(f"'dose_range' must be 0 < low < high, got {dose_range}")
    ratio = high / low
    bound_low = 1.0 + math.log(criterion[0]) / math.log(ratio)
    bound_high = 1.0 + math.log(criterion[1]) / math.log(ratio)
    b, lo, hi = result["b"], result["b_ci_low"], result["b_ci_high"]
    inside = (lo >= bound_low) & (hi <= bound_high)
    overlap = (hi >= bound_low) & (lo <= bound_high)
    ds = xr.Dataset(
        {
            "b": b, "b_ci_low": lo, "b_ci_high": hi,
            "bound_low": xr.full_like(b, bound_low), "bound_high": xr.full_like(b, bound_high),
            "proportional": inside & np.isfinite(lo) & np.isfinite(hi),
            "inconclusive": overlap & ~inside & np.isfinite(lo) & np.isfinite(hi),
        }
    )
    ds.attrs.update({"dose_range": list(dose_range), "criterion": list(criterion)})
    for name in ("bound_low", "bound_high"):
        ds[name].attrs["units"] = "dimensionless"
    return ds
```
Exports in `pkpdutils/fit/__init__.py`: `fit_table`, `fit_timecourses`, `proportionality_test`.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/fit -n 0 -q && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass. `test_fit_table_on_nca_result`: `nca` of an IV bolus batch gives exact `auc_inf_obs ∝ dose` with the default log-down rule on a mono-exponential, so `b == 1` within 0.01.

```bash
git add src/pkpdutils/fit tests/fit
git commit -q -m "Add the timecourse and table front ends and the dose proportionality test

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---
### Task 8: Figures of fits (`plot/fit.py`)

**Files:**
- Create: `src/pkpdutils/plot/fit.py`
- Modify: `src/pkpdutils/plot/__init__.py`
- Test: `tests/plot/test_plot_fit.py`

**Interfaces:**
- `plot_fit(result: FitResult, *, log_x=False, log_y=False, n_grid=200, title=None, style=DEFAULT_STYLE, **indexers) -> Figure`: one sample (indexers select it; none for a 0-D result); upper panel: the data points (`x_data`, `y_data`, with error bars from `sd_data` when the fit had `sd` - `build_result` stores `sd` as the point variable `sd_data`), the fitted curve on a fine grid between the smallest and largest finite `x` (log-spaced when `log_x`), the title with the model name, the parameters (`name = value ± se`) and the flags; lower panel: the weighted residuals against `x` with a zero line
- `plot_goodness_of_fit(result: FitResult, *, log=False, style=DEFAULT_STYLE) -> Figure`: predicted against observed for every sample (one colour per sample from the cmap), identity line, `R²` per sample in the legend when ≤ 8 samples
- `plot_dose_proportionality(result: FitResult, *, test: xr.Dataset | None = None, style=DEFAULT_STYLE, **indexers) -> Figure`: log-log data and power fit of one sample (or the single sample of a 0-D result); the slope `b` with its CI in the title; when `test` (from `proportionality_test`) is given, the acceptance bounds as a shaded wedge through the first data point and "proportional"/"inconclusive"/"not proportional" in the title

- [ ] **Step 1: Write the failing tests**

`tests/plot/test_plot_fit.py`:
```python
import matplotlib
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from pkpdutils.fit import FitOptions, Weighting, fit, fit_table, proportionality_test
from pkpdutils.fit.models import MonoExp, Power
from pkpdutils.plot import plot_dose_proportionality, plot_fit, plot_goodness_of_fit

matplotlib.use("Agg")
T = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])


def monoexp_result(n: int = 1):
    rng = np.random.default_rng(0)
    ys = np.stack([10 * np.exp(-(0.2 + 0.1 * i) * T) * rng.lognormal(0, 0.05, T.size) for i in range(n)])
    if n == 1:
        return fit(MonoExp(), T, ys[0], sd=0.05 * ys[0], options=FitOptions(weighting=Weighting.INV_SD), x_unit="hr", y_unit="mg/l")
    return fit(MonoExp(), T, ys, dims=("individual",), coords={"individual": [f"s{i}" for i in range(n)]}, x_unit="hr", y_unit="mg/l")


def test_plot_fit_single_and_batch() -> None:
    fig = plot_fit(monoexp_result(), log_y=True)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    assert fig.axes[0].get_yscale() == "log"
    assert "monoexp" in fig.axes[0].get_title() and "k" in fig.axes[0].get_title()
    assert fig.axes[0].get_xlabel() == "x [hr]" and fig.axes[0].get_ylabel() == "y [mg/l]"
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "fit" in labels
    fig2 = plot_fit(monoexp_result(3), individual="s1", title="s1")
    assert "s1" in fig2.axes[0].get_title()
    matplotlib.pyplot.close("all")


def test_plot_goodness_of_fit() -> None:
    fig = plot_goodness_of_fit(monoexp_result(3), log=True)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert any("identity" in line.get_label() for line in ax.get_lines())
    assert len(ax.collections) >= 3 or len(ax.get_lines()) >= 4
    matplotlib.pyplot.close(fig)


def test_plot_dose_proportionality() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    auc = 2.0 * doses**1.02
    ds = xr.Dataset({"auc": (("dose",), auc, {"units": "hr*mg/l"})}, coords={"dose": ("dose", doses, {"units": "mg"})})
    result = fit_table(Power(), ds, "dose", "auc", dim="dose")
    test = proportionality_test(result, dose_range=(25.0, 200.0))
    fig = plot_dose_proportionality(result, test=test)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert "b =" in ax.get_title()
    assert len(ax.collections) >= 1  # the acceptance wedge
    matplotlib.pyplot.close(fig)
```

Run: `uv run pytest tests/plot/test_plot_fit.py -n 0 -q` → FAIL with `ImportError`.

- [ ] **Step 2: Implement**

Add to `build_result` (engine.py): when `sd is not None`, `data_vars["sd_data"] = (point_dims, sd.reshape(point_shape), {"units": y_unit})`.

`src/pkpdutils/plot/fit.py`:
```python
"""Figures of fits: data and curve with residuals, goodness of fit, dose proportionality."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.fit.result import FitResult
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle


def _sample(result: FitResult, indexers: dict[str, Any]) -> xr.Dataset:
    """The dataset of one sample (every sample dimension indexed)."""
    missing = set(result.sample_dims) - set(indexers)
    if missing:
        raise ValueError(f"A label for every sample dimension is needed, missing {sorted(missing)}")
    return result.ds.sel(indexers)


def _parameter_text(result: FitResult, sample: xr.Dataset) -> str:
    """`name = value ± se` per parameter."""
    parts = []
    for name in result.model.parameter_names:
        value = float(sample[name].values)
        se = float(sample[f"{name}_se"].values)
        parts.append(f"{name} = {value:.3g} ± {se:.2g}" if np.isfinite(se) else f"{name} = {value:.3g}")
    return ", ".join(parts)


def plot_fit(
    result: FitResult,
    *,
    log_x: bool = False,
    log_y: bool = False,
    n_grid: int = 200,
    title: str | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Data, fitted curve and weighted residuals of one sample.

    Args:
        result: the fit
        log_x: logarithmic x axis (and log-spaced curve grid)
        log_y: logarithmic y axis
        n_grid: points of the curve grid
        title: title, the model name by default; the parameters and the flags are appended
        style: colors and markers
        **indexers: coordinate labels selecting the sample (none for a single fit)

    Returns:
        The figure with the fit panel and the residual panel.
    """
    sample = _sample(result, indexers)
    x = sample["x_data"].to_numpy()
    y = sample["y_data"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    fig, (ax, ax_res) = plt.subplots(nrows=2, ncols=1, figsize=(7, 6.5), height_ratios=[3, 1], sharex=True)
    fig.set_layout_engine("constrained")
    if "sd_data" in sample:
        sd = sample["sd_data"].to_numpy()
        ax.errorbar(x[ok], y[ok], yerr=sd[ok], marker=style.data_marker, linestyle="none", color=style.data_color, markersize=style.markersize, capsize=2, label="data")
    else:
        ax.plot(x[ok], y[ok], marker=style.data_marker, linestyle="none", color=style.data_color, markersize=style.markersize, label="data")
    p = np.array([float(sample[name].values) for name in result.model.parameter_names])
    if ok.any() and np.all(np.isfinite(p)):
        lo, hi = float(x[ok].min()), float(x[ok].max())
        grid = np.geomspace(max(lo, 1e-12), hi, n_grid) if log_x else np.linspace(lo, hi, n_grid)
        ax.plot(grid, result.model.predict(grid, p), "-", color=style.fit_color, linewidth=style.linewidth, label="fit")
    heading = title if title is not None else result.model.name
    flags = result.decode_flags(int(sample["flags"].values))
    ax.set_title(f"{heading}: {_parameter_text(result, sample)}" + (f" [{', '.join(flags)}]" if flags else ""), fontsize="small")
    ax.set_ylabel(f"y [{result.ds.attrs['y_unit']}]")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize="small")
    res = sample["residuals"].to_numpy()
    ax_res.axhline(0.0, color="gray", linewidth=1)
    ax_res.plot(x[ok], res[ok], marker=style.data_marker, linestyle="none", color=style.data_color, markersize=style.markersize)
    ax_res.set_xlabel(f"x [{result.ds.attrs['x_unit']}]")
    ax_res.set_ylabel("weighted residual")
    return fig


def plot_goodness_of_fit(result: FitResult, *, log: bool = False, style: PlotStyle = DEFAULT_STYLE) -> Figure:
    """Predicted against observed values of every sample with the identity line.

    Args:
        result: the fit
        log: logarithmic axes
        style: colors and markers

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.set_layout_engine("constrained")
    y = result["y_data"].to_numpy().reshape(-1, result.ds.sizes["point"])
    pred = result["y_pred"].to_numpy().reshape(-1, result.ds.sizes["point"])
    r2 = result["r2"].to_numpy().reshape(-1)
    labels = _sample_labels(result)
    cmap = plt.get_cmap(style.cmap)
    n = y.shape[0]
    for i in range(n):
        ok = np.isfinite(y[i]) & np.isfinite(pred[i])
        label = f"{labels[i]} (R² = {r2[i]:.3f})" if n <= 8 else None
        ax.scatter(y[i][ok], pred[i][ok], color=cmap(i / max(n - 1, 1)) if n > 1 else style.data_color, s=style.markersize**2 * 1.5, label=label)
    finite = np.concatenate([y[np.isfinite(y)], pred[np.isfinite(pred)]])
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="identity")
    unit = result.ds.attrs["y_unit"]
    ax.set_xlabel(f"observed [{unit}]")
    ax.set_ylabel(f"predicted [{unit}]")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend(fontsize="small")
    return fig


def _sample_labels(result: FitResult) -> list[str]:
    """One label per sample in C order of the sample dimensions."""
    if not result.sample_dims:
        return [result.model.name]
    labels = []
    for index in np.ndindex(*[result.ds.sizes[d] for d in result.sample_dims]):
        parts = [str(result.ds[d].values[i]) if d in result.ds.coords else str(i) for d, i in zip(result.sample_dims, index, strict=True)]
        labels.append("|".join(parts))
    return labels


def plot_dose_proportionality(
    result: FitResult,
    *,
    test: xr.Dataset | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Log-log exposure against dose with the power fit and the acceptance bounds.

    Args:
        result: fit of `Power` (parameters `a`, `b`)
        test: the dataset of `proportionality_test`, draws the acceptance wedge and the verdict
        style: colors and markers
        **indexers: coordinate labels selecting the sample

    Returns:
        The figure.
    """
    sample = _sample(result, indexers)
    x = sample["x_data"].to_numpy()
    y = sample["y_data"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.set_layout_engine("constrained")
    ax.plot(x[ok], y[ok], marker=style.data_marker, linestyle="none", color=style.data_color, markersize=style.markersize, label="data")
    a, b = float(sample["a"].values), float(sample["b"].values)
    lo, hi = float(x[ok].min()), float(x[ok].max())
    grid = np.geomspace(lo, hi, 100)
    ax.plot(grid, a * grid**b, "-", color=style.fit_color, linewidth=style.linewidth, label=f"a·x^b, b = {b:.3f}")
    heading = f"b = {b:.3f} [{float(sample['b_ci_low'].values):.3f}, {float(sample['b_ci_high'].values):.3f}]"
    if test is not None:
        t = test.sel(indexers) if indexers else test
        bound_low, bound_high = float(t["bound_low"].values), float(t["bound_high"].values)
        x0, y0 = grid[0], a * grid[0] ** b
        ax.fill_between(grid, y0 * (grid / x0) ** bound_low, y0 * (grid / x0) ** bound_high, color=style.auc_color, alpha=style.alpha, label=f"acceptance [{bound_low:.2f}, {bound_high:.2f}]")
        verdict = "proportional" if bool(t["proportional"].values) else "inconclusive" if bool(t["inconclusive"].values) else "not proportional"
        heading = f"{heading}: {verdict}"
    ax.set_title(heading)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"x [{result.ds.attrs['x_unit']}]")
    ax.set_ylabel(f"y [{result.ds.attrs['y_unit']}]")
    ax.legend(fontsize="small")
    return fig
```
`Axes` import unused: drop it. Export `plot_fit`, `plot_goodness_of_fit`, `plot_dose_proportionality` from `pkpdutils/plot/__init__.py`. `ParameterResult.decode_flags` (Task 1) is used for the title.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/plot tests/fit -n 0 -q -W error && uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest -q -W error`
Expected: all pass; the `x [hr]`/`y [mg/l]` labels come from the attrs set by `fit(..., x_unit=, y_unit=)`.

```bash
git add src/pkpdutils/plot src/pkpdutils/fit/engine.py tests/plot
git commit -q -m "Add the fit, goodness-of-fit and dose proportionality figures

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 9: Examples, exports, documentation

**Files:**
- Create: `examples/fitting_exponential.py`, `examples/emax.py`, `examples/dose_proportionality.py`, `examples/covariate.py`, `docs/fitting.md`, `docs/pd.md`, `docs/api/fit.md`, `docs/api/fit.models.md`, `docs/api/fit.engine.md`, `docs/api/fit.compare.md`, `docs/api/fit.proportionality.md`, `docs/api/result.md`
- Modify: `src/pkpdutils/__init__.py` (export `fit`, `fit_timecourses`, `fit_table`, `FitOptions`, `FitResult`, `compare_models`), `tests/examples/test_examples.py`, `examples/README.md`, `zensical.toml` (nav), `docs/api/index.md`, `docs/api/plot.md` (add `plot.fit`), `docs/index.md` (feature bullets), `docs/glossary.md` (fit variables), `docs/references.md` (Seber & Wild 1989; Burnham & Anderson 2002), `CLAUDE.md`
- Test: `tests/test_package.py` (export test)

- [ ] **Step 1: Exports and test**

Append to `tests/test_package.py`:
```python
def test_fit_exports() -> None:
    from pkpdutils import FitOptions, FitResult, compare_models, fit, fit_table, fit_timecourses

    assert callable(fit) and callable(fit_timecourses) and callable(fit_table) and callable(compare_models)
    assert FitOptions().n_starts == 1 and FitResult is not None
```
Add the imports to `src/pkpdutils/__init__.py` (`__all__` sorted). Run → PASS.

- [ ] **Step 2: Examples**

`examples/fitting_exponential.py`:
```python
"""Fitting exponential models to a concentration timecourse.

Run from the root of the repository with `python -m examples.fitting_exponential`.
Writes `fitting_exponential.png` and `fitting_gof.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, FitOptions, Route, Timecourse, Timecourses, compare_models, fit_timecourses
from pkpdutils.console import console
from pkpdutils.fit import Weighting
from pkpdutils.fit.models import Bateman, BiExp, MonoExp
from pkpdutils.plot import plot_fit, plot_goodness_of_fit

t = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24])
rng = np.random.default_rng(0)
true = Bateman().predict(t, np.array([12.0, 1.8, 0.25]))
value = true * rng.lognormal(0, 0.05, t.size)
tc = Timecourse(time=t, value=value, sd=0.05 * value, time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="drug", label="oral")

if __name__ == "__main__":
    console.rule("Bateman fit with 1/sd weighting and a residual bootstrap")
    result = fit_timecourses(Bateman(), Timecourses.from_timecourses([tc]), FitOptions(weighting=Weighting.INV_SD, bootstrap=200, seed=1))
    console.print(result.to_dataframe().T)
    plot_fit(result, individual="oral", log_y=True).savefig("fitting_exponential.png", dpi=120)

    console.rule("Which model? AICc of mono-, bi-exponential and Bateman")
    comparison = compare_models([MonoExp(), BiExp(), Bateman()], t, value, x_unit="hr", y_unit="mg/l", options=FitOptions(n_starts=5, seed=2))
    console.print(comparison.table)
    console.print("best:", str(comparison.best.values))
    plot_goodness_of_fit(comparison.results["bateman"]).savefig("fitting_gof.png", dpi=120)
    console.print("written: fitting_exponential.png, fitting_gof.png")
```

`examples/emax.py`:
```python
"""Concentration-effect relationship with the Emax family.

Run from the root of the repository with `python -m examples.emax`.
Writes `emax.png` into the working directory.
"""

import numpy as np

from pkpdutils import FitOptions, compare_models, fit
from pkpdutils.console import console
from pkpdutils.fit.models import Emax, Linear, SigmoidEmax
from pkpdutils.plot import plot_fit

concentration = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300])
rng = np.random.default_rng(3)
effect = SigmoidEmax().predict(concentration, np.array([5.0, 40.0, 8.0, 1.8])) + rng.normal(0, 1.0, concentration.size)

if __name__ == "__main__":
    console.rule("Sigmoid Emax fit")
    result = fit(SigmoidEmax(), concentration, effect, x_unit="ng/ml", y_unit="mmHg", options=FitOptions(n_starts=10, seed=0))
    for name in ("e0", "emax", "ec50", "hill", "ec90"):
        q = result.to_quantities()
        console.print(f"{name:<6} {q[name]:~P}  se {q[name + '_se']:~P}")
    console.print("flags:", result.flags())
    plot_fit(result, log_x=True).savefig("emax.png", dpi=120)

    console.rule("Emax versus sigmoid Emax versus linear")
    console.print(compare_models([Emax(), SigmoidEmax(), Linear()], concentration, effect, x_unit="ng/ml", y_unit="mmHg", options=FitOptions(n_starts=10, seed=0)).table)
    console.print("written: emax.png")
```

`examples/dose_proportionality.py`:
```python
"""Dose proportionality of the exposure from an NCA over a dose escalation.

Run from the root of the repository with `python -m examples.dose_proportionality`.
Writes `dose_proportionality.png` into the working directory.
"""

import numpy as np

from pkpdutils import Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.fit import fit_table, proportionality_test
from pkpdutils.fit.models import Power
from pkpdutils.plot import plot_dose_proportionality

time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
doses = np.array([25.0, 50.0, 100.0, 200.0, 400.0])
rng = np.random.default_rng(4)
values = np.stack([d ** 1.15 / 10 * np.exp(-0.25 * time) * rng.lognormal(0, 0.04, time.size) for d in doses])
batch = Timecourses.from_arrays(time, values, time_unit="hr", unit="mg/l", dims=("dose",), coords={"dose": doses}, dose={"amount": doses, "unit": "mg"}, route=Route.IV_BOLUS, substance="drug")

if __name__ == "__main__":
    result = nca(batch)
    ds = result.ds.assign_coords(dose=("dose", doses, {"units": "mg"}))
    power = fit_table(Power(), ds, "dose", "auc_inf_obs", dim="dose")
    test = proportionality_test(power, dose_range=(25.0, 400.0))
    console.rule("Power model AUC = a dose^b")
    console.print(power.to_dataframe().T.loc[["a", "b", "b_se", "b_ci_low", "b_ci_high", "r2"]])
    console.print(test.to_dataframe().T)
    plot_dose_proportionality(power, test=test).savefig("dose_proportionality.png", dpi=120)
    console.print("written: dose_proportionality.png")
```

`examples/covariate.py`:
```python
"""Clearance against body weight: allometric scaling with a free and a fixed exponent.

Run from the root of the repository with `python -m examples.covariate`.
"""

import numpy as np
import xarray as xr

from pkpdutils.console import console
from pkpdutils.fit import compare_models, fit_table
from pkpdutils.fit.models import Allometric, Linear

weights = np.array([45.0, 52, 60, 68, 75, 82, 90, 105, 120])
rng = np.random.default_rng(5)
clearance = 0.9 * weights**0.72 * rng.lognormal(0, 0.08, weights.size)
ds = xr.Dataset({"cl": (("individual",), clearance, {"units": "liter / hour"})}, coords={"weight": ("individual", weights, {"units": "kg"}), "individual": [f"s{i}" for i in range(weights.size)]})

if __name__ == "__main__":
    console.rule("Free exponent")
    free = fit_table(Allometric(), ds, "weight", "cl", dim="individual")
    console.print(free.to_dataframe().T.loc[["a", "b", "b_se", "b_ci_low", "b_ci_high", "r2"]])
    console.rule("Fixed exponent 0.75 and a linear model")
    fixed = fit_table(Allometric(exponent=0.75), ds, "weight", "cl", dim="individual")
    console.print(fixed.to_dataframe().T.loc[["a", "a_se", "r2"]])
    comparison = compare_models([Allometric(), Allometric(exponent=0.75), Linear()], weights, clearance, x_unit="kg", y_unit="liter/hour")
    console.print(comparison.table)
```
`compare_models` requires distinct names; `Allometric(exponent=0.75)` is named `allometric_0.75` (Task 4).

Register the four modules in `tests/examples/test_examples.py::SCRIPTS`; add rows to `examples/README.md`:
```markdown
| `examples/fitting_exponential.py` | Bateman fit of an oral timecourse with weighting and bootstrap, AICc comparison of exponential models, goodness of fit |
| `examples/emax.py` | sigmoid Emax fit of a concentration-effect relationship, comparison with Emax and linear |
| `examples/dose_proportionality.py` | power model of `auc_inf_obs` against the dose from an NCA batch and the Smith criterion |
| `examples/covariate.py` | allometric scaling of a clearance against body weight with a free and a fixed exponent |
```

Run every example from a temporary directory: `cd $(mktemp -d) && for m in fitting_exponential emax dose_proportionality covariate; do PYTHONPATH=/home/mkoenig/git/pkdb_analysis MPLBACKEND=Agg uv run --project /home/mkoenig/git/pkdb_analysis python -m examples.$m > /dev/null || echo "FAILED $m"; done; ls; cd -` → no `FAILED`.

- [ ] **Step 3: Documentation**

API pages: `docs/api/fit.md` (`::: pkpdutils.fit.engine` under `# fit`, with `## fit.result` → `::: pkpdutils.fit.result`, `## fit.frontends` → `::: pkpdutils.fit.frontends`, `## fit.options` → `::: pkpdutils.fit.options`, `## fit.model` → `::: pkpdutils.fit.model`), `docs/api/fit.models.md` (`::: pkpdutils.fit.models_exponential`, `::: pkpdutils.fit.models_response`, `::: pkpdutils.fit.models_linear` under subheadings), `docs/api/fit.compare.md` (`::: pkpdutils.fit.compare`), `docs/api/fit.proportionality.md` (`::: pkpdutils.fit.proportionality`), `docs/api/result.md` (`::: pkpdutils.result`); `docs/api/plot.md` gains `## plot.fit` → `::: pkpdutils.plot.fit`.

`zensical.toml` nav: user guide gains `{ "Curve fitting" = "fitting.md" }` and `{ "Pharmacodynamics" = "pd.md" }` after "Uncertainty"; API reference: `{ "result" = "api/result.md" }` in the `pkpdutils` block, a new block `{ "pkpdutils.fit" = [ { "fit" = "api/fit.md" }, { "models" = "api/fit.models.md" }, { "compare" = "api/fit.compare.md" }, { "proportionality" = "api/fit.proportionality.md" } ] }`. `docs/api/index.md`: rows for `result`, the four fit pages, and `plot.fit`.

`docs/references.md`: add under "Statistics":
```markdown
**Nonlinear regression.** The covariance of the parameters from the Jacobian, the t intervals and the residual bootstrap.

> Seber GAF, Wild CJ.
> **Nonlinear Regression.**
> Wiley; 1989.

**Model selection by AICc and Akaike weights.**

> Burnham KP, Anderson DR.
> **Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach.**
> 2nd edition. Springer; 2002.
```

`docs/fitting.md`:
````markdown
# Curve fitting

Non-compartmental analysis reads parameters from the observed points; some questions need a curve through them: the rate constants of the phases of a decline, the absorption rate of an oral curve, the concentration of half-maximal effect, whether the exposure grows in proportion to the dose, how a clearance scales with body weight. `pkpdutils.fit` fits small parametric models to one curve or to every curve of a batch with the same engine, reports standard errors, confidence intervals and goodness of fit, compares models and applies the dose proportionality criterion. The models are descriptive; compartmental interpretation of the coefficients and population (mixed effects) modelling are outside the scope of the package.

## Concepts

**Model.** A `Model` is a function \(y = f(x; p)\) with named parameters, bounds and an initial guess from the data. Every parameter states whether it is positive; positive parameters are searched on the logarithmic scale, which keeps them positive and makes the search insensitive to their magnitude (`FitOptions.parameter_scale`). The model library has three families: exponentials for concentration timecourses (`MonoExp`, `BiExp`, `TriExp`, `Bateman` with optional lag time), the Emax family for concentration-effect data (`Emax`, `SigmoidEmax`, `Imax`, `SigmoidImax`), and `Linear`, `LogLinear`, `Power` and `Allometric` for a parameter against a dose or a covariate.

**Weighting.** Concentrations span orders of magnitude and their error grows with their size, so an unweighted fit is dominated by the high points. `Weighting` names the variance model of the residuals: constant (`NONE`), proportional to \(y\) (`INV_Y`), proportional to \(y^2\) (constant CV, `INV_Y2`) or the reported standard deviations (`INV_SD`). Residuals are divided by the standard deviation of that model before the sum of squares.

**Uncertainty of the parameters.** The standard errors come from the Jacobian of the residuals at the optimum, a first order (Wald) approximation [^seber]; the confidence intervals use the t distribution with \(n - k\) degrees of freedom and are transformed from the search scale, so they are asymmetric for log parameters. Derived parameters (half-lives, areas, \(\mathrm{EC}_{90}\), \(t_\mathrm{max}\)) get their uncertainty by the delta method. `FitOptions(bootstrap=B)` replaces both by a residual bootstrap: the weighted residuals are resampled, the curve refitted \(B\) times, and the standard errors and percentile intervals read from the replicates.

**Multi-start.** Nonlinear least squares finds a local optimum. `FitOptions(n_starts=k)` starts from the initial guess and \(k - 1\) Latin hypercube points of a box around it and keeps the best converged solution; `n_starts_converged` says how many starts converged, `n_workers` spreads the samples over processes.

**Model comparison.** `compare_models` fits every model to the same data and ranks them per sample by the corrected Akaike information criterion; the Akaike weight is the probability that a model is the best of the set [^burnham]. AICc penalizes parameters, so a bi-exponential only wins over a mono-exponential when the second phase is supported by the data.

**Dose proportionality.** With \(\mathrm{AUC} = a D^b\) the exposure is proportional to the dose when \(b = 1\). `proportionality_test` applies the confidence interval criterion of Smith et al. [^smith]: over a dose range \(r = D_\mathrm{high} / D_\mathrm{low}\) the fit is proportional when the interval of \(b\) lies within \(1 + \ln(\theta) / \ln(r)\) for the acceptance limits \(\theta = 0.8\) and \(1.25\), inconclusive when it overlaps the bounds, and not proportional otherwise.

**Flags.** `NOT_CONVERGED`, `AT_BOUND` (a parameter sits on a bound), `TOO_FEW_POINTS`, `FLIP_FLOP` (Bateman with \(k_a < k_e\)), `SINGULAR` (no standard errors), `NO_DATA`.

## Math

Weighted residuals and the objective:

\[
r_i = \frac{y_i - f(x_i; p)}{\sqrt{v_i}}, \qquad v_i \in \{1,\ y_i,\ y_i^2,\ \mathrm{sd}_i^2\}, \qquad \min_p \tfrac12 \sum_i \rho(r_i^2)
\]

Covariance in the search space \(q\) (\(q = \log_{10} p\) for positive parameters), standard errors and intervals:

\[
\mathrm{cov}(q) = s^2 (J^\top J)^{-1},\quad s^2 = \frac{\sum r_i^2}{n - k}, \qquad
\mathrm{se}(p_j) = \mathrm{se}(q_j) \left|\frac{dp_j}{dq_j}\right|, \qquad
\mathrm{CI}(p_j) = p\!\left(q_j \pm t_{n-k,\,1-\alpha/2}\, \mathrm{se}(q_j)\right)
\]

Derived parameters \(d(p)\): \(\mathrm{se}(d) = \sqrt{g\, \mathrm{cov}(q)\, g^\top}\) with \(g = \partial d / \partial q\). Goodness of fit with \(e_i = y_i - f_i\):

\[
R^2 = 1 - \frac{\sum e_i^2}{\sum (y_i - \bar y)^2}, \quad
\mathrm{RMSE} = \sqrt{\tfrac1n \sum e_i^2}, \quad
\mathrm{AIC} = n \ln\!\frac{\sum r_i^2}{n} + 2k, \quad
\mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}, \quad
\mathrm{BIC} = n \ln\!\frac{\sum r_i^2}{n} + k \ln n
\]

Akaike weights: \(\Delta_i = \mathrm{AICc}_i - \min_j \mathrm{AICc}_j\), \(w_i = e^{-\Delta_i/2} / \sum_j e^{-\Delta_j/2}\). Dose proportionality bounds: \(b \in \left[1 + \frac{\ln 0.8}{\ln r},\ 1 + \frac{\ln 1.25}{\ln r}\right]\).

## Models

| model | curve | parameters | derived |
| --- | --- | --- | --- |
| `MonoExp` | \(a e^{-kx}\) | `a`, `k` | `thalf`, `auc` |
| `BiExp` | \(a_1 e^{-k_1 x} + a_2 e^{-k_2 x}\), \(k_1 > k_2\) | `a1`, `k1`, `a2`, `k2` | `lambda_z`, `thalf_1`, `thalf_2`, `auc` |
| `TriExp` | three phases | `a1..a3`, `k1..k3` | `lambda_z`, `thalf_1..3`, `auc` |
| `Bateman(lag)` | \(a \frac{k_a}{k_a - k_e}(e^{-k_e t} - e^{-k_a t})\), \(t = x - t_\mathrm{lag}\) | `a`, `ka`, `ke` (+ `tlag`) | `tmax`, `cmax`, `thalf`, `auc`, `flip_flop` |
| `Emax` | \(e_0 + e_\mathrm{max} \frac{x}{\mathrm{ec}_{50} + x}\) | `e0`, `emax`, `ec50` | `ec90` |
| `SigmoidEmax` | \(e_0 + e_\mathrm{max} \frac{x^n}{\mathrm{ec}_{50}^n + x^n}\) | + `hill` | `ec90` |
| `Imax`, `SigmoidImax` | \(e_0 \left(1 - i_\mathrm{max} \frac{x^n}{\mathrm{ic}_{50}^n + x^n}\right)\) | `e0`, `imax`, `ic50` (+ `hill`) | `ic90` |
| `Linear`, `LogLinear` | \(a + b x\), \(a + b \ln x\) | `intercept`, `slope` | |
| `Power` | \(a x^b\) | `a`, `b` | |
| `Allometric(exponent)` | \(a x^b\), \(b\) free or fixed | `a` (+ `b`) | |

Units of the parameters follow from the units of `x` and `y` (`k` in `1/[x]`, `a` in `[y]`, `auc` in `[y]·[x]`); volumes and clearances are normalized like in the NCA.

## API

One curve:

```python
import numpy as np
from pkpdutils import FitOptions, fit
from pkpdutils.fit import Weighting
from pkpdutils.fit.models import Bateman

result = fit(Bateman(), t, c, sd=sd, x_unit="hr", y_unit="mg/l", options=FitOptions(weighting=Weighting.INV_SD, n_starts=5, seed=0))
result.to_quantities()["ka"], result.to_quantities()["ka_ci_low"], result.to_quantities()["tmax"]
result.predict(np.linspace(0, 24, 100))
result.correlation()
result.flags()
```

A batch of timecourses (times relative to the dose, units from the batch) and a parameter against a dose or covariate across a sample dimension:

```python
from pkpdutils import compare_models, fit_table, fit_timecourses
from pkpdutils.fit import proportionality_test
from pkpdutils.fit.models import Allometric, BiExp, MonoExp, Power

fits = fit_timecourses(BiExp(), batch, FitOptions(n_starts=10, seed=1))    # FitResult over the sample dims
comparison = compare_models([MonoExp(), BiExp()], t, c, x_unit="hr", y_unit="mg/l")
comparison.table, comparison.best

power = fit_table(Power(), nca_result.ds, "dose", "auc_inf_obs", dim="dose")
test = proportionality_test(power, dose_range=(25, 400))                     # proportional / inconclusive
allometric = fit_table(Allometric(exponent=0.75), ds, "weight", "cl", dim="individual")
```

`FitResult.summarize(dim)` summarizes the parameters of individual fits like the NCA results. Figures: `plot_fit`, `plot_goodness_of_fit`, `plot_dose_proportionality` in [Plotting](plotting.md). Examples: `examples/fitting_exponential.py`, `examples/emax.py`, `examples/dose_proportionality.py`, `examples/covariate.py`; API: [fit](api/fit.md), [models](api/fit.models.md).

## References

[^seber]: Seber GAF, Wild CJ. *Nonlinear Regression*. Wiley; 1989. See [References](references.md#statistics).
[^burnham]: Burnham KP, Anderson DR. *Model Selection and Multimodel Inference*. 2nd ed. Springer; 2002. See [References](references.md#statistics).
[^smith]: Smith BP et al. Confidence interval criteria for assessment of dose proportionality. *Pharm Res.* 2000. See [References](references.md#statistics).
````

`docs/pd.md`:
````markdown
# Pharmacodynamics

A pharmacodynamic timecourse measures an effect over time, a concentration-effect relationship measures the effect against the concentration. `pkpdutils` treats both with the tools of the previous pages: effect timecourses go through the non-compartmental analysis with `Kind.EFFECT`, concentration-effect data are fitted with the Emax family.

## Effect timecourses

`NCAOptions(kind=Kind.EFFECT)` switches the analysis to effect parameters: the baseline `e0` (the first value), the observed maximum `emax_obs` and its time `temax`, the area under the effect curve `auec_last` (linear trapezoids, any sign), the baseline corrected `auec_baseline` and `emax_baseline`, and with `effect_threshold` the time the linearly interpolated curve spends above the threshold, `time_above`. No terminal phase and no dose parameters are computed. Group effect curves with `sd`/`se` get the same uncertainty variables as concentrations ([Uncertainty](uncertainty.md)); the bootstrap does not clip effect values at 0.

```python
from pkpdutils import NCAOptions, nca_single
from pkpdutils.nca import Kind

result = nca_single(effect_timecourse, NCAOptions(kind=Kind.EFFECT, effect_threshold=15.0))
result.to_quantities()["auec_baseline"], result.to_quantities()["time_above"]
```

## Concentration-effect relationships

The Emax model \(E = E_0 + E_\mathrm{max}\, C / (\mathrm{EC}_{50} + C)\) describes a saturable effect; the Hill coefficient \(n\) of the sigmoid form makes the transition steeper; the Imax forms describe inhibition, \(E = E_0 (1 - I_\mathrm{max}\, C^n / (\mathrm{IC}_{50}^n + C^n))\) [^gw]. \(\mathrm{EC}_{90} = 9^{1/n}\, \mathrm{EC}_{50}\) is the concentration of 90 % of the maximal effect. The models are fitted with the [curve fitting](fitting.md) engine; `compare_models` decides between Emax, sigmoid Emax and a linear relationship by AICc. The same models describe a pharmacokinetic parameter against an inhibitor dose (e.g. `Imax` for a clearance against the dose of an inhibitor).

```python
from pkpdutils import FitOptions, compare_models, fit
from pkpdutils.fit.models import Emax, Linear, SigmoidEmax

result = fit(SigmoidEmax(), concentration, effect, x_unit="ng/ml", y_unit="mmHg", options=FitOptions(n_starts=10, seed=0))
result.to_quantities()["ec50"], result.to_quantities()["hill"], result.to_quantities()["ec90"]
compare_models([Emax(), SigmoidEmax(), Linear()], concentration, effect).table
```

The example is `examples/emax.py`.

## References

[^gw]: Gabrielsson J, Weiner D. *Pharmacokinetic and Pharmacodynamic Data Analysis*. 5th ed. 2016, ch. 4. See [References](references.md#textbooks).
````

`docs/index.md`: feature bullets `- **[Curve fitting](fitting.md)** - exponential, Bateman, Emax, power and covariate models with standard errors, confidence intervals, bootstrap, model comparison and dose proportionality.` and `- **[Pharmacodynamics](pd.md)** - effect timecourses in the NCA and concentration-effect relationships with the Emax family.`; `docs/plotting.md`: a section "Fits" describing `plot_fit`, `plot_goodness_of_fit`, `plot_dose_proportionality` with a code block; `docs/glossary.md`: rows for `p_se`, `p_ci_low`/`p_ci_high`, `p_cv`, `cost`, `r2`, `rmse`, `aic`, `aicc`, `bic`, `n_points`, `n_parameters`, `n_starts_converged`, `n_bootstrap`, `y_pred`, `residuals`, `correlation`, `akaike_weight`, `bound_low`/`bound_high` pointing at `[Curve fitting](fitting.md)`; `CLAUDE.md`: commands for the four examples and an architecture paragraph `**\`fit/\` - curve fitting.**` naming `model.py` (`Model`, `ModelParameter`, `parameter_unit_expression`), `options.py`, `models_exponential.py`/`models_response.py`/`models_linear.py` (re-exported by `models.py`), `engine.py` (`fit_row`: scaled space, `least_squares`, Jacobian covariance, t intervals, delta method for derived parameters, statistics, residual bootstrap; `fit_rows`: seeds and pool; `build_result`), `result.py` (`FitResult(ParameterResult)`), `frontends.py`, `compare.py`, `proportionality.py`; and a sentence for `result.py` at the package root (`ParameterResult`, the base of `NCAResult` and `FitResult`).

- [ ] **Step 4: Build and check**

Run: `uv run zensical build --clean && uv run python scripts/llms_txt.py && uv run pytest -q -W error && uv run ruff check && uv run ruff format --check && uv run ty check && uv run pre-commit run --all-files && grep -rn $'\xe2\x80\x94' docs README.md CLAUDE.md src examples tests | grep -v docs/superpowers | wc -l`
Expected: site builds (the `pd.md` warning is gone), all pass, `0`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -q -m "Document the curve fitting and the pharmacodynamics, add the fitting examples

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 10: Matrix and pull request

- [ ] **Step 1: Tox**

Run: `uv run tox run-parallel` → `py3.13`, `py3.14`, `ty` OK.

- [ ] **Step 2: Push and open the pull request**

```bash
git push -u origin fitting
gh pr create --base develop --title "Add the curve fitting and the pharmacodynamics" --body-file - <<'EOF'
## Summary

Phase 5 of the pkpdutils redesign (`docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`, section 3): `pkpdutils.fit` with the model protocol, the exponential, Bateman, Emax, linear, power and allometric models, the engine (`fit`, `fit_timecourses`, `fit_table`, scaled search, Jacobian standard errors, t intervals, delta method, multi-start, worker pool, residual bootstrap), `compare_models` (AICc, Akaike weights), `proportionality_test` (Smith criterion), the shared `ParameterResult` base class, the figures `plot_fit`, `plot_goodness_of_fit`, `plot_dose_proportionality`, documentation (`fitting.md`, `pd.md`) and four examples.

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

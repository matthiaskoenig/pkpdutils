# pkpdutils Statistics on Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pkpdutils.stats`: parameter samples, significance tests, geometric mean ratios, bioequivalence (TOST, 2x2 crossover), drug-drug interaction classification and meta-analysis, plus the figures `plot_parameters`, `plot_ratio`, `plot_forest`, `plot_bland_altman`, the documentation page `statistics.md` and three examples (phase 6 of the spec).

**Architecture:** `ParameterSample` (`stats/sample.py`) is the one input type of every statistic: individual values over one dimension (with labels and coordinates such as `period` and `sequence`) or summary statistics (`mean`, `sd`, `n`, optionally `geomean`, `geocv`). It is built by the new `ParameterResult.sample(name, dim=..., **indexers)` from an `NCAResult` or `FitResult`, or directly from numbers. Log-normal parameters are analysed on the log scale (`Scale.LOG`, default); the log moments of summary data come from the geometric statistics when present, else from the log-normal moment relations. On top of it: `compare` (`stats/tests.py`, scipy tests and Welch t from summary data, effect sizes, `multiple_comparison`), `ratio` (`stats/ratio.py`, geometric mean ratio with a t interval), `bioequivalence` (`stats/bioequivalence.py`, TOST on the ratio, the 2x2 crossover as the period-difference analysis of Chow & Liu which is the ANOVA with sequence, period and subject effects), `ddi_classification` (`stats/ddi.py`, FDA and EMA thresholds, conservative bound of an interval), and the meta-analysis (`stats/meta.py`, effect sizes, fixed effect, DerSimonian-Laird random effects, heterogeneity). Every result is a frozen dataclass with a `to_dataframe` where a table makes sense. Figures in `pkpdutils.plot.parameters`, `plot.ratio`, `plot.meta`.

**Tech Stack:** numpy, scipy (`stats.t`, `stats.norm`, `stats.chi2`, `stats.ttest_ind`, `ttest_rel`, `ttest_ind_from_stats`, `mannwhitneyu`, `wilcoxon`, `permutation_test`, `false_discovery_control`), pandas, xarray, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md` section "4. Statistics on PK parameters", the items `plot_parameters`, `plot_forest`, `plot_ratio`, `plot_bland_altman` of section 5, the `sample(parameter, dim)` method of "Results" in section 1, the examples `bioequivalence.py`, `ddi.py`, `meta_analysis.py` and the page `statistics.md` of section 6. Plans 1-4 for the interfaces this plan consumes.

**Later plan:** release 1.0.0 (7).

## Global Constraints

- package `pkpdutils`, python 3.13 and 3.14, runtime dependencies stay `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib`, `rich` (no statsmodels, no R)
- ty `error-on-warning = true`: zero diagnostics on `src`, `tests`, `examples`, `scripts`; rule-specific `# ty: ignore[rule]` only, none unused; narrowing asserts in tests
- every module, class and function of `src/pkpdutils` has full type annotations and a google style docstring with the formula and a citation key of `docs/references.md` where a formula exists; `examples/`, `tests/`, `scripts/` exempt from `D`
- library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, never prints, never calls `plt.show()`
- test output pristine: `uv run pytest -q -W error` passes
- enumerations are `StrEnum` with lower case values (as `Weighting`, `ParameterScale` in `fit/options.py`); result objects are `@dataclass(frozen=True)`
- the default scale of every statistic is the log scale (`Scale.LOG`); a non-positive value on the log scale raises `ValueError`
- the interval of a ratio (`ratio`, `bioequivalence`) is a t interval at `ci_level = 0.90` by default, the interval of `compare` and of the meta-analysis at `0.95`
- the geometric coefficient of variation is `sqrt(exp(sigma_log^2) - 1)` (the definition used by `ParameterResult.summarize`, plan 3)
- text rules: never write the em dash character (U+2014), use "-"; commit messages carry NO `Co-Authored-By` line and end with the single line `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`; markdown has no hard line wraps; formulas `\(...\)`/`\[...\]`
- `develop` accepts only pull requests; work happens on the branch `statistics` off `develop`, merged with the checks `tests`, `ruff`, `ty`, `docs`
- commands run from the repository root with `uv run`

## Interfaces of plans 1-4 this plan consumes

- `pkpdutils.result.ParameterResult(ds)`: `ds`, `sample_dims`, `parameters`, `derived_variables`, `statistics`, `point_variables`, `units(name)`, `__getitem__`, `__contains__`, `_sample(indexers)`, `to_quantities`, `flags`, `to_dataframe`, `summarize(dim, ci_level)`; `UNCERTAINTY_SUFFIXES`, `SUMMARY_SUFFIXES`, `base_name`; a group result carries `x_sd`, `x_se`, `x_ci_low`, `x_ci_high`, `x_geomean`, `x_geocv` and `n`; a summary (`summarize`) carries additionally `x_n`, `x_median`, `x_q25`, `x_q75`
- `pkpdutils.nca.NCAResult`, `pkpdutils.nca.nca(timecourses, options)`, `nca_single`, `NCAOptions`; the result coordinates are built at `src/pkpdutils/nca/nca.py:596-600` (single dose) and `:699-703` (steady state) from the dimension coordinates only
- `pkpdutils.fit.FitResult`, `fit`, `fit_timecourses` (coordinates at `src/pkpdutils/fit/frontends.py:74-78`), `fit_table` (`:183`); `FitResult` has `x_data`, `y_data`, `y_pred` over `point` and `model`
- `pkpdutils.timecourse.Timecourses.from_arrays(time, values, *, time_unit, unit, dims, coords, sd, se, n, dose, route, substance)`: `coords` is passed to `xarray.Dataset`, so a non-dimension coordinate along a sample dimension is given as `coords={"individual": [...], "period": ("individual", [...])}`
- `pkpdutils.plot`: `PlotStyle`, `DEFAULT_STYLE`, `_figure_of(ax)` in `plot/timecourse.py`, `plot_goodness_of_fit(result, *, log, style)` in `plot/fit.py`
- `pkpdutils.units`: `parse_unit`

## File structure

| file | responsibility |
| --- | --- |
| `src/pkpdutils/stats/__init__.py` | exports of the subpackage |
| `src/pkpdutils/stats/sample.py` | `Scale`, `ParameterSample`, `Summary`, `summarize`, the log-normal moment relations |
| `src/pkpdutils/stats/tests.py` | `TestMethod`, `Alternative`, `AdjustMethod`, `TestResult`, `compare`, `multiple_comparison` |
| `src/pkpdutils/stats/ratio.py` | `RatioResult`, `ratio` |
| `src/pkpdutils/stats/bioequivalence.py` | `Design`, `BEParameter`, `BEResult`, `tost`, `bioequivalence` |
| `src/pkpdutils/stats/ddi.py` | `DDIKind`, `DDIStrength`, `Sensitivity`, `DDIThresholds`, `DDIResult`, `ddi_classification`, `substrate_sensitivity` |
| `src/pkpdutils/stats/meta.py` | `EffectKind`, `EffectSize`, `effect_size`, `Heterogeneity`, `heterogeneity`, `PooledEffect`, `fixed_effect`, `random_effects`, `Study`, `MetaResult`, `meta_analysis`, `meta_analysis_by` |
| `src/pkpdutils/result.py` | `sample_coordinates(ds, sample_dims)`, `ParameterResult.sample(...)` |
| `src/pkpdutils/plot/parameters.py` | `plot_parameters` |
| `src/pkpdutils/plot/ratio.py` | `plot_ratio` |
| `src/pkpdutils/plot/meta.py` | `plot_forest` |
| `src/pkpdutils/plot/fit.py` | `plot_bland_altman` |
| `tests/stats/*.py`, `tests/data/reference/meta_bcg.json` | tests and the meta-analysis reference |
| `docs/statistics.md`, `docs/api/stats*.md` | documentation |
| `examples/bioequivalence.py`, `examples/ddi.py`, `examples/meta_analysis.py` | examples |

---

### Task 1: Branch, `Scale`, `ParameterSample`, `Summary` and `summarize` (`stats/sample.py`)

**Files:**
- Create: `src/pkpdutils/stats/__init__.py`, `src/pkpdutils/stats/sample.py`
- Test: `tests/stats/__init__.py` (empty), `tests/stats/test_sample.py`

**Interfaces:**
- Produces `pkpdutils.stats.sample.Scale` (`LINEAR = "linear"`, `LOG = "log"`); `lognormal_from_moments(mean, sd) -> tuple[float, float]` (`mu`, `sigma` of the log); `lognormal_from_geometric(geomean, geocv) -> tuple[float, float]`; `moments_from_lognormal(mu, sigma) -> tuple[float, float]` (`mean`, `sd`).
- `ParameterSample` (frozen dataclass): fields `values: np.ndarray | None = None`, `labels: np.ndarray | None = None`, `coords: dict[str, np.ndarray]` (default empty), `mean`, `sd`, `n`, `geomean`, `geocv` (`float | None`), `name: str = "value"`, `unit: str = "dimensionless"`. Properties `is_individual: bool`, `finite_values: np.ndarray` (finite values only, empty for summary data), `finite_labels`, `size: int` (finite count or `n`), `log_values: np.ndarray` (log of the finite values, `ValueError` if any is `<= 0`). Methods `log_moments() -> tuple[float, float]` (`mu`, `sigma_log`; individual: mean and `ddof=1` standard deviation of `log_values`; summary: from `geomean`/`geocv` when both are present, else `lognormal_from_moments(mean, sd)`), `linear_moments() -> tuple[float, float]` (`mean`, `sd`; summary data without `mean`/`sd` via `moments_from_lognormal`), `moments(scale) -> tuple[float, float, int]` (center, spread, `size` on the given scale), `summary(scale=Scale.LOG, ci_level=0.95) -> Summary`, `select(mask) -> ParameterSample` (individual data only, keeps `labels`, `coords`).
- `Summary` (frozen dataclass): `n`, `mean`, `sd`, `se`, `cv`, `geomean`, `geocv`, `median`, `q25`, `q75`, `min`, `max`, `ci_low`, `ci_high`, `ci_level`, `scale`, `name`, `unit`; `to_dict()`.
- `summarize(values: ParameterSample | ArrayLike, *, scale=Scale.LOG, ci_level=0.95, name="value", unit="dimensionless") -> Summary`.

- [ ] **Step 1: Branch**

```bash
git switch develop && git pull -q && git switch -c statistics
mkdir -p src/pkpdutils/stats tests/stats && touch tests/stats/__init__.py
```

- [ ] **Step 2: Write the failing tests**

`tests/stats/test_sample.py`:
```python
import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils.stats.sample import (
    ParameterSample,
    Scale,
    Summary,
    lognormal_from_geometric,
    lognormal_from_moments,
    moments_from_lognormal,
    summarize,
)

VALUES = np.array([80.0, 95.0, 110.0, 120.0, 150.0, np.nan])


def test_lognormal_relations_round_trip() -> None:
    mu, sigma = lognormal_from_moments(100.0, 30.0)
    assert sigma == pytest.approx(np.sqrt(np.log1p(0.09)))
    assert mu == pytest.approx(np.log(100.0) - sigma**2 / 2)
    mean, sd = moments_from_lognormal(mu, sigma)
    assert (mean, sd) == pytest.approx((100.0, 30.0))
    mu2, sigma2 = lognormal_from_geometric(95.78262852211522, 0.3)
    assert (mu2, sigma2) == pytest.approx((mu, sigma))


def test_individual_sample() -> None:
    s = ParameterSample(values=VALUES, name="auc_inf_obs", unit="mg*hr/l")
    assert s.is_individual and s.size == 5
    assert s.finite_values.tolist() == VALUES[:5].tolist()
    mu, sigma = s.log_moments()
    logs = np.log(VALUES[:5])
    assert (mu, sigma) == pytest.approx((logs.mean(), logs.std(ddof=1)))
    mean, sd = s.linear_moments()
    assert (mean, sd) == pytest.approx((VALUES[:5].mean(), VALUES[:5].std(ddof=1)))
    assert s.moments(Scale.LINEAR) == pytest.approx((mean, sd, 5))
    assert s.moments(Scale.LOG) == pytest.approx((mu, sigma, 5))


def test_labels_and_coords_follow_the_values() -> None:
    s = ParameterSample(
        values=np.array([1.0, 2.0, np.nan]),
        labels=np.array(["a", "b", "c"]),
        coords={"period": np.array([1, 2, 1])},
    )
    assert s.finite_labels.tolist() == ["a", "b"]
    sub = s.select(np.array([True, False, True]))
    assert sub.labels is not None and sub.labels.tolist() == ["a", "c"]
    assert sub.coords["period"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="length"):
        ParameterSample(values=np.array([1.0, 2.0]), labels=np.array(["a"]))
    with pytest.raises(ValueError, match="length"):
        ParameterSample(values=np.array([1.0, 2.0]), coords={"p": np.array([1])})


def test_summary_sample_moments() -> None:
    s = ParameterSample(mean=100.0, sd=30.0, n=12)
    assert not s.is_individual and s.size == 12
    assert s.finite_values.size == 0
    assert s.log_moments() == pytest.approx(lognormal_from_moments(100.0, 30.0))
    assert s.linear_moments() == (100.0, 30.0)
    g = ParameterSample(mean=100.0, sd=30.0, n=12, geomean=96.0, geocv=0.28)
    assert g.log_moments() == pytest.approx(lognormal_from_geometric(96.0, 0.28))
    only_geo = ParameterSample(geomean=96.0, geocv=0.28, n=12)
    mu, sigma = only_geo.log_moments()
    assert only_geo.linear_moments() == pytest.approx(moments_from_lognormal(mu, sigma))


def test_sample_validation() -> None:
    with pytest.raises(ValueError, match="values or"):
        ParameterSample()
    with pytest.raises(ValueError, match="n"):
        ParameterSample(mean=1.0, sd=0.1)
    with pytest.raises(ValueError, match="1-D"):
        ParameterSample(values=np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive"):
        ParameterSample(values=np.array([1.0, -1.0])).log_values
    with pytest.raises(ValueError, match="individual"):
        ParameterSample(mean=1.0, sd=0.1, n=3).select(np.array([True]))


def test_summarize_individual_log_scale() -> None:
    summary = summarize(VALUES, scale=Scale.LOG, name="cl", unit="l/hr")
    assert isinstance(summary, Summary)
    v = VALUES[:5]
    logs = np.log(v)
    assert summary.n == 5
    assert summary.mean == pytest.approx(v.mean())
    assert summary.sd == pytest.approx(v.std(ddof=1))
    assert summary.se == pytest.approx(v.std(ddof=1) / np.sqrt(5))
    assert summary.cv == pytest.approx(v.std(ddof=1) / v.mean())
    assert summary.geomean == pytest.approx(np.exp(logs.mean()))
    assert summary.geocv == pytest.approx(np.sqrt(np.expm1(logs.var(ddof=1))))
    assert summary.median == pytest.approx(np.median(v))
    assert (summary.q25, summary.q75) == pytest.approx(tuple(np.percentile(v, [25, 75])))
    assert (summary.min, summary.max) == (80.0, 150.0)
    tq = student_t.ppf(0.975, 4)
    assert summary.ci_low == pytest.approx(np.exp(logs.mean() - tq * logs.std(ddof=1) / np.sqrt(5)))
    assert summary.ci_high == pytest.approx(np.exp(logs.mean() + tq * logs.std(ddof=1) / np.sqrt(5)))
    assert summary.scale is Scale.LOG and summary.name == "cl" and summary.unit == "l/hr"
    assert summary.to_dict()["geomean"] == summary.geomean


def test_summarize_linear_scale_interval() -> None:
    summary = summarize(VALUES, scale=Scale.LINEAR)
    v = VALUES[:5]
    tq = student_t.ppf(0.975, 4)
    se = v.std(ddof=1) / np.sqrt(5)
    assert (summary.ci_low, summary.ci_high) == pytest.approx((v.mean() - tq * se, v.mean() + tq * se))


def test_summarize_summary_data() -> None:
    summary = summarize(ParameterSample(mean=100.0, sd=30.0, n=12), scale=Scale.LOG)
    mu, sigma = lognormal_from_moments(100.0, 30.0)
    assert summary.n == 12 and summary.mean == 100.0 and summary.sd == 30.0
    assert summary.geomean == pytest.approx(np.exp(mu))
    assert summary.geocv == pytest.approx(np.sqrt(np.expm1(sigma**2)))
    assert np.isnan(summary.median) and np.isnan(summary.min)
    tq = student_t.ppf(0.975, 11)
    assert summary.ci_low == pytest.approx(np.exp(mu - tq * sigma / np.sqrt(12)))


def test_summarize_single_value() -> None:
    summary = summarize([3.0])
    assert summary.n == 1 and summary.mean == 3.0 and np.isnan(summary.sd)
    assert np.isnan(summary.ci_low) and np.isnan(summary.geocv)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/stats/test_sample.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.stats'`

- [ ] **Step 4: Implement `stats/sample.py`**

`src/pkpdutils/stats/__init__.py` (the exports of the later tasks are appended there):
```python
"""Statistics on parameters: samples, tests, ratios, bioequivalence, drug-drug interactions and meta-analysis."""

from pkpdutils.stats.sample import ParameterSample, Scale, Summary, summarize

__all__ = ["ParameterSample", "Scale", "Summary", "summarize"]
```

`src/pkpdutils/stats/sample.py`:
```python
"""Parameter samples: individual values or summary statistics, on the linear or the log scale.

A `ParameterSample` is the input of every function of `pkpdutils.stats`: the
values of one parameter over the individuals of a group, or the summary
statistics of a group as published (`mean`, `sd`, `n`, optionally `geomean`
and `geocv`). Pharmacokinetic parameters are log-normal, so the statistics
work on the log scale by default (`Scale.LOG`), where the summary statistics
are translated with the moment relations of the log-normal distribution
(Rowland & Tozer 2011, ch. 8; `lognormal_from_moments`).
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.stats import t as student_t


class Scale(StrEnum):
    """Scale of an analysis."""

    #: differences of arithmetic means
    LINEAR = "linear"
    #: ratios of geometric means: the analysis runs on the logarithms
    LOG = "log"


def lognormal_from_moments(mean: float, sd: float) -> tuple[float, float]:
    """Log-scale moments of a log-normal distribution with the given mean and standard deviation.

    \\(\\sigma^2 = \\ln(1 + \\mathrm{sd}^2 / \\mathrm{mean}^2)\\), \\(\\mu = \\ln \\mathrm{mean} - \\sigma^2 / 2\\).

    Args:
        mean: arithmetic mean, positive.
        sd: standard deviation, non-negative.

    Returns:
        `mu` and `sigma`, the mean and the standard deviation of the logarithm.

    Raises:
        ValueError: if `mean` is not positive or `sd` is negative.
    """
    if not mean > 0:
        raise ValueError(f"The mean of a log-normal sample must be positive, got {mean}")
    if sd < 0:
        raise ValueError(f"The standard deviation must not be negative, got {sd}")
    sigma2 = float(np.log1p((sd / mean) ** 2))
    return float(np.log(mean) - sigma2 / 2.0), float(np.sqrt(sigma2))


def lognormal_from_geometric(geomean: float, geocv: float) -> tuple[float, float]:
    """Log-scale moments from a geometric mean and a geometric coefficient of variation.

    \\(\\mu = \\ln \\mathrm{geomean}\\), \\(\\sigma = \\sqrt{\\ln(1 + \\mathrm{geocv}^2)}\\).

    Args:
        geomean: geometric mean, positive.
        geocv: geometric coefficient of variation \\(\\sqrt{e^{\\sigma^2} - 1}\\), non-negative.

    Returns:
        `mu` and `sigma`.

    Raises:
        ValueError: if `geomean` is not positive or `geocv` is negative.
    """
    if not geomean > 0:
        raise ValueError(f"The geometric mean must be positive, got {geomean}")
    if geocv < 0:
        raise ValueError(f"The geometric CV must not be negative, got {geocv}")
    return float(np.log(geomean)), float(np.sqrt(np.log1p(geocv**2)))


def moments_from_lognormal(mu: float, sigma: float) -> tuple[float, float]:
    """Arithmetic mean and standard deviation of a log-normal distribution.

    \\(\\mathrm{mean} = e^{\\mu + \\sigma^2/2}\\), \\(\\mathrm{sd} = \\mathrm{mean}\\sqrt{e^{\\sigma^2} - 1}\\).

    Args:
        mu: mean of the logarithm.
        sigma: standard deviation of the logarithm.

    Returns:
        The arithmetic mean and standard deviation.
    """
    mean = float(np.exp(mu + sigma**2 / 2.0))
    return mean, float(mean * np.sqrt(np.expm1(sigma**2)))


def _array(value: Any, name: str) -> np.ndarray:
    """A 1-D array of a field.

    Args:
        value: the field value.
        name: the field name for the error message.

    Returns:
        The array.

    Raises:
        ValueError: if `value` is not 1-D.
    """
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1-D, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class Summary:
    """Summary statistics of a parameter sample.

    Attributes:
        n: number of values (finite values, or `n` of summary data)
        mean: arithmetic mean
        sd: standard deviation (`ddof=1`)
        se: standard error of the mean, `sd / sqrt(n)`
        cv: coefficient of variation, `sd / mean`
        geomean: geometric mean \\(e^{\\mu}\\)
        geocv: geometric coefficient of variation \\(\\sqrt{e^{\\sigma^2} - 1}\\)
        median: median (`NaN` for summary data)
        q25: first quartile (`NaN` for summary data)
        q75: third quartile (`NaN` for summary data)
        min: minimum (`NaN` for summary data)
        max: maximum (`NaN` for summary data)
        ci_low: lower bound of the t interval of the mean (`LINEAR`) or of the geometric mean (`LOG`)
        ci_high: upper bound of the interval
        ci_level: level of the interval
        scale: scale of the interval
        name: name of the parameter
        unit: unit of the parameter
    """

    n: int
    mean: float
    sd: float
    se: float
    cv: float
    geomean: float
    geocv: float
    median: float
    q25: float
    q75: float
    min: float
    max: float
    ci_low: float
    ci_high: float
    ci_level: float
    scale: Scale
    name: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary.

        Returns:
            Field name to value.
        """
        return {
            "n": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "se": self.se,
            "cv": self.cv,
            "geomean": self.geomean,
            "geocv": self.geocv,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "min": self.min,
            "max": self.max,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "scale": str(self.scale),
            "name": self.name,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ParameterSample:
    """The values of one parameter over a group of individuals, or the summary statistics of the group.

    Individual data: `values` (1-D, `NaN` is skipped) with optional `labels`
    (the identity of the individuals, used to pair two samples) and `coords`
    (further attributes per individual, such as `period` and `sequence` of a
    crossover study). Summary data: `n` with `mean` and `sd` and/or `geomean`
    and `geocv`, as reported in a publication.

    Attributes:
        values: individual values, `None` for summary data
        labels: label per value, `None` without labels
        coords: name to array with one entry per value
        mean: arithmetic mean of summary data
        sd: standard deviation of summary data
        n: number of individuals of summary data
        geomean: geometric mean of summary data
        geocv: geometric coefficient of variation of summary data
        name: name of the parameter
        unit: unit of the parameter
    """

    values: np.ndarray | None = None
    labels: np.ndarray | None = None
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    mean: float | None = None
    sd: float | None = None
    n: float | None = None
    geomean: float | None = None
    geocv: float | None = None
    name: str = "value"
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        """Convert the arrays and validate the combination of fields.

        Raises:
            ValueError: for neither values nor summary data, summary data
                without `n` or without a pair of moments, an array which is
                not 1-D, or labels or coordinates of another length.
        """
        if self.values is not None:
            values = _array(self.values, "values").astype(np.float64)
            object.__setattr__(self, "values", values)
            if self.labels is not None:
                labels = _array(self.labels, "labels")
                if labels.size != values.size:
                    raise ValueError(
                        f"'labels' has length {labels.size}, 'values' has length {values.size}"
                    )
                object.__setattr__(self, "labels", labels)
            coords = {}
            for key, coord in self.coords.items():
                arr = _array(coord, key)
                if arr.size != values.size:
                    raise ValueError(
                        f"coordinate '{key}' has length {arr.size}, 'values' has length {values.size}"
                    )
                coords[key] = arr
            object.__setattr__(self, "coords", coords)
            return
        if self.n is None:
            raise ValueError("Summary data needs 'n'; give 'values or 'mean', 'sd' and 'n'")
        has_moments = self.mean is not None and self.sd is not None
        has_geometric = self.geomean is not None and self.geocv is not None
        if not (has_moments or has_geometric):
            raise ValueError(
                "Summary data needs 'mean' and 'sd' or 'geomean' and 'geocv'; give 'values or these"
            )
        if self.n < 1:
            raise ValueError(f"'n' must be at least 1, got {self.n}")

    @property
    def is_individual(self) -> bool:
        """Whether the sample holds individual values."""
        return self.values is not None

    @property
    def _finite(self) -> np.ndarray:
        """Mask of the finite values (all `False` for summary data)."""
        if self.values is None:
            return np.zeros(0, dtype=bool)
        return np.isfinite(self.values)

    @property
    def finite_values(self) -> np.ndarray:
        """The finite individual values, empty for summary data."""
        if self.values is None:
            return np.zeros(0, dtype=np.float64)
        return self.values[self._finite]

    @property
    def finite_labels(self) -> np.ndarray | None:
        """The labels of the finite values, `None` without labels."""
        if self.values is None or self.labels is None:
            return None
        return self.labels[self._finite]

    @property
    def size(self) -> int:
        """Number of finite values, or `n` of summary data."""
        if self.values is None:
            assert self.n is not None
            return int(self.n)
        return int(self._finite.sum())

    @property
    def log_values(self) -> np.ndarray:
        """The logarithms of the finite values.

        Raises:
            ValueError: if a finite value is not positive.
        """
        values = self.finite_values
        if np.any(values <= 0):
            raise ValueError(
                f"'{self.name}' has non-positive values, the log scale needs positive values"
            )
        return np.log(values)

    def log_moments(self) -> tuple[float, float]:
        """Mean and standard deviation of the logarithm.

        Individual data: the moments of `log_values` (`ddof=1`, `NaN` with a
        single value). Summary data: `lognormal_from_geometric` when
        `geomean` and `geocv` are given, else `lognormal_from_moments`.

        Returns:
            `mu` and `sigma`.
        """
        if self.values is not None:
            logs = self.log_values
            sigma = float(np.std(logs, ddof=1)) if logs.size > 1 else float("nan")
            return float(np.mean(logs)) if logs.size else float("nan"), sigma
        if self.geomean is not None and self.geocv is not None:
            return lognormal_from_geometric(self.geomean, self.geocv)
        assert self.mean is not None and self.sd is not None
        return lognormal_from_moments(self.mean, self.sd)

    def linear_moments(self) -> tuple[float, float]:
        """Arithmetic mean and standard deviation.

        Summary data given only as geometric statistics are translated with
        `moments_from_lognormal`.

        Returns:
            `mean` and `sd`.
        """
        if self.values is not None:
            values = self.finite_values
            sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
            return float(np.mean(values)) if values.size else float("nan"), sd
        if self.mean is not None and self.sd is not None:
            return float(self.mean), float(self.sd)
        return moments_from_lognormal(*self.log_moments())

    def moments(self, scale: Scale) -> tuple[float, float, int]:
        """Center, spread and size on a scale.

        Args:
            scale: `LINEAR` for the arithmetic moments, `LOG` for the log moments.

        Returns:
            The center, the standard deviation and the number of values.
        """
        center, spread = (
            self.log_moments() if scale is Scale.LOG else self.linear_moments()
        )
        return center, spread, self.size

    def select(self, mask: np.ndarray) -> "ParameterSample":
        """The individual data at a boolean mask, with its labels and coordinates.

        Args:
            mask: boolean array with one entry per value.

        Returns:
            The selected sample.

        Raises:
            ValueError: for summary data.
        """
        if self.values is None:
            raise ValueError("'select' needs individual data")
        mask = np.asarray(mask, dtype=bool)
        return ParameterSample(
            values=self.values[mask],
            labels=None if self.labels is None else self.labels[mask],
            coords={k: v[mask] for k, v in self.coords.items()},
            name=self.name,
            unit=self.unit,
        )

    def summary(self, scale: Scale = Scale.LOG, ci_level: float = 0.95) -> Summary:
        """The summary statistics, see `summarize`.

        Args:
            scale: scale of the confidence interval.
            ci_level: level of the confidence interval.

        Returns:
            The summary.
        """
        return summarize(self, scale=scale, ci_level=ci_level)


def summarize(
    values: ParameterSample | Any,
    *,
    scale: Scale = Scale.LOG,
    ci_level: float = 0.95,
    name: str = "value",
    unit: str = "dimensionless",
) -> Summary:
    """Summary statistics of a parameter sample.

    The arithmetic statistics, the geometric mean and the geometric CV, the
    quantiles of individual data, and a t interval: of the mean on the
    `LINEAR` scale, \\(\\bar x \\pm t_{1-\\alpha/2, n-1}\\,\\mathrm{sd}/\\sqrt{n}\\), and of the
    geometric mean on the `LOG` scale, \\(\\exp(\\mu \\pm t_{1-\\alpha/2, n-1}\\,\\sigma/\\sqrt{n})\\).
    The interval and the spread are `NaN` with a single value.

    Args:
        values: a sample, or individual values (`NaN` skipped).
        scale: scale of the interval.
        ci_level: level of the interval.
        name: name of the parameter (ignored for a sample, which carries its own).
        unit: unit of the parameter (ignored for a sample).

    Returns:
        The summary.
    """
    sample = (
        values
        if isinstance(values, ParameterSample)
        else ParameterSample(values=np.asarray(values, dtype=np.float64), name=name, unit=unit)
    )
    n = sample.size
    mean, sd = sample.linear_moments()
    mu, sigma = sample.log_moments()
    nan = float("nan")
    se = sd / np.sqrt(n) if n > 1 else nan
    if sample.is_individual:
        v = sample.finite_values
        median, q25, q75 = (
            (float(np.median(v)), *(float(q) for q in np.percentile(v, [25, 75])))
            if v.size
            else (nan, nan, nan)
        )
        low_high = (float(v.min()), float(v.max())) if v.size else (nan, nan)
    else:
        median, q25, q75 = nan, nan, nan
        low_high = (nan, nan)
    if n > 1:
        tq = float(student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, n - 1))
        if scale is Scale.LOG:
            ci = (float(np.exp(mu - tq * sigma / np.sqrt(n))), float(np.exp(mu + tq * sigma / np.sqrt(n))))
        else:
            ci = (mean - tq * se, mean + tq * se)
    else:
        ci = (nan, nan)
    return Summary(
        n=n,
        mean=mean,
        sd=sd,
        se=float(se),
        cv=float(sd / mean) if mean else nan,
        geomean=float(np.exp(mu)),
        geocv=float(np.sqrt(np.expm1(sigma**2))),
        median=median,
        q25=q25,
        q75=q75,
        min=low_high[0],
        max=low_high[1],
        ci_low=ci[0],
        ci_high=ci[1],
        ci_level=ci_level,
        scale=scale,
        name=sample.name,
        unit=sample.unit,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/stats/test_sample.py -q -W error`
Expected: PASS (9 tests). Run `uv run ruff check src tests && uv run ruff format src tests && uvx ty check` → clean. If `summarize([3.0])` warns about `ddof` (numpy `RuntimeWarning` on `std` of one value), the `n > 1` guards above prevent it; a `RuntimeWarning` from `np.log` of a summary with `sigma = NaN` is not raised because `log_moments` returns `NaN` without calling `log`.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/stats tests/stats
git commit -q -m "Add ParameterSample, the log-normal moment relations and summarize

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 2: `ParameterResult.sample` and the sample coordinates of a result

**Files:**
- Modify: `src/pkpdutils/result.py` (add `sample_coordinates`, `ParameterResult.sample`), `src/pkpdutils/nca/nca.py:596-600` and `:699-703`, `src/pkpdutils/fit/frontends.py:74-78`
- Test: `tests/test_result.py`, `tests/nca/test_nca.py`, `tests/fit/test_frontends.py`

**Interfaces:**
- Produces `pkpdutils.result.sample_coordinates(ds: xr.Dataset, sample_dims: Sequence[str]) -> dict[str, xr.DataArray]`: every coordinate of `ds` whose dimensions are a subset of `sample_dims` (the dimension coordinates and the non-dimension coordinates along the sample dimensions, e.g. `period` and `sequence` on `individual`).
- `ParameterResult.sample(name: str, dim: str | None = None, **indexers: Any) -> ParameterSample`: the values of the variable `name` along `dim` after selecting the other sample dimensions with `indexers`, `labels` from the coordinate of `dim` (the positions when it has none), `coords` from every non-dimension coordinate along `dim`, `name` and `unit` of the variable. Without `dim` the selection must leave no sample dimension and the result must be group data (`{name}_sd` or `{name}_se` with `n`): a summary sample with `mean = x`, `sd = x_sd` (or `x_se * sqrt(n)`), `n = x_n` when present else `n`, `geomean = x_geomean`, `geocv = x_geocv` when present. Any other case raises `ValueError`.
- The NCA and the fit of a batch keep the non-dimension coordinates of the batch on their result.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_result.py`:
```python
def test_sample_coordinates_keeps_non_dimension_coords() -> None:
    from pkpdutils.result import sample_coordinates

    ds = xr.Dataset(
        {"value": (("s", "time"), np.ones((3, 2)))},
        coords={
            "s": ["x", "y", "z"],
            "period": ("s", [1, 2, 1]),
            "time": [0.0, 1.0],
            "grid": ("time", [0, 1]),
        },
    )
    coords = sample_coordinates(ds, ("s",))
    assert set(coords) == {"s", "period"}
    assert coords["period"].to_numpy().tolist() == [1, 2, 1]


def test_sample_individual_values() -> None:
    r = make()
    ds = r.ds.assign_coords(period=("s", [1, 2, 1]))
    r = MyResult(ds)
    sample = r.sample("a", dim="s")
    assert sample.is_individual and sample.name == "a" and sample.unit == "mg"
    assert sample.values is not None and sample.values.tolist() == [1.0, 2.0, 4.0]
    assert sample.labels is not None and sample.labels.tolist() == ["x", "y", "z"]
    assert sample.coords["period"].tolist() == [1, 2, 1]
    with pytest.raises(ValueError, match="sample dimension"):
        r.sample("a", dim="point")
    with pytest.raises(ValueError, match="not a variable"):
        r.sample("b", dim="s")


def test_sample_two_dims_needs_indexers() -> None:
    ds = xr.Dataset(
        {
            "a": (("g", "s"), np.array([[1.0, 2.0], [3.0, 4.0]]), {"units": "mg"}),
            "flags": (("g", "s"), np.zeros((2, 2), dtype=int), {"units": "dimensionless"}),
        },
        coords={"g": ["c", "t"], "s": [1, 2]},
    )
    r = MyResult(ds)
    sample = r.sample("a", dim="s", g="t")
    assert sample.values is not None and sample.values.tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="remaining"):
        r.sample("a", dim="s")


def test_sample_summary_data() -> None:
    ds = xr.Dataset(
        {
            "a": ((), 10.0, {"units": "mg"}),
            "a_sd": ((), 2.0, {"units": "mg"}),
            "a_geomean": ((), 9.8, {"units": "mg"}),
            "a_geocv": ((), 0.2, {"units": "dimensionless"}),
            "n": ((), 12.0, {"units": "dimensionless"}),
            "flags": ((), 0, {"units": "dimensionless"}),
        }
    )
    sample = MyResult(ds).sample("a")
    assert not sample.is_individual
    assert (sample.mean, sample.sd, sample.n) == (10.0, 2.0, 12)
    assert (sample.geomean, sample.geocv) == (9.8, 0.2)
    ds_se = ds.drop_vars(["a_sd", "a_geomean", "a_geocv"]).assign(
        a_se=((), 0.5, {"units": "mg"}), a_n=((), 10.0, {"units": "dimensionless"})
    )
    sample_se = MyResult(ds_se).sample("a")
    assert sample_se.n == 10 and sample_se.sd == pytest.approx(0.5 * np.sqrt(10))
    with pytest.raises(ValueError, match="group data"):
        MyResult(ds.drop_vars(["a_sd", "a_geomean", "a_geocv"])).sample("a")
    with pytest.raises(ValueError, match="remaining"):
        make().sample("a")
```

Append to `tests/nca/test_nca.py` (import `Timecourses`, `Route`, `nca` are already imported there; check the head of the file):
```python
def test_nca_keeps_sample_coordinates() -> None:
    time = np.array([0.5, 1, 2, 4, 8, 12, 24])
    values = np.stack([10 * np.exp(-0.2 * time), 12 * np.exp(-0.25 * time)])
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"], "period": ("individual", [1, 2])},
        dose={"amount": np.array([100.0, 100.0]), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    result = nca(batch)
    assert result.ds["period"].to_numpy().tolist() == [1, 2]
    sample = result.sample("auc_inf_obs", dim="individual")
    assert sample.coords["period"].tolist() == [1, 2]
    assert sample.labels is not None and sample.labels.tolist() == ["a", "b"]
```

Append to `tests/fit/test_frontends.py` (uses the batch helper of that file; if none fits, build the batch as in the NCA test above and fit `MonoExp()` with `fit_timecourses`):
```python
def test_fit_timecourses_keeps_sample_coordinates() -> None:
    time = np.array([0.5, 1, 2, 4, 8, 12, 24])
    values = np.stack([10 * np.exp(-0.2 * time), 12 * np.exp(-0.25 * time)])
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"], "sequence": ("individual", ["RT", "TR"])},
        dose={"amount": np.array([100.0, 100.0]), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    result = fit_timecourses(MonoExp(), batch)
    assert result.ds["sequence"].to_numpy().tolist() == ["RT", "TR"]
    assert result.sample("k", dim="individual").coords["sequence"].tolist() == ["RT", "TR"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_result.py tests/nca/test_nca.py tests/fit/test_frontends.py -q -k "sample or coordinates"`
Expected: FAIL with `ImportError` / `AttributeError: ... has no attribute 'sample'` / `KeyError: 'period'`

- [ ] **Step 3: Implement**

In `src/pkpdutils/result.py` add after the imports (`Sequence` from `collections.abc`; the import of `ParameterSample` is placed inside the method to avoid the circular import `stats.sample` -> nothing today, but `pkpdutils.stats` will import `pkpdutils.result` for the type of `bioequivalence`):
```python
def sample_coordinates(
    ds: xr.Dataset, sample_dims: Sequence[str]
) -> dict[str, xr.DataArray]:
    """The coordinates of a dataset which live on the sample dimensions.

    The dimension coordinates of the sample dimensions and every
    non-dimension coordinate along them (the `period` or the `sequence` of
    the individuals of a crossover study, the weight of the subjects) are
    carried from the analysed batch to its result, so that
    `ParameterResult.sample` finds them.

    Args:
        ds: the dataset of the batch.
        sample_dims: the sample dimensions.

    Returns:
        Coordinate name to coordinate.
    """
    dims = set(sample_dims)
    return {
        str(name): coord
        for name, coord in ds.coords.items()
        if set(str(d) for d in coord.dims) <= dims
    }
```

Add the method to `ParameterResult` after `to_quantities`:
```python
    def sample(self, name: str, dim: str | None = None, **indexers: Any) -> "ParameterSample":
        """A parameter as a `ParameterSample` for the statistics of `pkpdutils.stats`.

        With `dim`, the individual values of `name` along `dim`, after the
        other sample dimensions were selected with `indexers`; the labels are
        the coordinate of `dim` and the coordinates along `dim` (`period`,
        `sequence`, ...) travel with the sample. Without `dim`, the summary
        statistics of a group result: `x` as the mean, `x_sd` (or `x_se`
        times the square root of `n`) as the standard deviation, `x_n` when
        present else `n` as the number of individuals, and `x_geomean`,
        `x_geocv` when present.

        Args:
            name: name of the parameter.
            dim: the sample dimension the values run over, `None` for a
                summary sample.
            **indexers: coordinate label per remaining sample dimension.

        Returns:
            The sample.

        Raises:
            ValueError: if `name` is not a variable, `dim` is not a sample
                dimension, a sample dimension besides `dim` is not indexed,
                or the summary sample has no group statistics.
        """
        from pkpdutils.stats.sample import ParameterSample

        if name not in self.ds.data_vars:
            raise ValueError(f"'{name}' is not a variable of the result")
        if dim is not None and dim not in self.sample_dims:
            raise ValueError(f"'{dim}' is not a sample dimension {self.sample_dims}")
        selected = self.ds.sel(indexers) if indexers else self.ds
        remaining = [d for d in selected["flags"].dims if d != dim]
        if remaining:
            raise ValueError(
                f"The remaining sample dimensions {remaining} need an indexer each"
            )
        unit = self.units(name)
        if dim is not None:
            da = selected[name]
            labels = (
                selected[dim].to_numpy() if dim in selected.coords else np.arange(da.sizes[dim])
            )
            coords = {
                str(key): coord.to_numpy()
                for key, coord in selected.coords.items()
                if key != dim and tuple(coord.dims) == (dim,)
            }
            return ParameterSample(
                values=da.to_numpy().astype(np.float64),
                labels=labels,
                coords=coords,
                name=name,
                unit=unit,
            )
        has_sd = f"{name}_sd" in selected
        has_se = f"{name}_se" in selected
        count_name = f"{name}_n" if f"{name}_n" in selected else "n"
        if not (has_sd or has_se) or count_name not in selected:
            raise ValueError(
                f"'{name}' has no group data ('{name}_sd' or '{name}_se' with 'n'); give 'dim' for individual values"
            )
        n = float(selected[count_name].values)
        sd = (
            float(selected[f"{name}_sd"].values)
            if has_sd
            else float(selected[f"{name}_se"].values) * np.sqrt(n)
        )
        geomean = float(selected[f"{name}_geomean"].values) if f"{name}_geomean" in selected else None
        geocv = float(selected[f"{name}_geocv"].values) if f"{name}_geocv" in selected else None
        return ParameterSample(
            mean=float(selected[name].values),
            sd=sd,
            n=n,
            geomean=geomean,
            geocv=geocv,
            name=name,
            unit=unit,
        )
```
Use `TYPE_CHECKING` for the annotation import: `if TYPE_CHECKING: from pkpdutils.stats.sample import ParameterSample` at the top of `result.py`, and the quoted return annotation. ty resolves the import cycle at check time without executing it.

In `src/pkpdutils/nca/nca.py` replace both `coords = {d: timecourses.ds[d] for d in timecourses.sample_dims if d in timecourses.ds.coords}` blocks (lines 596-600 and 699-703) by `coords = sample_coordinates(timecourses.ds, timecourses.sample_dims)` and import `sample_coordinates` from `pkpdutils.result`. In `src/pkpdutils/fit/frontends.py:74-78` do the same for the batch of `fit_timecourses`; `fit_table` (`:183`) keeps its dimension coordinates only (the dataset of `fit_table` is arbitrary).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_result.py tests/nca tests/fit -q -W error`
Expected: PASS (every earlier test unchanged). Run `uv run ruff check src tests && uv run ruff format src tests && uvx ty check` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/result.py src/pkpdutils/nca/nca.py src/pkpdutils/fit/frontends.py tests/test_result.py tests/nca/test_nca.py tests/fit/test_frontends.py
git commit -q -m "Add ParameterResult.sample and keep the sample coordinates of a batch on its result

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 3: Significance tests and effect sizes (`stats/tests.py`)

**Files:**
- Create: `src/pkpdutils/stats/tests.py`
- Modify: `src/pkpdutils/stats/__init__.py`
- Test: `tests/stats/test_tests.py`

**Interfaces:**
- Produces `TestMethod` (`AUTO`, `STUDENT_T`, `WELCH_T`, `PAIRED_T`, `MANN_WHITNEY`, `WILCOXON`, `PERMUTATION`), `Alternative` (`TWO_SIDED = "two-sided"`, `LESS`, `GREATER`), `AdjustMethod` (`HOLM`, `BONFERRONI`, `BH`).
- `TestResult` (frozen dataclass): `test: TestMethod` (the resolved method), `statistic`, `p_value`, `effect` (difference of means `a - b` on `LINEAR`, ratio of geometric means `a / b` on `LOG`; for the rank tests the difference/ratio of the medians), `ci_low`, `ci_high` (t interval of the effect for the t tests, `NaN` for the rank and permutation tests), `ci_level`, `scale`, `alternative`, `paired`, `df` (`NaN` for the non-t tests), `cohen_d`, `hedges_g` (on the analysis scale, pooled standard deviation), `n_a`, `n_b`, `name`, `unit`; `to_dict()`.
- `compare(a: ParameterSample, b: ParameterSample, *, test=TestMethod.AUTO, scale=Scale.LOG, paired=False, alternative=Alternative.TWO_SIDED, ci_level=0.95, n_perm=9999, seed: int | None = None) -> TestResult`.
- `multiple_comparison(p_values: ArrayLike, method=AdjustMethod.HOLM) -> np.ndarray` (adjusted p values, same order, clipped to 1).
- `hedges_correction(n_total: int) -> float`: \(J = 1 - 3/(4N - 9)\) (Hedges 1981, the approximation used by `esc`), reused by the meta-analysis.

- [ ] **Step 1: Write the failing tests**

`tests/stats/test_tests.py`:
```python
import numpy as np
import pytest
from scipy import stats

from pkpdutils.stats import ParameterSample, Scale
from pkpdutils.stats.tests import (
    AdjustMethod,
    Alternative,
    TestMethod,
    TestResult,
    compare,
    hedges_correction,
    multiple_comparison,
)

RNG = np.random.default_rng(7)
A = np.exp(RNG.normal(np.log(100), 0.3, 12))
B = np.exp(RNG.normal(np.log(130), 0.3, 10))
A_PAIRED = np.exp(RNG.normal(np.log(100), 0.3, 12))
B_PAIRED = A_PAIRED * np.exp(RNG.normal(0.2, 0.1, 12))


def sample(values: np.ndarray, labels: list[str] | None = None) -> ParameterSample:
    return ParameterSample(values=values, labels=None if labels is None else np.array(labels), name="auc", unit="mg*hr/l")


def test_auto_is_welch_on_log_scale() -> None:
    r = compare(sample(A), sample(B))
    assert isinstance(r, TestResult) and r.test is TestMethod.WELCH_T
    ref = stats.ttest_ind(np.log(A), np.log(B), equal_var=False)
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.df == pytest.approx(ref.df)
    assert r.effect == pytest.approx(np.exp(np.log(A).mean() - np.log(B).mean()))
    low, high = ref.confidence_interval(0.95)
    assert (r.ci_low, r.ci_high) == pytest.approx((np.exp(low), np.exp(high)))
    assert r.scale is Scale.LOG and r.n_a == 12 and r.n_b == 10 and r.name == "auc"


def test_student_t_linear() -> None:
    r = compare(sample(A), sample(B), test=TestMethod.STUDENT_T, scale=Scale.LINEAR)
    ref = stats.ttest_ind(A, B, equal_var=True)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    assert r.df == 20 and r.effect == pytest.approx(A.mean() - B.mean())
    low, high = ref.confidence_interval(0.95)
    assert (r.ci_low, r.ci_high) == pytest.approx((low, high))


def test_paired_t() -> None:
    r = compare(sample(A_PAIRED), sample(B_PAIRED), paired=True)
    assert r.test is TestMethod.PAIRED_T and r.paired
    ref = stats.ttest_rel(np.log(A_PAIRED), np.log(B_PAIRED))
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    assert r.df == 11
    d = np.log(A_PAIRED) - np.log(B_PAIRED)
    assert r.effect == pytest.approx(np.exp(d.mean()))


def test_paired_needs_equal_size() -> None:
    with pytest.raises(ValueError, match="paired"):
        compare(sample(A), sample(B), paired=True)


def test_one_sided_alternative() -> None:
    r = compare(sample(A), sample(B), alternative=Alternative.LESS)
    ref = stats.ttest_ind(np.log(A), np.log(B), equal_var=False, alternative="less")
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.ci_low == 0.0 and np.isfinite(r.ci_high)
    high = ref.confidence_interval(0.95).high
    assert r.ci_high == pytest.approx(np.exp(high))
    g = compare(sample(A), sample(B), scale=Scale.LINEAR, alternative=Alternative.GREATER)
    assert g.ci_high == np.inf and np.isfinite(g.ci_low)


def test_rank_tests() -> None:
    mw = compare(sample(A), sample(B), test=TestMethod.MANN_WHITNEY)
    ref = stats.mannwhitneyu(np.log(A), np.log(B))
    assert mw.statistic == pytest.approx(ref.statistic) and mw.p_value == pytest.approx(ref.pvalue)
    assert mw.effect == pytest.approx(np.median(A) / np.median(B))
    assert np.isnan(mw.ci_low) and np.isnan(mw.df)
    w = compare(sample(A_PAIRED), sample(B_PAIRED), test=TestMethod.WILCOXON, paired=True)
    ref_w = stats.wilcoxon(np.log(A_PAIRED), np.log(B_PAIRED))
    assert w.statistic == pytest.approx(ref_w.statistic) and w.p_value == pytest.approx(ref_w.pvalue)
    with pytest.raises(ValueError, match="paired"):
        compare(sample(A_PAIRED), sample(B_PAIRED), test=TestMethod.WILCOXON)


def test_permutation_reproducible() -> None:
    r1 = compare(sample(A), sample(B), test=TestMethod.PERMUTATION, n_perm=999, seed=3)
    r2 = compare(sample(A), sample(B), test=TestMethod.PERMUTATION, n_perm=999, seed=3)
    assert r1.p_value == r2.p_value and 0 < r1.p_value < 0.2
    assert r1.statistic == pytest.approx(np.log(A).mean() - np.log(B).mean())
    assert r1.effect == pytest.approx(np.exp(r1.statistic))
    paired = compare(sample(A_PAIRED), sample(B_PAIRED), test=TestMethod.PERMUTATION, paired=True, n_perm=499, seed=1)
    assert paired.p_value < 0.01


def test_summary_data_welch() -> None:
    a = ParameterSample(mean=100.0, sd=30.0, n=12)
    b = ParameterSample(mean=130.0, sd=35.0, n=10)
    r = compare(a, b, scale=Scale.LINEAR)
    ref = stats.ttest_ind_from_stats(100.0, 30.0, 12, 130.0, 35.0, 10, equal_var=False)
    assert r.test is TestMethod.WELCH_T
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    assert r.effect == -30.0
    log = compare(a, b)
    mu_a, s_a = a.log_moments()
    mu_b, s_b = b.log_moments()
    ref_log = stats.ttest_ind_from_stats(mu_a, s_a, 12, mu_b, s_b, 10, equal_var=False)
    assert log.p_value == pytest.approx(ref_log.pvalue)
    assert log.effect == pytest.approx(np.exp(mu_a - mu_b))
    with pytest.raises(ValueError, match="individual"):
        compare(a, b, test=TestMethod.MANN_WHITNEY)
    with pytest.raises(ValueError, match="individual"):
        compare(a, sample(B), paired=True)


def test_effect_sizes() -> None:
    r = compare(sample(A), sample(B), scale=Scale.LINEAR)
    sp = np.sqrt((11 * A.var(ddof=1) + 9 * B.var(ddof=1)) / 20)
    d = (A.mean() - B.mean()) / sp
    assert r.cohen_d == pytest.approx(d)
    assert r.hedges_g == pytest.approx(d * hedges_correction(22))
    assert hedges_correction(110) == pytest.approx(1 - 3 / (4 * 110 - 9))
    assert r.to_dict()["test"] == "welch_t"


def test_log_scale_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        compare(sample(np.array([1.0, -2.0, 3.0])), sample(B))


def test_multiple_comparison() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.2])
    bonf = multiple_comparison(p, AdjustMethod.BONFERRONI)
    assert bonf.tolist() == pytest.approx([0.04, 0.16, 0.12, 0.8])
    holm = multiple_comparison(p, AdjustMethod.HOLM)
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> 0.09 (monotone), 0.2*1 -> 0.2
    assert holm.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.2])
    bh = multiple_comparison(p, AdjustMethod.BH)
    assert bh.tolist() == pytest.approx(stats.false_discovery_control(p, method="bh").tolist())
    assert multiple_comparison([0.5], AdjustMethod.HOLM).tolist() == [0.5]
    assert multiple_comparison([0.9, 0.9], AdjustMethod.BONFERRONI).max() == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/stats/test_tests.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.stats.tests'`

- [ ] **Step 3: Implement `stats/tests.py`**

```python
"""Significance tests on parameter samples and the adjustment of p values.

`compare` runs the t tests (Student, Welch, paired), the rank tests
(Mann-Whitney U, Wilcoxon signed rank) and a permutation test of scipy on
two samples, on the log scale by default, and reports the effect with its
t interval and the standardized effect sizes (Cohen's d, Hedges' g; Hedges
1981). Summary data (`mean`, `sd`, `n`) is compared with the Welch t test
from the moments (`scipy.stats.ttest_ind_from_stats`), on the log scale with
the log-normal moments of `ParameterSample.log_moments`.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import t as student_t

from pkpdutils.stats.sample import ParameterSample, Scale


class TestMethod(StrEnum):
    """Test of `compare`."""

    #: paired t test for `paired=True`, else Welch t test
    AUTO = "auto"
    #: Student t test, equal variances
    STUDENT_T = "student_t"
    #: Welch t test, unequal variances
    WELCH_T = "welch_t"
    #: paired t test
    PAIRED_T = "paired_t"
    #: Mann-Whitney U rank test, unpaired
    MANN_WHITNEY = "mann_whitney"
    #: Wilcoxon signed rank test, paired
    WILCOXON = "wilcoxon"
    #: permutation test of the difference of means
    PERMUTATION = "permutation"


class Alternative(StrEnum):
    """Alternative hypothesis, as in scipy."""

    TWO_SIDED = "two-sided"
    #: the center of `a` is less than the center of `b`
    LESS = "less"
    #: the center of `a` is greater than the center of `b`
    GREATER = "greater"


class AdjustMethod(StrEnum):
    """Adjustment of p values for multiple comparisons."""

    #: Holm step-down (Holm 1979), controls the family-wise error rate
    HOLM = "holm"
    #: Bonferroni, `m p`
    BONFERRONI = "bonferroni"
    #: Benjamini-Hochberg step-up (Benjamini & Hochberg 1995), controls the false discovery rate
    BH = "bh"


#: the t tests: an effect with a t interval and degrees of freedom
_T_TESTS = frozenset({TestMethod.STUDENT_T, TestMethod.WELCH_T, TestMethod.PAIRED_T})
#: the paired tests
_PAIRED_TESTS = frozenset({TestMethod.PAIRED_T, TestMethod.WILCOXON})


@dataclass(frozen=True)
class TestResult:
    """Result of `compare`.

    Attributes:
        test: the test which was run (`AUTO` resolved)
        statistic: the test statistic
        p_value: the p value under `alternative`
        effect: difference of the means `a - b` (`LINEAR`) or ratio of the geometric means `a / b` (`LOG`); for the rank tests the difference or the ratio of the medians
        ci_low: lower bound of the interval of the effect (t tests; one-sided under a one-sided alternative), `NaN` otherwise
        ci_high: upper bound of the interval
        ci_level: level of the interval
        scale: scale of the analysis
        alternative: the alternative hypothesis
        paired: whether the samples were paired
        df: degrees of freedom of a t test, `NaN` otherwise
        cohen_d: standardized difference of the means on the analysis scale, pooled standard deviation
        hedges_g: `cohen_d` times the small sample correction `J`
        n_a: number of values of `a`
        n_b: number of values of `b`
        name: name of the parameter (of `a`)
        unit: unit of the parameter
    """

    test: TestMethod
    statistic: float
    p_value: float
    effect: float
    ci_low: float
    ci_high: float
    ci_level: float
    scale: Scale
    alternative: Alternative
    paired: bool
    df: float
    cohen_d: float
    hedges_g: float
    n_a: int
    n_b: int
    name: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the enumerations as strings.

        Returns:
            Field name to value.
        """
        return {
            "test": str(self.test),
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "scale": str(self.scale),
            "alternative": str(self.alternative),
            "paired": self.paired,
            "df": self.df,
            "cohen_d": self.cohen_d,
            "hedges_g": self.hedges_g,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "name": self.name,
            "unit": self.unit,
        }


def hedges_correction(n_total: int) -> float:
    """Small sample correction of the standardized mean difference.

    \\(J = 1 - 3 / (4N - 9)\\) (Hedges 1981, the approximation of the exact
    gamma expression), with \\(N\\) the total number of values.

    Args:
        n_total: total number of values of both samples.

    Returns:
        The factor `J`.
    """
    return 1.0 - 3.0 / (4.0 * n_total - 9.0)


def _values(sample: ParameterSample, scale: Scale) -> np.ndarray:
    """The finite values of an individual sample on the analysis scale.

    Args:
        sample: the sample.
        scale: the scale.

    Returns:
        The values (logarithms on the log scale).

    Raises:
        ValueError: for summary data.
    """
    if not sample.is_individual:
        raise ValueError(f"'{sample.name}' has no individual values, this test needs individual data")
    return sample.log_values if scale is Scale.LOG else sample.finite_values


def _back(value: float, scale: Scale) -> float:
    """An effect on the analysis scale as reported: exponentiated on the log scale.

    Args:
        value: the effect on the analysis scale.
        scale: the scale.

    Returns:
        The reported effect.
    """
    return float(np.exp(value)) if scale is Scale.LOG else float(value)


def _interval(
    center: float, se: float, df: float, ci_level: float, alternative: Alternative
) -> tuple[float, float]:
    """The t interval of an effect on the analysis scale.

    Two-sided: \\(\\hat\\theta \\pm t_{1-\\alpha/2, df}\\,\\mathrm{se}\\); under a one-sided
    alternative the interval is one-sided at `ci_level`, the other bound is
    infinite.

    Args:
        center: the estimate.
        se: its standard error.
        df: degrees of freedom.
        ci_level: level of the interval.
        alternative: the alternative hypothesis.

    Returns:
        The lower and the upper bound.
    """
    alpha = 1.0 - ci_level
    if alternative is Alternative.TWO_SIDED:
        tq = float(student_t.ppf(1.0 - alpha / 2.0, df))
        return center - tq * se, center + tq * se
    tq = float(student_t.ppf(1.0 - alpha, df))
    if alternative is Alternative.LESS:
        return -np.inf, center + tq * se
    return center - tq * se, np.inf


def _p_from_t(statistic: float, df: float, alternative: Alternative) -> float:
    """The p value of a t statistic under an alternative.

    Args:
        statistic: the t statistic.
        df: degrees of freedom.
        alternative: the alternative hypothesis.

    Returns:
        The p value.
    """
    if alternative is Alternative.TWO_SIDED:
        return float(2.0 * student_t.sf(abs(statistic), df))
    if alternative is Alternative.LESS:
        return float(student_t.cdf(statistic, df))
    return float(student_t.sf(statistic, df))


def _effect_sizes(
    mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int
) -> tuple[float, float]:
    """Cohen's d and Hedges' g from the moments of two samples.

    \\(d = (\\bar a - \\bar b) / s_p\\), \\(s_p^2 = ((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2)\\),
    \\(g = J d\\).

    Args:
        mean_a: mean of `a`.
        sd_a: standard deviation of `a`.
        n_a: size of `a`.
        mean_b: mean of `b`.
        sd_b: standard deviation of `b`.
        n_b: size of `b`.

    Returns:
        `d` and `g` (`NaN` with fewer than 3 values in total).
    """
    if n_a + n_b < 3:
        return float("nan"), float("nan")
    pooled = np.sqrt(((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2))
    d = float((mean_a - mean_b) / pooled) if pooled > 0 else float("nan")
    return d, d * hedges_correction(n_a + n_b)


def _welch_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """Welch-Satterthwaite degrees of freedom.

    \\(\\nu = (s_a^2/n_a + s_b^2/n_b)^2 / ((s_a^2/n_a)^2/(n_a-1) + (s_b^2/n_b)^2/(n_b-1))\\).

    Args:
        var_a: variance of `a`.
        n_a: size of `a`.
        var_b: variance of `b`.
        n_b: size of `b`.

    Returns:
        The degrees of freedom.
    """
    va, vb = var_a / n_a, var_b / n_b
    return float((va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1)))


def compare(
    a: ParameterSample,
    b: ParameterSample,
    *,
    test: TestMethod = TestMethod.AUTO,
    scale: Scale = Scale.LOG,
    paired: bool = False,
    alternative: Alternative = Alternative.TWO_SIDED,
    ci_level: float = 0.95,
    n_perm: int = 9999,
    seed: int | None = None,
) -> TestResult:
    """Compare two samples of a parameter.

    `AUTO` runs the paired t test for `paired=True` and the Welch t test
    otherwise; summary data is compared with the Welch t test from its
    moments. On the log scale the tests run on the logarithms and the effect
    is the ratio of the geometric means with the exponentiated t interval.
    The paired tests need individual data of equal size (matched by
    position). The permutation test permutes the group labels (or the signs
    of the paired differences) of the difference of the means, with
    `n_perm` resamples (Efron & Tibshirani 1993, ch. 15).

    Args:
        a: the first sample.
        b: the second sample.
        test: the test.
        scale: scale of the analysis.
        paired: whether the values of `a` and `b` are paired by position.
        alternative: the alternative hypothesis.
        ci_level: level of the interval of the effect.
        n_perm: number of resamples of the permutation test.
        seed: seed of the permutation test.

    Returns:
        The result.

    Raises:
        ValueError: for a paired test on unpaired or unequal samples, a
            non-t test on summary data, or non-positive values on the log scale.
    """
    method = test
    if method is TestMethod.AUTO:
        method = TestMethod.PAIRED_T if paired else TestMethod.WELCH_T
    if method in _PAIRED_TESTS and not paired:
        raise ValueError(f"{method} needs paired=True")
    if paired and method not in _PAIRED_TESTS and method is not TestMethod.PERMUTATION:
        raise ValueError(f"{method} is not a paired test, use PAIRED_T, WILCOXON or PERMUTATION")
    if not (a.is_individual and b.is_individual):
        if method is not TestMethod.WELCH_T:
            raise ValueError(f"{method} needs individual data; summary data allows WELCH_T only")
        return _welch_from_moments(a, b, scale, alternative, ci_level)
    x, y = _values(a, scale), _values(b, scale)
    n_a, n_b = x.size, y.size
    if paired and n_a != n_b:
        raise ValueError(f"paired samples need equal sizes, got {n_a} and {n_b}")
    mean_a, mean_b = float(x.mean()), float(y.mean())
    sd_a = float(x.std(ddof=1)) if n_a > 1 else float("nan")
    sd_b = float(y.std(ddof=1)) if n_b > 1 else float("nan")
    d, g = _effect_sizes(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
    nan = float("nan")
    center = mean_a - mean_b
    if method in _T_TESTS:
        if method is TestMethod.PAIRED_T:
            diff = x - y
            se = float(diff.std(ddof=1) / np.sqrt(n_a))
            df = float(n_a - 1)
        elif method is TestMethod.WELCH_T:
            se = float(np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b))
            df = _welch_df(sd_a**2, n_a, sd_b**2, n_b)
        else:
            pooled = ((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2)
            se = float(np.sqrt(pooled * (1.0 / n_a + 1.0 / n_b)))
            df = float(n_a + n_b - 2)
        statistic = center / se
        p_value = _p_from_t(statistic, df, alternative)
        low, high = _interval(center, se, df, ci_level, alternative)
        ci = (_back(low, scale), _back(high, scale))
        effect = _back(center, scale)
    elif method is TestMethod.MANN_WHITNEY:
        res = stats.mannwhitneyu(x, y, alternative=str(alternative))
        statistic, p_value, df = float(res.statistic), float(res.pvalue), nan
        effect = _back(float(np.median(x) - np.median(y)), scale)
        ci = (nan, nan)
    elif method is TestMethod.WILCOXON:
        res = stats.wilcoxon(x, y, alternative=str(alternative))
        statistic, p_value, df = float(res.statistic), float(res.pvalue), nan
        effect = _back(float(np.median(x - y)), scale)
        ci = (nan, nan)
    else:
        res = stats.permutation_test(
            (x, y),
            lambda u, v, axis: np.mean(u, axis=axis) - np.mean(v, axis=axis),
            permutation_type="samples" if paired else "independent",
            n_resamples=n_perm,
            alternative=str(alternative),
            rng=np.random.default_rng(seed),
        )
        statistic, p_value, df = float(res.statistic), float(res.pvalue), nan
        effect = _back(center, scale)
        ci = (nan, nan)
    return TestResult(
        test=method,
        statistic=float(statistic),
        p_value=p_value,
        effect=effect,
        ci_low=ci[0],
        ci_high=ci[1],
        ci_level=ci_level,
        scale=scale,
        alternative=alternative,
        paired=paired,
        df=df,
        cohen_d=d,
        hedges_g=g,
        n_a=n_a,
        n_b=n_b,
        name=a.name,
        unit=a.unit,
    )


def _welch_from_moments(
    a: ParameterSample,
    b: ParameterSample,
    scale: Scale,
    alternative: Alternative,
    ci_level: float,
) -> TestResult:
    """Welch t test from the moments of two samples (summary data).

    Args:
        a: the first sample.
        b: the second sample.
        scale: scale of the analysis (log-normal moments on the log scale).
        alternative: the alternative hypothesis.
        ci_level: level of the interval.

    Returns:
        The result.
    """
    mean_a, sd_a, n_a = a.moments(scale)
    mean_b, sd_b, n_b = b.moments(scale)
    se = float(np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b))
    df = _welch_df(sd_a**2, n_a, sd_b**2, n_b)
    center = mean_a - mean_b
    statistic = center / se
    low, high = _interval(center, se, df, ci_level, alternative)
    d, g = _effect_sizes(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
    return TestResult(
        test=TestMethod.WELCH_T,
        statistic=float(statistic),
        p_value=_p_from_t(statistic, df, alternative),
        effect=_back(center, scale),
        ci_low=_back(low, scale),
        ci_high=_back(high, scale),
        ci_level=ci_level,
        scale=scale,
        alternative=alternative,
        paired=False,
        df=df,
        cohen_d=d,
        hedges_g=g,
        n_a=n_a,
        n_b=n_b,
        name=a.name,
        unit=a.unit,
    )


def multiple_comparison(
    p_values: Any, method: AdjustMethod = AdjustMethod.HOLM
) -> np.ndarray:
    """Adjust p values for multiple comparisons.

    Bonferroni: \\(\\min(1, m p_i)\\). Holm (step-down): sort ascending,
    \\(\\tilde p_{(i)} = \\max_{j \\le i} \\min(1, (m - j + 1) p_{(j)})\\).
    Benjamini-Hochberg (step-up): \\(\\tilde p_{(i)} = \\min_{j \\ge i} \\min(1, m p_{(j)} / j)\\).

    Args:
        p_values: the p values.
        method: the adjustment.

    Returns:
        The adjusted p values in the order of the input.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return p
    if method is AdjustMethod.BONFERRONI:
        return np.minimum(1.0, m * p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    adjusted = np.empty(m)
    if method is AdjustMethod.HOLM:
        stepped = np.minimum(1.0, (m - ranks + 1) * p[order])
        adjusted[order] = np.maximum.accumulate(stepped)
    else:
        stepped = np.minimum(1.0, m * p[order] / ranks)
        adjusted[order] = np.minimum.accumulate(stepped[::-1])[::-1]
    return adjusted
```

Export from `src/pkpdutils/stats/__init__.py`: `AdjustMethod`, `Alternative`, `TestMethod`, `TestResult`, `compare`, `multiple_comparison` (keep `__all__` sorted).

`_back` of `-np.inf` on the log scale gives `0.0` and of `np.inf` gives `inf`, which is what the one-sided test expects. The `lambda` of the permutation test is a module level function if ruff (`E731`) complains: define `def _mean_difference(u, v, axis): return np.mean(u, axis=axis) - np.mean(v, axis=axis)` with full annotations (`np.ndarray`, `int`) and a one-line docstring.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/stats -q -W error`
Expected: PASS. The `scipy.stats.permutation_test` keyword is `rng` in scipy >= 1.15 (`random_state` is deprecated and would warn under `-W error`). Run `uv run ruff check src tests && uv run ruff format src tests && uvx ty check` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/stats tests/stats
git commit -q -m "Add compare with the t, rank and permutation tests, effect sizes and the p value adjustments

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 4: Geometric mean ratio (`stats/ratio.py`) and DDI classification (`stats/ddi.py`)

**Files:**
- Create: `src/pkpdutils/stats/ratio.py`, `src/pkpdutils/stats/ddi.py`
- Modify: `src/pkpdutils/stats/__init__.py`
- Test: `tests/stats/test_ratio.py`, `tests/stats/test_ddi.py`

**Interfaces:**
- `RatioResult` (frozen dataclass): `gmr`, `ci_low`, `ci_high`, `ci_level`, `log_ratio` (\(\ln \mathrm{GMR}\)), `se_log`, `df`, `paired`, `n_test`, `n_reference`, `name`, `unit`; `to_dict()`.
- `ratio(test: ParameterSample, reference: ParameterSample, *, ci_level=0.90, paired: bool | None = None) -> RatioResult`. `paired=None` pairs the samples when both are individual, have labels, and the label sets are equal (the values are then matched by label); `paired=True` on samples without labels pairs by position and needs equal sizes; `paired=False` is Welch on the logs; summary data is Welch from the log moments.
- `DDIKind` (`INHIBITOR`, `INDUCER`, `NONE`), `DDIStrength` (`STRONG`, `MODERATE`, `WEAK`, `NONE`), `Sensitivity` (`SENSITIVE`, `MODERATELY_SENSITIVE`, `NONE`).
- `DDIThresholds` (frozen dataclass): `inhibitor_weak = 1.25`, `inhibitor_moderate = 2.0`, `inhibitor_strong = 5.0`, `inducer_weak = 0.8`, `inducer_moderate = 0.5`, `inducer_strong = 0.2`, `sensitive = 5.0`, `moderately_sensitive = 2.0`, `source: str`; `fda()`, `ema()` classmethods; `classify(auc_ratio) -> tuple[DDIKind, DDIStrength]`.
- `DDIResult` (frozen dataclass): `kind`, `strength`, `auc_ratio`, `cmax_ratio`, `ci_low`, `ci_high`, `uncertain`, `kind_low`, `strength_low`, `kind_high`, `strength_high`, `thresholds`; `to_dict()`.
- `ddi_classification(auc_ratio: float | RatioResult, *, cmax_ratio: float | RatioResult | None = None, ci: tuple[float, float] | None = None, thresholds: DDIThresholds = DDIThresholds.fda()) -> DDIResult`; `substrate_sensitivity(auc_ratio: float | RatioResult, thresholds=DDIThresholds.fda()) -> Sensitivity`.

- [ ] **Step 1: Write the failing tests**

`tests/stats/test_ratio.py`:
```python
import numpy as np
import pytest
from scipy import stats
from scipy.stats import t as student_t

from pkpdutils.stats import ParameterSample
from pkpdutils.stats.ratio import RatioResult, ratio

RNG = np.random.default_rng(11)
REF = np.exp(RNG.normal(np.log(100), 0.25, 12))
TEST = REF * np.exp(RNG.normal(np.log(0.95), 0.12, 12))
LABELS = np.array([f"s{i}" for i in range(12)])


def test_paired_by_label_order_independent() -> None:
    t = ParameterSample(values=TEST, labels=LABELS, name="auc_inf_obs", unit="mg*hr/l")
    perm = RNG.permutation(12)
    r = ParameterSample(values=REF[perm], labels=LABELS[perm])
    res = ratio(t, r)
    assert isinstance(res, RatioResult) and res.paired
    d = np.log(TEST) - np.log(REF)
    se = d.std(ddof=1) / np.sqrt(12)
    tq = student_t.ppf(0.95, 11)
    assert res.gmr == pytest.approx(np.exp(d.mean()))
    assert res.log_ratio == pytest.approx(d.mean())
    assert (res.ci_low, res.ci_high) == pytest.approx((np.exp(d.mean() - tq * se), np.exp(d.mean() + tq * se)))
    assert res.se_log == pytest.approx(se) and res.df == 11
    assert res.n_test == 12 and res.n_reference == 12 and res.ci_level == 0.90
    assert res.name == "auc_inf_obs" and res.unit == "mg*hr/l"


def test_unpaired_is_welch_on_logs() -> None:
    res = ratio(ParameterSample(values=TEST), ParameterSample(values=REF[:10]))
    assert not res.paired
    ref = stats.ttest_ind(np.log(TEST), np.log(REF[:10]), equal_var=False)
    low, high = ref.confidence_interval(0.90)
    assert (res.ci_low, res.ci_high) == pytest.approx((np.exp(low), np.exp(high)))
    assert res.df == pytest.approx(ref.df)
    assert res.gmr == pytest.approx(np.exp(np.log(TEST).mean() - np.log(REF[:10]).mean()))


def test_paired_rules() -> None:
    with pytest.raises(ValueError, match="equal sizes"):
        ratio(ParameterSample(values=TEST), ParameterSample(values=REF[:10]), paired=True)
    by_position = ratio(ParameterSample(values=TEST), ParameterSample(values=REF), paired=True)
    assert by_position.paired and by_position.df == 11
    forced_unpaired = ratio(
        ParameterSample(values=TEST, labels=LABELS), ParameterSample(values=REF, labels=LABELS), paired=False
    )
    assert not forced_unpaired.paired
    other_labels = ParameterSample(values=REF, labels=np.array([f"x{i}" for i in range(12)]))
    assert not ratio(ParameterSample(values=TEST, labels=LABELS), other_labels).paired
    with pytest.raises(ValueError, match="labels"):
        ratio(ParameterSample(values=TEST, labels=LABELS), other_labels, paired=True)


def test_summary_data() -> None:
    t = ParameterSample(geomean=95.0, geocv=0.3, n=12)
    r = ParameterSample(mean=100.0, sd=25.0, n=12)
    res = ratio(t, r)
    mu_t, s_t = t.log_moments()
    mu_r, s_r = r.log_moments()
    ref = stats.ttest_ind_from_stats(mu_t, s_t, 12, mu_r, s_r, 12, equal_var=False)
    assert res.gmr == pytest.approx(np.exp(mu_t - mu_r))
    assert res.df == pytest.approx(ref.df)
    assert res.to_dict()["gmr"] == res.gmr
    with pytest.raises(ValueError, match="individual"):
        ratio(t, ParameterSample(values=REF), paired=True)
```

`tests/stats/test_ddi.py`:
```python
import pytest

from pkpdutils.stats import ParameterSample
from pkpdutils.stats.ddi import (
    DDIKind,
    DDIResult,
    DDIStrength,
    DDIThresholds,
    Sensitivity,
    ddi_classification,
    substrate_sensitivity,
)
from pkpdutils.stats.ratio import ratio


@pytest.mark.parametrize(
    ("auc_ratio", "kind", "strength"),
    [
        (5.0, DDIKind.INHIBITOR, DDIStrength.STRONG),
        (7.3, DDIKind.INHIBITOR, DDIStrength.STRONG),
        (4.99, DDIKind.INHIBITOR, DDIStrength.MODERATE),
        (2.0, DDIKind.INHIBITOR, DDIStrength.MODERATE),
        (1.99, DDIKind.INHIBITOR, DDIStrength.WEAK),
        (1.25, DDIKind.INHIBITOR, DDIStrength.WEAK),
        (1.24, DDIKind.NONE, DDIStrength.NONE),
        (1.0, DDIKind.NONE, DDIStrength.NONE),
        (0.81, DDIKind.NONE, DDIStrength.NONE),
        (0.8, DDIKind.INDUCER, DDIStrength.WEAK),
        (0.51, DDIKind.INDUCER, DDIStrength.WEAK),
        (0.5, DDIKind.INDUCER, DDIStrength.MODERATE),
        (0.21, DDIKind.INDUCER, DDIStrength.MODERATE),
        (0.2, DDIKind.INDUCER, DDIStrength.STRONG),
        (0.05, DDIKind.INDUCER, DDIStrength.STRONG),
    ],
)
def test_thresholds_at_the_boundaries(auc_ratio: float, kind: DDIKind, strength: DDIStrength) -> None:
    assert DDIThresholds.fda().classify(auc_ratio) == (kind, strength)
    res = ddi_classification(auc_ratio)
    assert isinstance(res, DDIResult)
    assert (res.kind, res.strength) == (kind, strength) and not res.uncertain
    assert res.auc_ratio == auc_ratio and res.cmax_ratio is None


def test_interval_uses_the_bound_closer_to_one() -> None:
    res = ddi_classification(2.5, ci=(1.6, 3.9))
    assert (res.kind, res.strength) == (DDIKind.INHIBITOR, DDIStrength.WEAK)
    assert res.uncertain and (res.kind_low, res.strength_low) == (DDIKind.INHIBITOR, DDIStrength.WEAK)
    assert (res.kind_high, res.strength_high) == (DDIKind.INHIBITOR, DDIStrength.MODERATE)
    certain = ddi_classification(2.5, ci=(2.1, 3.0))
    assert (certain.kind, certain.strength) == (DDIKind.INHIBITOR, DDIStrength.MODERATE) and not certain.uncertain
    inducer = ddi_classification(0.4, ci=(0.3, 0.55))
    assert (inducer.kind, inducer.strength) == (DDIKind.INDUCER, DDIStrength.WEAK) and inducer.uncertain
    spanning = ddi_classification(1.3, ci=(0.9, 1.9))
    assert (spanning.kind, spanning.strength) == (DDIKind.NONE, DDIStrength.NONE) and spanning.uncertain
    with pytest.raises(ValueError, match="ci"):
        ddi_classification(2.0, ci=(3.0, 1.0))


def test_ratio_result_input() -> None:
    inhibited = ParameterSample(values=[300.0, 320.0, 280.0, 310.0, 290.0])
    control = ParameterSample(values=[100.0, 105.0, 95.0, 110.0, 90.0])
    r = ratio(inhibited, control)
    res = ddi_classification(r, cmax_ratio=ratio(control, control))
    assert res.auc_ratio == r.gmr and (res.ci_low, res.ci_high) == (r.ci_low, r.ci_high)
    assert res.cmax_ratio == 1.0
    assert (res.kind, res.strength) == (DDIKind.INHIBITOR, DDIStrength.MODERATE)
    assert res.to_dict()["kind"] == "inhibitor"


def test_ema_and_sensitivity() -> None:
    ema = DDIThresholds.ema()
    assert ema.source.startswith("EMA") and ema.inhibitor_strong == 5.0
    assert ddi_classification(3.0, thresholds=ema).thresholds is ema
    assert substrate_sensitivity(5.0) is Sensitivity.SENSITIVE
    assert substrate_sensitivity(2.0) is Sensitivity.MODERATELY_SENSITIVE
    assert substrate_sensitivity(1.9) is Sensitivity.NONE
    with pytest.raises(ValueError, match="positive"):
        ddi_classification(0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/stats/test_ratio.py tests/stats/test_ddi.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `stats/ratio.py`**

```python
"""Geometric mean ratio of a parameter between a test and a reference sample."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from pkpdutils.stats.sample import ParameterSample, Scale
from pkpdutils.stats.tests import _welch_df


@dataclass(frozen=True)
class RatioResult:
    """Geometric mean ratio with its t interval on the log scale.

    Attributes:
        gmr: geometric mean ratio test / reference
        ci_low: lower bound of the interval of the ratio
        ci_high: upper bound of the interval
        ci_level: level of the interval
        log_ratio: \\(\\ln \\mathrm{GMR}\\)
        se_log: standard error of `log_ratio`
        df: degrees of freedom of the t interval
        paired: whether the samples were paired
        n_test: number of values of the test sample
        n_reference: number of values of the reference sample
        name: name of the parameter
        unit: unit of the parameter
    """

    gmr: float
    ci_low: float
    ci_high: float
    ci_level: float
    log_ratio: float
    se_log: float
    df: float
    paired: bool
    n_test: int
    n_reference: int
    name: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary.

        Returns:
            Field name to value.
        """
        return {
            "gmr": self.gmr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "log_ratio": self.log_ratio,
            "se_log": self.se_log,
            "df": self.df,
            "paired": self.paired,
            "n_test": self.n_test,
            "n_reference": self.n_reference,
            "name": self.name,
            "unit": self.unit,
        }


def _pair(test: ParameterSample, reference: ParameterSample) -> tuple[np.ndarray, np.ndarray]:
    """The log values of two paired samples, matched by label when both have labels.

    Args:
        test: the test sample.
        reference: the reference sample.

    Returns:
        The logarithms of the test and the reference values in matching order.

    Raises:
        ValueError: for summary data, unequal sizes, or labels which do not match.
    """
    if not (test.is_individual and reference.is_individual):
        raise ValueError("A paired ratio needs individual data of both samples")
    x, y = test.log_values, reference.log_values
    lx, ly = test.finite_labels, reference.finite_labels
    if lx is not None and ly is not None:
        if set(lx.tolist()) != set(ly.tolist()) or lx.size != ly.size:
            raise ValueError(
                f"The labels of '{test.name}' and '{reference.name}' do not match; pairing needs the same individuals"
            )
        order = {label: i for i, label in enumerate(ly.tolist())}
        y = y[[order[label] for label in lx.tolist()]]
    elif x.size != y.size:
        raise ValueError(f"paired samples need equal sizes, got {x.size} and {y.size}")
    return x, y


def _labels_match(test: ParameterSample, reference: ParameterSample) -> bool:
    """Whether both samples are individual, labelled and hold the same labels.

    Args:
        test: the test sample.
        reference: the reference sample.

    Returns:
        `True` if the samples can be paired by label.
    """
    lx, ly = test.finite_labels, reference.finite_labels
    if lx is None or ly is None:
        return False
    return lx.size == ly.size and set(lx.tolist()) == set(ly.tolist())


def ratio(
    test: ParameterSample,
    reference: ParameterSample,
    *,
    ci_level: float = 0.90,
    paired: bool | None = None,
) -> RatioResult:
    """Geometric mean ratio of a parameter, test over reference, with a t interval.

    Paired (crossover design): \\(d_i = \\ln t_i - \\ln r_i\\), \\(\\ln \\mathrm{GMR} = \\bar d\\),
    \\(\\mathrm{se} = s_d / \\sqrt{n}\\), \\(n - 1\\) degrees of freedom. Unpaired (parallel
    groups): the Welch t interval of \\(\\bar{\\ln t} - \\bar{\\ln r}\\). The interval of
    the ratio is the exponentiated interval (FDA 2001; Schuirmann 1987).
    Summary data uses the log moments of `ParameterSample.log_moments`.

    Args:
        test: the test sample.
        reference: the reference sample.
        ci_level: level of the interval, 0.90 by default as in bioequivalence.
        paired: pair the samples (by label when both have labels, else by
            position); `None` pairs when both samples carry the same labels.

    Returns:
        The ratio.

    Raises:
        ValueError: for a paired ratio on summary data, unequal sizes or
            labels which do not match.
    """
    is_paired = _labels_match(test, reference) if paired is None else paired
    alpha = 1.0 - ci_level
    if is_paired:
        x, y = _pair(test, reference)
        d = x - y
        n = d.size
        center = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        df = float(n - 1)
        n_test = n_reference = n
    else:
        mu_t, s_t, n_test = test.moments(Scale.LOG)
        mu_r, s_r, n_reference = reference.moments(Scale.LOG)
        center = mu_t - mu_r
        se = float(np.sqrt(s_t**2 / n_test + s_r**2 / n_reference))
        df = _welch_df(s_t**2, n_test, s_r**2, n_reference)
    tq = float(student_t.ppf(1.0 - alpha / 2.0, df)) if df > 0 else float("nan")
    return RatioResult(
        gmr=float(np.exp(center)),
        ci_low=float(np.exp(center - tq * se)),
        ci_high=float(np.exp(center + tq * se)),
        ci_level=ci_level,
        log_ratio=center,
        se_log=se,
        df=df,
        paired=bool(is_paired),
        n_test=int(n_test),
        n_reference=int(n_reference),
        name=test.name,
        unit=test.unit,
    )
```

`_welch_df` is imported from `stats/tests.py`; rename it to `welch_df` (public, documented) there if ruff or ty object to the private import (they do not; keep the underscore).

- [ ] **Step 4: Implement `stats/ddi.py`**

```python
"""Classification of drug-drug interactions by the change of the exposure.

The FDA guidance (FDA 2020) classifies a perpetrator by the ratio of the
AUC of a sensitive substrate with and without it: a strong, moderate or
weak inhibitor raises the AUC at least 5-fold, 2- to 5-fold or 1.25- to
2-fold; a strong, moderate or weak inducer lowers it by at least 80 %,
50-80 % or 20-50 %. A substrate is sensitive when a strong inhibitor raises
its AUC at least 5-fold and moderately sensitive at 2- to 5-fold. The EMA
guideline (EMA 2012) uses the same thresholds.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pkpdutils.stats.ratio import RatioResult


class DDIKind(StrEnum):
    """Direction of an interaction."""

    INHIBITOR = "inhibitor"
    INDUCER = "inducer"
    NONE = "none"


class DDIStrength(StrEnum):
    """Strength of an interaction."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class Sensitivity(StrEnum):
    """Sensitivity of a substrate to a strong inhibitor."""

    SENSITIVE = "sensitive"
    MODERATELY_SENSITIVE = "moderately_sensitive"
    NONE = "none"


@dataclass(frozen=True)
class DDIThresholds:
    """Thresholds of the classification, as AUC ratios with / without the perpetrator.

    An inhibitor threshold is the smallest ratio of its class, an inducer
    threshold the largest (`inducer_strong = 0.2` is an 80 % decrease).

    Attributes:
        inhibitor_weak: weak inhibitor at or above this ratio
        inhibitor_moderate: moderate inhibitor at or above this ratio
        inhibitor_strong: strong inhibitor at or above this ratio
        inducer_weak: weak inducer at or below this ratio
        inducer_moderate: moderate inducer at or below this ratio
        inducer_strong: strong inducer at or below this ratio
        sensitive: sensitive substrate at or above this ratio
        moderately_sensitive: moderately sensitive substrate at or above this ratio
        source: the guidance the thresholds come from
    """

    inhibitor_weak: float = 1.25
    inhibitor_moderate: float = 2.0
    inhibitor_strong: float = 5.0
    inducer_weak: float = 0.8
    inducer_moderate: float = 0.5
    inducer_strong: float = 0.2
    sensitive: float = 5.0
    moderately_sensitive: float = 2.0
    source: str = "FDA 2020"

    @classmethod
    def fda(cls) -> "DDIThresholds":
        """The thresholds of the FDA clinical drug interaction guidance (FDA 2020).

        Returns:
            The thresholds.
        """
        return cls()

    @classmethod
    def ema(cls) -> "DDIThresholds":
        """The thresholds of the EMA guideline on the investigation of drug interactions (EMA 2012), which equal the FDA ones.

        Returns:
            The thresholds.
        """
        return cls(source="EMA 2012")

    def classify(self, auc_ratio: float) -> tuple[DDIKind, DDIStrength]:
        """Kind and strength of an interaction from an AUC ratio.

        Args:
            auc_ratio: AUC with / without the perpetrator, positive.

        Returns:
            The kind and the strength.

        Raises:
            ValueError: if the ratio is not positive.
        """
        if not auc_ratio > 0:
            raise ValueError(f"The AUC ratio must be positive, got {auc_ratio}")
        if auc_ratio >= self.inhibitor_strong:
            return DDIKind.INHIBITOR, DDIStrength.STRONG
        if auc_ratio >= self.inhibitor_moderate:
            return DDIKind.INHIBITOR, DDIStrength.MODERATE
        if auc_ratio >= self.inhibitor_weak:
            return DDIKind.INHIBITOR, DDIStrength.WEAK
        if auc_ratio <= self.inducer_strong:
            return DDIKind.INDUCER, DDIStrength.STRONG
        if auc_ratio <= self.inducer_moderate:
            return DDIKind.INDUCER, DDIStrength.MODERATE
        if auc_ratio <= self.inducer_weak:
            return DDIKind.INDUCER, DDIStrength.WEAK
        return DDIKind.NONE, DDIStrength.NONE


@dataclass(frozen=True)
class DDIResult:
    """Classification of an interaction.

    Attributes:
        kind: the classification (from the bound of the interval closer to 1 when an interval is given)
        strength: the strength
        auc_ratio: the AUC ratio
        cmax_ratio: the Cmax ratio, reported only
        ci_low: lower bound of the interval of the AUC ratio, `NaN` without one
        ci_high: upper bound of the interval
        uncertain: whether the interval spans a boundary of the classes
        kind_low: classification of `ci_low`
        strength_low: strength of `ci_low`
        kind_high: classification of `ci_high`
        strength_high: strength of `ci_high`
        thresholds: the thresholds used
    """

    kind: DDIKind
    strength: DDIStrength
    auc_ratio: float
    cmax_ratio: float | None
    ci_low: float
    ci_high: float
    uncertain: bool
    kind_low: DDIKind
    strength_low: DDIStrength
    kind_high: DDIKind
    strength_high: DDIStrength
    thresholds: DDIThresholds

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the enumerations as strings.

        Returns:
            Field name to value.
        """
        return {
            "kind": str(self.kind),
            "strength": str(self.strength),
            "auc_ratio": self.auc_ratio,
            "cmax_ratio": self.cmax_ratio,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "uncertain": self.uncertain,
            "kind_low": str(self.kind_low),
            "strength_low": str(self.strength_low),
            "kind_high": str(self.kind_high),
            "strength_high": str(self.strength_high),
            "source": self.thresholds.source,
        }


def _ratio_value(value: float | RatioResult) -> float:
    """The point estimate of a ratio given as a number or as a `RatioResult`.

    Args:
        value: the ratio.

    Returns:
        The number.
    """
    return value.gmr if isinstance(value, RatioResult) else float(value)


def ddi_classification(
    auc_ratio: float | RatioResult,
    *,
    cmax_ratio: float | RatioResult | None = None,
    ci: tuple[float, float] | None = None,
    thresholds: DDIThresholds = DDIThresholds.fda(),
) -> DDIResult:
    """Classify a perpetrator by the AUC ratio of a substrate with and without it.

    With an interval (given as `ci` or carried by a `RatioResult`) the
    classification is conservative: it uses the bound closer to 1 (the lower
    bound of an increase, the upper bound of a decrease), an interval which
    contains 1 gives no interaction, and `uncertain` is set when the two
    bounds fall into different classes.

    Args:
        auc_ratio: AUC ratio with / without the perpetrator, or the `ratio` result.
        cmax_ratio: Cmax ratio, reported next to the classification.
        ci: interval of the AUC ratio; overrides the interval of a `RatioResult`.
        thresholds: the thresholds, FDA 2020 by default.

    Returns:
        The classification.

    Raises:
        ValueError: if the interval is reversed or a ratio is not positive.
    """
    value = _ratio_value(auc_ratio)
    if ci is None and isinstance(auc_ratio, RatioResult):
        ci = (auc_ratio.ci_low, auc_ratio.ci_high)
    kind, strength = thresholds.classify(value)
    if ci is None:
        return DDIResult(
            kind=kind,
            strength=strength,
            auc_ratio=value,
            cmax_ratio=None if cmax_ratio is None else _ratio_value(cmax_ratio),
            ci_low=float("nan"),
            ci_high=float("nan"),
            uncertain=False,
            kind_low=kind,
            strength_low=strength,
            kind_high=kind,
            strength_high=strength,
            thresholds=thresholds,
        )
    low, high = float(ci[0]), float(ci[1])
    if not low <= high:
        raise ValueError(f"'ci' must be (low, high), got {ci}")
    kind_low, strength_low = thresholds.classify(low)
    kind_high, strength_high = thresholds.classify(high)
    closer = low if low > 1.0 else high if high < 1.0 else 1.0
    kind, strength = thresholds.classify(closer)
    return DDIResult(
        kind=kind,
        strength=strength,
        auc_ratio=value,
        cmax_ratio=None if cmax_ratio is None else _ratio_value(cmax_ratio),
        ci_low=low,
        ci_high=high,
        uncertain=(kind_low, strength_low) != (kind_high, strength_high),
        kind_low=kind_low,
        strength_low=strength_low,
        kind_high=kind_high,
        strength_high=strength_high,
        thresholds=thresholds,
    )


def substrate_sensitivity(
    auc_ratio: float | RatioResult, thresholds: DDIThresholds = DDIThresholds.fda()
) -> Sensitivity:
    """Sensitivity of a substrate from its AUC ratio with a strong inhibitor.

    Args:
        auc_ratio: AUC ratio with / without the strong inhibitor.
        thresholds: the thresholds.

    Returns:
        `SENSITIVE` at or above `thresholds.sensitive`, `MODERATELY_SENSITIVE`
        at or above `thresholds.moderately_sensitive`, else `NONE`.

    Raises:
        ValueError: if the ratio is not positive.
    """
    value = _ratio_value(auc_ratio)
    if not value > 0:
        raise ValueError(f"The AUC ratio must be positive, got {value}")
    if value >= thresholds.sensitive:
        return Sensitivity.SENSITIVE
    if value >= thresholds.moderately_sensitive:
        return Sensitivity.MODERATELY_SENSITIVE
    return Sensitivity.NONE
```

Ruff `B008` flags `DDIThresholds.fda()` as a default argument: use `thresholds: DDIThresholds | None = None` and `thresholds = thresholds or DDIThresholds.fda()` in the body if it does (a frozen dataclass default is harmless but the rule does not know); the test `ddi_classification(3.0, thresholds=ema).thresholds is ema` stays valid.

Export from `stats/__init__.py`: `RatioResult`, `ratio`, `DDIKind`, `DDIResult`, `DDIStrength`, `DDIThresholds`, `Sensitivity`, `ddi_classification`, `substrate_sensitivity`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/stats -q -W error`
Expected: PASS. Run `uv run ruff check src tests && uv run ruff format src tests && uvx ty check` → clean.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/stats tests/stats
git commit -q -m "Add the geometric mean ratio and the classification of drug-drug interactions

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 5: Bioequivalence: TOST, paired, parallel and 2x2 crossover (`stats/bioequivalence.py`)

**Files:**
- Create: `src/pkpdutils/stats/bioequivalence.py`
- Modify: `src/pkpdutils/stats/__init__.py`
- Test: `tests/stats/test_bioequivalence.py`

**Interfaces:**
- `Design` (`PARALLEL`, `PAIRED`, `CROSSOVER`).
- `BEParameter` (frozen dataclass): `name`, `unit`, `gmr`, `ci_low`, `ci_high`, `ci_level`, `limits`, `bioequivalent`, `p_lower`, `p_upper`, `p_value` (the larger of the two), `log_ratio`, `se_log`, `df`, `design`, `cv_intra` (within-subject CV, `NaN` for `PARALLEL`), `p_period`, `p_sequence` (`NaN` unless `CROSSOVER`), `n_test`, `n_reference`; `to_dict()`.
- `BEResult` (frozen dataclass): `parameters: dict[str, BEParameter]`, `bioequivalent: bool` (all parameters), `limits`, `ci_level`; `__getitem__(name)`, `to_dataframe()` (one row per parameter).
- `tost(test: ParameterSample, reference: ParameterSample, *, limits=(0.8, 1.25), ci_level=0.90, design: Design | None = None) -> BEParameter`. `design=None`: `CROSSOVER` when both samples carry the coordinates `period` and `sequence` and matching labels, `PAIRED` when the labels match, else `PARALLEL`.
- `bioequivalence(test: ParameterResult, reference: ParameterResult, parameters: Sequence[str] = ("auc_inf_obs", "cmax"), *, dim="individual", limits=(0.8, 1.25), ci_level=0.90, design=None, **indexers) -> BEResult`: `tost` on `result.sample(name, dim, **indexers)` of every parameter.

- [ ] **Step 1: Write the failing tests**

`tests/stats/test_bioequivalence.py`:
```python
import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils import Route, Timecourses, nca
from pkpdutils.stats import ParameterSample
from pkpdutils.stats.bioequivalence import BEParameter, BEResult, Design, bioequivalence, tost
from pkpdutils.stats.ratio import ratio

N = 12
RNG = np.random.default_rng(21)
SUBJECTS = np.array([f"s{i:02d}" for i in range(N)])
# sequence RT: reference in period 1, test in period 2; TR the other way round
SEQUENCE = np.array(["RT"] * 6 + ["TR"] * 6)
PERIOD_TEST = np.where(SEQUENCE == "RT", 2, 1)
PERIOD_REF = 3 - PERIOD_TEST
SUBJECT_EFFECT = RNG.normal(0, 0.3, N)
PERIOD_EFFECT = np.array([0.0, 0.2])  # period 2 raises the log exposure by 0.2
TRUE_LOG_RATIO = np.log(0.95)


def log_value(treatment_effect: float, period: np.ndarray) -> np.ndarray:
    return np.log(100.0) + SUBJECT_EFFECT + treatment_effect + PERIOD_EFFECT[period - 1] + RNG.normal(0, 0.08, N)


TEST_VALUES = np.exp(log_value(TRUE_LOG_RATIO, PERIOD_TEST))
REF_VALUES = np.exp(log_value(0.0, PERIOD_REF))


def crossover_samples() -> tuple[ParameterSample, ParameterSample]:
    test = ParameterSample(
        values=TEST_VALUES,
        labels=SUBJECTS,
        coords={"period": PERIOD_TEST, "sequence": SEQUENCE},
        name="auc_inf_obs",
        unit="mg*hr/l",
    )
    perm = RNG.permutation(N)
    reference = ParameterSample(
        values=REF_VALUES[perm],
        labels=SUBJECTS[perm],
        coords={"period": PERIOD_REF[perm], "sequence": SEQUENCE[perm]},
        name="auc_inf_obs",
        unit="mg*hr/l",
    )
    return test, reference


def ols_crossover() -> tuple[float, float, float]:
    """Treatment effect, its standard error and the residual variance of the ANOVA with subject, period and treatment effects."""
    y = np.concatenate([np.log(TEST_VALUES), np.log(REF_VALUES)])
    treatment = np.concatenate([np.ones(N), np.zeros(N)])
    period2 = np.concatenate([PERIOD_TEST == 2, PERIOD_REF == 2]).astype(float)
    subject = np.concatenate([np.eye(N), np.eye(N)])[:, 1:]  # drop one subject dummy
    x = np.column_stack([np.ones(2 * N), treatment, period2, subject])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    mse = residuals @ residuals / (2 * N - x.shape[1])
    cov = mse * np.linalg.inv(x.T @ x)
    return float(beta[1]), float(np.sqrt(cov[1, 1])), float(mse)


def test_crossover_equals_the_anova() -> None:
    test, reference = crossover_samples()
    res = tost(test, reference)
    assert isinstance(res, BEParameter) and res.design is Design.CROSSOVER
    effect, se, mse = ols_crossover()
    assert res.log_ratio == pytest.approx(effect)
    assert res.se_log == pytest.approx(se)
    assert res.df == N - 2
    assert res.cv_intra == pytest.approx(np.sqrt(np.expm1(mse)))
    tq = student_t.ppf(0.95, N - 2)
    assert (res.ci_low, res.ci_high) == pytest.approx((np.exp(effect - tq * se), np.exp(effect + tq * se)))
    assert res.gmr == pytest.approx(np.exp(effect))
    # the period effect of 0.2 is detected (p = 7e-4 for this seed), the sequence (carryover) effect is absent (p = 0.68)
    assert res.p_period < 0.01 and res.p_sequence > 0.05
    assert res.n_test == N and res.n_reference == N and res.name == "auc_inf_obs"


def test_tost_p_values_and_verdict() -> None:
    test, reference = crossover_samples()
    res = tost(test, reference)
    t_low = (res.log_ratio - np.log(0.8)) / res.se_log
    t_high = (np.log(1.25) - res.log_ratio) / res.se_log
    assert res.p_lower == pytest.approx(student_t.sf(t_low, N - 2))
    assert res.p_upper == pytest.approx(student_t.sf(t_high, N - 2))
    assert res.p_value == max(res.p_lower, res.p_upper)
    assert res.bioequivalent == (0.8 <= res.ci_low and res.ci_high <= 1.25)
    assert res.bioequivalent == (res.p_value < 0.05)
    wide = tost(test, reference, limits=(0.5, 2.0))
    assert wide.bioequivalent and wide.limits == (0.5, 2.0)
    narrow = tost(test, reference, limits=(0.99, 1.01))
    assert not narrow.bioequivalent


def test_paired_and_parallel_designs() -> None:
    test = ParameterSample(values=TEST_VALUES, labels=SUBJECTS)
    reference = ParameterSample(values=REF_VALUES, labels=SUBJECTS)
    paired = tost(test, reference)
    assert paired.design is Design.PAIRED
    r = ratio(test, reference)
    assert (paired.gmr, paired.ci_low, paired.ci_high) == (r.gmr, r.ci_low, r.ci_high)
    d = np.log(TEST_VALUES) - np.log(REF_VALUES)
    assert paired.cv_intra == pytest.approx(np.sqrt(np.expm1(d.var(ddof=1) / 2)))
    assert np.isnan(paired.p_period)
    parallel = tost(ParameterSample(values=TEST_VALUES), ParameterSample(values=REF_VALUES[:10]))
    assert parallel.design is Design.PARALLEL and np.isnan(parallel.cv_intra)
    r2 = ratio(ParameterSample(values=TEST_VALUES), ParameterSample(values=REF_VALUES[:10]))
    assert parallel.se_log == pytest.approx(r2.se_log)
    forced = tost(test, reference, design=Design.PARALLEL)
    assert forced.design is Design.PARALLEL
    summary = tost(ParameterSample(mean=95.0, sd=20.0, n=12), ParameterSample(mean=100.0, sd=22.0, n=12))
    assert summary.design is Design.PARALLEL and np.isfinite(summary.p_value)


def test_crossover_validation() -> None:
    test, reference = crossover_samples()
    bad_period = ParameterSample(
        values=REF_VALUES, labels=SUBJECTS, coords={"period": PERIOD_TEST, "sequence": SEQUENCE}
    )
    with pytest.raises(ValueError, match="period"):
        tost(test, bad_period)
    one_sequence = ParameterSample(
        values=REF_VALUES, labels=SUBJECTS, coords={"period": PERIOD_REF, "sequence": np.array(["RT"] * N)}
    )
    with pytest.raises(ValueError, match="sequence"):
        tost(test, one_sequence)
    with pytest.raises(ValueError, match="crossover"):
        tost(ParameterSample(values=TEST_VALUES), ParameterSample(values=REF_VALUES), design=Design.CROSSOVER)
    with pytest.raises(ValueError, match="limits"):
        tost(test, reference, limits=(1.25, 0.8))


def batch(values: np.ndarray, period: np.ndarray) -> Timecourses:
    time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
    curves = np.stack([v / 5 * np.exp(-0.2 * time) for v in values])
    return Timecourses.from_arrays(
        time,
        curves,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": SUBJECTS, "period": ("individual", period), "sequence": ("individual", SEQUENCE)},
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
    )


def test_bioequivalence_of_nca_results() -> None:
    test = nca(batch(TEST_VALUES, PERIOD_TEST))
    reference = nca(batch(REF_VALUES, PERIOD_REF))
    res = bioequivalence(test, reference)
    assert isinstance(res, BEResult)
    assert set(res.parameters) == {"auc_inf_obs", "cmax"}
    assert res["auc_inf_obs"].design is Design.CROSSOVER
    assert res.bioequivalent == all(p.bioequivalent for p in res.parameters.values())
    direct = tost(test.sample("cmax", "individual"), reference.sample("cmax", "individual"))
    assert res["cmax"].gmr == direct.gmr
    df = res.to_dataframe()
    assert list(df["parameter"]) == ["auc_inf_obs", "cmax"]
    assert {"gmr", "ci_low", "ci_high", "bioequivalent", "p_value", "design", "cv_intra"} <= set(df.columns)
    with pytest.raises(ValueError, match="not a variable"):
        bioequivalence(test, reference, parameters=["auc_inf"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/stats/test_bioequivalence.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `stats/bioequivalence.py`**

```python
"""Average bioequivalence: the two one-sided tests on the geometric mean ratio.

Test and reference are bioequivalent when the 90 % confidence interval of
the geometric mean ratio of the exposure lies within 80-125 % (FDA 2001),
which is the two one-sided tests procedure of Schuirmann (1987) at
\\(\\alpha = 0.05\\). The interval comes from the design of the study: a 2x2
crossover (each subject receives both formulations in two periods, in one
of two sequences) is analysed with the period differences of Chow & Liu
(2009, ch. 3), which is the analysis of variance with sequence, period and
subject-within-sequence effects on the log scale; a paired design uses the
within-subject differences; parallel groups use the Welch interval.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from pkpdutils.result import ParameterResult
from pkpdutils.stats.ratio import RatioResult, _pair, ratio
from pkpdutils.stats.sample import ParameterSample


class Design(StrEnum):
    """Design of a bioequivalence study."""

    #: two groups of different subjects
    PARALLEL = "parallel"
    #: every subject receives both formulations, without period information
    PAIRED = "paired"
    #: 2x2 crossover with the coordinates `period` and `sequence`
    CROSSOVER = "crossover"


@dataclass(frozen=True)
class BEParameter:
    """Bioequivalence of one parameter.

    Attributes:
        name: name of the parameter
        unit: unit of the parameter
        gmr: geometric mean ratio test / reference
        ci_low: lower bound of the interval of the ratio
        ci_high: upper bound of the interval
        ci_level: level of the interval
        limits: acceptance limits of the ratio
        bioequivalent: whether the interval lies within the limits
        p_lower: p value of the test against the lower limit
        p_upper: p value of the test against the upper limit
        p_value: the larger of the two, the p value of the TOST procedure
        log_ratio: \\(\\ln \\mathrm{GMR}\\)
        se_log: standard error of `log_ratio`
        df: degrees of freedom
        design: the design of the analysis
        cv_intra: within-subject coefficient of variation \\(\\sqrt{e^{\\sigma_e^2} - 1}\\), `NaN` for a parallel design
        p_period: p value of the period effect (crossover), `NaN` otherwise
        p_sequence: p value of the sequence (carryover) effect (crossover), `NaN` otherwise
        n_test: number of test values
        n_reference: number of reference values
    """

    name: str
    unit: str
    gmr: float
    ci_low: float
    ci_high: float
    ci_level: float
    limits: tuple[float, float]
    bioequivalent: bool
    p_lower: float
    p_upper: float
    p_value: float
    log_ratio: float
    se_log: float
    df: float
    design: Design
    cv_intra: float
    p_period: float
    p_sequence: float
    n_test: int
    n_reference: int

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the enumerations as strings.

        Returns:
            Field name to value.
        """
        return {
            "parameter": self.name,
            "unit": self.unit,
            "gmr": self.gmr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "limit_low": self.limits[0],
            "limit_high": self.limits[1],
            "bioequivalent": self.bioequivalent,
            "p_lower": self.p_lower,
            "p_upper": self.p_upper,
            "p_value": self.p_value,
            "log_ratio": self.log_ratio,
            "se_log": self.se_log,
            "df": self.df,
            "design": str(self.design),
            "cv_intra": self.cv_intra,
            "p_period": self.p_period,
            "p_sequence": self.p_sequence,
            "n_test": self.n_test,
            "n_reference": self.n_reference,
        }


@dataclass(frozen=True)
class BEResult:
    """Bioequivalence of several parameters.

    Attributes:
        parameters: parameter name to its result
        bioequivalent: whether every parameter is bioequivalent
        limits: the acceptance limits
        ci_level: the level of the intervals
    """

    parameters: dict[str, BEParameter]
    bioequivalent: bool
    limits: tuple[float, float]
    ci_level: float

    def __getitem__(self, name: str) -> BEParameter:
        """The result of one parameter.

        Args:
            name: name of the parameter.

        Returns:
            The result.
        """
        return self.parameters[name]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per parameter.

        Returns:
            The dataframe.
        """
        return pd.DataFrame([p.to_dict() for p in self.parameters.values()])


def _detect_design(test: ParameterSample, reference: ParameterSample) -> Design:
    """The design the samples allow.

    Args:
        test: the test sample.
        reference: the reference sample.

    Returns:
        `CROSSOVER` with `period` and `sequence` coordinates and matching
        labels, `PAIRED` with matching labels, else `PARALLEL`.
    """
    from pkpdutils.stats.ratio import _labels_match

    if not _labels_match(test, reference):
        return Design.PARALLEL
    keys = {"period", "sequence"}
    if keys <= set(test.coords) and keys <= set(reference.coords):
        return Design.CROSSOVER
    return Design.PAIRED


def _crossover(
    test: ParameterSample, reference: ParameterSample
) -> tuple[float, float, float, float, float, float]:
    """The period-difference analysis of a 2x2 crossover on the log scale.

    For subject \\(i\\) with the log values \\(y_{i1}, y_{i2}\\) of the two periods,
    \\(d_i = (y_{i2} - y_{i1}) / 2\\) and the total \\(u_i = y_{i1} + y_{i2}\\). With
    the sequences A (test in period 2) and B (test in period 1):
    treatment effect \\(\\hat F = \\bar d_A - \\bar d_B\\), period effect
    \\(\\hat P = \\bar d_A + \\bar d_B\\), both with the variance
    \\(\\sigma_d^2 (1/n_A + 1/n_B)\\) of the pooled within-sequence variance
    \\(\\sigma_d^2\\) with \\(n_A + n_B - 2\\) degrees of freedom; the sequence
    (carryover) effect \\(\\hat C = \\bar u_A - \\bar u_B\\) with the pooled
    variance of the totals. The residual variance of the ANOVA is
    \\(\\sigma_e^2 = 2 \\sigma_d^2\\) (Chow & Liu 2009, ch. 3).

    Args:
        test: the test sample with `period` and `sequence` coordinates.
        reference: the reference sample with `period` and `sequence` coordinates.

    Returns:
        `log_ratio`, `se_log`, `df`, `sigma_e2`, `p_period`, `p_sequence`.

    Raises:
        ValueError: if a subject has no two different periods, the periods
            are not 1 and 2, a sequence mixes the order, or there are not
            exactly two sequences with at least two subjects each.
    """
    x, y = _pair(test, reference)
    labels = test.finite_labels
    assert labels is not None
    finite_t = np.isfinite(test.values) if test.values is not None else np.zeros(0, bool)
    finite_r = np.isfinite(reference.values) if reference.values is not None else np.zeros(0, bool)
    period_t = np.asarray(test.coords["period"])[finite_t]
    sequence = np.asarray(test.coords["sequence"])[finite_t]
    ref_labels = reference.finite_labels
    assert ref_labels is not None
    order = {label: i for i, label in enumerate(ref_labels.tolist())}
    idx = [order[label] for label in labels.tolist()]
    period_r = np.asarray(reference.coords["period"])[finite_r][idx]
    if set(np.unique(period_t).tolist()) | set(np.unique(period_r).tolist()) != {1, 2} or np.any(period_t == period_r):
        raise ValueError("A 2x2 crossover needs the periods 1 and 2, test and reference in different periods of every subject")
    sequences = np.unique(sequence)
    if sequences.size != 2:
        raise ValueError(f"A 2x2 crossover needs exactly two sequences, got {sequences.tolist()}")
    y1 = np.where(period_t == 1, x, y)
    y2 = np.where(period_t == 2, x, y)
    d = (y2 - y1) / 2.0
    u = y1 + y2
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    test_second: list[bool] = []
    for seq in sequences:
        mask = sequence == seq
        in_second = np.unique(period_t[mask])
        if in_second.size != 1:
            raise ValueError(f"sequence '{seq}' mixes the order of test and reference")
        if mask.sum() < 2:
            raise ValueError(f"sequence '{seq}' needs at least two subjects")
        groups.append((d[mask], u[mask]))
        test_second.append(bool(in_second[0] == 2))
    (d_a, u_a), (d_b, u_b) = (groups if test_second[0] else groups[::-1])
    n_a, n_b = d_a.size, d_b.size
    df = float(n_a + n_b - 2)
    factor = 1.0 / n_a + 1.0 / n_b
    sigma_d2 = ((n_a - 1) * d_a.var(ddof=1) + (n_b - 1) * d_b.var(ddof=1)) / df
    sigma_u2 = ((n_a - 1) * u_a.var(ddof=1) + (n_b - 1) * u_b.var(ddof=1)) / df
    se = float(np.sqrt(sigma_d2 * factor))
    effect = float(d_a.mean() - d_b.mean())
    period = float(d_a.mean() + d_b.mean())
    carryover = float(u_a.mean() - u_b.mean())
    se_u = float(np.sqrt(sigma_u2 * factor))
    p_period = float(2.0 * student_t.sf(abs(period / se), df))
    p_sequence = float(2.0 * student_t.sf(abs(carryover / se_u), df))
    return effect, se, df, float(2.0 * sigma_d2), p_period, p_sequence


def tost(
    test: ParameterSample,
    reference: ParameterSample,
    *,
    limits: tuple[float, float] = (0.8, 1.25),
    ci_level: float = 0.90,
    design: Design | None = None,
) -> BEParameter:
    """Two one-sided tests of the geometric mean ratio against the acceptance limits.

    \\(t_L = (\\ln \\mathrm{GMR} - \\ln \\theta_L) / \\mathrm{se}\\),
    \\(t_U = (\\ln \\theta_U - \\ln \\mathrm{GMR}) / \\mathrm{se}\\), each tested one-sided
    with the degrees of freedom of the design at \\(\\alpha = (1 - \\mathrm{ci\\_level}) / 2\\);
    rejecting both is the same as the interval at `ci_level` lying within
    the limits (Schuirmann 1987).

    Args:
        test: the test sample.
        reference: the reference sample.
        limits: acceptance limits of the ratio.
        ci_level: level of the interval, 0.90 for the usual \\(\\alpha = 0.05\\).
        design: the design, detected from the samples by default.

    Returns:
        The result of the parameter.

    Raises:
        ValueError: for reversed limits or a design the samples do not support.
    """
    if not 0 < limits[0] < limits[1]:
        raise ValueError(f"'limits' must be (low, high) with 0 < low < high, got {limits}")
    resolved = design if design is not None else _detect_design(test, reference)
    nan = float("nan")
    p_period = p_sequence = cv_intra = nan
    if resolved is Design.CROSSOVER:
        if not ({"period", "sequence"} <= set(test.coords) and {"period", "sequence"} <= set(reference.coords)):
            raise ValueError("A crossover analysis needs the coordinates 'period' and 'sequence' on both samples")
        log_ratio, se, df, sigma_e2, p_period, p_sequence = _crossover(test, reference)
        cv_intra = float(np.sqrt(np.expm1(sigma_e2)))
        n_test = n_reference = test.size
    else:
        r: RatioResult = ratio(test, reference, ci_level=ci_level, paired=resolved is Design.PAIRED)
        log_ratio, se, df = r.log_ratio, r.se_log, r.df
        n_test, n_reference = r.n_test, r.n_reference
        if resolved is Design.PAIRED:
            # the variance of a within-subject difference is twice the residual variance
            cv_intra = float(np.sqrt(np.expm1(se**2 * n_test / 2.0)))
    alpha = (1.0 - ci_level) / 2.0
    tq = float(student_t.ppf(1.0 - alpha, df))
    ci = (float(np.exp(log_ratio - tq * se)), float(np.exp(log_ratio + tq * se)))
    p_lower = float(student_t.sf((log_ratio - np.log(limits[0])) / se, df))
    p_upper = float(student_t.sf((np.log(limits[1]) - log_ratio) / se, df))
    return BEParameter(
        name=test.name,
        unit=test.unit,
        gmr=float(np.exp(log_ratio)),
        ci_low=ci[0],
        ci_high=ci[1],
        ci_level=ci_level,
        limits=(float(limits[0]), float(limits[1])),
        bioequivalent=bool(limits[0] <= ci[0] and ci[1] <= limits[1]),
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=max(p_lower, p_upper),
        log_ratio=log_ratio,
        se_log=se,
        df=df,
        design=resolved,
        cv_intra=cv_intra,
        p_period=p_period,
        p_sequence=p_sequence,
        n_test=int(n_test),
        n_reference=int(n_reference),
    )


def bioequivalence(
    test: ParameterResult,
    reference: ParameterResult,
    parameters: Sequence[str] = ("auc_inf_obs", "cmax"),
    *,
    dim: str = "individual",
    limits: tuple[float, float] = (0.8, 1.25),
    ci_level: float = 0.90,
    design: Design | None = None,
    **indexers: Any,
) -> BEResult:
    """Average bioequivalence of the parameters of two results.

    Every parameter is taken with `ParameterResult.sample(name, dim, **indexers)`
    from both results and tested with `tost`; the study is bioequivalent
    when every parameter is.

    Args:
        test: the result of the test formulation.
        reference: the result of the reference formulation.
        parameters: the parameters to test.
        dim: the sample dimension of the individuals.
        limits: acceptance limits of the ratio.
        ci_level: level of the intervals.
        design: the design, detected from the samples by default.
        **indexers: coordinate label per remaining sample dimension.

    Returns:
        The result.
    """
    results = {
        name: tost(
            test.sample(name, dim, **indexers),
            reference.sample(name, dim, **indexers),
            limits=limits,
            ci_level=ci_level,
            design=design,
        )
        for name in parameters
    }
    return BEResult(
        parameters=results,
        bioequivalent=all(p.bioequivalent for p in results.values()),
        limits=(float(limits[0]), float(limits[1])),
        ci_level=ci_level,
    )
```

Move the import of `_labels_match` to the module top (it is in `ratio.py`, no cycle). If ty complains about `test.values` being `None` in `_crossover`, guard with `assert test.values is not None and reference.values is not None` right after `_pair` (which already raised for summary data).

Export from `stats/__init__.py`: `BEParameter`, `BEResult`, `Design`, `bioequivalence`, `tost`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/stats -q -W error`
Expected: PASS. The crossover test compares against the OLS with subject dummies, which is exact: `log_ratio` and `se_log` agree to `1e-6`. Run `uv run ruff check src tests && uv run ruff format src tests && uvx ty check` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/stats tests/stats
git commit -q -m "Add bioequivalence with the two one-sided tests and the 2x2 crossover analysis

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 6: Meta-analysis (`stats/meta.py`) with the BCG reference fixture

**Files:**
- Create: `src/pkpdutils/stats/meta.py`, `tests/data/reference/meta_bcg.json`
- Modify: `src/pkpdutils/stats/__init__.py`
- Test: `tests/stats/test_meta.py`

**Interfaces:**
- `EffectKind` (`HEDGES_G`, `MEAN_DIFF`, `LOG_RATIO`).
- `EffectSize` (frozen dataclass): `estimate`, `variance`, `se`, `ci_low`, `ci_high`, `ci_level`, `kind`, `n_control`, `n_treatment`, `label`; `to_dict()`.
- `effect_size(control: ParameterSample, treatment: ParameterSample, kind=EffectKind.HEDGES_G, *, ci_level=0.95, label="") -> EffectSize` (normal interval \(\hat\theta \pm z\,\mathrm{se}\)).
- `Heterogeneity` (frozen dataclass): `q`, `df`, `p_value`, `i2`, `h2`, `tau2`.
- `heterogeneity(effects: Sequence[EffectSize]) -> Heterogeneity`.
- `PooledEffect` (frozen dataclass): `estimate`, `se`, `ci_low`, `ci_high`, `ci_level`, `z`, `p_value`, `weights: np.ndarray` (normalized to 1), `model: str` (`"fixed"` / `"random"`), `tau2`; `to_dict()` (without `weights`).
- `fixed_effect(effects, *, ci_level=0.95) -> PooledEffect`, `random_effects(effects, *, ci_level=0.95) -> PooledEffect` (DerSimonian-Laird).
- `Study` (frozen dataclass): `label: str`, `control: ParameterSample`, `treatment: ParameterSample`, `category: str | None = None`.
- `MetaResult` (frozen dataclass): `kind`, `effects: tuple[EffectSize, ...]`, `fixed`, `random`, `heterogeneity`, `ci_level`; properties `labels`, `n_studies`; `to_dataframe()` (one row per study with `label`, `estimate`, `se`, `ci_low`, `ci_high`, `n_control`, `n_treatment`, `weight_fixed`, `weight_random`).
- `meta_analysis(studies: Sequence[Study], kind=EffectKind.HEDGES_G, *, ci_level=0.95) -> MetaResult`; `meta_analysis_by(studies, kind=..., *, ci_level=0.95) -> dict[str, MetaResult]` (grouped by `Study.category`, `None` grouped under `""`).
- `effects_from_arrays(estimates, variances, labels=None, kind=..., ci_level=0.95) -> list[EffectSize]`: effect sizes computed elsewhere (a log risk ratio of a 2x2 table), for the pooling functions.

- [ ] **Step 1: Write the reference fixture**

`tests/data/reference/meta_bcg.json`: the 13 BCG vaccine trials of Colditz et al. 1994 (the `dat.bcg` example of the R package `metafor`), with the log risk ratios and their variances and the results of `metafor::rma(yi, vi, method="FE")` and `method="DL"` as printed in the metafor documentation (`Q = 152.2330`, `tau^2 = 0.3088`, `I^2 = 92.12 %`, fixed effect `-0.4303` with `se = 0.0405`, random effects `-0.7141` with `se = 0.1787`); the fixture stores the same numbers with six digits, recomputed from the counts on 2026-09-15 with an independent numpy script. The effect of a study is `log((tpos / (tpos + tneg)) / (cpos / (cpos + cneg)))` with the variance `1/tpos - 1/(tpos + tneg) + 1/cpos - 1/(cpos + cneg)`:
```json
{
  "source": "Colditz et al. 1994, JAMA 271:698-702; metafor dat.bcg; results of rma(yi, vi, method='FE') and method='DL' from the metafor documentation, recomputed from the counts",
  "columns": ["trial", "tpos", "tneg", "cpos", "cneg"],
  "trials": [
    ["Aronson 1948", 4, 119, 11, 128],
    ["Ferguson & Simes 1949", 6, 300, 29, 274],
    ["Rosenthal 1960", 3, 228, 11, 209],
    ["Hart & Sutherland 1977", 62, 13536, 248, 12619],
    ["Frimodt-Moller 1973", 33, 5036, 47, 5761],
    ["Stein & Aronson 1953", 180, 1361, 372, 1079],
    ["Vandiviere 1973", 8, 2537, 10, 619],
    ["TPT Madras 1980", 505, 87886, 499, 87892],
    ["Coetzee & Berjak 1968", 29, 7470, 45, 7232],
    ["Rosenthal 1961", 17, 1699, 65, 1600],
    ["Comstock 1974", 186, 50448, 141, 27197],
    ["Comstock & Webster 1969", 5, 2493, 3, 2338],
    ["Comstock 1976", 27, 16886, 29, 17825]
  ],
  "fixed": {"estimate": -0.430285, "se": 0.040499},
  "random_dl": {"estimate": -0.714117, "se": 0.178742, "tau2": 0.308760},
  "heterogeneity": {"q": 152.233008, "df": 12, "i2": 92.1173, "h2": 12.6861}
}
```

- [ ] **Step 2: Write the failing tests**

`tests/stats/test_meta.py`:
```python
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, norm

from pkpdutils.stats import ParameterSample
from pkpdutils.stats.meta import (
    EffectKind,
    EffectSize,
    Heterogeneity,
    MetaResult,
    PooledEffect,
    Study,
    effect_size,
    effects_from_arrays,
    fixed_effect,
    heterogeneity,
    meta_analysis,
    meta_analysis_by,
    random_effects,
)
from pkpdutils.stats.tests import hedges_correction

BCG = json.loads((Path(__file__).parent.parent / "data" / "reference" / "meta_bcg.json").read_text())


def bcg_effects() -> list[EffectSize]:
    trials = np.array([row[1:] for row in BCG["trials"]], dtype=float)
    a, b, c, d = trials.T
    yi = np.log((a / (a + b)) / (c / (c + d)))
    vi = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)
    return effects_from_arrays(yi, vi, labels=[row[0] for row in BCG["trials"]], kind=EffectKind.LOG_RATIO)


def test_fixed_effect_against_metafor() -> None:
    effects = bcg_effects()
    fixed = fixed_effect(effects)
    assert isinstance(fixed, PooledEffect) and fixed.model == "fixed" and fixed.tau2 == 0.0
    assert fixed.estimate == pytest.approx(BCG["fixed"]["estimate"], abs=1e-6)
    assert fixed.se == pytest.approx(BCG["fixed"]["se"], abs=1e-6)
    assert fixed.weights.sum() == pytest.approx(1.0)
    assert fixed.z == pytest.approx(fixed.estimate / fixed.se)
    assert fixed.p_value == pytest.approx(2 * norm.sf(abs(fixed.z)))
    assert (fixed.ci_low, fixed.ci_high) == pytest.approx((fixed.estimate - 1.959964 * fixed.se, fixed.estimate + 1.959964 * fixed.se), abs=1e-6)


def test_random_effects_and_heterogeneity_against_metafor() -> None:
    effects = bcg_effects()
    het = heterogeneity(effects)
    assert isinstance(het, Heterogeneity)
    assert het.q == pytest.approx(BCG["heterogeneity"]["q"], abs=1e-5)
    assert het.df == 12
    assert het.p_value == pytest.approx(chi2.sf(het.q, 12))
    assert het.i2 == pytest.approx(BCG["heterogeneity"]["i2"], abs=1e-3)
    assert het.h2 == pytest.approx(BCG["heterogeneity"]["h2"], abs=1e-3)
    assert het.tau2 == pytest.approx(BCG["random_dl"]["tau2"], abs=1e-6)
    random = random_effects(effects)
    assert random.model == "random" and random.tau2 == het.tau2
    assert random.estimate == pytest.approx(BCG["random_dl"]["estimate"], abs=1e-6)
    assert random.se == pytest.approx(BCG["random_dl"]["se"], abs=1e-6)
    # random effects weights are more even than fixed effect weights
    assert random.weights.max() < fixed_effect(effects).weights.max()


def test_tau2_is_clipped_at_zero() -> None:
    effects = effects_from_arrays([0.1, 0.12, 0.09], [0.04, 0.05, 0.04])
    het = heterogeneity(effects)
    assert het.tau2 == 0.0 and het.i2 == 0.0
    random = random_effects(effects)
    fixed = fixed_effect(effects)
    assert random.estimate == pytest.approx(fixed.estimate) and random.se == pytest.approx(fixed.se)
    single = heterogeneity(effects[:1])
    assert single.q == 0.0 and single.df == 0 and np.isnan(single.p_value) and single.i2 == 0.0
    with pytest.raises(ValueError, match="at least one"):
        fixed_effect([])


def test_hedges_g_matches_the_esc_port() -> None:
    control = ParameterSample(mean=10.0, sd=1.5, n=50)
    treatment = ParameterSample(mean=12.0, sd=2.5, n=60)
    es = effect_size(control, treatment, EffectKind.HEDGES_G)
    # esc_mean_sd(grp1m=10, grp1sd=1.5, grp1n=50, grp2m=12, grp2sd=2.5, grp2n=60): d = 0.94967, var(d) = 0.040766
    d, var_d = 0.9496730565857971, 0.04076611627759853
    j = hedges_correction(110)
    assert es.estimate == pytest.approx(d * j)
    assert es.variance == pytest.approx(var_d * j**2)
    assert es.se == pytest.approx(np.sqrt(var_d) * j)
    assert (es.n_control, es.n_treatment) == (50, 60) and es.kind is EffectKind.HEDGES_G
    assert (es.ci_low, es.ci_high) == pytest.approx((es.estimate - 1.959964 * es.se, es.estimate + 1.959964 * es.se), abs=1e-6)
    individual = effect_size(ParameterSample(values=[9.0, 10.0, 11.0, 10.0]), ParameterSample(values=[11.0, 12.0, 13.0, 12.0]))
    assert individual.estimate == pytest.approx((12 - 10) / np.sqrt(2 / 3) * hedges_correction(8))


def test_mean_difference_and_log_ratio() -> None:
    control = ParameterSample(mean=100.0, sd=30.0, n=12, name="cl", unit="l/hr")
    treatment = ParameterSample(mean=130.0, sd=35.0, n=10)
    md = effect_size(control, treatment, EffectKind.MEAN_DIFF)
    assert md.estimate == 30.0 and md.variance == pytest.approx(30.0**2 / 12 + 35.0**2 / 10)
    lr = effect_size(control, treatment, EffectKind.LOG_RATIO)
    mu_c, s_c = control.log_moments()
    mu_t, s_t = treatment.log_moments()
    assert lr.estimate == pytest.approx(mu_t - mu_c)
    assert lr.variance == pytest.approx(s_t**2 / 10 + s_c**2 / 12)
    assert lr.to_dict()["kind"] == "log_ratio"


def test_meta_analysis_result() -> None:
    rng = np.random.default_rng(5)
    studies = [
        Study(
            label=f"study {i}",
            control=ParameterSample(values=np.exp(rng.normal(np.log(100), 0.3, 10))),
            treatment=ParameterSample(values=np.exp(rng.normal(np.log(150), 0.3, 12))),
            category="smokers" if i % 2 else "contraceptives",
        )
        for i in range(6)
    ]
    result = meta_analysis(studies, EffectKind.LOG_RATIO)
    assert isinstance(result, MetaResult) and result.n_studies == 6
    assert result.labels == tuple(f"study {i}" for i in range(6))
    assert result.fixed.estimate == pytest.approx(np.log(1.5), abs=0.15)
    assert result.random.ci_low < result.random.estimate < result.random.ci_high
    df = result.to_dataframe()
    assert list(df.columns) == ["label", "estimate", "se", "ci_low", "ci_high", "n_control", "n_treatment", "weight_fixed", "weight_random"]
    assert df["weight_fixed"].sum() == pytest.approx(1.0)
    by = meta_analysis_by(studies, EffectKind.LOG_RATIO)
    assert set(by) == {"smokers", "contraceptives"} and by["smokers"].n_studies == 3
    with pytest.raises(ValueError, match="at least one"):
        meta_analysis([])
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/stats/test_meta.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `stats/meta.py`**

```python
"""Meta-analysis of a parameter over studies: effect sizes, fixed effect and random effects pooling.

An effect size per study (Hedges' g, the mean difference or the log ratio
of the geometric means, the effect native to pharmacokinetics) is pooled
with inverse variance weights: the fixed effect model assumes one true
effect, the random effects model of DerSimonian & Laird (1986) adds the
between-study variance \\(\\tau^2\\) to every weight. The heterogeneity
statistics \\(Q\\), \\(I^2\\) and \\(H^2\\) follow Higgins & Thompson (2002).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from pkpdutils.stats.sample import ParameterSample, Scale
from pkpdutils.stats.tests import hedges_correction


class EffectKind(StrEnum):
    """Effect size of a study."""

    #: standardized mean difference with the small sample correction (Hedges 1981)
    HEDGES_G = "hedges_g"
    #: difference of the arithmetic means, treatment minus control
    MEAN_DIFF = "mean_diff"
    #: log of the ratio of the geometric means, treatment over control
    LOG_RATIO = "log_ratio"


@dataclass(frozen=True)
class EffectSize:
    """Effect size of one study.

    Attributes:
        estimate: the effect
        variance: its variance
        se: its standard error
        ci_low: lower bound of the normal interval
        ci_high: upper bound of the normal interval
        ci_level: level of the interval
        kind: the kind of effect
        n_control: size of the control group
        n_treatment: size of the treatment group
        label: label of the study
    """

    estimate: float
    variance: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float
    kind: EffectKind
    n_control: int
    n_treatment: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the kind as a string.

        Returns:
            Field name to value.
        """
        return {
            "label": self.label,
            "estimate": self.estimate,
            "variance": self.variance,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "kind": str(self.kind),
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
        }


@dataclass(frozen=True)
class Heterogeneity:
    """Heterogeneity of the effects of the studies.

    Attributes:
        q: Cochran's \\(Q = \\sum w_i (\\theta_i - \\hat\\theta_F)^2\\)
        df: \\(k - 1\\)
        p_value: p value of \\(Q\\) under \\(\\chi^2_{k-1}\\), `NaN` for one study
        i2: \\(I^2 = \\max(0, (Q - df) / Q)\\) in percent
        h2: \\(H^2 = Q / df\\), `NaN` for one study
        tau2: between-study variance \\(\\tau^2 = \\max(0, (Q - df) / C)\\), \\(C = \\sum w_i - \\sum w_i^2 / \\sum w_i\\)
    """

    q: float
    df: int
    p_value: float
    i2: float
    h2: float
    tau2: float


@dataclass(frozen=True)
class PooledEffect:
    """Pooled effect of a meta-analysis.

    Attributes:
        estimate: the pooled effect \\(\\sum w_i \\theta_i / \\sum w_i\\)
        se: its standard error \\(1 / \\sqrt{\\sum w_i}\\)
        ci_low: lower bound of the normal interval
        ci_high: upper bound
        ci_level: level of the interval
        z: \\(\\hat\\theta / \\mathrm{se}\\)
        p_value: two-sided p value of `z`
        weights: the weights of the studies, normalized to 1
        model: `"fixed"` or `"random"`
        tau2: between-study variance used in the weights (0 for the fixed effect)
    """

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float
    z: float
    p_value: float
    weights: np.ndarray
    model: str
    tau2: float

    def to_dict(self) -> dict[str, Any]:
        """The scalar fields as a dictionary.

        Returns:
            Field name to value.
        """
        return {
            "model": self.model,
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "z": self.z,
            "p_value": self.p_value,
            "tau2": self.tau2,
        }


@dataclass(frozen=True)
class Study:
    """A study with a control and a treatment sample of one parameter.

    Attributes:
        label: label of the study
        control: the control sample
        treatment: the treatment sample
        category: category of the study for `meta_analysis_by`
    """

    label: str
    control: ParameterSample
    treatment: ParameterSample
    category: str | None = None


@dataclass(frozen=True)
class MetaResult:
    """Result of a meta-analysis.

    Attributes:
        kind: the kind of effect
        effects: the effect per study
        fixed: the fixed effect pooling
        random: the random effects pooling
        heterogeneity: the heterogeneity statistics
        ci_level: level of the intervals
    """

    kind: EffectKind
    effects: tuple[EffectSize, ...]
    fixed: PooledEffect
    random: PooledEffect
    heterogeneity: Heterogeneity
    ci_level: float

    @property
    def labels(self) -> tuple[str, ...]:
        """The labels of the studies."""
        return tuple(e.label for e in self.effects)

    @property
    def n_studies(self) -> int:
        """Number of studies."""
        return len(self.effects)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per study with its effect, interval, sizes and weights.

        Returns:
            The dataframe.
        """
        rows = []
        for i, e in enumerate(self.effects):
            rows.append(
                {
                    "label": e.label,
                    "estimate": e.estimate,
                    "se": e.se,
                    "ci_low": e.ci_low,
                    "ci_high": e.ci_high,
                    "n_control": e.n_control,
                    "n_treatment": e.n_treatment,
                    "weight_fixed": float(self.fixed.weights[i]),
                    "weight_random": float(self.random.weights[i]),
                }
            )
        return pd.DataFrame(rows)


def _z(ci_level: float) -> float:
    """The normal quantile of a two-sided interval.

    Args:
        ci_level: level of the interval.

    Returns:
        \\(z_{1 - \\alpha/2}\\).
    """
    return float(norm.ppf(1.0 - (1.0 - ci_level) / 2.0))


def _effect(estimate: float, variance: float, kind: EffectKind, n_control: int, n_treatment: int, label: str, ci_level: float) -> EffectSize:
    """Build an effect size with its normal interval.

    Args:
        estimate: the effect.
        variance: its variance.
        kind: the kind of effect.
        n_control: size of the control group.
        n_treatment: size of the treatment group.
        label: label of the study.
        ci_level: level of the interval.

    Returns:
        The effect size.
    """
    se = float(np.sqrt(variance))
    z = _z(ci_level)
    return EffectSize(
        estimate=float(estimate),
        variance=float(variance),
        se=se,
        ci_low=float(estimate - z * se),
        ci_high=float(estimate + z * se),
        ci_level=ci_level,
        kind=kind,
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        label=label,
    )


def effect_size(
    control: ParameterSample,
    treatment: ParameterSample,
    kind: EffectKind = EffectKind.HEDGES_G,
    *,
    ci_level: float = 0.95,
    label: str = "",
) -> EffectSize:
    """Effect size of a treatment against a control.

    Hedges' g: \\(d = (\\bar x_T - \\bar x_C) / s_p\\) with the pooled standard deviation,
    \\(\\mathrm{var}(d) = N / (n_C n_T) + d^2 / (2N)\\), \\(g = J d\\),
    \\(\\mathrm{var}(g) = J^2 \\mathrm{var}(d)\\) (Hedges 1981). Mean difference:
    \\(\\bar x_T - \\bar x_C\\) with \\(s_T^2 / n_T + s_C^2 / n_C\\). Log ratio:
    \\(\\mu_T - \\mu_C\\) of the log moments with \\(\\sigma_T^2 / n_T + \\sigma_C^2 / n_C\\).

    Args:
        control: the control sample.
        treatment: the treatment sample.
        kind: the kind of effect.
        ci_level: level of the interval.
        label: label of the study.

    Returns:
        The effect size.
    """
    scale = Scale.LOG if kind is EffectKind.LOG_RATIO else Scale.LINEAR
    m_c, s_c, n_c = control.moments(scale)
    m_t, s_t, n_t = treatment.moments(scale)
    if kind is EffectKind.HEDGES_G:
        total = n_c + n_t
        pooled = np.sqrt(((n_c - 1) * s_c**2 + (n_t - 1) * s_t**2) / (total - 2))
        d = (m_t - m_c) / pooled
        var_d = total / (n_c * n_t) + d**2 / (2.0 * total)
        j = hedges_correction(total)
        return _effect(d * j, var_d * j**2, kind, n_c, n_t, label, ci_level)
    return _effect(m_t - m_c, s_t**2 / n_t + s_c**2 / n_c, kind, n_c, n_t, label, ci_level)


def effects_from_arrays(
    estimates: Any,
    variances: Any,
    labels: Sequence[str] | None = None,
    kind: EffectKind = EffectKind.LOG_RATIO,
    ci_level: float = 0.95,
) -> list[EffectSize]:
    """Effect sizes from estimates and variances computed elsewhere.

    Args:
        estimates: the effects.
        variances: their variances.
        labels: labels of the studies, the positions by default.
        kind: the kind of effect.
        ci_level: level of the intervals.

    Returns:
        The effect sizes (`n_control` and `n_treatment` are 0).

    Raises:
        ValueError: if the lengths differ.
    """
    est = np.asarray(estimates, dtype=np.float64).ravel()
    var = np.asarray(variances, dtype=np.float64).ravel()
    if est.size != var.size:
        raise ValueError(f"{est.size} estimates but {var.size} variances")
    names = [str(i) for i in range(est.size)] if labels is None else list(labels)
    if len(names) != est.size:
        raise ValueError(f"{est.size} estimates but {len(names)} labels")
    return [_effect(e, v, kind, 0, 0, name, ci_level) for e, v, name in zip(est, var, names, strict=True)]


def _arrays(effects: Sequence[EffectSize]) -> tuple[np.ndarray, np.ndarray]:
    """The estimates and the variances of the effects.

    Args:
        effects: the effect sizes.

    Returns:
        The estimates and the variances.

    Raises:
        ValueError: without effects.
    """
    if not effects:
        raise ValueError("A meta-analysis needs at least one study")
    return (
        np.array([e.estimate for e in effects], dtype=np.float64),
        np.array([e.variance for e in effects], dtype=np.float64),
    )


def _pool(theta: np.ndarray, w: np.ndarray, model: str, tau2: float, ci_level: float) -> PooledEffect:
    """Inverse variance pooling.

    Args:
        theta: the effects.
        w: the weights.
        model: `"fixed"` or `"random"`.
        tau2: the between-study variance of the weights.
        ci_level: level of the interval.

    Returns:
        The pooled effect.
    """
    estimate = float((w * theta).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    z = estimate / se
    zq = _z(ci_level)
    return PooledEffect(
        estimate=estimate,
        se=se,
        ci_low=estimate - zq * se,
        ci_high=estimate + zq * se,
        ci_level=ci_level,
        z=float(z),
        p_value=float(2.0 * norm.sf(abs(z))),
        weights=w / w.sum(),
        model=model,
        tau2=tau2,
    )


def fixed_effect(effects: Sequence[EffectSize], *, ci_level: float = 0.95) -> PooledEffect:
    """Fixed effect pooling with the weights \\(w_i = 1 / v_i\\).

    Args:
        effects: the effect sizes.
        ci_level: level of the interval.

    Returns:
        The pooled effect.
    """
    theta, v = _arrays(effects)
    return _pool(theta, 1.0 / v, "fixed", 0.0, ci_level)


def heterogeneity(effects: Sequence[EffectSize]) -> Heterogeneity:
    """Heterogeneity statistics of the effects.

    \\(Q = \\sum w_i (\\theta_i - \\hat\\theta_F)^2\\) with \\(w_i = 1/v_i\\),
    \\(C = \\sum w_i - \\sum w_i^2 / \\sum w_i\\), \\(\\tau^2 = \\max(0, (Q - (k-1)) / C)\\)
    (DerSimonian & Laird 1986), \\(I^2 = \\max(0, (Q - (k-1)) / Q)\\), \\(H^2 = Q / (k-1)\\)
    (Higgins & Thompson 2002).

    Args:
        effects: the effect sizes.

    Returns:
        The statistics.
    """
    theta, v = _arrays(effects)
    w = 1.0 / v
    k = theta.size
    theta_f = (w * theta).sum() / w.sum()
    q = float((w * (theta - theta_f) ** 2).sum())
    df = k - 1
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if df > 0 and c > 0 else 0.0
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return Heterogeneity(
        q=q,
        df=df,
        p_value=float(chi2.sf(q, df)) if df > 0 else float("nan"),
        i2=i2,
        h2=q / df if df > 0 else float("nan"),
        tau2=tau2,
    )


def random_effects(effects: Sequence[EffectSize], *, ci_level: float = 0.95) -> PooledEffect:
    """Random effects pooling of DerSimonian & Laird with the weights \\(w_i^* = 1 / (v_i + \\tau^2)\\).

    Args:
        effects: the effect sizes.
        ci_level: level of the interval.

    Returns:
        The pooled effect.
    """
    theta, v = _arrays(effects)
    tau2 = heterogeneity(effects).tau2
    return _pool(theta, 1.0 / (v + tau2), "random", tau2, ci_level)


def meta_analysis(
    studies: Sequence[Study],
    kind: EffectKind = EffectKind.HEDGES_G,
    *,
    ci_level: float = 0.95,
) -> MetaResult:
    """Meta-analysis of a parameter over studies.

    Args:
        studies: the studies.
        kind: the kind of effect.
        ci_level: level of the intervals.

    Returns:
        The per study effects, the fixed effect and random effects pooling and the heterogeneity.

    Raises:
        ValueError: without studies.
    """
    if not studies:
        raise ValueError("A meta-analysis needs at least one study")
    effects = tuple(
        effect_size(s.control, s.treatment, kind, ci_level=ci_level, label=s.label) for s in studies
    )
    return MetaResult(
        kind=kind,
        effects=effects,
        fixed=fixed_effect(effects, ci_level=ci_level),
        random=random_effects(effects, ci_level=ci_level),
        heterogeneity=heterogeneity(effects),
        ci_level=ci_level,
    )


def meta_analysis_by(
    studies: Sequence[Study],
    kind: EffectKind = EffectKind.HEDGES_G,
    *,
    ci_level: float = 0.95,
) -> dict[str, MetaResult]:
    """One meta-analysis per category of the studies.

    Args:
        studies: the studies; a study without a category is grouped under `""`.
        kind: the kind of effect.
        ci_level: level of the intervals.

    Returns:
        Category to result, in the order of first appearance.
    """
    groups: dict[str, list[Study]] = {}
    for study in studies:
        groups.setdefault(study.category or "", []).append(study)
    return {key: meta_analysis(group, kind, ci_level=ci_level) for key, group in groups.items()}
```

Export from `stats/__init__.py`: `EffectKind`, `EffectSize`, `Heterogeneity`, `MetaResult`, `PooledEffect`, `Study`, `effect_size`, `effects_from_arrays`, `fixed_effect`, `heterogeneity`, `meta_analysis`, `meta_analysis_by`, `random_effects`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/stats -q -W error`
Expected: PASS, the BCG numbers within `1e-6` of the fixture (the fixture carries six digits). Run `uv run ruff check src tests && uv run ruff format src tests && uvx ty check` → clean.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/stats tests/stats tests/data/reference/meta_bcg.json
git commit -q -m "Add the meta-analysis: effect sizes, fixed effect, DerSimonian-Laird random effects and heterogeneity

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 7: Figures: `plot_parameters`, `plot_ratio`, `plot_forest`, `plot_bland_altman`

**Files:**
- Create: `src/pkpdutils/plot/parameters.py`, `src/pkpdutils/plot/ratio.py`, `src/pkpdutils/plot/meta.py`
- Modify: `src/pkpdutils/plot/fit.py` (append `plot_bland_altman`), `src/pkpdutils/plot/__init__.py`, `src/pkpdutils/plot/style.py` (add `group_colors`, `limit_color`, `pooled_color`)
- Test: `tests/plot/test_plot_parameters.py`, `tests/plot/test_plot_ratio.py`, `tests/plot/test_plot_meta.py`, `tests/plot/test_plot_fit.py`

**Interfaces:**
- `PlotStyle` gains `limit_color: str = "tab:red"` (acceptance limits, thresholds), `pooled_color: str = "tab:orange"` (pooled effects), `summary_color: str = "tab:blue"` (means and intervals over points).
- `plot_parameters(result: ParameterResult, name: str, dim: str, *, by: str | None = None, log: bool = False, scale: Scale = Scale.LOG, ci_level: float = 0.95, ax: Axes | None = None, style: PlotStyle = DEFAULT_STYLE, **indexers: Any) -> Figure`: jittered points of every individual per group (the groups are the values of the coordinate `by` along `dim`, one group without `by`), a box plot per group and the geometric mean (`scale=LOG`) or the mean with its interval of `summarize` as a marker with an error bar; y label `name [unit]`, x tick labels the groups.
- `plot_ratio(ratios: Mapping[str, RatioResult | BEParameter] | BEResult, *, limits: tuple[float, float] | None = (0.8, 1.25), thresholds: DDIThresholds | None = None, ax=None, style=DEFAULT_STYLE) -> Figure`: one row per entry with the point estimate and its interval on a logarithmic x axis, the unity line, the acceptance limits as dashed lines (`limits`) and the DDI thresholds as dotted lines with the class labels (`thresholds`); the title names the level of the intervals.
- `plot_forest(result: MetaResult, *, ax=None, style=DEFAULT_STYLE, exp: bool | None = None) -> Figure`: one row per study (marker area proportional to the random effects weight, interval line), the pooled fixed and random effects as diamonds below, a vertical line at the null (0, or 1 with `exp`); `exp=None` exponentiates a `LOG_RATIO` analysis (ratio axis, log scale) and leaves the other kinds.
- `plot_bland_altman(result: FitResult, *, log: bool = False, style=DEFAULT_STYLE) -> Figure`: difference `y_pred - y_data` (log ratio with `log`) against the mean of both, over every sample and point, the mean difference and the `mean +- 1.96 sd` limits of agreement as horizontal lines.

- [ ] **Step 1: Write the failing tests**

`tests/plot/test_plot_parameters.py`:
```python
import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib.figure import Figure

from pkpdutils import Route, Timecourses, nca
from pkpdutils.plot import plot_parameters
from pkpdutils.stats import Scale

matplotlib.use("Agg")


def nca_result():
    time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
    rng = np.random.default_rng(2)
    n = 8
    curves = np.stack([rng.lognormal(np.log(10), 0.2) * np.exp(-0.2 * time) for _ in range(n)])
    batch = Timecourses.from_arrays(
        time,
        curves,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": [f"s{i}" for i in range(n)], "sex": ("individual", ["F", "M"] * 4)},
        dose={"amount": np.full(n, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    return nca(batch)


def test_plot_parameters_groups() -> None:
    result = nca_result()
    fig = plot_parameters(result, "auc_inf_obs", "individual", by="sex", log=True)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["F", "M"]
    assert ax.get_ylabel().startswith("auc_inf_obs [")
    matplotlib.pyplot.close("all")


def test_plot_parameters_single_group_linear() -> None:
    result = nca_result()
    fig, ax = matplotlib.pyplot.subplots()
    out = plot_parameters(result, "cmax", "individual", scale=Scale.LINEAR, ax=ax)
    assert out is fig
    assert [t.get_text() for t in ax.get_xticklabels()] == ["cmax"]
    assert len(ax.containers) >= 1  # the error bar of the mean
    matplotlib.pyplot.close("all")
```

`tests/plot/test_plot_ratio.py`:
```python
import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib.figure import Figure

from pkpdutils.plot import plot_ratio
from pkpdutils.stats import DDIThresholds, ParameterSample, ratio
from pkpdutils.stats.bioequivalence import tost, BEResult

matplotlib.use("Agg")
RNG = np.random.default_rng(1)
REF = np.exp(RNG.normal(np.log(100), 0.2, 10))
TEST = REF * np.exp(RNG.normal(np.log(0.97), 0.1, 10))


def test_plot_ratio_of_ratio_results() -> None:
    ratios = {
        "auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF)),
        "cmax": ratio(ParameterSample(values=TEST * 1.1), ParameterSample(values=REF)),
    }
    fig = plot_ratio(ratios)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["auc", "cmax"]
    assert "90" in ax.get_title()
    xs = sorted(line.get_xdata()[0] for line in ax.get_lines() if line.get_linestyle() == "--")
    assert xs == [0.8, 1.25]
    matplotlib.pyplot.close("all")


def test_plot_ratio_of_bioequivalence_and_thresholds() -> None:
    be = BEResult(
        parameters={"auc": tost(ParameterSample(values=TEST), ParameterSample(values=REF))},
        bioequivalent=True,
        limits=(0.8, 1.25),
        ci_level=0.90,
    )
    fig = plot_ratio(be, limits=None, thresholds=DDIThresholds.fda())
    ax = fig.axes[0]
    dotted = [line.get_xdata()[0] for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert sorted(dotted) == [0.2, 0.5, 0.8, 1.25, 2.0, 5.0]
    assert any("strong" in t.get_text() for t in ax.texts)
    matplotlib.pyplot.close("all")
```

`tests/plot/test_plot_meta.py`:
```python
import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib.figure import Figure

from pkpdutils.plot import plot_forest
from pkpdutils.stats import EffectKind, ParameterSample, Study, meta_analysis

matplotlib.use("Agg")


def studies() -> list[Study]:
    rng = np.random.default_rng(9)
    return [
        Study(
            label=f"study {i}",
            control=ParameterSample(values=np.exp(rng.normal(np.log(100), 0.3, 10))),
            treatment=ParameterSample(values=np.exp(rng.normal(np.log(140), 0.3, 10))),
        )
        for i in range(4)
    ]


def test_plot_forest_log_ratio_is_a_ratio_axis() -> None:
    result = meta_analysis(studies(), EffectKind.LOG_RATIO)
    fig = plot_forest(result)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[:4] == ["study 0", "study 1", "study 2", "study 3"]
    assert "fixed" in labels[4].lower() and "random" in labels[5].lower()
    assert ax.get_xlabel().startswith("ratio")
    matplotlib.pyplot.close("all")


def test_plot_forest_hedges_g_linear_axis() -> None:
    result = meta_analysis(studies(), EffectKind.HEDGES_G)
    fig, ax = matplotlib.pyplot.subplots()
    plot_forest(result, ax=ax)
    assert ax.get_xscale() == "linear"
    assert "Hedges" in ax.get_xlabel()
    assert "I2" in ax.get_title() or "I²" in ax.get_title()
    matplotlib.pyplot.close("all")
```

Append to `tests/plot/test_plot_fit.py` (add `plot_bland_altman` to its import from `pkpdutils.plot`):
```python
def test_plot_bland_altman() -> None:
    fig = plot_bland_altman(monoexp_result(3))
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("mean") and ax.get_ylabel().startswith("difference")
    # the mean difference and the two limits of agreement
    assert sum(1 for line in ax.get_lines() if line.get_linestyle() in ("--", ":")) >= 3
    fig_log = plot_bland_altman(monoexp_result(), log=True)
    assert fig_log.axes[0].get_ylabel().startswith("log ratio")
    matplotlib.pyplot.close("all")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/plot -q`
Expected: FAIL with `ImportError: cannot import name 'plot_parameters'`

- [ ] **Step 3: Implement**

Add to `PlotStyle` in `src/pkpdutils/plot/style.py` (with attribute docs):
```python
    limit_color: str = "tab:red"
    pooled_color: str = "tab:orange"
    summary_color: str = "tab:blue"
```

`src/pkpdutils/plot/parameters.py`:
```python
"""Distribution of a parameter over the individuals, by group."""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import _figure_of
from pkpdutils.result import ParameterResult
from pkpdutils.stats.sample import ParameterSample, Scale, summarize


def plot_parameters(
    result: ParameterResult,
    name: str,
    dim: str,
    *,
    by: str | None = None,
    log: bool = False,
    scale: Scale = Scale.LOG,
    ci_level: float = 0.95,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Strip and box plot of a parameter over a sample dimension, with the mean and its interval per group.

    Every individual is a jittered point, every group a box plot, and the
    geometric mean (`scale=LOG`) or the arithmetic mean with the t interval
    of `summarize` at `ci_level` a marker with an error bar.

    Args:
        result: the result the parameter is taken from.
        name: name of the parameter.
        dim: the sample dimension of the individuals.
        by: a coordinate along `dim` which groups the individuals, one
            group named after the parameter without it.
        log: logarithmic y axis.
        scale: scale of the mean and its interval.
        ci_level: level of the interval.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.
        **indexers: coordinate label per remaining sample dimension.

    Returns:
        The figure.

    Raises:
        ValueError: if `by` is not a coordinate along `dim`.
    """
    sample = result.sample(name, dim, **indexers)
    assert sample.values is not None
    if by is None:
        groups: dict[str, ParameterSample] = {name: sample}
    else:
        if by not in sample.coords:
            raise ValueError(f"'{by}' is not a coordinate along '{dim}': {sorted(sample.coords)}")
        keys = sample.coords[by]
        groups = {str(key): sample.select(keys == key) for key in dict.fromkeys(keys.tolist())}
    fig, ax = _figure_of(ax)
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(groups) + 1)
    values = [g.finite_values for g in groups.values()]
    ax.boxplot(values, positions=positions, widths=0.5, showfliers=False, zorder=1)
    for pos, (label, group) in zip(positions, groups.items(), strict=True):
        v = group.finite_values
        ax.plot(
            pos + rng.uniform(-0.15, 0.15, v.size),
            v,
            linestyle="none",
            marker=style.data_marker,
            color=style.data_color,
            markersize=style.markersize,
            alpha=0.6,
            zorder=2,
            label=label,
        )
        if v.size:
            s = summarize(group, scale=scale, ci_level=ci_level)
            center = s.geomean if scale is Scale.LOG else s.mean
            err = np.array([[center - s.ci_low], [s.ci_high - center]]) if np.isfinite(s.ci_low) else None
            ax.errorbar(
                [pos + 0.3],
                [center],
                yerr=err,
                marker="D",
                color=style.summary_color,
                capsize=3,
                linestyle="none",
                zorder=3,
            )
    ax.set_xticks(positions, list(groups))
    ax.set_ylabel(f"{name} [{sample.unit}]")
    if by is not None:
        ax.set_xlabel(by)
    if log:
        ax.set_yscale("log")
    return fig
```

`src/pkpdutils/plot/ratio.py`:
```python
"""Ratios with their intervals against acceptance limits and interaction thresholds."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import _figure_of
from pkpdutils.stats.bioequivalence import BEResult
from pkpdutils.stats.ddi import DDIThresholds


class RatioLike(Protocol):
    """A ratio with an interval: `RatioResult`, `BEParameter`."""

    gmr: float
    ci_low: float
    ci_high: float
    ci_level: float


def plot_ratio(
    ratios: Mapping[str, RatioLike] | BEResult,
    *,
    limits: tuple[float, float] | None = (0.8, 1.25),
    thresholds: DDIThresholds | None = None,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Point estimates and intervals of ratios on a logarithmic axis, with limits and thresholds.

    Args:
        ratios: name to ratio, or a bioequivalence result (its parameters).
        limits: acceptance limits drawn as dashed lines, `None` for none.
        thresholds: interaction thresholds drawn as dotted lines with the
            class names, `None` for none.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.

    Returns:
        The figure.
    """
    entries: Mapping[str, RatioLike] = ratios.parameters if isinstance(ratios, BEResult) else ratios
    fig, ax = _figure_of(ax)
    names = list(entries)
    ys = np.arange(len(names))
    for y, name in zip(ys, names, strict=True):
        r = entries[name]
        ax.errorbar(
            [r.gmr],
            [y],
            xerr=[[r.gmr - r.ci_low], [r.ci_high - r.gmr]],
            marker=style.data_marker,
            color=style.data_color,
            capsize=3,
            linestyle="none",
            markersize=style.markersize,
        )
    ax.axvline(1.0, color="gray", linewidth=1.0)
    if limits is not None:
        for limit in limits:
            ax.axvline(limit, color=style.limit_color, linestyle="--", linewidth=style.linewidth)
    if thresholds is not None:
        classes = {
            thresholds.inhibitor_weak: "weak inhibitor",
            thresholds.inhibitor_moderate: "moderate inhibitor",
            thresholds.inhibitor_strong: "strong inhibitor",
            thresholds.inducer_weak: "weak inducer",
            thresholds.inducer_moderate: "moderate inducer",
            thresholds.inducer_strong: "strong inducer",
        }
        for value, label in classes.items():
            ax.axvline(value, color=style.limit_color, linestyle=":", linewidth=1.0)
            ax.text(value, len(names) - 0.4, label, rotation=90, fontsize="x-small", ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yticks(ys, names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("ratio test / reference")
    levels = {int(round(entries[n].ci_level * 100)) for n in names}
    ax.set_title(f"geometric mean ratios with {', '.join(str(lv) for lv in sorted(levels))} % intervals", fontsize="small")
    return fig
```

`src/pkpdutils/plot/meta.py`:
```python
"""Forest plot of a meta-analysis."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import _figure_of
from pkpdutils.stats.meta import EffectKind, MetaResult

#: axis label per kind of effect
_LABELS = {
    EffectKind.HEDGES_G: "Hedges' g",
    EffectKind.MEAN_DIFF: "mean difference",
    EffectKind.LOG_RATIO: "log ratio",
}


def plot_forest(
    result: MetaResult,
    *,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    exp: bool | None = None,
) -> Figure:
    """Forest plot: the effect of every study with its interval, the pooled effects as diamonds.

    The marker area of a study is proportional to its random effects
    weight. A `LOG_RATIO` analysis is shown as ratios on a logarithmic axis
    (`exp=None` or `True`); the other kinds stay on their scale.

    Args:
        result: the meta-analysis.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.
        exp: exponentiate the effects; `None` does so for `LOG_RATIO`.

    Returns:
        The figure.
    """
    use_exp = result.kind is EffectKind.LOG_RATIO if exp is None else exp
    transform = np.exp if use_exp else (lambda v: np.asarray(v, dtype=float))
    fig, ax = _figure_of(ax)
    n = result.n_studies
    weights = result.random.weights
    for i, e in enumerate(result.effects):
        est, low, high = (float(transform(v)) for v in (e.estimate, e.ci_low, e.ci_high))
        ax.plot([low, high], [i, i], color=style.data_color, linewidth=style.linewidth)
        ax.scatter([est], [i], s=40 + 400 * weights[i], marker="s", color=style.data_color, zorder=3)
    for j, pooled in enumerate((result.fixed, result.random)):
        y = n + j
        est, low, high = (float(transform(v)) for v in (pooled.estimate, pooled.ci_low, pooled.ci_high))
        ax.fill([low, est, high, est], [y, y - 0.3, y, y + 0.3], color=style.pooled_color, zorder=3)
    null = 1.0 if use_exp else 0.0
    ax.axvline(null, color="gray", linewidth=1.0)
    ax.set_yticks(
        np.arange(n + 2),
        [*result.labels, "fixed effect", f"random effects (tau2 = {result.random.tau2:.3g})"],
    )
    ax.set_ylim(-0.6, n + 1.6)
    ax.invert_yaxis()
    if use_exp:
        ax.set_xscale("log")
        ax.set_xlabel("ratio treatment / control")
    else:
        ax.set_xlabel(_LABELS[result.kind])
    het = result.heterogeneity
    ax.set_title(
        f"Q = {het.q:.2f} (df = {het.df}, p = {het.p_value:.2g}), I² = {het.i2:.1f} %",
        fontsize="small",
    )
    return fig
```

Append to `src/pkpdutils/plot/fit.py`:
```python
def plot_bland_altman(
    result: FitResult, *, log: bool = False, style: PlotStyle = DEFAULT_STYLE
) -> Figure:
    """Bland-Altman plot of the predictions against the data of every sample.

    The difference `y_pred - y_data` (the log ratio with `log`) against the
    mean of both per point, with the mean difference and the limits of
    agreement `mean +- 1.96 sd` as horizontal lines (Bland & Altman 1986).

    Args:
        result: the fit.
        log: use the log ratio and the log mean; non-positive points are masked out.
        style: colors and markers.

    Returns:
        The figure.
    """
    y = result.ds["y_data"].to_numpy().ravel()
    pred = result.ds["y_pred"].to_numpy().ravel()
    ok = np.isfinite(y) & np.isfinite(pred)
    if log:
        ok &= (y > 0) & (pred > 0)
        mean = np.exp((np.log(y[ok]) + np.log(pred[ok])) / 2.0)
        diff = np.log(pred[ok]) - np.log(y[ok])
    else:
        mean = (y[ok] + pred[ok]) / 2.0
        diff = pred[ok] - y[ok]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean, diff, linestyle="none", marker=style.data_marker, color=style.data_color, markersize=style.markersize)
    if diff.size:
        center, sd = float(diff.mean()), float(diff.std(ddof=1)) if diff.size > 1 else 0.0
        ax.axhline(center, color=style.fit_color, linestyle="--", linewidth=style.linewidth)
        ax.axhline(center + 1.96 * sd, color=style.limit_color, linestyle=":", linewidth=1.0)
        ax.axhline(center - 1.96 * sd, color=style.limit_color, linestyle=":", linewidth=1.0)
    ax.axhline(0.0, color="gray", linewidth=1.0)
    unit = result.ds.attrs["y_unit"]
    if log:
        ax.set_xscale("log")
        ax.set_xlabel(f"mean of observed and predicted [{unit}]")
        ax.set_ylabel("log ratio predicted / observed")
    else:
        ax.set_xlabel(f"mean of observed and predicted [{unit}]")
        ax.set_ylabel(f"difference predicted - observed [{unit}]")
    return fig
```

Export `plot_bland_altman`, `plot_forest`, `plot_parameters`, `plot_ratio` from `src/pkpdutils/plot/__init__.py` (sorted `__all__`). `pkpdutils.plot` now imports `pkpdutils.stats`; `pkpdutils.stats` does not import `pkpdutils.plot`, so there is no cycle. Add Bland & Altman 1986 to `docs/references.md` in Task 8.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/plot -q -W error`
Expected: PASS. The tests assert axis scales, tick labels, line styles and texts, not pixels. Run `uv run ruff check src tests && uv run ruff format src tests && uvx ty check` → clean (ty: the `lambda` in `plot_forest` needs a `Callable[[Any], np.ndarray]` annotation on `transform`, `from collections.abc import Callable`).

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/plot tests/plot
git commit -q -m "Add the parameter, ratio, forest and Bland-Altman figures

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 8: Examples and exports

**Files:**
- Create: `examples/bioequivalence.py`, `examples/ddi.py`, `examples/meta_analysis.py`
- Modify: `src/pkpdutils/__init__.py`, `tests/test_package.py`, `tests/examples/test_examples.py`, `examples/README.md`

- [ ] **Step 1: Exports and the export test**

Append to `tests/test_package.py`:
```python
def test_stats_exports() -> None:
    from pkpdutils import (
        ParameterSample,
        bioequivalence,
        compare,
        ddi_classification,
        meta_analysis,
        ratio,
    )

    assert callable(compare) and callable(ratio) and callable(bioequivalence)
    assert callable(ddi_classification) and callable(meta_analysis)
    assert ParameterSample(values=[1.0, 2.0]).size == 2
```
Run: `uv run pytest tests/test_package.py -q` → FAIL (`ImportError`). Add to `src/pkpdutils/__init__.py` the import `from pkpdutils.stats import ParameterSample, bioequivalence, compare, ddi_classification, meta_analysis, ratio`, the six names to `__all__` (sorted), and the sentence "`pkpdutils.stats` holds the statistics on parameters (tests, ratios, bioequivalence, drug-drug interactions, meta-analysis)." to the module docstring. Run again → PASS.

- [ ] **Step 2: `examples/bioequivalence.py`**

```python
"""Average bioequivalence of a test against a reference formulation in a 2x2 crossover.

Run from the root of the repository with `python -m examples.bioequivalence`.
Writes `bioequivalence.png` and `bioequivalence_parameters.png` into the working directory.
"""

import numpy as np

from pkpdutils import Route, Timecourses, bioequivalence, nca
from pkpdutils.console import console
from pkpdutils.plot import plot_parameters, plot_ratio

# 12 subjects, sequence RT receives the reference in period 1 and the test in
# period 2, sequence TR the other way round; the test formulation has a
# slightly lower bioavailability (ratio 0.93) and a slower absorption
N = 12
TIME = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24])
SUBJECTS = [f"s{i:02d}" for i in range(N)]
SEQUENCE = ["RT"] * 6 + ["TR"] * 6
PERIOD_TEST = np.where(np.array(SEQUENCE) == "RT", 2, 1)
PERIOD_REF = 3 - PERIOD_TEST
rng = np.random.default_rng(12)
subject_scale = rng.lognormal(0, 0.25, N)  # between-subject variability of the exposure


def curves(bioavailability: float, ka: float, period: np.ndarray) -> np.ndarray:
    ke = 0.15
    period_effect = np.where(period == 2, 1.05, 1.0)  # period 2 runs 5 % higher
    values = []
    for scale, p in zip(subject_scale * period_effect, period, strict=True):
        c = bioavailability * scale * 100 * ka / (ka - ke) * (np.exp(-ke * TIME) - np.exp(-ka * TIME)) / 30
        values.append(c * rng.lognormal(0, 0.06, TIME.size))
    return np.stack(values)


def batch(values: np.ndarray, period: np.ndarray) -> Timecourses:
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": SUBJECTS,
            "period": ("individual", period),
            "sequence": ("individual", SEQUENCE),
        },
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )


if __name__ == "__main__":
    reference = nca(batch(curves(1.0, 1.5, PERIOD_REF), PERIOD_REF))
    test = nca(batch(curves(0.93, 0.9, PERIOD_TEST), PERIOD_TEST))

    console.rule("Two one-sided tests, 2x2 crossover, 90 % intervals, 80-125 %")
    result = bioequivalence(test, reference, parameters=["auc_inf_obs", "auc_last", "cmax"])
    console.print(
        result.to_dataframe()[
            ["parameter", "gmr", "ci_low", "ci_high", "bioequivalent", "p_value", "cv_intra", "p_period", "p_sequence", "design"]
        ]
    )
    console.print("bioequivalent:", result.bioequivalent)
    plot_ratio(result).savefig("bioequivalence.png", dpi=120)
    plot_parameters(test, "cmax", "individual", by="sequence", log=True).savefig(
        "bioequivalence_parameters.png", dpi=120
    )
    console.print("written: bioequivalence.png, bioequivalence_parameters.png")
```

- [ ] **Step 3: `examples/ddi.py`**

```python
"""Classification of a drug-drug interaction from the exposure with and without a perpetrator.

Run from the root of the repository with `python -m examples.ddi`.
Writes `ddi.png` into the working directory.
"""

import numpy as np

from pkpdutils import Route, Timecourses, compare, ddi_classification, nca, ratio
from pkpdutils.console import console
from pkpdutils.plot import plot_ratio
from pkpdutils.stats import DDIThresholds, substrate_sensitivity

# a substrate given alone (control) and with a moderate CYP inhibitor to two
# parallel groups of 10 subjects: the inhibitor lowers the clearance to 35 %
TIME = np.array([0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48])
N = 10
rng = np.random.default_rng(8)


def batch(clearance_factor: float, label: str) -> Timecourses:
    ke = 0.12 * clearance_factor
    values = np.stack(
        [
            rng.lognormal(np.log(8), 0.2) * np.exp(-ke * TIME) * rng.lognormal(0, 0.05, TIME.size)
            for _ in range(N)
        ]
    )
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": [f"{label}{i}" for i in range(N)], "treatment": ("individual", [label] * N)},
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="substrate",
    )


if __name__ == "__main__":
    control = nca(batch(1.0, "control"))
    inhibited = nca(batch(0.35, "inhibitor"))

    console.rule("Exposure ratios with / without the inhibitor")
    auc_ratio = ratio(inhibited.sample("auc_inf_obs", "individual"), control.sample("auc_inf_obs", "individual"))
    cmax_ratio = ratio(inhibited.sample("cmax", "individual"), control.sample("cmax", "individual"))
    for r in (auc_ratio, cmax_ratio):
        console.print(f"{r.name:<12} GMR {r.gmr:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] (90 %)")

    console.rule("Welch t test of the log AUC")
    test = compare(inhibited.sample("auc_inf_obs", "individual"), control.sample("auc_inf_obs", "individual"))
    console.print(f"{test.test}: t = {test.statistic:.2f}, p = {test.p_value:.2g}, Hedges' g = {test.hedges_g:.2f}")

    console.rule("Classification (FDA 2020)")
    ddi = ddi_classification(auc_ratio, cmax_ratio=cmax_ratio)
    console.print(f"{ddi.strength} {ddi.kind}, uncertain: {ddi.uncertain}")
    console.print("EMA:", ddi_classification(auc_ratio, thresholds=DDIThresholds.ema()).to_dict())
    console.print("substrate sensitivity:", substrate_sensitivity(auc_ratio))
    plot_ratio({"auc_inf_obs": auc_ratio, "cmax": cmax_ratio}, limits=None, thresholds=DDIThresholds.fda()).savefig("ddi.png", dpi=120)
    console.print("written: ddi.png")
```

- [ ] **Step 4: `examples/meta_analysis.py`**

```python
"""Meta-analysis of the effect of smoking on the clearance of caffeine over published studies.

Run from the root of the repository with `python -m examples.meta_analysis`.
Writes `meta_analysis.png` into the working directory.
"""

from pkpdutils import ParameterSample, meta_analysis
from pkpdutils.console import console
from pkpdutils.plot import plot_forest
from pkpdutils.stats import EffectKind, Study

# clearance of caffeine (ml/min/kg) in non-smokers (control) and smokers as
# mean, sd and n; illustrative values in the range of the literature
STUDIES = [
    Study("study A", ParameterSample(mean=1.2, sd=0.4, n=10), ParameterSample(mean=2.0, sd=0.6, n=10), "smokers"),
    Study("study B", ParameterSample(mean=1.4, sd=0.5, n=8), ParameterSample(mean=2.3, sd=0.8, n=9), "smokers"),
    Study("study C", ParameterSample(mean=1.1, sd=0.3, n=12), ParameterSample(mean=1.6, sd=0.5, n=12), "smokers"),
    Study("study D", ParameterSample(mean=1.3, sd=0.4, n=15), ParameterSample(mean=2.4, sd=0.9, n=14), "smokers"),
    Study("study E", ParameterSample(mean=1.2, sd=0.4, n=6), ParameterSample(mean=1.5, sd=0.4, n=6), "smokers"),
]

if __name__ == "__main__":
    for kind in (EffectKind.LOG_RATIO, EffectKind.HEDGES_G):
        console.rule(f"{kind}")
        result = meta_analysis(STUDIES, kind)
        console.print(result.to_dataframe())
        console.print("fixed effect:  ", result.fixed.to_dict())
        console.print("random effects:", result.random.to_dict())
        het = result.heterogeneity
        console.print(f"Q = {het.q:.2f}, df = {het.df}, p = {het.p_value:.2g}, I2 = {het.i2:.1f} %, tau2 = {het.tau2:.3g}")
    plot_forest(meta_analysis(STUDIES, EffectKind.LOG_RATIO)).savefig("meta_analysis.png", dpi=120)
    console.print("written: meta_analysis.png")
```

- [ ] **Step 5: Register and run the examples**

Add `"examples.bioequivalence"`, `"examples.ddi"`, `"examples.meta_analysis"` to `SCRIPTS` in `tests/examples/test_examples.py`. Add three rows to the table of `examples/README.md`:

| path | content |
| --- | --- |
| `examples/bioequivalence.py` | average bioequivalence of a test against a reference formulation in a 2x2 crossover from two NCA batches, ratio and parameter figures |
| `examples/ddi.py` | exposure ratios with and without an inhibitor, Welch t test, FDA and EMA classification of the interaction, substrate sensitivity |
| `examples/meta_analysis.py` | fixed effect and random effects meta-analysis of published summary statistics with a forest plot |

Run: `uv run pytest tests/examples -q -W error` → PASS. Run each example once from the scratch directory to read its output: `cd /tmp && uv run --project /home/mkoenig/git/pkdb_analysis python -m examples.bioequivalence` (with `PYTHONPATH=/home/mkoenig/git/pkdb_analysis`), and look at the three PNG files: the bioequivalence plot shows the three ratios within or near 80-125 %, the DDI plot the AUC ratio near 2.9 in the moderate inhibitor band, the forest plot the ratios above 1 with the two diamonds. Fix what looks off (overlapping labels, clipped texts) before committing.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/__init__.py tests/test_package.py tests/examples/test_examples.py examples
git commit -q -m "Add the bioequivalence, drug-drug interaction and meta-analysis examples and the stats exports

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 9: Documentation

**Files:**
- Create: `docs/statistics.md`, `docs/api/stats.md`, `docs/api/stats.bioequivalence.md`, `docs/api/stats.ddi.md`, `docs/api/stats.meta.md`
- Modify: `docs/api/plot.md`, `docs/api/index.md`, `docs/api/result.md` (no change needed, `sample` is a method and renders), `docs/plotting.md`, `docs/glossary.md`, `docs/references.md`, `docs/index.md`, `zensical.toml`, `CLAUDE.md`

- [ ] **Step 1: API pages**

`docs/api/stats.md`:
```markdown
# stats

## stats.sample

::: pkpdutils.stats.sample

## stats.tests

::: pkpdutils.stats.tests

## stats.ratio

::: pkpdutils.stats.ratio
```

`docs/api/stats.bioequivalence.md`:
```markdown
# stats.bioequivalence

::: pkpdutils.stats.bioequivalence
```

`docs/api/stats.ddi.md`:
```markdown
# stats.ddi

::: pkpdutils.stats.ddi
```

`docs/api/stats.meta.md`:
```markdown
# stats.meta

::: pkpdutils.stats.meta
```

Append to `docs/api/plot.md`:
```markdown

## plot.parameters

::: pkpdutils.plot.parameters

## plot.ratio

::: pkpdutils.plot.ratio

## plot.meta

::: pkpdutils.plot.meta
```

In `docs/api/index.md` add after the `pkpdutils.fit` table:
```markdown
## pkpdutils.stats

Statistics on parameters, see [Statistics](../statistics.md).

| module | description |
| --- | --- |
| [stats](stats.md) | `ParameterSample`, `Scale`, `summarize`, `compare`, `multiple_comparison`, `ratio`: samples, tests and the geometric mean ratio |
| [stats.bioequivalence](stats.bioequivalence.md) | `bioequivalence`, `tost`, `Design`: the two one-sided tests, paired, parallel and 2x2 crossover designs |
| [stats.ddi](stats.ddi.md) | `ddi_classification`, `substrate_sensitivity`, `DDIThresholds`: the FDA and EMA classification of interactions |
| [stats.meta](stats.meta.md) | `effect_size`, `fixed_effect`, `random_effects`, `heterogeneity`, `meta_analysis`: the meta-analysis |
```
and extend the `plot` row: `plot_parameters`, `plot_ratio`, `plot_forest`, `plot_bland_altman`. In the `pkpdutils` table extend the `result` row with "`sample` gives a `ParameterSample`".

In `zensical.toml` add `{ "Statistics" = "statistics.md" },` after the `Pharmacodynamics` entry of the user guide, and the block
```toml
    { "pkpdutils.stats" = [
      { "stats" = "api/stats.md" },
      { "bioequivalence" = "api/stats.bioequivalence.md" },
      { "ddi" = "api/stats.ddi.md" },
      { "meta" = "api/stats.meta.md" },
    ] },
```
after the `pkpdutils.fit` block.

- [ ] **Step 2: References**

Add to `docs/references.md` under "Regulatory guidance" nothing new (FDA 2020, EMA 2012, FDA 2001 exist); under "Statistics" append:
```markdown
**Crossover designs.** The period-difference analysis of the 2x2 crossover, the tests of the period and the carryover effect and the within-subject CV.

> Chow SC, Liu JP.
> **Design and Analysis of Bioavailability and Bioequivalence Studies.**
> 3rd edition. Chapman & Hall/CRC; 2009.

**Heterogeneity.** \(I^2\) and \(H^2\) of a meta-analysis.

> Higgins JPT, Thompson SG.
> **Quantifying heterogeneity in a meta-analysis.**
> *Statistics in Medicine.* 2002;21(11):1539-1558.
> [doi:10.1002/sim.1186](https://doi.org/10.1002/sim.1186)

**Multiple comparisons.**

> Holm S.
> **A simple sequentially rejective multiple test procedure.**
> *Scandinavian Journal of Statistics.* 1979;6(2):65-70.

> Benjamini Y, Hochberg Y.
> **Controlling the false discovery rate: a practical and powerful approach to multiple testing.**
> *Journal of the Royal Statistical Society B.* 1995;57(1):289-300.
> [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

**Agreement of two measurements.** The Bland-Altman plot of `plot_bland_altman`.

> Bland JM, Altman DG.
> **Statistical methods for assessing agreement between two methods of clinical measurement.**
> *The Lancet.* 1986;327(8476):307-310.
> [doi:10.1016/S0140-6736(86)90837-8](https://doi.org/10.1016/S0140-6736(86)90837-8)

**Meta-analysis reference data.** The BCG vaccine trials used as the regression reference of the meta-analysis (`tests/data/reference/meta_bcg.json`), analysed with the R package `metafor`.

> Colditz GA, Brewer TF, Berkey CS, et al.
> **Efficacy of BCG vaccine in the prevention of tuberculosis: meta-analysis of the published literature.**
> *JAMA.* 1994;271(9):698-702.

> Viechtbauer W.
> **Conducting meta-analyses in R with the metafor package.**
> *Journal of Statistical Software.* 2010;36(3):1-48.
> [doi:10.18637/jss.v036.i03](https://doi.org/10.18637/jss.v036.i03)
```
Extend the "Effect sizes" entry (Hedges 1981) with the sentence "The small sample correction \(J\) of Hedges' g and its variance." and the "Random effects meta-analysis" entry with "The between-study variance \(\tau^2\) of the random effects model.".

- [ ] **Step 3: `docs/statistics.md`**

````markdown
# Statistics

Statistics on pharmacokinetic parameters: comparisons of two groups, geometric mean ratios, average bioequivalence, the classification of drug-drug interactions and the meta-analysis of published studies. Every function of `pkpdutils.stats` works on a `ParameterSample`, the values of one parameter over the individuals of a group or the summary statistics of the group, taken from an `NCAResult` or a `FitResult` with `sample` or typed in from a publication.

## Concepts

**Log-normal parameters.** Exposure, clearance, volume and half-life are positive and skewed: their logarithms are close to normal. The statistics therefore run on the log scale by default (`Scale.LOG`): differences of the logarithms are ratios of geometric means, intervals are symmetric on the log scale and asymmetric around the ratio, and the geometric mean and the geometric coefficient of variation \(\mathrm{CV}_g = \sqrt{e^{\sigma^2} - 1}\) describe a group. `Scale.LINEAR` compares arithmetic means, for parameters like \(t_\mathrm{max}\) or an effect which may be negative.

**Individual and summary data.** With the individual values of a group every test of scipy is available; a publication often gives only the mean, the standard deviation and the number of subjects. A summary sample is analysed with the Welch t test from its moments; on the log scale the moments of the logarithm follow from the log-normal relations \(\sigma^2 = \ln(1 + \mathrm{sd}^2/\mathrm{mean}^2)\) and \(\mu = \ln\mathrm{mean} - \sigma^2/2\), or directly from the geometric mean and CV when they are reported.

**Designs.** In a parallel design two groups of different subjects are compared with the Welch interval. In a paired or crossover design every subject receives both treatments, and the within-subject differences remove the between-subject variability: a 2x2 crossover (two sequences RT and TR, two periods) is analysed with sequence, period and subject effects, which also tests the period and the carryover effect and gives the within-subject CV.

**Bioequivalence.** Two formulations are bioequivalent when the 90 % confidence interval of the geometric mean ratio of \(\mathrm{AUC}\) and \(C_\mathrm{max}\) lies within 80-125 %[^fda_be]. That is the two one-sided tests procedure of Schuirmann at \(\alpha = 0.05\)[^schuirmann].

**Drug-drug interactions.** A perpetrator is classified by how much it changes the \(\mathrm{AUC}\) of a sensitive substrate: a strong, moderate or weak inhibitor raises it at least 5-fold, 2- to 5-fold or 1.25- to 2-fold, a strong, moderate or weak inducer lowers it by at least 80 %, 50-80 % or 20-50 %[^fda_ddi]; the EMA guideline uses the same thresholds[^ema_ddi]. With an interval of the ratio the classification is conservative and marked as uncertain when the interval spans a boundary.

**Meta-analysis.** Effects of several studies (Hedges' g, a mean difference or the log ratio of geometric means, the effect native to pharmacokinetics) are pooled with inverse variance weights. The fixed effect model assumes one true effect; the random effects model of DerSimonian and Laird adds the between-study variance \(\tau^2\) to every weight and widens the interval when the studies disagree[^dl]. \(Q\), \(I^2\) and \(H^2\) measure that disagreement[^higgins].

## Math

**Two-sample t tests.** With the means \(\bar a\), \(\bar b\), the variances \(s_a^2\), \(s_b^2\) and the sizes \(n_a\), \(n_b\) (on the analysis scale), Welch's statistic is \(t = (\bar a - \bar b) / \sqrt{s_a^2/n_a + s_b^2/n_b}\) with the Welch-Satterthwaite degrees of freedom; Student's uses the pooled variance \(s_p^2 = ((n_a-1)s_a^2 + (n_b-1)s_b^2)/(n_a+n_b-2)\) with \(n_a + n_b - 2\); the paired test is the one-sample test of the differences. The interval of the effect is \(\hat\theta \pm t_{1-\alpha/2,\nu}\,\mathrm{se}\), exponentiated on the log scale. The standardized effect sizes are Cohen's \(d = (\bar a - \bar b)/s_p\) and Hedges' \(g = J d\) with \(J = 1 - 3/(4N - 9)\), \(N = n_a + n_b\)[^hedges].

**Geometric mean ratio.** Paired: \(d_i = \ln t_i - \ln r_i\), \(\ln\mathrm{GMR} = \bar d\), \(\mathrm{se} = s_d/\sqrt{n}\), \(n - 1\) degrees of freedom. Parallel: \(\ln\mathrm{GMR} = \bar{\ln t} - \bar{\ln r}\) with the Welch standard error. The interval of the ratio is \(\exp(\ln\mathrm{GMR} \pm t\,\mathrm{se})\).

**2x2 crossover.** With the log values \(y_{i1}\), \(y_{i2}\) of subject \(i\) in the two periods, the period differences \(d_i = (y_{i2} - y_{i1})/2\) and the totals \(u_i = y_{i1} + y_{i2}\), and the sequences A (test in period 2) and B (test in period 1)[^chow]:

\[\hat F = \bar d_A - \bar d_B, \qquad \hat P = \bar d_A + \bar d_B, \qquad \hat C = \bar u_A - \bar u_B,\]

with \(\mathrm{var}(\hat F) = \mathrm{var}(\hat P) = \sigma_d^2 (1/n_A + 1/n_B)\) from the pooled within-sequence variance \(\sigma_d^2\) with \(n_A + n_B - 2\) degrees of freedom, and \(\mathrm{var}(\hat C)\) from the pooled variance of the totals. \(\hat F\) is the treatment effect of the analysis of variance with sequence, period and subject-within-sequence effects, its residual variance is \(\sigma_e^2 = 2\sigma_d^2\), and the within-subject CV is \(\sqrt{e^{\sigma_e^2} - 1}\).

**Two one-sided tests.** For the limits \(\theta_L < 1 < \theta_U\), \(t_L = (\ln\mathrm{GMR} - \ln\theta_L)/\mathrm{se}\) and \(t_U = (\ln\theta_U - \ln\mathrm{GMR})/\mathrm{se}\) are tested one-sided at \(\alpha\); rejecting both is the same as the \(1 - 2\alpha\) interval lying within the limits[^schuirmann].

**Multiple comparisons.** Bonferroni \(\tilde p_i = \min(1, m p_i)\); Holm sorts the p values and takes \(\tilde p_{(i)} = \max_{j \le i}\min(1, (m-j+1)p_{(j)})\)[^holm]; Benjamini-Hochberg takes \(\tilde p_{(i)} = \min_{j \ge i}\min(1, m p_{(j)}/j)\)[^bh].

**Effect sizes of a study.** Hedges' g as above with \(\mathrm{var}(d) = N/(n_C n_T) + d^2/(2N)\) and \(\mathrm{var}(g) = J^2\mathrm{var}(d)\); the mean difference \(\bar x_T - \bar x_C\) with \(s_T^2/n_T + s_C^2/n_C\); the log ratio \(\mu_T - \mu_C\) with \(\sigma_T^2/n_T + \sigma_C^2/n_C\).

**Pooling.** With \(w_i = 1/v_i\): \(\hat\theta_F = \sum w_i\theta_i/\sum w_i\), \(\mathrm{se} = 1/\sqrt{\sum w_i}\). Heterogeneity \(Q = \sum w_i(\theta_i - \hat\theta_F)^2\), \(C = \sum w_i - \sum w_i^2/\sum w_i\), \(\tau^2 = \max(0, (Q - (k-1))/C)\), \(I^2 = \max(0, (Q-(k-1))/Q)\), \(H^2 = Q/(k-1)\). Random effects: \(w_i^* = 1/(v_i + \tau^2)\) and the same pooling[^dl][^higgins]. The intervals of the effects and the pooled effects are normal, \(\hat\theta \pm z_{1-\alpha/2}\,\mathrm{se}\).

## Results

| function | result | fields |
| --- | --- | --- |
| `summarize` | `Summary` | `n`, `mean`, `sd`, `se`, `cv`, `geomean`, `geocv`, `median`, `q25`, `q75`, `min`, `max`, `ci_low`, `ci_high` |
| `compare` | `TestResult` | `test`, `statistic`, `p_value`, `effect`, `ci_low`, `ci_high`, `df`, `cohen_d`, `hedges_g`, `n_a`, `n_b` |
| `ratio` | `RatioResult` | `gmr`, `ci_low`, `ci_high`, `log_ratio`, `se_log`, `df`, `paired`, `n_test`, `n_reference` |
| `tost`, `bioequivalence` | `BEParameter`, `BEResult` | `gmr`, `ci_low`, `ci_high`, `bioequivalent`, `p_lower`, `p_upper`, `p_value`, `design`, `cv_intra`, `p_period`, `p_sequence` |
| `ddi_classification` | `DDIResult` | `kind`, `strength`, `auc_ratio`, `cmax_ratio`, `ci_low`, `ci_high`, `uncertain`, `kind_low`, `strength_low`, `kind_high`, `strength_high` |
| `effect_size` | `EffectSize` | `estimate`, `variance`, `se`, `ci_low`, `ci_high`, `kind`, `n_control`, `n_treatment`, `label` |
| `fixed_effect`, `random_effects` | `PooledEffect` | `estimate`, `se`, `ci_low`, `ci_high`, `z`, `p_value`, `weights`, `model`, `tau2` |
| `heterogeneity` | `Heterogeneity` | `q`, `df`, `p_value`, `i2`, `h2`, `tau2` |
| `meta_analysis` | `MetaResult` | `effects`, `fixed`, `random`, `heterogeneity`, `to_dataframe()` |

The `effect` of `compare` is `a - b` on the linear scale and the ratio of the geometric means `a / b` on the log scale; `ratio`, `tost` and `ddi_classification` report test over reference and with over without the perpetrator. `p_value` of a `BEParameter` is the larger of the two one-sided p values, `bioequivalent` is `p_value < (1 - ci_level) / 2`, which is the interval within the limits.

## API

A sample from a result or from numbers:

```python
from pkpdutils import ParameterSample
from pkpdutils.stats import Scale, summarize

auc = result.sample("auc_inf_obs", "individual")  # individual values with labels and coordinates
published = ParameterSample(mean=45.2, sd=12.1, n=12, name="cl", unit="l/hr")  # summary data
geometric = ParameterSample(geomean=42.0, geocv=0.28, n=12)
summarize(auc)  # geometric mean with its interval, quantiles, CV
summarize(auc, scale=Scale.LINEAR).ci_low
```

Compare two groups:

```python
from pkpdutils import compare
from pkpdutils.stats import Alternative, TestMethod, multiple_comparison

smokers = result.sample("cl", "individual", group="smokers")
non_smokers = result.sample("cl", "individual", group="non-smokers")
test = compare(smokers, non_smokers)  # Welch t on the log scale, effect = ratio of geometric means
test.p_value, test.effect, (test.ci_low, test.ci_high), test.hedges_g
compare(smokers, non_smokers, test=TestMethod.MANN_WHITNEY)
compare(before, after, paired=True, alternative=Alternative.LESS)
compare(published_a, published_b)  # Welch t from mean, sd, n
multiple_comparison([t.p_value for t in tests])  # Holm
```

Ratios and bioequivalence:

```python
from pkpdutils import bioequivalence, ratio

r = ratio(test_auc, reference_auc)  # paired by label when both carry the same subjects, 90 % interval
be = bioequivalence(test_result, reference_result, parameters=["auc_inf_obs", "cmax"])
be.bioequivalent, be["cmax"].gmr, be.to_dataframe()
```

A 2x2 crossover is recognized from the coordinates `period` (1 or 2) and `sequence` along the individual dimension of both batches; they are given to `Timecourses.from_arrays` as `coords={"individual": ids, "period": ("individual", periods), "sequence": ("individual", sequences)}` and travel through the NCA to the result. Without them two results with the same individuals are paired, otherwise the groups are parallel; `design=Design.PARALLEL` overrides the detection.

Drug-drug interactions:

```python
from pkpdutils import ddi_classification
from pkpdutils.stats import DDIThresholds, substrate_sensitivity

ddi = ddi_classification(ratio(inhibited_auc, control_auc), cmax_ratio=ratio(inhibited_cmax, control_cmax))
ddi.kind, ddi.strength, ddi.uncertain
ddi_classification(3.2, ci=(2.4, 4.3), thresholds=DDIThresholds.ema())
substrate_sensitivity(6.1)
```

Meta-analysis:

```python
from pkpdutils import meta_analysis
from pkpdutils.stats import EffectKind, Study, effects_from_arrays, random_effects

studies = [Study("Smith 1990", control=ParameterSample(mean=1.2, sd=0.4, n=10), treatment=ParameterSample(mean=2.0, sd=0.6, n=10))]
meta = meta_analysis(studies, EffectKind.LOG_RATIO)
meta.random.estimate, meta.heterogeneity.i2, meta.to_dataframe()
random_effects(effects_from_arrays(log_ratios, variances, labels))  # effects computed elsewhere
```

Figures: `plot_parameters`, `plot_ratio` and `plot_forest`, see [Plotting](plotting.md). Examples: `examples/bioequivalence.py`, `examples/ddi.py` and `examples/meta_analysis.py`. The reference of the modules is in [API: stats](api/stats.md), [API: stats.bioequivalence](api/stats.bioequivalence.md), [API: stats.ddi](api/stats.ddi.md) and [API: stats.meta](api/stats.meta.md).

## References

[^fda_be]: U.S. Food and Drug Administration. *Statistical Approaches to Establishing Bioequivalence.* 2001. See [References](references.md#regulatory-guidance).
[^schuirmann]: Schuirmann DJ. *J Pharmacokinet Biopharm.* 1987;15:657-680. See [References](references.md#statistics).
[^fda_ddi]: U.S. Food and Drug Administration. *Clinical Drug Interaction Studies.* 2020. See [References](references.md#regulatory-guidance).
[^ema_ddi]: European Medicines Agency. *Guideline on the investigation of drug interactions.* 2012. See [References](references.md#regulatory-guidance).
[^chow]: Chow SC, Liu JP. *Design and Analysis of Bioavailability and Bioequivalence Studies.* 3rd ed. 2009, ch. 3. See [References](references.md#statistics).
[^hedges]: Hedges LV. *J Educ Stat.* 1981;6:107-128. See [References](references.md#statistics).
[^dl]: DerSimonian R, Laird N. *Control Clin Trials.* 1986;7:177-188. See [References](references.md#statistics).
[^higgins]: Higgins JPT, Thompson SG. *Stat Med.* 2002;21:1539-1558. See [References](references.md#statistics).
[^holm]: Holm S. *Scand J Stat.* 1979;6:65-70. See [References](references.md#statistics).
[^bh]: Benjamini Y, Hochberg Y. *J R Stat Soc B.* 1995;57:289-300. See [References](references.md#statistics).
````

- [ ] **Step 4: Plotting, glossary, index, CLAUDE.md**

Insert in `docs/plotting.md` before `## Style`:
````markdown
## Parameters, ratios and forest plots

`plot_parameters` draws the individual values of a parameter of a result as jittered points with a box plot per group (`by` names a coordinate along the sample dimension) and the geometric mean with its interval. `plot_ratio` draws geometric mean ratios with their intervals on a logarithmic axis against the acceptance limits of bioequivalence or the thresholds of the interaction classes, from a dictionary of `ratio` results or a `bioequivalence` result. `plot_forest` is the forest plot of a `meta_analysis`: the effect of every study with its interval and a marker sized by its random effects weight, the pooled fixed and random effects as diamonds, the heterogeneity in the title. `plot_bland_altman` shows the agreement of the predictions of a fit with the data.

```python
from pkpdutils.plot import plot_bland_altman, plot_forest, plot_parameters, plot_ratio
from pkpdutils.stats import DDIThresholds

fig = plot_parameters(result, "auc_inf_obs", "individual", by="sex", log=True)
fig = plot_ratio(be)  # a BEResult with the 80-125 % limits
fig = plot_ratio({"auc": auc_ratio, "cmax": cmax_ratio}, limits=None, thresholds=DDIThresholds.fda())
fig = plot_forest(meta)
fig = plot_bland_altman(fits, log=True)
```

The images are written by `examples/bioequivalence.py`, `examples/ddi.py` and `examples/meta_analysis.py`.
````
(the inner code fence uses three backticks as everywhere in the docs; the outer fence of this plan is the only reason it is shown nested here).

Append to the table of `docs/glossary.md`:
```markdown
| `gmr`, `log_ratio`, `se_log` | \(\mathrm{GMR}\) | geometric mean ratio test / reference, its logarithm and the standard error of the logarithm | –, – | [Statistics](statistics.md) |
| `effect`, `cohen_d`, `hedges_g` | \(d\), \(g\) | effect of a comparison and the standardized effect sizes | unit of the parameter (ratio: –), – | [Statistics](statistics.md) |
| `bioequivalent`, `p_lower`, `p_upper`, `cv_intra`, `p_period`, `p_sequence` | | verdict and the two one-sided p values of the bioequivalence test, within-subject CV, period and carryover p values of a crossover | – | [Statistics](statistics.md) |
| `kind`, `strength`, `uncertain` | | class of an interaction (inhibitor, inducer), its strength and whether the interval spans a boundary | – | [Statistics](statistics.md) |
| `estimate`, `variance`, `weight_fixed`, `weight_random` | \(\theta_i\), \(v_i\), \(w_i\) | effect of a study, its variance and its normalized weights in the pooling | – | [Statistics](statistics.md) |
| `q`, `i2`, `h2`, `tau2` | \(Q\), \(I^2\), \(H^2\), \(\tau^2\) | heterogeneity statistics of a meta-analysis | –, %, –, – | [Statistics](statistics.md) |
```

In `docs/index.md` add the feature bullet after "Pharmacodynamics": `- **[Statistics](statistics.md)** - significance tests, geometric mean ratios, bioequivalence, drug-drug interaction classification and meta-analysis on the parameters of groups and studies.` and extend the "Plotting" bullet with "parameter distributions, ratio and forest plots". The README already lists the statistics and the figures.

In `CLAUDE.md` add after the `fit/` paragraph of the architecture:
```markdown
**`stats/` - statistics on parameters.** `sample.py` holds `ParameterSample` (individual values with `labels` and `coords`, or summary `mean`/`sd`/`n`, `geomean`/`geocv`; `log_moments` translates summary data with the log-normal relations), `Scale`, `Summary` and `summarize`; `ParameterResult.sample(name, dim, **indexers)` builds one from a result (the non-dimension coordinates of a batch travel through `nca` and `fit_timecourses` via `result.sample_coordinates`). `tests.py` has `compare` (scipy t, rank and permutation tests, Welch from moments for summary data, `TestResult` with the effect, its t interval, Cohen's d and Hedges' g) and `multiple_comparison` (Holm, Bonferroni, BH); `ratio.py` the geometric mean ratio with a t interval (paired by label or Welch); `bioequivalence.py` `tost` and `bioequivalence` (TOST; `Design` detected from the samples: 2x2 crossover from the `period`/`sequence` coordinates as the period-difference analysis of Chow & Liu, paired, parallel); `ddi.py` `DDIThresholds` (FDA 2020 and EMA 2012, identical values) with `ddi_classification` (conservative bound of an interval, `uncertain`) and `substrate_sensitivity`; `meta.py` `effect_size` (Hedges' g, mean difference, log ratio), `fixed_effect`, `random_effects` (DerSimonian-Laird), `heterogeneity`, `meta_analysis` over `Study` objects and `effects_from_arrays`. Results are frozen dataclasses. `tests/data/reference/meta_bcg.json` holds the BCG trials with the metafor results as the regression reference.
```
Extend the `plot/` paragraph with `parameters.py` (`plot_parameters`), `ratio.py` (`plot_ratio`), `meta.py` (`plot_forest`) and `plot_bland_altman` in `fit.py`, and the Project paragraph with nothing (it already lists the statistics). Add `python -m examples.bioequivalence`, `python -m examples.ddi`, `python -m examples.meta_analysis` to the commands.

- [ ] **Step 5: Build the docs**

Run: `uv run zensical build --clean 2>&1 | grep -i -E "warning|error"` → no output. Run `uv run python scripts/llms_txt.py` → OK. Open `site/statistics/index.html` and `site/api/stats/index.html` in a text dump (`grep -c "ParameterSample" site/api/stats/index.html` > 0) to see that the formulas and the API render. Check `git status` shows no changes in `site/` (gitignored).

- [ ] **Step 6: Commit**

```bash
git add docs zensical.toml CLAUDE.md
git commit -q -m "Document the statistics on parameters

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 10: Matrix and pull request

- [ ] **Step 1: Full checks**

Run: `uv run ruff check && uv run ruff format --check && uv run tox run-parallel` → `py3.13`, `py3.14`, `ty` OK. Run `uv run pytest -q -W error` once more in the working tree → PASS, no warnings.

- [ ] **Step 2: Push and open the pull request**

```bash
git push -u origin statistics
gh pr create --base develop --title "Add the statistics on parameters" --body-file - <<'EOF'
## Summary

Phase 6 of the pkpdutils redesign (`docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`, section 4): `pkpdutils.stats` with `ParameterSample` (individual values or summary statistics, log-normal moment relations), `ParameterResult.sample`, `compare` (t, rank and permutation tests, Welch from moments, effect sizes) and `multiple_comparison`, `ratio` (geometric mean ratio), `bioequivalence`/`tost` (two one-sided tests; parallel, paired and 2x2 crossover designs with period and carryover tests), `ddi_classification` (FDA 2020 / EMA 2012 thresholds, conservative bound of an interval) and `substrate_sensitivity`, the meta-analysis (`effect_size`, `fixed_effect`, `random_effects`, `heterogeneity`, `meta_analysis`) with the BCG trials of metafor as regression reference, the figures `plot_parameters`, `plot_ratio`, `plot_forest`, `plot_bland_altman`, the sample coordinates of a batch on the NCA and fit results, the documentation page `statistics.md` and three examples.

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

## Deviations from the spec, for the release notes

- `compare` returns `ci_low`/`ci_high` instead of a `ci` tuple, and reports `cohen_d`/`hedges_g` with every test; the rank tests report the difference or ratio of the medians as the effect without an interval.
- `bioequivalence` takes two `ParameterResult` objects and a `dim` (the spec left the input open); `tost` is the per-parameter function on samples. The crossover analysis is the period-difference method of Chow & Liu, which equals the ANOVA with sequence, period and subject effects (verified against an OLS with subject dummies in the tests); no Chow & Liu textbook fixture, the reference is the OLS.
- `ddi_classification` reports `cmax_ratio` but classifies on `auc_ratio` only, as the guidances do.
- `meta_analysis` takes `Study` objects; `meta_analysis_by` groups by `Study.category` (the spec's `by="category"`). The variance of Hedges' g is \(J^2 \mathrm{var}(d)\) (Hedges 1981), where the old `effect_analysis.py` reported \(\mathrm{var}(d)\) for g; the fixture is the metafor BCG example, not `esc` output.
- `summarize` of `pkpdutils.stats` is a function returning a `Summary`; `ParameterResult.summarize(dim)` (plan 3) is unchanged.
- `plot_fit` has no bootstrap band (plan 4); `plot_bland_altman` lives in `plot/fit.py`.

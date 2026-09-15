"""Tests of the fitting engine and the fit result."""

from concurrent.futures.process import BrokenProcessPool
from typing import Any

import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils.fit import FitOptions, FitResult, ParameterScale, Weighting, engine, fit
from pkpdutils.fit.engine import (
    fit_row,
    fit_rows,
    from_scale,
    scale_derivative,
    to_scale,
    variance_of,
)
from pkpdutils.fit.model import Model, ModelParameter
from pkpdutils.fit.models import Bateman, BiExp, Emax, Linear, MonoExp
from pkpdutils.parallel import ExecutorKind
from pkpdutils.units import ureg

T = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24])
BIEXP_TRUTH = np.array([8.0, 2.0, 2.0, 0.2])


def noisy_monoexp(
    rng: np.random.Generator,
    a: float = 10.0,
    k: float = 0.3,
    cv: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """A monoexponential curve and a log-normally perturbed copy of it.

    Args:
        rng: random generator of the noise
        a: value at `x = 0`
        k: rate constant
        cv: coefficient of variation of the noise

    Returns:
        The exact curve and the noisy curve at `T`.
    """
    y = a * np.exp(-k * T)
    return y, y * rng.lognormal(0.0, cv, size=T.size)


def noisy_biexp(rng: np.random.Generator, cv: float = 0.02) -> np.ndarray:
    """A biexponential curve with the phases 2 and 0.2, log-normally perturbed.

    Args:
        rng: random generator of the noise
        cv: coefficient of variation of the noise

    Returns:
        The noisy curve at `T`.
    """
    return BiExp().predict(T, BIEXP_TRUTH) * rng.lognormal(0.0, cv, T.size)


def test_scale_helpers() -> None:
    """The scale transformations and their derivatives."""
    p = np.array([10.0, -2.0, 0.5])
    scales = [ParameterScale.LOG10, ParameterScale.LINEAR, ParameterScale.LOG]
    q = to_scale(p, scales)
    np.testing.assert_allclose(q, [1.0, -2.0, np.log(0.5)])
    np.testing.assert_allclose(from_scale(q, scales), p)
    np.testing.assert_allclose(
        scale_derivative(p, scales), [10.0 * np.log(10.0), 1.0, 0.5]
    )


def test_variance_of() -> None:
    """The variance models of the weighting."""
    y = np.array([1.0, 4.0, 0.0])
    np.testing.assert_allclose(variance_of(y, None, Weighting.NONE), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        variance_of(y, None, Weighting.INV_Y2), [1.0, 16.0, 1.0]
    )  # y = 0 -> smallest positive
    np.testing.assert_allclose(variance_of(y, None, Weighting.INV_Y), [1.0, 4.0, 1.0])
    np.testing.assert_allclose(
        variance_of(y, np.array([0.1, 0.2, 0.3]), Weighting.INV_SD), [0.01, 0.04, 0.09]
    )
    with pytest.raises(ValueError, match="sd"):
        variance_of(y, None, Weighting.INV_SD)


def test_monoexp_recovery_and_statistics() -> None:
    """A monoexponential fit recovers the parameters and reports the statistics."""
    _, y = noisy_monoexp(np.random.default_rng(0))
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
    assert (
        np.isfinite(q["aic"].magnitude)
        and q["aicc"].magnitude > q["aic"].magnitude
        and np.isfinite(q["bic"].magnitude)
    )
    assert result.flags() == []
    assert result.point_variables == [
        "x_data",
        "y_data",
        "sd_data",
        "y_pred",
        "residuals",
        "correlation",
    ]
    np.testing.assert_allclose(result["y_pred"].values, result.predict(T), rtol=1e-12)
    corr = result.correlation()
    assert (
        corr.shape == (2, 2)
        and corr.loc["a", "a"] == pytest.approx(1.0)
        and abs(corr.loc["a", "k"]) < 1.0
    )
    np.testing.assert_allclose(
        result.parameter_vector(), [q["a"].magnitude, q["k"].magnitude]
    )


def test_exact_data_gives_tiny_uncertainty_and_t_interval() -> None:
    """Almost exact data gives a tiny standard error and a symmetric t interval."""
    truth, _ = noisy_monoexp(np.random.default_rng(1))
    result = fit(
        MonoExp(),
        T,
        truth * (1 + 1e-6 * np.sin(T)),
        options=FitOptions(parameter_scale=ParameterScale.LINEAR),
    )
    q = result.to_quantities()
    assert q["k_se"].magnitude < 1e-4
    tq = student_t.ppf(0.975, T.size - 2)
    assert q["k_ci_high"].magnitude == pytest.approx(
        q["k"].magnitude + tq * q["k_se"].magnitude, rel=1e-6
    )


def test_log_scale_interval_is_asymmetric() -> None:
    """On the log scale the interval is symmetric in the logarithm."""
    _, y = noisy_monoexp(np.random.default_rng(2), cv=0.2)
    q = fit(MonoExp(), T, y).to_quantities()
    k, low, high = q["k"].magnitude, q["k_ci_low"].magnitude, q["k_ci_high"].magnitude
    assert np.log(k / low) == pytest.approx(np.log(high / k), rel=1e-6)


def test_pooled_fit_matches_serial() -> None:
    """The rows of a batch fit give the same result in the pool as in the process."""
    rng = np.random.default_rng(17)
    n_rows = 300
    a = rng.uniform(8.0, 12.0, n_rows)[:, None]
    k = rng.uniform(0.2, 0.4, n_rows)[:, None]
    y = a * np.exp(-k * T[None, :]) * rng.normal(1.0, 0.02, (n_rows, T.size))
    x = np.broadcast_to(T, y.shape)
    serial = fit(MonoExp(), x, y, options=FitOptions(seed=11, n_workers=1))
    pooled = fit(MonoExp(), x, y, options=FitOptions(seed=11, n_workers=2))
    assert set(pooled.ds.data_vars) == set(serial.ds.data_vars)
    for variable in serial.ds.data_vars:
        name = str(variable)
        np.testing.assert_allclose(
            pooled[name].values, serial[name].values, equal_nan=True, err_msg=name
        )


def test_pooled_fit_retries_a_broken_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that dies breaks the shared pool; the batch is fitted in a fresh one."""
    rng = np.random.default_rng(5)
    y = np.stack([noisy_monoexp(rng)[1] for _ in range(4)])
    x = np.broadcast_to(T, y.shape).copy()
    options = FitOptions(seed=3, n_workers=2)
    serial = fit_rows(
        MonoExp(), x, y, None, options.model_copy(update={"n_workers": 1})
    )

    class BrokenPool:
        def map(self, fn: Any, rows: Any, chunksize: int = 1) -> Any:
            raise BrokenProcessPool("a worker of the pool died")

    real = engine.executor
    calls: list[tuple[str, int]] = []

    def fake_executor(kind: ExecutorKind, n_workers: int) -> Any:
        calls.append((kind, n_workers))
        if len(calls) == 1:
            return BrokenPool()
        return real(kind, n_workers)

    evicted: list[tuple[str, int]] = []
    monkeypatch.setattr(engine, "executor", fake_executor)
    monkeypatch.setattr(
        engine, "evict", lambda kind, n_workers: evicted.append((kind, n_workers))
    )
    rows = fit_rows(MonoExp(), x, y, None, options)
    assert calls == [("process", 2), ("process", 2)]
    assert evicted == [("process", 2)]
    for got, want in zip(rows, serial, strict=True):
        np.testing.assert_allclose(got.p, want.p)


def test_batch_rows_dims_and_nan_padding() -> None:
    """Many rows with sample dimensions, coordinates and NaN padded points."""
    ys = np.stack(
        [
            noisy_monoexp(np.random.default_rng(i), k=k)[1]
            for i, k in enumerate((0.2, 0.3, 0.5))
        ]
    )
    ys[2, -2:] = np.nan
    result = fit(
        MonoExp(),
        T,
        ys,
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        x_unit="hr",
        y_unit="mg/l",
    )
    assert result.sample_dims == ("individual",)
    np.testing.assert_allclose(result["k"].values, [0.2, 0.3, 0.5], rtol=0.15)
    assert result["n_points"].values[2] == T.size - 2
    assert result["y_pred"].dims == ("individual", "point")
    assert np.isnan(result["residuals"].values[2, -2:]).all()
    assert result["correlation"].dims == ("individual", "parameter", "parameter_")
    df = result.to_dataframe()
    assert (
        "k_se" in df.columns
        and "y_pred" not in df.columns
        and list(df.columns)[-1] == "flags"
    )
    # one row per sample, not one per (sample, point, parameter pair)
    assert len(df) == 3
    single = fit(MonoExp(), T, ys[1], x_unit="hr", y_unit="mg/l")
    assert single.to_quantities()["k"].magnitude == pytest.approx(result["k"].values[1])
    assert result.predict(T, individual="b").shape == T.shape
    assert result.predict_all(T).dims == ("individual", "x")


def test_weighting_inv_sd_and_fixed_and_bounds() -> None:
    """The `sd` weighting, a fixed parameter and a bound override."""
    _, y = noisy_monoexp(np.random.default_rng(6))
    sd = 0.05 * y
    weighted = fit(
        MonoExp(), T, y, sd=sd, options=FitOptions(weighting=Weighting.INV_SD)
    )
    assert weighted.flags() == []
    fixed = fit(MonoExp(), T, y, options=FitOptions(fixed={"k": 0.3}))
    q = fixed.to_quantities()
    assert (
        q["k"].magnitude == 0.3
        and np.isnan(q["k_se"].magnitude)
        and q["n_parameters"].magnitude == 1
    )
    bounded = fit(MonoExp(), T, y, options=FitOptions(bounds={"k": (0.5, 1.0)}))
    assert bounded.to_quantities()["k"].magnitude == pytest.approx(0.5, rel=1e-3)
    assert "AT_BOUND" in bounded.flags()


def test_too_few_points_and_no_data() -> None:
    """Too few points and an empty row are flagged without a fit."""
    r = fit(Emax(), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert "TOO_FEW_POINTS" in r.flags() and np.isnan(
        r.to_quantities()["ec50"].magnitude
    )
    r = fit(MonoExp(), T, np.full(T.size, np.nan))
    assert "NO_DATA" in r.flags()


def test_n_points_of_a_row_without_a_fit_counts_the_used_points() -> None:
    """`n_points` is the number of points the fit used, not the width of the row."""
    x = np.arange(1.0, 10.0)  # nine points
    empty = fit(MonoExp(), x, np.full(x.size, np.nan))
    assert int(empty["n_points"].values) == 0
    assert empty["y_data"].sizes["point"] == 9
    y = np.full(x.size, np.nan)
    y[:2] = [5.0, 4.0]
    too_few = fit(Emax(), x, y)
    assert "TOO_FEW_POINTS" in too_few.flags()
    assert int(too_few["n_points"].values) == 2
    assert too_few["y_data"].sizes["point"] == 9


def test_bateman_flip_flop_flag_and_derived() -> None:
    """The Bateman fit reports its derived parameters and the flip-flop."""
    y = Bateman().predict(T, np.array([10.0, 0.2, 0.6])) * np.random.default_rng(
        3
    ).lognormal(0, 0.02, T.size)
    result = fit(Bateman(), T, y, options=FitOptions(initial={"ka": 0.2, "ke": 0.6}))
    q = result.to_quantities()
    assert "FLIP_FLOP" in result.flags() or q["ka"].magnitude > q["ke"].magnitude
    assert "tmax" in result and "cmax" in result and "flip_flop" in result


def test_biexp_phase_ordering() -> None:
    """The phases of a sum of exponentials are ordered by decreasing rate."""
    y = noisy_biexp(np.random.default_rng(4))
    result = fit(
        BiExp(),
        T,
        y,
        options=FitOptions(initial={"a1": 2.0, "k1": 0.2, "a2": 8.0, "k2": 2.0}),
    )
    q = result.to_quantities()
    assert q["k1"].magnitude > q["k2"].magnitude
    assert q["lambda_z"].magnitude == pytest.approx(q["k2"].magnitude)


def test_linear_scale_for_non_positive_parameters() -> None:
    """Parameters which are not positive are fitted on the linear scale."""
    x = np.linspace(0, 10, 11)
    y = -3.0 + 0.5 * x + np.random.default_rng(5).normal(0, 0.1, x.size)
    q = fit(Linear(), x, y).to_quantities()
    assert q["intercept"].magnitude == pytest.approx(-3.0, abs=0.2)
    assert q["slope"].magnitude == pytest.approx(0.5, abs=0.05)


def test_fit_row_returns_rowfit() -> None:
    """`fit_row` returns the arrays of one row."""
    _, y = noisy_monoexp(np.random.default_rng(7))
    row = fit_row(MonoExp(), T, y, None, FitOptions(), np.random.default_rng(0))
    assert row.p.shape == (2,) and row.cov_q.shape == (2, 2) and row.flags == 0
    assert row.y_pred.shape == T.shape and row.n_starts_converged == 1


def test_fixed_phase_parameter_keeps_its_label() -> None:
    """A fixed phase parameter is not moved by the phase ordering."""
    y = noisy_biexp(np.random.default_rng(8))
    result = fit(
        BiExp(),
        T,
        y,
        options=FitOptions(
            fixed={"k1": 0.2}, initial={"a1": 2.0, "a2": 8.0, "k2": 2.0}
        ),
    )
    q = result.to_quantities()
    assert q["k1"].magnitude == 0.2
    assert np.isnan(q["k1_se"].magnitude)
    assert np.isfinite(q["k2_se"].magnitude) and q["k2_se"].magnitude > 0
    assert q["k2"].magnitude == pytest.approx(2.0, rel=0.1)
    # the phases keep the user's labelling, so they are not ordered here and
    # `lambda_z` is the slowest rate, not the rate of the last phase
    assert q["lambda_z"].magnitude == 0.2


def test_bounded_phase_parameter_flags_its_own_bound() -> None:
    """A bounded phase parameter keeps its label and its `AT_BOUND` flag."""
    y = noisy_biexp(np.random.default_rng(9))
    result = fit(
        BiExp(),
        T,
        y,
        options=FitOptions(
            bounds={"k1": (0.05, 0.19)},
            initial={"a1": 2.0, "k1": 0.1, "a2": 8.0, "k2": 2.0},
        ),
    )
    q = result.to_quantities()
    assert q["k1"].magnitude == pytest.approx(0.19, rel=1e-3)
    assert "AT_BOUND" in result.flags()


def test_phase_permutation_keeps_the_uncertainties_aligned() -> None:
    """A fit started with swapped phases matches one started in the right order."""
    y = noisy_biexp(np.random.default_rng(10))
    swapped = fit(
        BiExp(),
        T,
        y,
        options=FitOptions(initial={"a1": 2.0, "k1": 0.2, "a2": 8.0, "k2": 2.0}),
    )
    direct = fit(
        BiExp(),
        T,
        y,
        options=FitOptions(initial={"a1": 8.0, "k1": 2.0, "a2": 2.0, "k2": 0.2}),
    )
    assert float(swapped["k1"].values) > float(swapped["k2"].values)
    for name in ("a1", "k1", "a2", "k2"):
        assert float(swapped[name].values) == pytest.approx(
            float(direct[name].values), rel=1e-4
        )
        assert float(swapped[f"{name}_se"].values) == pytest.approx(
            float(direct[f"{name}_se"].values), rel=1e-3
        )
    np.testing.assert_allclose(
        swapped["correlation"].values, direct["correlation"].values, atol=1e-4
    )


def test_parameters_keep_the_raw_units_of_the_data() -> None:
    """Fit parameters are reported in the units of `x` and `y`, unnormalized."""
    _, y = noisy_monoexp(np.random.default_rng(11))
    result = fit(MonoExp(), T, y, x_unit="hr", y_unit="ml")
    np.testing.assert_allclose(result.predict(T), result["y_pred"].values, rtol=1e-12)
    assert result.units("a") == "milliliter"
    assert ureg(result.units("auc")) == ureg("ml*hr")


def test_cv_is_not_a_parameter() -> None:
    """The `_cv` variables are uncertainties, not parameters of their own."""
    ys = np.stack(
        [
            noisy_monoexp(np.random.default_rng(i), k=k)[1]
            for i, k in ((12, 0.2), (13, 0.4))
        ]
    )
    result = fit(MonoExp(), T, ys, dims=("individual",))
    assert [name for name in result.parameters if name.endswith("_cv")] == []
    assert "k_cv" in result.derived_variables
    summary = result.summarize("individual")
    assert "k_cv_sd" not in summary and "k_sd" in summary


def test_statistics_are_not_parameters() -> None:
    """The goodness-of-fit statistics and the counts are `statistics`, not `parameters`."""
    ys = np.stack(
        [
            noisy_monoexp(np.random.default_rng(i), k=k)[1]
            for i, k in ((16, 0.2), (17, 0.3), (18, 0.4))
        ]
    )
    result = fit(MonoExp(), T, ys, dims=("individual",))
    assert set(result.statistics) == {
        "cost",
        "r2",
        "rmse",
        "aic",
        "aicc",
        "bic",
        "n_points",
        "n_parameters",
        "n_starts_converged",
        "n_bootstrap",
    }
    assert not set(result.parameters) & set(result.statistics)
    assert result.parameters == ["a", "k", "thalf", "auc"]
    # they stay in the data frame, which reports the individual fits
    assert set(result.to_dataframe().columns) >= set(result.statistics)


def test_summarize_drops_the_statistics() -> None:
    """A summary over samples averages the parameters, never the statistics of the single fits."""
    ys = np.stack(
        [
            Bateman().predict(T, np.array([10.0, 1.5, k]))
            * np.random.default_rng(i).lognormal(0.0, 0.02, T.size)
            for i, k in ((19, 0.2), (20, 0.3), (21, 0.4))
        ]
    )
    summary = fit(Bateman(), T, ys, dims=("individual",)).summarize("individual")
    for name in ("aic", "cost", "n_points", "r2", "rmse", "aicc", "bic"):
        assert name not in summary
        for suffix in ("_se", "_sd", "_ci_high", "_q75", "_n"):
            assert f"{name}{suffix}" not in summary
    assert "tmax" in summary and "tmax_sd" in summary and "ka_ci_low" in summary
    assert float(summary["n"].values) == 3.0


def test_at_bound_at_a_zero_lower_bound() -> None:
    """A rate driven to its lower bound of zero is flagged, although that bound is -inf on the log scale."""
    result = fit(
        MonoExp(), T, np.full(T.size, 5.0), options=FitOptions(initial={"k": 0.1})
    )
    assert float(result["k"].values) < 1e-6
    assert "AT_BOUND" in result.flags()


def test_discrete_derived_parameters_carry_no_uncertainty() -> None:
    """An indicator such as `flip_flop` has no `_se`, `_ci_low`, `_ci_high`, `_cv`."""
    y = Bateman().predict(T, np.array([10.0, 0.2, 0.6])) * np.random.default_rng(
        14
    ).lognormal(0.0, 0.02, T.size)
    result = fit(Bateman(), T, y)
    assert "flip_flop" in result
    for suffix in ("_se", "_ci_low", "_ci_high", "_cv"):
        assert f"flip_flop{suffix}" not in result
    assert "tmax_se" in result


class ReservedSuffixModel(Model):
    """A model whose parameter name collides with the suffixes of the derived variables."""

    name = "reserved"
    parameters = (
        ModelParameter("a", "[y]", description="amplitude"),
        ModelParameter("k_n", "1/[x]", description="a rate named like a count"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(-p[1] * x)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([float(y[0]), 0.1])


def test_a_parameter_with_a_reserved_suffix_is_rejected() -> None:
    """`k_n` would read as the count of a parameter `k`, so the model is rejected."""
    _, y = noisy_monoexp(np.random.default_rng(22))
    with pytest.raises(ValueError, match=r"k_n.*reserved suffix '_n'"):
        fit(ReservedSuffixModel(), T, y)


def test_a_derived_parameter_with_a_reserved_suffix_is_rejected() -> None:
    """The derived parameters are checked like the fitted ones."""

    class DerivedSuffixModel(ReservedSuffixModel):
        name = "reserved_derived"
        parameters = (ModelParameter("a", "[y]", description="amplitude"),)
        derived_units = {"a_se": "[y]"}  # noqa: RUF012

        def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
            return np.full_like(x, p[0])

        def derived(self, p: np.ndarray) -> dict[str, float]:
            return {"a_se": float(p[0])}

        def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.array([float(y[0])])

    _, y = noisy_monoexp(np.random.default_rng(23))
    with pytest.raises(ValueError, match=r"a_se.*reserved suffix '_se'"):
        fit(DerivedSuffixModel(), T, y)


def test_fit_validates_the_sample_dimensions_before_fitting() -> None:
    """Several sample dimensions are rejected before any row is fitted."""
    ys = np.zeros((2, T.size))
    with pytest.raises(ValueError, match="one sample dimension"):
        fit(MonoExp(), T, ys, dims=("a", "b"))


def test_fit_rejects_a_sample_dimension_that_collides_with_a_result_variable() -> None:
    """A sample dimension named like a parameter or a reserved dimension is rejected."""
    _, y = noisy_monoexp(np.random.default_rng(15))
    with pytest.raises(ValueError, match="collides"):
        fit(MonoExp(), T, y[None, :], dims=("k",))
    with pytest.raises(ValueError, match="collides"):
        fit(MonoExp(), T, y[None, :], dims=("parameter",))

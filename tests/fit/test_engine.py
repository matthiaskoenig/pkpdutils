"""Tests of the fitting engine and the fit result."""

import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils.fit import FitOptions, FitResult, ParameterScale, Weighting, fit
from pkpdutils.fit.engine import (
    fit_row,
    from_scale,
    scale_derivative,
    to_scale,
    variance_of,
)
from pkpdutils.fit.models import Bateman, BiExp, Emax, Linear, MonoExp

T = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24])
RNG = np.random.default_rng(0)


def noisy_monoexp(
    a: float = 10.0,
    k: float = 0.3,
    cv: float = 0.05,
    rng: np.random.Generator = RNG,
) -> tuple[np.ndarray, np.ndarray]:
    """A monoexponential curve and a log-normally perturbed copy of it.

    Args:
        a: value at `x = 0`
        k: rate constant
        cv: coefficient of variation of the noise
        rng: random generator of the noise

    Returns:
        The exact curve and the noisy curve at `T`.
    """
    y = a * np.exp(-k * T)
    return y, y * rng.lognormal(0.0, cv, size=T.size)


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
    _, y = noisy_monoexp()
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
    truth, _ = noisy_monoexp()
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
    _, y = noisy_monoexp(cv=0.2)
    q = fit(MonoExp(), T, y).to_quantities()
    k, low, high = q["k"].magnitude, q["k_ci_low"].magnitude, q["k_ci_high"].magnitude
    assert np.log(k / low) == pytest.approx(np.log(high / k), rel=1e-6)


def test_batch_rows_dims_and_nan_padding() -> None:
    """Many rows with sample dimensions, coordinates and NaN padded points."""
    ys = np.stack(
        [
            noisy_monoexp(k=k, rng=np.random.default_rng(i))[1]
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
    single = fit(MonoExp(), T, ys[1], x_unit="hr", y_unit="mg/l")
    assert single.to_quantities()["k"].magnitude == pytest.approx(result["k"].values[1])
    assert result.predict(T, individual="b").shape == T.shape
    assert result.predict_all(T).dims == ("individual", "x")


def test_weighting_inv_sd_and_fixed_and_bounds() -> None:
    """The `sd` weighting, a fixed parameter and a bound override."""
    _, y = noisy_monoexp()
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
    y = BiExp().predict(T, np.array([8.0, 2.0, 2.0, 0.2])) * np.random.default_rng(
        4
    ).lognormal(0, 0.02, T.size)
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
    _, y = noisy_monoexp()
    row = fit_row(MonoExp(), T, y, None, FitOptions(), np.random.default_rng(0))
    assert row.p.shape == (2,) and row.cov_q.shape == (2, 2) and row.flags == 0
    assert row.y_pred.shape == T.shape and row.n_starts_converged == 1

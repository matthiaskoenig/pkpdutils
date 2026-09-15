"""Tests of multi-start robustness, the worker pool, the bootstrap and model comparison."""

import numpy as np
import pytest

from pkpdutils.fit import FitFlag, FitOptions, compare_models, fit
from pkpdutils.fit.engine import fit_row
from pkpdutils.fit.models import BiExp, Emax, Linear, MonoExp, SigmoidEmax

T = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24])


def data(seed: int = 0, cv: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 10.0 * np.exp(-0.3 * T) * rng.lognormal(0.0, cv, size=T.size)


def test_multistart_recovers_from_a_bad_start() -> None:
    y = BiExp().predict(T, np.array([8.0, 2.0, 2.0, 0.2])) * np.random.default_rng(
        1
    ).lognormal(0, 0.02, T.size)
    bad = FitOptions(
        initial={"a1": 0.01, "k1": 30.0, "a2": 0.01, "k2": 10.0},
        n_starts=1,
        max_nfev=50,
    )
    good = FitOptions(
        initial={"a1": 0.01, "k1": 30.0, "a2": 0.01, "k2": 10.0}, n_starts=20, seed=0
    )
    single = fit(BiExp(), T, y, options=bad).to_quantities()
    multi = fit(BiExp(), T, y, options=good)
    q = multi.to_quantities()
    assert q["cost"].magnitude <= single["cost"].magnitude
    assert q["k2"].magnitude == pytest.approx(0.2, rel=0.1)
    assert q["n_starts_converged"].magnitude >= 1


def test_workers_equal_serial() -> None:
    ys = np.stack([data(seed=i) for i in range(4)])
    serial = fit(MonoExp(), T, ys, options=FitOptions(n_starts=3, seed=5))
    parallel = fit(
        MonoExp(), T, ys, options=FitOptions(n_starts=3, seed=5, n_workers=2)
    )
    for name in serial.parameters:
        np.testing.assert_allclose(
            parallel[name].values, serial[name].values, equal_nan=True
        )
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
    assert list(table.columns) == [
        "model",
        "n_parameters",
        "aicc",
        "delta_aicc",
        "akaike_weight",
        "best",
    ]
    assert table["akaike_weight"].sum() == pytest.approx(1.0)
    assert str(comparison.best.values) == "monoexp"
    assert table.loc[table["best"], "model"].item() == "monoexp"
    assert table.loc[table["model"] == "monoexp", "delta_aicc"].item() == 0.0


def test_compare_models_batch() -> None:
    ys = np.stack([data(seed=i) for i in range(3)])
    comparison = compare_models(
        [MonoExp(), BiExp()],
        T,
        ys,
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
    )
    assert comparison.best.dims == ("individual",)
    assert next(iter(comparison.table.columns)) == "individual"
    assert len(comparison.table) == 6
    weights = comparison.table.groupby("individual")["akaike_weight"].sum()
    np.testing.assert_allclose(weights.values, 1.0)


def test_compare_models_with_a_failed_model() -> None:
    c = np.array([0.5, 1, 2, 5, 10, 20, 50])
    e = Emax().predict(c, np.array([1.0, 8.0, 5.0]))
    comparison = compare_models([Emax(), SigmoidEmax()], c, e)
    assert set(comparison.results) == {"emax", "sigmoid_emax"}
    assert comparison.table["akaike_weight"].sum() == pytest.approx(1.0)


def test_bootstrap_residuals_are_centered_and_inflated() -> None:
    """The bootstrap `slope_se` of a linear fit with known gaussian noise agrees with the Jacobian one.

    Efron & Tibshirani 1993, ch. 9: before resampling, the weighted residuals
    are centered and inflated by `sqrt(n / (n - k))` so their variance
    matches the residual variance of the fit (`n = 25` points, `k = 2` free
    parameters here).
    """
    rng = np.random.default_rng(21)
    x = np.linspace(0.0, 10.0, 25)
    y = 2.0 + 3.0 * x + rng.normal(0.0, 1.0, size=x.size)
    jac = fit(Linear(), x, y).to_quantities()
    boot = fit(
        Linear(), x, y, options=FitOptions(bootstrap=400, seed=22)
    ).to_quantities()
    assert boot["slope_se"].magnitude == pytest.approx(
        jac["slope_se"].magnitude, rel=0.15
    )


def test_bootstrap_fallback_flag_with_a_single_replicate() -> None:
    """`bootstrap=1` can never give 2 replicates, so the Jacobian uncertainties are kept and flagged."""
    y = data(seed=2)
    jac = fit(MonoExp(), T, y).to_quantities()
    result = fit(MonoExp(), T, y, options=FitOptions(bootstrap=1, seed=3))
    q = result.to_quantities()
    assert q["k_se"].magnitude == jac["k_se"].magnitude
    assert "BOOTSTRAP_FALLBACK" in result.flags()


def test_aic_aicc_bic_count_the_residual_variance_as_a_parameter() -> None:
    """AIC/AICc/BIC use `K = k + 1` estimated parameters (Burnham & Anderson 2002, sec. 2.2, 6.9.6)."""
    y = data(seed=0)
    result = fit(MonoExp(), T, y)
    q = result.to_quantities()
    n = int(q["n_points"].magnitude)
    k = int(q["n_parameters"].magnitude)
    big_k = k + 1
    cost = q["cost"].magnitude
    ln_term = n * np.log(2.0 * cost / n)
    expected_aic = ln_term + 2 * big_k
    expected_bic = ln_term + big_k * np.log(n)
    expected_aicc = expected_aic + 2 * big_k * (big_k + 1) / (n - big_k - 1)
    assert q["aic"].magnitude == pytest.approx(expected_aic)
    assert q["bic"].magnitude == pytest.approx(expected_bic)
    assert q["aicc"].magnitude == pytest.approx(expected_aicc)
    assert q["n_parameters"].magnitude == k


def test_from_scale_overflow_fails_the_start_without_a_warning() -> None:
    """A start giving a parameter at the `float64` maximum overflows the `log10` round trip.

    `from_scale` must not let this escape as a `RuntimeWarning` under
    `-W error`: the initial residual evaluation is then not finite and the
    start fails cleanly (`NOT_CONVERGED`) instead of raising.
    """
    value = float(np.finfo(np.float64).max)
    options = FitOptions(
        initial={"a1": value, "k1": value, "a2": value, "k2": value}, n_starts=1
    )
    y = BiExp().predict(T, np.array([8.0, 2.0, 2.0, 0.2]))
    row = fit_row(BiExp(), T, y, None, options, np.random.default_rng(0))
    assert row.flags & int(FitFlag.NOT_CONVERGED)

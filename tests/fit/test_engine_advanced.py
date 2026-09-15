"""Tests of multi-start robustness, the worker pool, the bootstrap and model comparison."""

import numpy as np
import pytest

from pkpdutils.fit import FitOptions, compare_models, fit
from pkpdutils.fit.models import BiExp, Emax, MonoExp, SigmoidEmax

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

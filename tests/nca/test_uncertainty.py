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
        time=T,
        value=c,
        sd=cv * c,
        n=n,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
        substance="x",
        label=label,
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
    np.testing.assert_allclose(
        resolve_spread(no_n, NCAOptions(bootstrap_spread=BootstrapSpread.SD)),
        0.1 * C0 * np.exp(-K * T)[None, :],
    )


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
    reps = {
        "auc_last": np.exp(rng.normal(np.log(100.0), 0.1, size=(2, 5000))),
        "tmax": np.full((2, 5000), 2.0),
        "flags": np.zeros((2, 5000)),
    }
    point = {
        "auc_last": np.array([100.0, 100.0]),
        "tmax": np.array([2.0, 2.0]),
        "flags": np.zeros(2),
    }
    out = reduce_replicates(
        reps,
        point,
        spread_kind=BootstrapSpread.SE,
        n_subjects=np.array([4.0, np.nan]),
        ci_level=0.95,
    )
    assert set(out) == {
        "auc_last_sd",
        "auc_last_se",
        "auc_last_ci_low",
        "auc_last_ci_high",
        "auc_last_geomean",
        "auc_last_geocv",
    }
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
    assert "auc_inf_obs_se" in a and "auc_inf_obs_ci_low" in a
    assert "auc_inf_obs_geocv" in a
    assert "tmax_se" not in a  # discrete
    assert "lambda_z_r2_se" not in a  # regression diagnostic
    assert "auc_last" in a.parameters and "auc_last_se" not in a.parameters
    assert "auc_last_se" in a.derived_variables
    for name in a.derived_variables:
        np.testing.assert_allclose(a[name].values, b[name].values, equal_nan=True)
    q = a.to_quantities()
    plain = nca_single(
        group_curve(),
        NCAOptions(uncertainty=UncertaintyMethod.NONE, auc_method=AUCMethod.LOG),
    )
    assert q["auc_inf_obs"].magnitude == pytest.approx(
        plain.to_quantities()["auc_inf_obs"].magnitude
    )
    assert str(q["auc_inf_obs_se"].units) == str(q["auc_inf_obs"].units)
    assert str(q["auc_inf_obs_geocv"].units) == "dimensionless"
    assert q["n"].magnitude == 12
    assert (
        q["auc_inf_obs_ci_low"].magnitude
        < q["auc_inf_obs"].magnitude
        < q["auc_inf_obs_ci_high"].magnitude
    )


def test_bootstrap_se_scales_with_input_uncertainty() -> None:
    options = NCAOptions(seed=1, n_boot=2000, auc_method=AUCMethod.LINEAR)
    small = (
        nca_single(group_curve(cv=0.05), options)
        .to_quantities()["auc_last_se"]
        .magnitude
    )
    large = (
        nca_single(group_curve(cv=0.10), options)
        .to_quantities()["auc_last_se"]
        .magnitude
    )
    assert large / small == pytest.approx(2.0, rel=0.1)


def test_bootstrap_sd_spread_and_sd_se_relation() -> None:
    options = NCAOptions(seed=1, n_boot=2000, bootstrap_spread=BootstrapSpread.SD)
    q = nca_single(group_curve(n=16), options).to_quantities()
    assert q["auc_last_se"].magnitude == pytest.approx(q["auc_last_sd"].magnitude / 4.0)


def test_bootstrap_batch_shape_and_chunking() -> None:
    curves = [
        group_curve(label="a"),
        group_curve(cv=0.2, label="b"),
        group_curve(label="c"),
    ]
    tcs = Timecourses.from_timecourses(curves)
    options = NCAOptions(seed=3, n_boot=200)
    result = nca(tcs, options)
    assert result["auc_last_se"].dims == ("individual",)
    assert result["auc_last_se"].values[1] > result["auc_last_se"].values[0]
    chunked = nca(tcs, options.model_copy(update={"chunk_rows": 7}))
    for name in result.derived_variables:
        np.testing.assert_allclose(
            chunked[name].values, result[name].values, equal_nan=True
        )
    df = result.to_dataframe()
    assert "auc_last_se" in df.columns and "n" in df.columns
    assert list(df.columns)[-1] == "flags"


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
    for diagnostic in (
        "lambda_z_stderr",
        "lambda_z_r2",
        "lambda_z_r2_adj",
        "lambda_z_intercept",
    ):
        assert diagnostic in DISCRETE_PARAMETERS
    assert "auc_last" in LOGNORMAL_PARAMETERS
    assert "auc_extrap_fraction" not in LOGNORMAL_PARAMETERS

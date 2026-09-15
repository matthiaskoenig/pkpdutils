import numpy as np
import pytest
import xarray as xr
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
        curves.append(
            Timecourse(
                time=T,
                value=c0 * np.exp(-K * T),
                time_unit="hr",
                unit="mg/l",
                dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
                label=f"s{i}",
            )
        )
    return Timecourses.from_timecourses(curves)


def test_summarize_statistics() -> None:
    result = nca(individuals(), NCAOptions(auc_method=AUCMethod.LOG))
    summary = result.summarize("individual")
    assert summary.sample_dims == ()
    values = result["auc_inf_obs"].values
    q = summary.to_quantities()
    assert q["auc_inf_obs"].magnitude == pytest.approx(values.mean())
    assert q["auc_inf_obs_sd"].magnitude == pytest.approx(values.std(ddof=1))
    assert q["auc_inf_obs_se"].magnitude == pytest.approx(
        values.std(ddof=1) / np.sqrt(6)
    )
    tq = student_t.ppf(0.975, 5)
    assert q["auc_inf_obs_ci_high"].magnitude == pytest.approx(
        values.mean() + tq * values.std(ddof=1) / np.sqrt(6)
    )
    assert q["auc_inf_obs_median"].magnitude == pytest.approx(np.median(values))
    assert q["auc_inf_obs_q25"].magnitude == pytest.approx(np.percentile(values, 25))
    assert q["auc_inf_obs_geomean"].magnitude == pytest.approx(
        np.exp(np.log(values).mean())
    )
    assert q["auc_inf_obs_geocv"].magnitude == pytest.approx(
        np.sqrt(np.expm1(np.log(values).var(ddof=1)))
    )
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
    two = Timecourses(
        xr.concat([tcs.ds, tcs.ds], dim="study").assign_coords(study=["s1", "s2"])
    )
    result2 = nca(two)
    summary = result2.summarize("individual")
    assert summary.sample_dims == ("study",)
    assert summary["auc_last"].shape == (2,)
    short = Timecourse(
        time=[1, 2, 3],
        value=[1, 3, 2],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
        label="short",
    )
    mixed = Timecourses.from_timecourses([*list(tcs), short])
    flags = nca(mixed).summarize("individual").flags()
    assert "TOO_FEW_POINTS" in flags
    q = nca(mixed).summarize("individual").to_quantities()
    # `n` counts the samples along the dimension, `x_n` the finite values of `x`
    assert q["lambda_z_n"].magnitude == 4
    assert q["n"].magnitude == 5
    assert q["lambda_z_n"].magnitude < q["n"].magnitude
    assert q["lambda_z_se"].magnitude == pytest.approx(
        q["lambda_z_sd"].magnitude / np.sqrt(q["lambda_z_n"].magnitude)
    )
    with pytest.raises(ValueError, match="dim"):
        result.summarize("study")


def test_summarize_reports_no_uncertainty_for_discrete_parameters() -> None:
    # B17: a confidence interval of a point count or of an adjusted R² is not a
    # quantity; the bootstrap skips them and so does `summarize`
    summary = nca(individuals()).summarize("individual")
    names = set(summary.ds.data_vars)
    for name in ("lambda_z_n_points", "lambda_z_r2_adj", "tmax", "tlast"):
        assert name in names
        for suffix in ("_sd", "_se", "_ci_low", "_ci_high", "_geomean", "_geocv"):
            assert f"{name}{suffix}" not in names
        for suffix in ("_median", "_q25", "_q75", "_n"):
            assert f"{name}{suffix}" in names
    # a continuous parameter keeps its statistics
    for suffix in ("_sd", "_se", "_ci_low", "_ci_high", "_median", "_n"):
        assert f"auc_last{suffix}" in names

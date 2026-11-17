import warnings

import numpy as np
import pkpdutils_core
import pytest
import xarray as xr
from scipy.stats import t as student_t

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils import result as result_module
from pkpdutils.nca import AUCMethod, NCAOptions, engine, nca
from pkpdutils.result import ParameterResult
from scripts.validation import analyse, case

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
    result = nca(individuals(), options=NCAOptions(auc_method=AUCMethod.LOG))
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
    result = nca(Timecourses(ds), options=NCAOptions())
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


def multiple_dose(n: int = 4) -> Timecourses:
    tau, n_doses = 12.0, 3
    times = np.arange(0.0, 36.1, 1.0)
    rng = np.random.default_rng(1)
    curves = []
    for i in range(n):
        c0 = 10.0 * rng.lognormal(0, 0.2)
        value = np.zeros_like(times)
        for k in range(n_doses):
            since = times - k * tau
            value += np.where(
                since >= 0, c0 * np.exp(-K * np.clip(since, 0, None)), 0.0
            )
        curves.append(
            Timecourse(
                time=times,
                value=value,
                time_unit="hr",
                unit="mg/l",
                dosing=Dosing.regimen(
                    Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
                    interval=tau,
                    n_doses=n_doses,
                ),
                label=f"s{i}",
            )
        )
    return Timecourses.from_timecourses(curves)


def test_summarize_reports_no_uncertainty_for_discrete_interval_parameters() -> None:
    # the time of the maximum of an interval and the bounds of the interval are
    # observed or given, an uncertainty of them is not a quantity
    result = nca(multiple_dose())
    assert result.has_intervals
    summary = result.summarize("individual")
    names = set(summary.ds.data_vars)
    for name in (
        "interval_tmax",
        "interval_start",
        "interval_end",
        "interval_dose",
        "interval_n_points",
    ):
        assert name in names
        for suffix in ("_sd", "_se", "_cv", "_ci_low", "_ci_high"):
            assert f"{name}{suffix}" not in names
        for suffix in ("_median", "_q25", "_q75", "_min", "_max", "_n"):
            assert f"{name}{suffix}" in names
    # a measured interval parameter keeps its statistics
    for suffix in ("_sd", "_se", "_cv", "_ci_low", "_ci_high"):
        assert f"interval_ctrough{suffix}" in names


# the summary of a hand-made result, under each engine (`summary_engine`)
class Table(ParameterResult):
    lognormal_parameters = frozenset({"ln"})
    discrete_parameters = frozenset({"k"})


def table(**columns: list[float]) -> Table:
    n = len(next(iter(columns.values())))
    return Table(
        xr.Dataset(
            {
                **{
                    name: (("s",), np.array(values), {"units": "mg"})
                    for name, values in columns.items()
                },
                "flags": (("s",), np.zeros(n, dtype=int), {"units": "dimensionless"}),
            },
            coords={"s": [f"s{i}" for i in range(n)]},
        )
    )


def scalar(summary: ParameterResult, name: str) -> float:
    return float(summary.ds[name].values)


def test_a_group_of_one_sample_reports_no_spread_and_no_interval(
    summary_engine: str,
) -> None:
    # one sample, and a column of three samples with a single finite value:
    # the masks are on the finite count, not on the length of the axis
    for name, column in (("a", [5.0]), ("b", [7.0, np.nan, np.inf])):
        summary = table(**{name: column}).summarize("s")
        for statistic in ("sd", "se", "cv", "ci_low", "ci_high"):
            assert np.isnan(scalar(summary, f"{name}_{statistic}")), (
                summary_engine,
                name,
                statistic,
            )
        value = column[0]
        for statistic in ("", "_median", "_min", "_max", "_q25", "_q75"):
            assert scalar(summary, f"{name}{statistic}") == value, (name, statistic)
        assert scalar(summary, f"{name}_n") == 1.0
    # contrast: two samples give a finite interval, t(0.975, 1) = 12.706...
    pair = table(a=[1.0, 3.0]).summarize("s")
    assert scalar(pair, "a_sd") == pytest.approx(np.sqrt(2.0), rel=1e-14)
    assert scalar(pair, "a_ci_high") == pytest.approx(
        2.0 + student_t.isf(0.025, 1) * 1.0, rel=1e-12
    )
    assert scalar(pair, "a_ci_high") == pytest.approx(14.706204736, rel=1e-9)


def test_a_parameter_which_is_nan_for_every_sample_counts_zero(
    summary_engine: str,
) -> None:
    result = table(ln=[np.nan, np.nan, np.nan], other=[1.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = result.summarize("s")
    assert [str(w.message) for w in caught] == [], summary_engine
    assert scalar(summary, "ln_n") == 0.0
    for statistic in (
        "",
        "_sd",
        "_se",
        "_cv",
        "_ci_low",
        "_ci_high",
        "_median",
        "_q25",
        "_q75",
        "_min",
        "_max",
        "_geomean",
        "_geocv",
    ):
        assert np.isnan(scalar(summary, f"ln{statistic}")), statistic
    # the other column of the same request is untouched
    assert scalar(summary, "other") == 2.0
    assert scalar(summary, "other_n") == 3.0


def test_a_lognormal_parameter_with_a_zero_uses_the_positive_values_alone(
    summary_engine: str,
) -> None:
    summary = table(ln=[0.0, 2.0, 8.0]).summarize("s")
    # the geometric mean of 2 and 8 is 4, the zero is left out of it alone
    assert scalar(summary, "ln_geomean") == pytest.approx(4.0, rel=1e-14)
    assert scalar(summary, "ln_geocv") == pytest.approx(
        np.sqrt(np.expm1(2.0 * np.log(2.0) ** 2)), rel=1e-12
    )
    # and the row is not dropped from the arithmetic statistics
    assert scalar(summary, "ln_n") == 3.0
    assert scalar(summary, "ln") == pytest.approx(10.0 / 3.0, rel=1e-14)
    assert scalar(summary, "ln_min") == 0.0
    # one positive value gives the geometric mean but no coefficient of variation
    one = table(ln=[-1.0, 3.0]).summarize("s")
    assert scalar(one, "ln_geomean") == pytest.approx(3.0, rel=1e-14)
    assert np.isnan(scalar(one, "ln_geocv"))
    # none positive gives neither
    none = table(ln=[0.0, -1.0]).summarize("s")
    assert np.isnan(scalar(none, "ln_geomean"))
    assert np.isnan(scalar(none, "ln_geocv"))
    assert scalar(none, "ln_n") == 2.0


def test_the_classification_stays_with_the_python_class(summary_engine: str) -> None:
    # the kernel computes every statistic, the class decides which are written
    summary = table(
        ln=[1.0, 2.0, 4.0], k=[1.0, 2.0, 4.0], plain=[1.0, 2.0, 4.0]
    ).summarize("s")
    names = set(summary.ds.data_vars)
    assert "ln_geomean" in names
    assert not {"k_geomean", "k_sd", "k_ci_low", "plain_geomean"} & names
    assert {"k_median", "k_n", "plain_sd"} <= names


needs_kernel_summary = pytest.mark.skipif(
    not hasattr(pkpdutils_core, "summarize"),
    reason="this build of pkpdutils-core carries no `summarize`",
)


@needs_kernel_summary
def test_the_kernel_and_the_python_summary_agree_on_the_validation_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = analyse("theoph-pknca-summary")
    dim = case("theoph-pknca-summary")["dim"]
    summaries = {}
    for forced in (False, True):
        monkeypatch.setattr(engine, "_force_python", forced)
        summaries[forced] = result.summarize(dim)
    kernel, python = summaries[False].ds, summaries[True].ds
    assert set(kernel.data_vars) == set(python.data_vars)
    assert len(kernel.data_vars) > 100
    for name in kernel.data_vars:
        np.testing.assert_allclose(
            kernel[name].values, python[name].values, rtol=1e-12, atol=0.0, err_msg=name
        )


@needs_kernel_summary
def test_a_malformed_summary_request_is_not_a_refusal() -> None:
    request: dict[str, object] = {
        "values": np.array([1.0, 2.0, 4.0]),
        "n_vars": 1,
        "n_groups": 1,
        "n_samples": 4,
        "ci_level": 0.95,
        "lognormal": [False],
        "discrete": [False],
    }
    with pytest.raises(
        ValueError, match="3 values for 1 variables x 1 groups x 4 samples"
    ) as raised:
        pkpdutils_core.summarize(request)  # ty: ignore[missing-attribute]
    assert not isinstance(raised.value, engine.UnsupportedError)
    request["n_samples"] = 3
    request["ci_level"] = 1.0
    with pytest.raises(ValueError, match="ci_level") as raised:
        pkpdutils_core.summarize(request)  # ty: ignore[missing-attribute]
    assert not isinstance(raised.value, engine.UnsupportedError)


@needs_kernel_summary
def test_a_malformed_request_from_the_host_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a host which sent a wrong request would get the kernel's ValueError out of
    # `summarize`, not a silent second answer from scipy
    real = pkpdutils_core.summarize  # ty: ignore[missing-attribute]
    python_runs: list[int] = []
    original = result_module._python_statistics
    monkeypatch.setattr(
        result_module,
        "_python_statistics",
        lambda *args: (python_runs.append(1), original(*args))[1],
    )
    monkeypatch.setattr(
        pkpdutils_core,
        "summarize",
        lambda request: real({**request, "n_samples": request["n_samples"] + 1}),
    )
    with pytest.raises(ValueError, match="values for 1 variables x 1 groups") as raised:
        table(a=[1.0, 2.0, 4.0]).summarize("s")
    assert not isinstance(raised.value, engine.UnsupportedError)
    assert python_runs == []


def test_a_kernel_without_summarize_is_summarized_in_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # what a wheel of the declared floor (0.3.0) does: the python path answers,
    # with the numbers of the oracle
    columns = {"ln": [0.0, 2.0, 8.0], "k": [1.0, 2.0, 4.0]}
    expected = table(**columns).summarize("s")
    monkeypatch.delattr(pkpdutils_core, "summarize", raising=False)
    python_runs: list[int] = []
    original = result_module._python_statistics
    monkeypatch.setattr(
        result_module,
        "_python_statistics",
        lambda *args: (python_runs.append(1), original(*args))[1],
    )
    summary = table(**columns).summarize("s")
    assert len(python_runs) == 2
    assert scalar(summary, "ln_geomean") == pytest.approx(4.0, rel=1e-14)
    assert scalar(summary, "ln") == pytest.approx(10.0 / 3.0, rel=1e-14)
    half = student_t.isf(0.025, 2) * np.std([0.0, 2.0, 8.0], ddof=1) / np.sqrt(3)
    assert scalar(summary, "ln_ci_high") == pytest.approx(10.0 / 3.0 + half, rel=1e-12)
    assert scalar(summary, "k_median") == 2.0
    for name in expected.ds.data_vars:
        np.testing.assert_allclose(
            summary.ds[name].values, expected.ds[name].values, rtol=1e-12, err_msg=name
        )


def test_an_empty_sample_axis_is_a_group_of_no_value(summary_engine: str) -> None:
    result = Table(
        xr.Dataset(
            {
                "ln": (("s",), np.zeros(0), {"units": "mg"}),
                "flags": (("s",), np.zeros(0, dtype=int), {"units": "dimensionless"}),
            },
            coords={"s": np.array([], dtype=str)},
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = result.summarize("s")
    assert [str(w.message) for w in caught] == [], summary_engine
    assert scalar(summary, "ln_n") == 0.0
    for name in summary.ds.data_vars:
        if name not in ("ln_n", "n", "flags"):
            assert np.isnan(scalar(summary, str(name))), (summary_engine, name)

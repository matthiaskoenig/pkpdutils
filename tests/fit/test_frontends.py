import numpy as np
import pytest
import xarray as xr

from pkpdutils import Dose, Route, Timecourse, Timecourses, nca
from pkpdutils.fit import (
    FitOptions,
    Weighting,
    fit_table,
    fit_timecourse,
    fit_timecourses,
    proportionality_test,
)
from pkpdutils.fit.models import Allometric, MonoExp, Power

T = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])


def curves(n: int = 3) -> Timecourses:
    rng = np.random.default_rng(0)
    tcs = []
    for i in range(n):
        k = 0.2 + 0.1 * i
        y = 10 * np.exp(-k * T) * rng.lognormal(0, 0.03, T.size)
        tcs.append(
            Timecourse(
                time=T + 5.0,
                value=y,
                sd=0.05 * y,
                time_unit="hr",
                unit="mg/l",
                dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS, time=5.0),
                substance="x",
                label=f"s{i}",
            )
        )
    return Timecourses.from_timecourses(tcs)


def test_fit_timecourse_of_a_single_curve() -> None:
    """One curve gives a result without a sample dimension."""
    tc = Timecourse(
        time=T + 5.0,
        value=10.0 * np.exp(-0.25 * T),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS, time=5.0),
        substance="x",
        label="single",
    )
    result = fit_timecourse(MonoExp(), tc, options=FitOptions(n_starts=3, seed=0))
    assert result.sample_dims == ()
    q = result.to_quantities()  # no indexer needed
    assert q["k"].magnitude == pytest.approx(0.25, rel=1e-3)
    assert result.units("k") == "1 / hour"
    np.testing.assert_allclose(result["x_data"].values, T)
    assert result.flags() == []
    assert len(result.to_dataframe()) == 1
    assert result.predict(T).shape == T.shape


def test_fit_timecourses_relative_to_dose_with_units() -> None:
    result = fit_timecourses(
        MonoExp(), curves(), options=FitOptions(weighting=Weighting.INV_SD)
    )
    assert result.sample_dims == ("individual",)
    np.testing.assert_allclose(result["k"].values, [0.2, 0.3, 0.4], rtol=0.1)
    assert result.units("k") == "1 / hour" and result.units("a") == "milligram / liter"
    assert result.ds.attrs["x_unit"] == "hr"
    np.testing.assert_allclose(result["x_data"].values[0], T)
    assert list(result.ds["individual"].values) == ["s0", "s1", "s2"]


def test_fit_timecourses_two_sample_dims() -> None:
    time = T
    values = np.stack(
        [np.stack([10 * np.exp(-k * time) * d for k in (0.2, 0.4)]) for d in (1.0, 2.0)]
    )  # (dose, rate, time)
    tcs = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "rate"),
        coords={"dose": [1.0, 2.0], "rate": [0.2, 0.4]},
    )
    result = fit_timecourses(MonoExp(), tcs)
    assert result.sample_dims == ("dose", "rate")
    assert "k" in result.ds.data_vars
    np.testing.assert_allclose(result["k"].values, [[0.2, 0.4], [0.2, 0.4]], rtol=1e-4)
    np.testing.assert_allclose(result["a"].values, [[10, 10], [20, 20]], rtol=1e-4)


def test_fit_table_dose_proportionality() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    individuals = ["a", "b", "c"]
    rng = np.random.default_rng(1)
    auc = 2.0 * doses[:, None] ** 1.05 * rng.lognormal(0, 0.05, (4, 3))
    ds = xr.Dataset(
        {
            "auc_inf_obs": (
                ("dose", "individual"),
                auc,
                {"units": "hour * milligram / liter"},
            )
        },
        coords={
            "dose": ("dose", doses, {"units": "milligram"}),
            "individual": individuals,
        },
    )
    result = fit_table(Power(), ds, "dose", "auc_inf_obs", dim="dose")
    assert result.sample_dims == ("individual",)
    np.testing.assert_allclose(result["b"].values, 1.05, atol=0.15)
    assert result.units("a") == "hour * milligram / liter"
    pooled = fit_table(
        Power(), ds.mean("individual"), "dose", "auc_inf_obs", dim="dose"
    )
    assert pooled.sample_dims == ()
    test = proportionality_test(pooled, dose_range=(25.0, 200.0))
    assert set(test.to_dict()) == {
        "slope",
        "ci_low",
        "ci_high",
        "bounds",
        "proportional",
        "inconclusive",
        "dose_range",
        "criterion",
    }
    r = 200.0 / 25.0
    assert test.bounds[0] == pytest.approx(1 + np.log(0.8) / np.log(r))
    assert test.bounds[1] == pytest.approx(1 + np.log(1.25) / np.log(r))
    assert float(test.slope) == pytest.approx(float(pooled["b"]))
    assert test.dose_range == (25.0, 200.0) and test.criterion == (0.8, 1.25)
    assert bool(test.proportional) or bool(test.inconclusive)


def test_fit_table_row_pairing_independent_of_dim_order() -> None:
    """`x` and `y` are paired by sample, not by the dimension order left over from broadcasting."""
    cohort = ["p", "q"]
    subject = ["r", "s", "t"]
    doses = np.array([1.0, 2.0, 4.0])
    a_true = np.array([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]])  # (cohort, subject)
    scale = np.array(
        [[100.0, 110.0, 120.0], [210.0, 220.0, 230.0]]
    )  # (cohort, subject)
    x_true = scale[:, :, None] * doses[None, None, :]  # (cohort, subject, dose)
    y_true = a_true[:, :, None] * x_true  # power law with exponent 1
    ds = xr.Dataset(
        {
            "y": (("cohort", "subject", "dose"), y_true),
            "x": (("subject", "cohort", "dose"), np.transpose(x_true, (1, 0, 2))),
        },
        coords={"cohort": cohort, "subject": subject, "dose": doses},
    )
    result = fit_table(
        Power(), ds, "x", "y", dim="dose", options=FitOptions(fixed={"b": 1.0})
    )
    assert result.sample_dims == ("cohort", "subject")
    np.testing.assert_allclose(result["a"].values, a_true, rtol=1e-6)
    np.testing.assert_allclose(result["x_data"].values, x_true, rtol=1e-6)


def test_fit_table_extra_dimension_raises() -> None:
    doses = np.array([1.0, 2.0, 4.0])
    ds = xr.Dataset(
        {
            "y": (("dose",), 2.0 * doses),
            "x": (("dose", "extra"), np.stack([doses, doses * 2], axis=-1)),
            "sd": (("dose", "extra"), np.ones((3, 2))),
        },
        coords={"dose": doses},
    )
    with pytest.raises(ValueError, match="extra"):
        fit_table(Power(), ds, "x", "y", dim="dose")
    with pytest.raises(ValueError, match="extra"):
        fit_table(Power(), ds, "dose", "y", dim="dose", sd="sd")


def test_proportionality_test_invalid_criterion_raises() -> None:
    doses = np.array([10.0, 20.0, 50.0, 100.0])
    result = fit_table(
        Power(),
        xr.Dataset({"auc": (("dose",), 2.0 * doses)}, coords={"dose": doses}),
        "dose",
        "auc",
        dim="dose",
    )
    with pytest.raises(ValueError, match="criterion"):
        proportionality_test(result, dose_range=(10.0, 100.0), criterion=(1.25, 0.8))


def test_proportionality_test_detects_nonproportional() -> None:
    doses = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 400.0])
    auc = 3.0 * doses**1.6
    result = fit_table(
        Power(),
        xr.Dataset({"auc": (("dose",), auc)}, coords={"dose": doses}),
        "dose",
        "auc",
        dim="dose",
    )
    test = proportionality_test(result, dose_range=(10.0, 400.0))
    assert not bool(test.proportional) and not bool(test.inconclusive)
    assert test.to_dict()["proportional"] is False
    with pytest.raises(ValueError, match="b"):
        proportionality_test(
            fit_table(
                Allometric(exponent=0.75),
                xr.Dataset({"cl": (("w",), 3.0 * doses**0.75)}, coords={"w": doses}),
                "w",
                "cl",
                dim="w",
            ),
            dose_range=(10.0, 400.0),
        )


def test_fit_table_on_nca_result() -> None:
    doses = np.array([50.0, 100.0, 200.0])
    time = T
    values = np.stack([d / 10 * np.exp(-0.3 * time) for d in doses])
    tcs = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose",),
        coords={"dose": doses},
        dose={"amount": doses, "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    result = nca(tcs)
    prop = fit_table(Power(), result.ds, "dose", "auc_inf_obs", dim="dose")
    assert prop.sample_dims == ()
    assert prop.to_quantities()["b"].magnitude == pytest.approx(1.0, abs=0.01)


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
    assert result.sample("k", dim="individual").coords["sequence"].tolist() == [
        "RT",
        "TR",
    ]


def test_fit_timecourses_rejects_coordinate_named_like_a_parameter() -> None:
    time = np.array([0.5, 1, 2, 4, 8, 12, 24])
    values = np.stack([10 * np.exp(-0.2 * time), 12 * np.exp(-0.25 * time)])
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"], "k": ("individual", [1.0, 2.0])},
        dose={"amount": np.array([100.0, 100.0]), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    with pytest.raises(ValueError, match="collides"):
        fit_timecourses(MonoExp(), batch)

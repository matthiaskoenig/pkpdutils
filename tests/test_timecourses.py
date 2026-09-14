import logging
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pkpdutils.timecourse import Dose, Route, Timecourse, Timecourses

T = np.array([0.0, 1.0, 2.0, 4.0])
V = np.array([[0.0, 2.0, 1.5, 0.5], [0.0, 3.0, 2.0, 1.0], [0.0, 1.0, 0.8, 0.3]])


def make_batch() -> Timecourses:
    return Timecourses.from_arrays(
        T,
        V,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )


def test_from_arrays_layout() -> None:
    tcs = make_batch()
    assert tcs.sample_dims == ("individual",)
    assert tcs.sample_shape == (3,)
    assert tcs.n_samples == 3 and len(tcs) == 3
    assert tcs.n_time == 4
    assert tcs.time_unit == "hr" and tcs.unit == "mg/l"
    assert tcs.substance == "caffeine"
    assert tcs.route is Route.ORAL
    assert tcs.ds["value"].dims == ("individual", "time")
    assert tcs.ds["value"].attrs["units"] == "mg/l"
    assert tcs.ds["time"].attrs["units"] == "hr"
    np.testing.assert_allclose(tcs.times, np.broadcast_to(T, (3, 4)))
    np.testing.assert_allclose(tcs.values, V)
    assert tcs.has_dose and not tcs.has_uncertainty
    dose_amount = tcs.dose_amount
    assert dose_amount is not None
    np.testing.assert_allclose(dose_amount, [100, 100, 100])
    dose_time = tcs.dose_time
    assert dose_time is not None
    np.testing.assert_allclose(dose_time, [0, 0, 0])
    assert tcs.dose_unit == "mg"
    assert tcs.sd is None and tcs.n is None


def test_from_arrays_two_sample_dims() -> None:
    values = np.stack([V, 2 * V])  # (dose, individual, time)
    tcs = Timecourses.from_arrays(
        T,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "individual"),
        coords={"dose": [50, 100], "individual": ["a", "b", "c"]},
        dose={"amount": np.array([[50, 50, 50], [100, 100, 100]]), "unit": "mg"},
        route=Route.ORAL,
    )
    assert tcs.sample_dims == ("dose", "individual")
    assert tcs.n_samples == 6
    dose_amount = tcs.dose_amount
    assert dose_amount is not None
    np.testing.assert_allclose(dose_amount[1], [100, 100, 100])
    tc = tcs.sel(dose=100, individual="b")
    np.testing.assert_allclose(tc.value, 2 * V[1])
    assert tc.dose is not None and tc.dose.amount == 100


def test_from_arrays_uncertainty() -> None:
    sd = 0.1 * V
    tcs = Timecourses.from_arrays(
        T, V, time_unit="hr", unit="mg/l", sd=sd, n=np.array([5, 6, 7])
    )
    assert tcs.has_uncertainty
    tcs_sd = tcs.sd
    assert tcs_sd is not None
    np.testing.assert_allclose(tcs_sd, sd)
    tcs_se = tcs.se
    assert tcs_se is not None
    np.testing.assert_allclose(tcs_se, sd / np.sqrt([[5], [6], [7]]))
    tcs_n = tcs.n
    assert tcs_n is not None
    np.testing.assert_allclose(tcs_n, [5, 6, 7])
    tc = tcs.isel(individual=1)
    assert tc.n is not None
    np.testing.assert_allclose(tc.n, 6)


def test_from_arrays_per_sample_times() -> None:
    times = np.array([[0, 1, 2, 4], [0, 2, 4, 8], [0, 0.5, 1, 2]])
    tcs = Timecourses.from_arrays(times, V, time_unit="hr", unit="mg/l")
    assert "times" in tcs.ds
    np.testing.assert_allclose(tcs.times, times)
    np.testing.assert_allclose(tcs.isel(individual=1).time, [0, 2, 4, 8])


def test_from_arrays_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        Timecourses.from_arrays(T, V[:, :3], time_unit="hr", unit="mg/l")


def test_iteration_yields_timecourses() -> None:
    tcs = make_batch()
    items = list(tcs)
    assert len(items) == 3
    assert all(isinstance(tc, Timecourse) for tc in items)
    assert items[2].label == "c"
    np.testing.assert_allclose(items[2].value, V[2])
    assert items[2].dose is not None and items[2].dose.route is Route.ORAL
    assert items[2].substance == "caffeine"


def test_from_timecourses_shared_grid() -> None:
    tcs = Timecourses.from_timecourses(
        [
            Timecourse(time=T, value=V[0], time_unit="hr", unit="mg/l", label="a"),
            Timecourse(time=T, value=V[1], time_unit="hr", unit="mg/l", label="b"),
        ]
    )
    assert "times" not in tcs.ds
    assert list(tcs.ds["individual"].values) == ["a", "b"]
    np.testing.assert_allclose(tcs.values, V[:2])


def test_from_timecourses_ragged() -> None:
    tcs = Timecourses.from_timecourses(
        [
            Timecourse(time=[0, 1, 2], value=[0, 2, 1], time_unit="hr", unit="mg/l"),
            Timecourse(
                time=[0, 1, 2, 4, 8],
                value=[0, 3, 2, 1, 0.5],
                time_unit="hr",
                unit="mg/l",
            ),
        ],
        dim="subject",
    )
    assert tcs.sample_dims == ("subject",)
    assert tcs.n_time == 5
    assert np.isnan(tcs.values[0, 3:]).all()
    assert np.isnan(tcs.times[0, 3:]).all()
    tc = tcs.isel(subject=0)
    assert tc.size == 3
    np.testing.assert_allclose(tc.time, [0, 1, 2])


def test_from_timecourses_requires_same_units() -> None:
    with pytest.raises(ValueError, match="unit"):
        Timecourses.from_timecourses(
            [
                Timecourse(time=T, value=V[0], time_unit="hr", unit="mg/l"),
                Timecourse(time=T, value=V[1], time_unit="hr", unit="ng/ml"),
            ]
        )


def test_from_timecourses_uncertainty_and_dose() -> None:
    tcs = Timecourses.from_timecourses(
        [
            Timecourse(
                time=T,
                value=V[0],
                sd=0.1 * V[0],
                n=4,
                time_unit="hr",
                unit="mg/l",
                dose=Dose(amount=50, unit="mg"),
            ),
            Timecourse(
                time=T,
                value=V[1],
                sd=0.1 * V[1],
                n=6,
                time_unit="hr",
                unit="mg/l",
                dose=Dose(amount=100, unit="mg"),
            ),
        ]
    )
    assert tcs.has_uncertainty and tcs.has_dose
    tcs_n = tcs.n
    assert tcs_n is not None
    np.testing.assert_allclose(tcs_n, [4, 6])
    dose_amount = tcs.dose_amount
    assert dose_amount is not None
    np.testing.assert_allclose(dose_amount, [50, 100])


def test_to_dataframe_long() -> None:
    df = make_batch().to_dataframe()
    assert set(df.columns) == {"individual", "time", "value"}
    assert len(df) == 12
    assert df["value"].sum() == pytest.approx(V.sum())


def test_to_dataframe_with_uncertainty_columns() -> None:
    tcs = Timecourses.from_arrays(
        T,
        V,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        sd=0.1 * V,
        n=np.array([5, 6, 7]),
    )
    df = tcs.to_dataframe()
    assert list(df.columns) == ["individual", "time", "value", "sd", "se", "n"]
    assert len(df) == 12
    rows = df[df["individual"] == "b"]
    np.testing.assert_allclose(rows["time"].to_numpy(), T)
    np.testing.assert_allclose(rows["value"].to_numpy(), V[1])
    np.testing.assert_allclose(rows["sd"].to_numpy(), 0.1 * V[1])
    np.testing.assert_allclose(rows["se"].to_numpy(), 0.1 * V[1] / np.sqrt(6))
    np.testing.assert_allclose(rows["n"].to_numpy(), 6)


def test_to_dataframe_ragged_transposed_times() -> None:
    # 'times' comes in with the dimensions in the order (time, individual)
    times = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    values = np.array([[0.0, 0.0], [2.0, 3.0], [1.0, 1.5]])
    ds = xr.Dataset(
        data_vars={
            "value": (("individual", "time"), values.T, {"units": "mg/l"}),
            "times": (("time", "individual"), times, {"units": "hr"}),
        },
        coords={
            "time": ("time", np.arange(3), {"units": "hr"}),
            "individual": ("individual", ["a", "b"]),
        },
    )
    tcs = Timecourses(ds)
    assert tcs.ds["times"].dims == ("individual", "time")
    df = tcs.to_dataframe()
    for i, name in enumerate(["a", "b"]):
        rows = df[df["individual"] == name]
        np.testing.assert_allclose(rows["time"].to_numpy(), times[:, i])
        np.testing.assert_allclose(rows["value"].to_numpy(), values[:, i])


def test_dataset_validation() -> None:
    ds = xr.Dataset({"foo": (("time",), np.zeros(3))}, coords={"time": [0, 1, 2]})
    with pytest.raises(ValueError, match="value"):
        Timecourses(ds)


def test_from_dataframe_shared_grid() -> None:
    rows = []
    for i, name in enumerate(["a", "b", "c"]):
        for t, v in zip(T, V[i], strict=True):
            rows.append({"individual": name, "time": t, "value": v, "dose": 100})
    df = pd.DataFrame(rows)
    tcs = Timecourses.from_dataframe(
        df,
        sample=["individual"],
        time_unit="hr",
        unit="mg/l",
        dose_amount="dose",
        dose_unit="mg",
        route=Route.ORAL,
    )
    assert tcs.sample_dims == ("individual",)
    assert list(tcs.ds["individual"].values) == ["a", "b", "c"]
    np.testing.assert_allclose(tcs.values, V)
    dose_amount = tcs.dose_amount
    assert dose_amount is not None
    np.testing.assert_allclose(dose_amount, [100, 100, 100])
    assert "times" not in tcs.ds


def test_from_dataframe_ragged() -> None:
    df = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "b", "b"],
            "time": [0, 1, 2, 0, 4],
            "value": [0, 2, 1, 0, 3],
            "sd": [0, 0.2, 0.1, 0, 0.3],
            "n": [5, 5, 5, 8, 8],
        }
    )
    tcs = Timecourses.from_dataframe(
        df, sample=["subject"], time_unit="hr", unit="mg/l", sd="sd", n="n"
    )
    assert "times" in tcs.ds
    assert tcs.n_time == 3
    tcs_n = tcs.n
    assert tcs_n is not None
    np.testing.assert_allclose(tcs_n, [5, 8])
    tc = tcs.sel(subject="b")
    np.testing.assert_allclose(tc.time, [0, 4])
    tc_sd = tc.sd
    assert tc_sd is not None
    np.testing.assert_allclose(tc_sd, [0, 0.3])


def test_from_dataframe_two_sample_columns() -> None:
    rows = []
    for dose in (50, 100):
        for name in ("a", "b"):
            for t in T:
                rows.append(
                    {"dose": dose, "individual": name, "time": t, "value": dose * t}
                )
    tcs = Timecourses.from_dataframe(
        pd.DataFrame(rows), sample=["dose", "individual"], time_unit="hr", unit="mg/l"
    )
    assert tcs.sample_dims == ("dose", "individual")
    assert tcs.sample_shape == (2, 2)
    np.testing.assert_allclose(tcs.sel(dose=100, individual="b").value, 100 * T)
    assert tcs.unit == "mg/l" and tcs.time_unit == "hr"


def test_from_dataframe_missing_combination() -> None:
    rows = []
    for dose in (50, 100):
        for name in ("a", "b"):
            if dose == 100 and name == "b":
                continue  # this combination is not measured
            for t in T:
                rows.append(
                    {"dose": dose, "individual": name, "time": t, "value": dose * t}
                )
    tcs = Timecourses.from_dataframe(
        pd.DataFrame(rows),
        sample=["dose", "individual"],
        time_unit="hr",
        unit="mg/l",
        dose_amount="dose",
        dose_unit="mg",
        route=Route.ORAL,
    )
    assert tcs.sample_shape == (2, 2)
    missing = tcs.sel(dose=100, individual="b")
    assert missing.dose is None
    assert np.isnan(missing.value).all()
    assert len(list(tcs)) == 4
    for dose_amount, name in ((50, "a"), (50, "b"), (100, "a")):
        tc = tcs.sel(dose=dose_amount, individual=name)
        assert tc.dose is not None
        assert tc.dose.amount == dose_amount
        np.testing.assert_allclose(tc.value, dose_amount * T)


def test_from_timecourses_mixed_doses_raise() -> None:
    with pytest.raises(ValueError, match="dose") as excinfo:
        Timecourses.from_timecourses(
            [
                Timecourse(
                    time=T,
                    value=V[0],
                    time_unit="hr",
                    unit="mg/l",
                    label="a",
                    dose=Dose(amount=50, unit="mg"),
                ),
                Timecourse(time=T, value=V[1], time_unit="hr", unit="mg/l", label="b"),
            ]
        )
    assert "b" in str(excinfo.value)


def test_from_timecourses_mixed_routes_raise() -> None:
    with pytest.raises(ValueError, match="separate batches"):
        Timecourses.from_timecourses(
            [
                Timecourse(
                    time=T,
                    value=V[0],
                    time_unit="hr",
                    unit="mg/l",
                    dose=Dose(amount=50, unit="mg", route=Route.ORAL),
                ),
                Timecourse(
                    time=T,
                    value=V[1],
                    time_unit="hr",
                    unit="mg/l",
                    dose=Dose(amount=50, unit="mg", route=Route.IV_BOLUS),
                ),
            ]
        )


def test_from_timecourses_varying_n_warns(caplog: pytest.LogCaptureFixture) -> None:
    timecourses = [
        Timecourse(
            time=T,
            value=V[0],
            sd=0.1 * V[0],
            n=[4, 4, 6, 6],
            time_unit="hr",
            unit="mg/l",
            label="a",
        ),
        Timecourse(
            time=T,
            value=V[1],
            sd=0.1 * V[1],
            n=8,
            time_unit="hr",
            unit="mg/l",
            label="b",
        ),
    ]
    with caplog.at_level(logging.WARNING, logger="pkpdutils.timecourse"):
        tcs = Timecourses.from_timecourses(timecourses)
    tcs_n = tcs.n
    assert tcs_n is not None
    np.testing.assert_allclose(tcs_n, [6, 8])
    assert "'a'" in caplog.text
    assert "'n'" in caplog.text


def test_from_dataset_scan() -> None:
    time = np.linspace(0, 10, 11)
    scan = np.array([1.0, 2.0])
    values = np.exp(-scan[None, :] * time[:, None] / 5)  # (_time, dim_dose)
    ds = xr.Dataset(
        {"[Cve_mid]": (("_time", "dim_dose"), values)},
        coords={"_time": time, "dim_dose": scan},
    )
    tcs = Timecourses.from_dataset(
        ds, "[Cve_mid]", unit="mmol/l", time_unit="min", substance="midazolam"
    )
    assert tcs.sample_dims == ("dim_dose",)
    assert tcs.n_time == 11
    np.testing.assert_allclose(tcs.values, values.T)
    np.testing.assert_allclose(tcs.ds["time"].values, time)
    assert tcs.unit == "mmol/l" and tcs.time_unit == "min"


def test_from_xresult_duck_typed() -> None:
    time = np.linspace(0, 10, 11)
    ds = xr.Dataset(
        {"[Cve_mid]": (("_time",), np.exp(-time / 5))}, coords={"_time": time}
    )

    class FakeXResult:
        xds: ClassVar[xr.Dataset] = ds
        uinfo: ClassVar[dict[str, str]] = {"[Cve_mid]": "mmol/l", "time": "min"}

    tcs = Timecourses.from_xresult(
        FakeXResult(),
        "[Cve_mid]",
        dose=Dose(amount=7.5, unit="mg", route=Route.IV_BOLUS),
    )
    assert tcs.sample_dims == ()
    assert tcs.n_samples == 1
    assert tcs.unit == "mmol/l"
    tc = tcs.isel()
    assert tc.dose is not None and tc.dose.route is Route.IV_BOLUS
    np.testing.assert_allclose(tc.time, time)

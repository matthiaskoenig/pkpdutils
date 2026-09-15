import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pkpdutils.timecourse import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.units import Q_

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
    assert tcs.n_dose == 1
    np.testing.assert_allclose(dose_amount, [[100], [100], [100]])
    dose_time = tcs.dose_time
    assert dose_time is not None
    np.testing.assert_allclose(dose_time, [[0], [0], [0]])
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
    np.testing.assert_allclose(dose_amount[1], [[100], [100], [100]])
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
    np.testing.assert_allclose(dose_amount, [[50], [100]])


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
    np.testing.assert_allclose(dose_amount, [[100], [100], [100]])
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
    assert missing.dose is None and missing.dosing is None
    assert np.isnan(missing.value).all()
    assert tcs.dosing_of(dose=100, individual="b") is None
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


def test_from_timecourses_partial_spread_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    timecourses = [
        Timecourse(
            time=T,
            value=V[0],
            sd=0.1 * V[0],
            n=6,
            time_unit="hr",
            unit="mg/l",
            label="a",
        ),
        Timecourse(time=T, value=V[1], time_unit="hr", unit="mg/l", label="b"),
    ]
    with caplog.at_level(logging.WARNING, logger="pkpdutils.timecourse"):
        tcs = Timecourses.from_timecourses(timecourses)
    assert tcs.sd is None
    assert tcs.se is None
    assert tcs.n is None
    assert not tcs.has_uncertainty
    assert "'sd'" in caplog.text
    assert "'b'" in caplog.text


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


def test_batch_carries_protocols_padded() -> None:
    t = np.array([1.0, 2.0, 4.0, 13.0, 25.0])
    a = Timecourse(
        time=t,
        value=[1, 2, 1, 3, 3],
        time_unit="hr",
        unit="mg/l",
        label="a",
        dosing=Dosing(amounts=[100, 100], times=[0, 12], unit="mg"),
    )
    b = Timecourse(
        time=t,
        value=[1, 2, 1, 3, 3],
        time_unit="hr",
        unit="mg/l",
        label="b",
        dose=Dose(amount=50, unit="mg", time=0.0),
    )
    batch = Timecourses.from_timecourses([a, b], labels=["a", "b"])
    assert batch.n_dose == 2 and batch.has_dose
    assert batch.dose_amount is not None and batch.dose_amount.shape == (2, 2)
    assert batch.dose_amount[0].tolist() == [100.0, 100.0]
    assert batch.dose_amount[1][0] == 50.0 and np.isnan(batch.dose_amount[1][1])
    assert batch.dose_time is not None and batch.dose_time[0].tolist() == [0.0, 12.0]
    assert batch.n_doses is not None and batch.n_doses.tolist() == [2, 1]
    assert batch.last_dose_time is not None and batch.last_dose_time.tolist() == [
        12.0,
        0.0,
    ]
    assert batch.first_dose_amount is not None and batch.first_dose_amount.tolist() == [
        100.0,
        50.0,
    ]
    assert list(batch.ds["dose_amount"].dims) == ["individual", "dose_index"]
    assert batch.dosing_of(individual="a") == a.dosing
    assert batch.dosing_of(individual="b") == b.dosing
    assert batch.sel(individual="a") == a
    assert [tc.dosing for tc in batch] == [a.dosing, b.dosing]


def test_from_arrays_with_a_protocol_and_with_padded_mapping() -> None:
    time = np.array([1.0, 2.0, 4.0, 13.0])
    values = np.ones((3, 4))
    protocol = Dosing(amounts=[10, 10], times=[0, 12], unit="mg", route=Route.IV_BOLUS)
    batch = Timecourses.from_arrays(
        time, values, time_unit="hr", unit="mg/l", dose=protocol, route=None
    )
    assert (
        batch.dose_amount is not None
        and batch.dose_amount.shape == (3, 2)
        and batch.route is Route.IV_BOLUS
    )
    mapping = {
        "amount": np.array([[10, 10], [20, np.nan], [10, 5]]),
        "time": np.array([[0, 12], [0, np.nan], [0, 24]]),
        "unit": "mg",
    }
    batch2 = Timecourses.from_arrays(
        time, values, time_unit="hr", unit="mg/l", dose=mapping, route=Route.ORAL
    )
    assert batch2.n_doses is not None and batch2.n_doses.tolist() == [2, 1, 2]
    assert batch2.dosing_of(individual=2) == Dosing(
        amounts=[10, 5], times=[0, 24], unit="mg", route=Route.ORAL
    )
    with pytest.raises(ValueError, match="time"):
        Timecourses.from_arrays(
            time,
            values,
            time_unit="hr",
            unit="mg/l",
            dose={"amount": mapping["amount"], "unit": "mg"},
            route=Route.ORAL,
        )


def test_from_dataframe_builds_protocols_from_dose_rows() -> None:
    rows = []
    for subject, doses in (
        ("s1", [(0.0, 100.0), (12.0, 100.0)]),
        ("s2", [(0.0, 50.0)]),
    ):
        for i, t in enumerate([1.0, 2.0, 13.0]):
            dt, da = doses[min(i, len(doses) - 1)]
            rows.append(
                {
                    "subject": subject,
                    "time": t,
                    "value": 1.0 + i,
                    "dose_amount": da,
                    "dose_time": dt,
                }
            )
    df = pd.DataFrame(rows)
    batch = Timecourses.from_dataframe(
        df,
        sample=["subject"],
        time_unit="hr",
        unit="mg/l",
        dose_amount="dose_amount",
        dose_unit="mg",
        dose_time="dose_time",
        route=Route.ORAL,
    )
    assert batch.dosing_of(subject="s1") == Dosing(
        amounts=[100, 100], times=[0, 12], unit="mg", route=Route.ORAL
    )
    assert batch.dosing_of(subject="s2") == Dosing(
        amounts=[50], times=[0], unit="mg", route=Route.ORAL
    )


def test_single_dose_batch_layout_is_one_column() -> None:
    time = np.array([1.0, 2.0])
    batch = Timecourses.from_arrays(
        time,
        np.ones((2, 2)),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=1, unit="mg"),
        route=None,
    )
    assert batch.n_dose == 1 and batch.dose_time is not None
    assert batch.dose_time.shape == (2, 1)
    assert batch.dosing_of(individual=0) == Dosing.single(Dose(amount=1, unit="mg"))


def test_from_arrays_mapping_sorts_rows_by_time() -> None:
    batch = Timecourses.from_arrays(
        np.array([1.0, 2.0]),
        np.ones((1, 2)),
        time_unit="hr",
        unit="mg/l",
        dose={
            "amount": np.array([[100.0, 50.0]]),
            "time": np.array([[12.0, 0.0]]),
            "unit": "mg",
        },
        route=Route.ORAL,
    )
    assert batch.first_dose_amount is not None
    assert batch.first_dose_amount.tolist() == [50.0]
    assert batch.last_dose_amount is not None
    assert batch.last_dose_amount.tolist() == [100.0]
    assert batch.dose_time is not None and batch.dose_time[0].tolist() == [0.0, 12.0]
    assert batch.dosing_of(individual=0) == Dosing(
        amounts=[50, 100], times=[0, 12], unit="mg", route=Route.ORAL
    )


def test_from_arrays_mapping_rejects_interleaved_nan_and_duplicates() -> None:
    time = np.array([1.0, 2.0])
    values = np.ones((1, 2))
    with pytest.raises(ValueError, match="NaN"):
        Timecourses.from_arrays(
            time,
            values,
            time_unit="hr",
            unit="mg/l",
            dose={
                "amount": np.array([[np.nan, 10.0]]),
                "time": np.array([[0.0, np.nan]]),
                "unit": "mg",
            },
            route=Route.ORAL,
        )
    with pytest.raises(ValueError, match="Duplicate"):
        Timecourses.from_arrays(
            time,
            values,
            time_unit="hr",
            unit="mg/l",
            dose={
                "amount": np.array([[10.0, 10.0]]),
                "time": np.array([[0.0, 0.0]]),
                "unit": "mg",
            },
            route=Route.ORAL,
        )


def test_dose_index_is_a_reserved_sample_dimension() -> None:
    with pytest.raises(ValueError, match="dose_index"):
        Timecourses.from_arrays(
            T,
            V,
            time_unit="hr",
            unit="mg/l",
            dims=("dose_index",),
            dose=Dose(amount=100, unit="mg"),
        )


def test_tissue_survives_the_batch_round_trip() -> None:
    # B5: `tissue` was the only field of `Timecourse` a batch dropped
    tc = Timecourse(
        time=np.array([0.5, 1.0, 2.0, 4.0]),
        value=np.array([1.0, 2.0, 1.5, 0.5]),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg"),
        substance="caffeine",
        label="A",
        tissue="plasma",
    )
    batch = Timecourses.from_timecourses([tc])
    assert batch.tissue == "plasma"
    assert batch.ds.attrs["tissue"] == "plasma"
    back = batch.sel(individual="A")
    assert back.tissue == "plasma"
    assert back == tc
    assert next(iter(batch)) == tc


def test_from_timecourses_rejects_differing_tissues() -> None:
    def curve(tissue: str | None) -> Timecourse:
        return Timecourse(
            time=T,
            value=V[0],
            time_unit="hr",
            unit="mg/l",
            label=str(tissue),
            tissue=tissue,
        )

    with pytest.raises(ValueError, match="tissue"):
        Timecourses.from_timecourses([curve("plasma"), curve("urine")])
    without = Timecourses.from_timecourses([curve(None), curve(None)])
    assert without.tissue is None
    assert "tissue" not in without.ds.attrs


def test_from_arrays_and_from_dataframe_carry_the_tissue() -> None:
    batch = Timecourses.from_arrays(T, V, time_unit="hr", unit="mg/l", tissue="plasma")
    assert batch.tissue == "plasma"
    df = pd.DataFrame(
        {
            "individual": np.repeat(["a", "b"], 4),
            "time": np.tile(T, 2),
            "value": np.concatenate([V[0], V[1]]),
        }
    )
    from_df = Timecourses.from_dataframe(
        df, sample=["individual"], time_unit="hr", unit="mg/l", tissue="serum"
    )
    assert from_df.tissue == "serum"
    assert from_df.sel(individual="a").tissue == "serum"


def test_ragged_batch_survives_the_dataframe_round_trip() -> None:
    # B7: the `NaN` time padding was written into the frame and rejected on the
    # way back; B31: the samples came back sorted instead of in their order
    tc1 = Timecourse(
        time=np.array([0.5, 1.0, 2.0, 4.0]),
        value=np.array([1.0, 2.0, 1.5, 0.8]),
        time_unit="hr",
        unit="ng/ml",
        label="b",
    )
    tc2 = Timecourse(
        time=np.array([0.25, 1.0, 3.0]),
        value=np.array([0.5, 2.1, 1.1]),
        time_unit="hr",
        unit="ng/ml",
        label="a",
    )
    batch = Timecourses.from_timecourses([tc1, tc2])
    df = batch.to_dataframe()
    assert not df["time"].isna().any()
    assert len(df) == 7
    back = Timecourses.from_dataframe(
        df, sample=["individual"], time_unit="hr", unit="ng/ml"
    )
    assert back.ds["individual"].to_numpy().tolist() == ["b", "a"]
    assert back == batch


def test_shared_grid_batch_keeps_the_sample_order_through_the_dataframe() -> None:
    batch = Timecourses.from_arrays(
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        time_unit="hr",
        unit="ng/ml",
        coords={"individual": ["b", "a"]},
    )
    back = Timecourses.from_dataframe(
        batch.to_dataframe(), sample=["individual"], time_unit="hr", unit="ng/ml"
    )
    assert back.ds["individual"].to_numpy().tolist() == ["b", "a"]
    assert back == batch


def test_batch_constructors_coerce_a_route_string() -> None:
    # B20: `from_arrays` crashed with an `AttributeError`, `from_dataframe` worked
    batch = Timecourses.from_arrays(
        T,
        V,
        time_unit="hr",
        unit="mg/l",
        dose={"amount": 100.0, "unit": "mg"},
        route="oral",
    )
    assert batch.route is Route.ORAL
    ds = xr.Dataset(
        {"c": (("scan", "_time"), V, {"units": "mg/l"})},
        coords={"_time": T, "scan": [0, 1, 2]},
    )
    scan = Timecourses.from_dataset(
        ds,
        "c",
        unit="mg/l",
        time_unit="hr",
        dose={"amount": 100.0, "unit": "mg"},
        route="IV_BOLUS",
    )
    assert scan.route is Route.IV_BOLUS
    df = pd.DataFrame(
        {
            "id": ["a"] * 4,
            "time": T,
            "value": V[0],
            "dose": [100.0] * 4,
        }
    )
    from_df = Timecourses.from_dataframe(
        df,
        sample=["id"],
        time_unit="hr",
        unit="mg/l",
        dose_amount="dose",
        dose_unit="mg",
        route="oral",
    )
    assert from_df.route is Route.ORAL


def test_from_arrays_rejects_an_empty_unit() -> None:
    # B4
    with pytest.raises(ValueError, match="dimensionless"):
        Timecourses.from_arrays(T, V, time_unit="hr", unit="")
    with pytest.raises(ValueError, match="dimensionless"):
        Timecourses.from_arrays(T, V, time_unit="", unit="mg/l")


def curves_of_every_kind() -> list[Timecourse]:
    """One curve per case the batch arrays have to carry: ragged, spread, doses."""
    return [
        Timecourse(
            time=[0.5, 1.0, 2.0, 4.0],
            value=[1.0, 2.0, 1.5, 0.8],
            sd=[0.1, 0.2, 0.15, 0.08],
            n=8.0,
            time_unit="hr",
            unit="ng/ml",
            label="a",
            tissue="plasma",
            dosing=Dosing(amounts=[100.0], times=[0.0], unit="mg", route=Route.ORAL),
        ),
        Timecourse(
            time=[0.25, 1.0, 3.0],
            value=[0.5, 2.1, 1.1],
            sd=[0.05, 0.2, 0.1],
            n=6.0,
            time_unit="hr",
            unit="ng/ml",
            label="b",
            tissue="plasma",
            dosing=Dosing(
                amounts=[100.0, 50.0], times=[0.0, 12.0], unit="mg", route=Route.ORAL
            ),
        ),
    ]


def test_iteration_rebuilds_the_curves_the_batch_was_built_from() -> None:
    # B3: the curves are built from the arrays of the batch with
    # `model_construct`, which must give exactly what the validating
    # constructor of `from_timecourses` was given
    curves = curves_of_every_kind()
    batch = Timecourses.from_timecourses(curves)
    assert list(batch) == curves
    assert batch.isel(individual=1) == curves[1]
    assert batch.sel(individual="b") == curves[1]
    assert batch.dosing_of(individual="b") == curves[1].dosing


def test_iteration_rebuilds_an_infusion_protocol() -> None:
    curve = Timecourse(
        time=[0.5, 1.0, 2.0],
        value=[1.0, 2.0, 1.5],
        time_unit="hr",
        unit="ng/ml",
        label="a",
        dosing=Dosing(
            amounts=[100.0, 100.0],
            times=[0.0, 12.0],
            durations=[0.5, 0.25],
            unit="mg",
            route=Route.IV_INFUSION,
        ),
    )
    batch = Timecourses.from_timecourses([curve])
    rebuilt = batch.isel(individual=0)
    assert rebuilt == curve
    assert rebuilt.dosing is not None and rebuilt.dosing.durations is not None


def test_iteration_validates_a_batch_which_does_not_hold_the_invariants(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # the fast path is only taken for a row which is already valid; a dataset
    # built by hand still goes through the validation of `Timecourse`
    duplicate = Timecourses.from_arrays(
        np.array([1.0, 1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0]]),
        time_unit="hr",
        unit="mg/l",
    )
    with pytest.raises(ValueError, match="duplicate"):
        duplicate.isel(individual=0)
    unsorted = Timecourses.from_arrays(
        np.array([2.0, 1.0]),
        np.array([[1.0, 2.0]]),
        time_unit="hr",
        unit="mg/l",
    )
    with caplog.at_level(logging.WARNING, logger="pkpdutils.timecourse"):
        curve = unsorted.isel(individual=0)
    assert "Unsorted time points" in caplog.text
    np.testing.assert_allclose(curve.time, [1.0, 2.0])
    np.testing.assert_allclose(curve.value, [2.0, 1.0])
    one_point = Timecourses.from_arrays(
        np.array([1.0, 2.0]),
        np.array([[1.0, np.nan]]),
        time_unit="hr",
        unit="mg/l",
        sd=np.array([[0.1, np.nan]]),
        n=np.array([4.0]),
    )
    assert one_point.isel(individual=0).size == 2  # a NaN value keeps its time


def test_iteration_derives_the_missing_spread_of_a_hand_built_batch() -> None:
    # `Timecourse` derives `se` from `sd` and `n`, which the fast path of the
    # iteration must not skip: such a batch takes the validating path
    ds = Timecourses.from_arrays(
        T, V, time_unit="hr", unit="mg/l", sd=0.1 * V, n=np.array([4.0, 4.0, 4.0])
    ).ds
    batch = Timecourses(ds.drop_vars("se"))
    curve = batch.isel(individual=0)
    assert curve.se is not None
    np.testing.assert_allclose(curve.se, 0.1 * V[0] / 2.0)


def test_from_dataframe_matches_the_curves_of_the_batch() -> None:
    # B3: the frame is read into the padded arrays directly; the batch must be
    # the one the per curve path built
    batch = Timecourses.from_timecourses(curves_of_every_kind())
    back = Timecourses.from_dataframe(
        batch.to_dataframe(),
        sample=["individual"],
        time_unit="hr",
        unit="ng/ml",
        sd="sd",
        se="se",
        n="n",
        tissue="plasma",
    )
    assert back.ds["individual"].to_numpy().tolist() == ["a", "b"]
    for name in ("value", "sd", "se", "n", "times"):
        np.testing.assert_allclose(
            back.ds[name].to_numpy(), batch.ds[name].to_numpy(), equal_nan=True
        )


def test_from_dataframe_names_the_sample_of_every_check() -> None:
    def frame(**changes: Any) -> pd.DataFrame:
        rows = pd.DataFrame(
            {
                "subject": ["a", "a", "b", "b"],
                "time": [0.0, 1.0, 0.0, 1.0],
                "value": [1.0, 2.0, 3.0, 4.0],
                "dose": [100.0, 100.0, 50.0, 50.0],
                "dose_time": [0.0, 0.0, 0.0, 0.0],
            }
        )
        for column, values in changes.items():
            rows[column] = values
        return rows

    base: dict[str, Any] = {"sample": ["subject"], "time_unit": "hr", "unit": "mg/l"}
    with pytest.raises(ValueError, match=r"sample b: 'time' contains duplicate"):
        Timecourses.from_dataframe(frame(time=[0.0, 1.0, 2.0, 2.0]), **base)
    with pytest.raises(ValueError, match=r"sample b: 'time' contains NaN"):
        Timecourses.from_dataframe(frame(time=[0.0, 1.0, 0.0, np.nan]), **base)
    with pytest.raises(ValueError, match=r"sample b: a timecourse needs at least 2"):
        Timecourses.from_dataframe(frame().iloc[:3], **base)
    doses: dict[str, Any] = {
        **base,
        "dose_amount": "dose",
        "dose_unit": "mg",
        "route": Route.ORAL,
    }
    with pytest.raises(ValueError, match=r"sample a: the dose must be constant"):
        Timecourses.from_dataframe(frame(dose=[100.0, 200.0, 50.0, 50.0]), **doses)
    with pytest.raises(ValueError, match=r"sample a: 'amounts' must be non-negative"):
        Timecourses.from_dataframe(frame(dose=-1.0), **doses)
    protocol: dict[str, Any] = {**doses, "dose_time": "dose_time"}
    with pytest.raises(ValueError, match=r"sample b: the sample has no dose"):
        Timecourses.from_dataframe(
            frame(dose=[100.0, 100.0, np.nan, np.nan]), **protocol
        )
    with pytest.raises(
        ValueError, match=r"sample a: the dose must be constant per dose"
    ):
        Timecourses.from_dataframe(frame(dose=[100.0, 200.0, 50.0, 50.0]), **protocol)


def test_from_dataframe_rejects_a_value_which_is_not_a_number() -> None:
    # a cell which is neither missing nor a number must not be read as a
    # missing value: the dose record would be dropped from the protocol
    rows = pd.DataFrame(
        {
            "subject": ["a", "a", "b", "b"],
            "time": [0.0, 1.0, 0.0, 1.0],
            "value": [1.0, 2.0, 3.0, 4.0],
            "dose": ["abc", 100.0, 50.0, 50.0],
            "dose_time": [0.0, 0.0, 0.0, 0.0],
        }
    )
    doses: dict[str, Any] = {
        "sample": ["subject"],
        "time_unit": "hr",
        "unit": "mg/l",
        "dose_amount": "dose",
        "dose_unit": "mg",
        "route": Route.ORAL,
    }
    message = r"sample a: the column 'dose' has the non-numeric value 'abc'"
    with pytest.raises(ValueError, match=message):
        Timecourses.from_dataframe(rows, **doses)
    with pytest.raises(ValueError, match=message):
        Timecourses.from_dataframe(rows, **doses, dose_time="dose_time")
    with pytest.raises(
        ValueError, match=r"sample b: the column 'time' has the non-numeric value 'x'"
    ):
        Timecourses.from_dataframe(
            rows.assign(dose=100.0, time=[0.0, 1.0, 0.0, "x"]),
            sample=["subject"],
            time_unit="hr",
            unit="mg/l",
        )
    with pytest.raises(
        ValueError, match=r"sample b: the column 'value' has the non-numeric value 'x'"
    ):
        Timecourses.from_dataframe(
            rows.assign(dose=100.0, value=[1.0, 2.0, "x", 4.0]),
            sample=["subject"],
            time_unit="hr",
            unit="mg/l",
        )
    # a missing value stays a missing value
    batch = Timecourses.from_dataframe(
        rows.assign(dose=100.0, value=[1.0, 2.0, None, 4.0]),
        sample=["subject"],
        time_unit="hr",
        unit="mg/l",
    )
    assert np.isnan(batch.values[1, 0])


def test_from_dataframe_of_an_empty_frame() -> None:
    empty = pd.DataFrame({"subject": [], "time": [], "value": []})
    with pytest.raises(ValueError, match="At least one timecourse"):
        Timecourses.from_dataframe(
            empty, sample=["subject"], time_unit="hr", unit="mg/l"
        )


def test_relative_to_dose_of_a_shared_grid_keeps_it_shared() -> None:
    """Every row shifted by the same dose time keeps the shared time coordinate."""
    batch = Timecourses.from_arrays(
        T + 2.0,
        V,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL, time=2.0),
        substance="x",
    )
    shifted = batch.relative_to_dose()
    np.testing.assert_allclose(shifted.ds["time"].to_numpy(), T)
    assert "times" not in shifted.ds
    dose_time = shifted.dose_time
    assert dose_time is not None
    np.testing.assert_allclose(dose_time, np.zeros((3, 1)))
    np.testing.assert_allclose(shifted.values, batch.values)
    assert shifted.ds["time"].attrs["units"] == "hr"
    assert shifted.unit == "mg/l" and shifted.substance == "x"


def test_relative_to_dose_of_different_dose_times_uses_the_union_grid() -> None:
    """Rows shifted by their own dose time are placed on the union of the grids."""
    curves = [
        Timecourse(
            time=np.array([0.0, 1.0, 2.0]) + offset,
            value=np.array([1.0, 2.0, 3.0]) * (i + 1),
            time_unit="hr",
            unit="mg/l",
            dose=Dose(amount=100, unit="mg", route=Route.ORAL, time=offset),
            label=label,
        )
        for i, (label, offset) in enumerate((("a", 0.0), ("b", 0.5)))
    ]
    batch = Timecourses.from_timecourses(curves)
    shifted = batch.relative_to_dose()
    np.testing.assert_allclose(shifted.times[0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(shifted.times[1], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(shifted.values[1], [2.0, 4.0, 6.0])
    dose_time = shifted.dose_time
    assert dose_time is not None
    np.testing.assert_allclose(dose_time, np.zeros((2, 1)))
    # a row which no longer aligns keeps its own points and is NaN elsewhere
    ragged = Timecourses.from_timecourses(
        [
            curves[0],
            curves[1].model_copy(update={"time": np.array([0.5, 1.7, 2.5])}),
        ]
    )
    on_grid = ragged.relative_to_dose()
    np.testing.assert_allclose(on_grid.ds["time"].to_numpy(), [0.0, 1.0, 1.2, 2.0])
    np.testing.assert_allclose(on_grid.values[0], [1.0, 2.0, np.nan, 3.0])
    np.testing.assert_allclose(on_grid.values[1], [2.0, np.nan, 4.0, 6.0])


def test_relative_to_dose_last_shifts_by_the_last_dose() -> None:
    protocol = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.ORAL), interval=12.0, n_doses=2
    )
    tc = Timecourse(
        time=np.array([0.0, 6.0, 12.0, 18.0]),
        value=np.array([1.0, 2.0, 3.0, 4.0]),
        time_unit="hr",
        unit="mg/l",
        dosing=protocol,
    )
    batch = tc.to_batch()
    last = batch.relative_to_dose(which="last")
    np.testing.assert_allclose(last.times[0], [-12.0, -6.0, 0.0, 6.0])
    dose_time = last.dose_time
    assert dose_time is not None
    np.testing.assert_allclose(dose_time[0], [-12.0, 0.0])
    # a batch without doses is returned unchanged
    plain = Timecourses.from_arrays(T, V, time_unit="hr", unit="mg/l")
    assert plain.relative_to_dose() is plain


def test_relative_to_dose_keeps_the_uncertainty_and_the_coordinates() -> None:
    curves = [
        Timecourse(
            time=np.array([0.0, 1.0, 2.0]) + offset,
            value=np.array([1.0, 2.0, 3.0]),
            sd=np.array([0.1, 0.2, 0.3]),
            n=6,
            time_unit="hr",
            unit="mg/l",
            dose=Dose(amount=100, unit="mg", route=Route.ORAL, time=offset),
            label=label,
        )
        for label, offset in (("a", 0.0), ("b", 0.5))
    ]
    batch = Timecourses.from_timecourses(curves)
    batch.ds.coords["sex"] = ("individual", ["m", "f"])
    shifted = batch.relative_to_dose()
    assert shifted.sd is not None
    np.testing.assert_allclose(shifted.sd[1], [0.1, 0.2, 0.3])
    subjects = shifted.n
    assert subjects is not None
    np.testing.assert_allclose(subjects, [6, 6])
    assert list(shifted.ds.coords["sex"].to_numpy()) == ["m", "f"]
    assert shifted.ds["sd"].attrs["units"] == "mg/l"


def test_to_batch_of_a_single_timecourse() -> None:
    tc = Timecourse(
        time=T,
        value=V[0],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        label="s1",
        substance="x",
    )
    batch = tc.to_batch()
    assert isinstance(batch, Timecourses)
    assert batch.sample_dims == ("individual",)
    assert list(batch.ds.coords["individual"].to_numpy()) == ["s1"]
    assert batch.sel(individual="s1") == tc
    named = tc.to_batch(dim="subject", label="other")
    assert named.sample_dims == ("subject",)
    assert list(named.ds.coords["subject"].to_numpy()) == ["other"]


def test_from_dataframe_keeps_the_label_dtype() -> None:
    """An integer subject column gives integer labels, as `from_timecourses` does."""
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 1, 2, 2, 2],
            "time": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    batch = Timecourses.from_dataframe(
        frame, sample=["subject"], time_unit="hr", unit="mg/l"
    )
    labels = batch.ds.coords["subject"].to_numpy()
    assert labels.dtype.kind == "i"
    assert list(labels) == [1, 2]
    assert batch.sel(subject=2).value[0] == 4.0
    curves = [
        Timecourse(time=T, value=V[i], time_unit="hr", unit="mg/l") for i in range(2)
    ]
    from_curves = Timecourses.from_timecourses(curves, labels=[1, 2])
    assert from_curves.ds.coords["individual"].to_numpy().dtype.kind == "i"


def make_study() -> Timecourses:
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0, 5.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )
    return Timecourses.from_arrays(
        T,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": ["s1", "s2", "s3", "s4"],
            "arm": ("individual", ["a", "a", "b", "b"]),
            "weight": ("individual", [70.0, 80.0, 90.0, 100.0]),
        },
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )


def test_select_by_label_list_slice_and_coordinate() -> None:
    tcs = make_study()
    one = tcs.select(individual="s2")
    assert isinstance(one, Timecourses)
    assert one.sample_dims == ("individual",) and one.n_samples == 1
    assert one.ds["individual"].to_numpy().tolist() == ["s2"]
    assert tcs.select(individual=["s1", "s3"]).ds["individual"].to_numpy().tolist() == [
        "s1",
        "s3",
    ]
    assert tcs.select(individual=slice("s2", "s3")).ds[
        "individual"
    ].to_numpy().tolist() == [
        "s2",
        "s3",
    ]
    arm = tcs.select(arm="a")
    assert arm.ds["individual"].to_numpy().tolist() == ["s1", "s2"]
    np.testing.assert_allclose(arm.values, [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    # both bounds of a coordinate slice are included
    assert tcs.select(weight=slice(80.0, 90.0)).ds[
        "individual"
    ].to_numpy().tolist() == [
        "s2",
        "s3",
    ]
    # several indexers are combined
    assert tcs.select(arm="b", weight=slice(0.0, 95.0)).ds[
        "individual"
    ].to_numpy().tolist() == ["s3"]


def test_select_errors() -> None:
    tcs = make_study()
    with pytest.raises(ValueError, match="neither a sample dimension"):
        tcs.select(study="x")
    with pytest.raises(ValueError, match="no sample of the batch"):
        tcs.select(arm="c")
    # a label no sample carries is the same error, not a KeyError of the index
    with pytest.raises(ValueError, match="no sample of the batch has individual"):
        tcs.select(individual="zzz")
    with pytest.raises(ValueError, match="no sample of the batch"):
        tcs.select(individual=["zzz", "yyy"])


def test_groupby_partitions_in_order_of_appearance() -> None:
    tcs = make_study()
    groups = list(tcs.groupby("arm"))
    assert [value for value, _ in groups] == ["a", "b"]
    assert [group.n_samples for _, group in groups] == [2, 2]
    assert groups[1][1].ds["individual"].to_numpy().tolist() == ["s3", "s4"]
    assert sum(group.n_samples for _, group in groups) == tcs.n_samples
    # a sample dimension groups by its own labels, one sample per group
    assert [value for value, _ in tcs.groupby("individual")] == ["s1", "s2", "s3", "s4"]
    with pytest.raises(ValueError, match="neither a sample dimension"):
        list(tcs.groupby("nope"))


def test_mean_of_known_curves() -> None:
    group = make_study().select(arm="a").mean("individual")
    assert group.sample_dims == () and group.n_samples == 1
    np.testing.assert_allclose(group.values, [2.0, 3.0, 4.0, 5.0])
    sd = group.sd
    se = group.se
    n = group.n
    assert sd is not None and se is not None and n is not None
    np.testing.assert_allclose(sd, np.full(4, np.sqrt(2.0)))
    np.testing.assert_allclose(se, np.full(4, 1.0))
    np.testing.assert_allclose(n, 2.0)
    assert group.unit == "mg/l" and group.time_unit == "hr"
    assert group.substance == "caffeine"
    # the protocol of the group is the shared protocol of its samples
    dose_amount = group.dose_amount
    assert dose_amount is not None
    np.testing.assert_allclose(dose_amount, [100.0])
    assert group.route is Route.ORAL


def test_mean_keeps_the_remaining_dimensions_and_their_coordinates() -> None:
    values = np.arange(24, dtype=float).reshape(2, 3, 4)
    tcs = Timecourses.from_arrays(
        T,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "individual"),
        coords={"dose": [50, 100], "individual": ["s1", "s2", "s3"]},
        substance="caffeine",
    )
    group = tcs.mean("individual")
    assert group.sample_dims == ("dose",)
    assert group.ds["dose"].to_numpy().tolist() == [50, 100]
    np.testing.assert_allclose(group.values, values.mean(axis=1))


def test_mean_with_missing_points_and_min_n() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, np.nan, 5.0, 6.0]])
    tcs = Timecourses.from_arrays(
        T,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["s1", "s2"]},
        substance="caffeine",
    )
    group = tcs.mean("individual")
    n = group.n
    sd = group.sd
    se = group.se
    assert n is not None and sd is not None and se is not None
    np.testing.assert_allclose(group.values, [2.0, 2.0, 4.0, 5.0])
    assert np.isnan(sd[1])  # one value, no scatter
    np.testing.assert_allclose(n, 2.0)
    # se = sd / sqrt(n) with the stored n
    np.testing.assert_allclose(se[0], sd[0] / np.sqrt(2.0))
    # `min_n` drops the points which not every sample covers
    strict = tcs.mean("individual", min_n=2)
    assert np.isnan(strict.values[1])
    np.testing.assert_allclose(strict.values[[0, 2, 3]], [2.0, 4.0, 5.0])
    # with `spread="se"` the standard error uses the count of its own point
    by_se = tcs.mean("individual", spread="se")
    se_by_se = by_se.se
    sd_by_se = by_se.sd
    assert se_by_se is not None and sd_by_se is not None
    np.testing.assert_allclose(se_by_se[0], np.std([1.0, 3.0], ddof=1) / np.sqrt(2.0))
    np.testing.assert_allclose(sd_by_se[0], se_by_se[0] * np.sqrt(2.0))


def test_mean_warns_on_different_protocols(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tcs = Timecourses.from_arrays(
        T,
        V,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        dose={"amount": [100.0, 50.0, 100.0], "unit": "mg", "time": [0.0, 0.0, 0.0]},
        route=Route.ORAL,
        substance="caffeine",
    )
    with caplog.at_level(logging.WARNING, logger="pkpdutils.timecourse"):
        group = tcs.mean("individual")
    assert "dosing protocol" in caplog.text
    dose_amount = group.dose_amount
    assert dose_amount is not None
    np.testing.assert_allclose(dose_amount, [100.0])


def test_mean_of_a_ragged_batch_uses_the_union_grid() -> None:
    times = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 2.0]])
    values = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    tcs = Timecourses.from_arrays(
        times,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"]},
        substance="caffeine",
    )
    group = tcs.mean("individual")
    np.testing.assert_allclose(group.times, [0.0, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(group.values, [2.0, 2.0, 4.0, 4.0])


def test_mean_rejects_an_unknown_dimension() -> None:
    with pytest.raises(ValueError, match="not a sample dimension"):
        make_study().mean("group")
    with pytest.raises(ValueError, match="'min_n'"):
        make_study().mean("individual", min_n=0)


def test_dose_normalized_values_and_unit() -> None:
    tcs = Timecourses.from_arrays(
        T,
        V,
        time_unit="hr",
        unit="ng/ml",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        dose={"amount": [100.0, 50.0, 200.0], "unit": "mg", "time": [0.0, 0.0, 0.0]},
        route=Route.ORAL,
        substance="caffeine",
    )
    normalized = tcs.dose_normalized()
    assert normalized.unit == str((Q_(1.0, "ng/ml") / Q_(1.0, "mg")).units)
    np.testing.assert_allclose(
        normalized.values, V / np.array([100.0, 50.0, 200.0])[:, None]
    )
    # the doses themselves are kept
    dose_amount = normalized.dose_amount
    assert dose_amount is not None
    np.testing.assert_allclose(dose_amount[:, 0], [100.0, 50.0, 200.0])
    # one reference dose for every sample
    reference = tcs.dose_normalized(reference=100.0)
    np.testing.assert_allclose(reference.values, V / 100.0)
    # a quantity is converted to the dose unit of the batch
    quantity = tcs.dose_normalized(reference=Q_(0.1, "g"))
    np.testing.assert_allclose(quantity.values, V / 100.0)


def test_dose_normalized_scales_the_uncertainty() -> None:
    sd = np.abs(V) * 0.1
    tcs = Timecourses.from_arrays(
        T,
        V,
        sd=sd,
        n=6,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        dose=Dose(amount=50, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )
    normalized = tcs.dose_normalized()
    normalized_sd = normalized.sd
    normalized_se = normalized.se
    assert normalized_sd is not None and normalized_se is not None
    np.testing.assert_allclose(normalized_sd, sd / 50.0)
    np.testing.assert_allclose(normalized_se, sd / 50.0 / np.sqrt(6.0))
    assert normalized.ds["sd"].attrs["units"] == normalized.unit


def test_dose_normalized_without_doses() -> None:
    tcs = Timecourses.from_arrays(
        T,
        V,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        substance="caffeine",
    )
    with pytest.raises(ValueError, match="no doses"):
        tcs.dose_normalized()


def test_select_a_dimension_without_labels_by_position() -> None:
    tcs = Timecourses.from_arrays(
        T,
        V,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        substance="caffeine",
    )
    assert tcs.select(individual=1).n_samples == 1
    np.testing.assert_allclose(tcs.select(individual=[0, 2]).values, V[[0, 2]])
    # without labels a slice is the python slice, its stop is exclusive
    np.testing.assert_allclose(tcs.select(individual=slice(0, 2)).values, V[:2])

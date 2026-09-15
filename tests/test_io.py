from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkpdutils import Dosing, Route, Timecourses
from pkpdutils.io import read_adnca, read_events, read_pknca, write_events

DATA = Path(__file__).parent / "data" / "formats"


def events() -> pd.DataFrame:
    return pd.read_csv(DATA / "multiple_dose_events.csv", na_values=".")


def test_read_events_expands_addl_and_ss_and_keeps_covariates() -> None:
    batch = read_events(
        events(), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
    )
    assert batch.sample_dims == ("individual",)
    assert batch.ds["individual"].to_numpy().tolist() == [1, 2, 3]
    assert batch.dosing_of(individual=1) == Dosing(
        amounts=np.array([100.0] * 4),
        times=np.array([0.0, 12.0, 24.0, 36.0]),
        unit="mg",
        route=Route.ORAL,
    )
    assert batch.dosing_of(individual=2) == Dosing(
        amounts=np.array([100.0, 100.0]),
        times=np.array([0.0, 12.0]),
        unit="mg",
        route=Route.ORAL,
    )
    ss = batch.dosing_of(individual=3)
    assert ss is not None
    assert ss.times.tolist() == [-60, -48, -36, -24, -12, 0]
    assert batch.ds.attrs["steady_state_marker"] is True
    assert batch.ds["WT"].to_numpy().tolist() == [70, 82, 65]
    tc = batch.sel(individual=2)
    assert tc.time.tolist() == [1, 13, 24]
    assert tc.value.tolist() == [4.8, 6.0, 1.9]
    assert (
        Timecourses.from_events(
            events(), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
        )
        == batch
    )


def test_read_events_monolix_aliases_and_infusion() -> None:
    df = pd.read_csv(DATA / "monolix_events.csv", na_values=".")
    batch = read_events(
        df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION
    )
    d = batch.dosing_of(individual=1)
    assert d is not None and d.route is Route.IV_INFUSION
    assert d.durations is not None and d.durations.tolist() == [1.0]
    assert batch.sel(individual=1).value.tolist() == [2.1, 4.0, 3.1, 1.0]
    table = write_events(batch)
    assert table["RATE"].tolist() == [50.0, 0.0, 0.0, 0.0, 0.0]
    assert table["AMT"].tolist() == [50.0, 0.0, 0.0, 0.0, 0.0]


def test_read_events_drops_other_events_and_selects_covariates() -> None:
    df = events()
    reset = df.iloc[[1]].copy()
    reset["EVID"] = 3
    df = pd.concat([df, reset], ignore_index=True)
    batch = read_events(
        df,
        time_unit="hr",
        unit="mg/l",
        dose_unit="mg",
        route=Route.ORAL,
        covariates=["wt"],  # the lookup is case-insensitive
    )
    assert batch.ds["WT"].to_numpy().tolist() == [70, 82, 65]
    assert batch.sel(individual=1).time.tolist() == [1, 4, 12, 37, 48]


def test_read_events_rate_and_errors() -> None:
    df = pd.DataFrame(
        {
            "ID": np.array([1, 1]),
            "TIME": np.array([0.0, 1.0]),
            "DV": np.array([np.nan, 2.0]),
            "AMT": np.array([30.0, 0.0]),
            "EVID": np.array([1, 0]),
            "RATE": np.array([15.0, 0.0]),
        }
    )
    d = read_events(
        df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION
    ).dosing_of(individual=1)
    assert d is not None and d.durations is not None and d.durations.tolist() == [2.0]
    df["RATE"] = np.array([-1.0, 0.0])
    with pytest.raises(ValueError, match="Modelled rates"):
        read_events(
            df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION
        )
    with pytest.raises(ValueError, match="column"):
        read_events(
            df.drop(columns=["TIME"]),
            time_unit="hr",
            unit="mg/l",
            dose_unit="mg",
            route=Route.ORAL,
        )


def test_events_round_trip() -> None:
    batch = read_events(
        events(), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
    )
    table = write_events(batch)
    assert list(table.columns)[:7] == ["ID", "TIME", "DV", "AMT", "EVID", "MDV", "RATE"]
    assert "WT" in table.columns
    again = read_events(
        table, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
    )
    for i in (1, 2, 3):
        assert again.dosing_of(individual=i) == batch.dosing_of(individual=i)
        assert again.sel(individual=i) == batch.sel(individual=i)
    assert batch.to_events().equals(table)


def test_read_pknca() -> None:
    conc = pd.read_csv(DATA / "pknca_conc.csv")
    dose = pd.read_csv(DATA / "pknca_dose.csv")
    batch = read_pknca(
        conc,
        dose,
        time_unit="hr",
        unit="mg/l",
        dose_unit="mg",
        route=Route.ORAL,
        groups=["treatment"],
    )
    assert batch.ds["treatment"].to_numpy().tolist() == ["A", "B"]
    assert batch.dosing_of(individual=2) == Dosing(
        amounts=np.array([100.0, 100.0]),
        times=np.array([0.0, 12.0]),
        unit="mg",
        route=Route.ORAL,
    )
    assert batch.sel(individual=1).value.tolist() == [0.0, 4.2, 3.0, 1.0]
    assert (
        Timecourses.from_pknca(
            conc,
            dose,
            time_unit="hr",
            unit="mg/l",
            dose_unit="mg",
            route=Route.ORAL,
            groups=["treatment"],
        )
        == batch
    )


def test_read_pknca_subject_without_doses() -> None:
    conc = pd.read_csv(DATA / "pknca_conc.csv")
    dose = pd.read_csv(DATA / "pknca_dose.csv")
    batch = read_pknca(
        conc,
        dose[dose["subject"] != 1],
        time_unit="hr",
        unit="mg/l",
        dose_unit="mg",
        route=Route.ORAL,
    )
    assert batch.has_dose
    assert batch.dosing_of(individual=1) is None
    assert batch.dosing_of(individual=2) is not None
    assert batch.sel(individual=1).dosing is None


def test_read_adnca_analyte_route_and_units() -> None:
    df = pd.read_csv(DATA / "adnca.csv")
    two = pd.concat(
        [df, df.assign(PARAMCD="MET", AVAL=df["AVAL"] / 2)], ignore_index=True
    )
    batch = read_adnca(two, analyte="MET", route=Route.IV_BOLUS, unit="ng/ml")
    assert batch.substance == "MET" and batch.route is Route.IV_BOLUS
    assert batch.unit == "ng/ml"
    assert batch.sel(individual="S1").value.tolist() == [0.025, 2.1, 0.5]
    with pytest.raises(ValueError, match="route"):
        read_adnca(df.assign(ROUTE="SUBLINGUAL"))


def test_read_adnca() -> None:
    df = pd.read_csv(DATA / "adnca.csv")
    batch = read_adnca(df)
    assert batch.unit in ("ng/mL", "nanogram / milliliter")
    assert batch.route is Route.ORAL and batch.substance == "XAN"
    assert batch.dosing_of(individual="S2") == Dosing(
        amounts=np.array([100.0, 100.0]),
        times=np.array([0.0, 12.0]),
        unit="mg",
        route=Route.ORAL,
    )
    # the COPY row is dropped
    assert batch.sel(individual="S2").time.tolist() == [1, 12, 13, 24]
    assert batch.ds["lloq"].to_numpy().tolist() == [0.1, 0.1]
    with pytest.raises(ValueError, match="analyte"):
        read_adnca(pd.concat([df, df.assign(PARAMCD="OTHER")]))
    assert Timecourses.from_adnca(df) == batch

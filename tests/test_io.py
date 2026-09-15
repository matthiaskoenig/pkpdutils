from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkpdutils import Dose, Dosing, Route, Timecourses, nca
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


def test_read_events_repeated_doses_need_an_interval() -> None:
    df = events()
    df.loc[df["ID"] == 1, "II"] = 0  # ADDL 3 without an interdose interval
    with pytest.raises(ValueError, match=r"'ADDL'/'SS' need a positive 'II'"):
        read_events(df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)
    df = events()
    df["ADDL"] = 0
    df = df.drop(columns=["II"])  # SS 1 of subject 3 without an interval column
    with pytest.raises(ValueError, match="subject '3'") as excinfo:
        read_events(df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)
    assert "'ADDL'/'SS' need a positive 'II'" in str(excinfo.value)


def test_read_events_without_evid_reads_a_dose_row_as_a_dose_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # a bolus written without an `EVID` column: the dose row carries `DV = 0`,
    # which is not an observation of the subject (NM-TRAN). Read as one, it
    # would add the point (0, 0) and destroy `c0` and `auc_last`
    df = pd.DataFrame(
        {
            "ID": np.array([1, 1, 1, 1]),
            "TIME": np.array([0.0, 1.0, 4.0, 8.0]),
            "DV": np.array([0.0, 4.0, 2.0, 1.0]),
            "AMT": np.array([100.0, 0.0, 0.0, 0.0]),
        }
    )
    with caplog.at_level("WARNING", logger="pkpdutils.io"):
        batch = read_events(
            df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_BOLUS
        )
    tc = batch.sel(individual=1)
    assert tc.time.tolist() == [1.0, 4.0, 8.0]
    assert tc.value.tolist() == [4.0, 2.0, 1.0]
    assert batch.dosing_of(individual=1) == Dosing(
        amounts=np.array([100.0]),
        times=np.array([0.0]),
        unit="mg",
        route=Route.IV_BOLUS,
    )
    assert (
        "1 dose rows carry a DV value which is ignored (no EVID column)" in caplog.text
    )


def test_read_events_rejects_evid_4() -> None:
    df = events()
    df.loc[df["ID"] == 2, "EVID"] = df.loc[df["ID"] == 2, "EVID"].replace(1, 4)
    with pytest.raises(ValueError, match=r"EVID 4 \(reset and dose\) at subject 2"):
        read_events(df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)


def test_read_events_rejects_an_unknown_ss_value() -> None:
    df = events()
    df.loc[df["ID"] == 3, "SS"] = df.loc[df["ID"] == 3, "SS"].replace(1, 2)
    with pytest.raises(ValueError, match=r"Unknown 'SS' value 2 at subject 3"):
        read_events(df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)


def test_read_events_warns_about_a_varying_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = events()
    df["OCCASION"] = np.arange(len(df))  # not constant within a subject
    with caplog.at_level("WARNING", logger="pkpdutils.io"):
        batch = read_events(
            df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
        )
    assert "OCCASION" not in batch.ds.coords
    assert "OCCASION varies within a subject" in caplog.text
    assert "WT" in batch.ds.coords


def test_read_events_monolix_spelled_out_aliases() -> None:
    df = pd.DataFrame(
        {
            "ID": np.array([1, 1, 1]),
            "TIME": np.array([0.0, 1.0, 4.0]),
            "OBSERVATION": np.array([np.nan, 4.0, 2.0]),
            "AMOUNT": np.array([100.0, np.nan, np.nan]),
            "INFUSION DURATION": np.array([0.5, np.nan, np.nan]),
            "ADDITIONAL DOSES": np.array([1.0, np.nan, np.nan]),
            "INTERDOSE INTERVAL": np.array([12.0, np.nan, np.nan]),
            "STEADY STATE": np.array([0.0, np.nan, np.nan]),
        }
    )
    batch = read_events(
        df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION
    )
    d = batch.dosing_of(individual=1)
    assert d is not None and d.times.tolist() == [0.0, 12.0]
    assert d.durations is not None and d.durations.tolist() == [0.5, 0.5]
    assert batch.sel(individual=1).value.tolist() == [4.0, 2.0]


def test_read_events_rate_and_errors() -> None:
    df = pd.DataFrame(
        {
            "ID": np.array([1, 1, 1]),
            "TIME": np.array([0.0, 1.0, 2.0]),
            "DV": np.array([np.nan, 2.0, 1.0]),
            "AMT": np.array([30.0, 0.0, 0.0]),
            "EVID": np.array([1, 0, 0]),
            "RATE": np.array([15.0, 0.0, 0.0]),
        }
    )
    d = read_events(
        df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION
    ).dosing_of(individual=1)
    assert d is not None and d.durations is not None and d.durations.tolist() == [2.0]
    df["RATE"] = np.array([-1.0, 0.0, 0.0])
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
    # the subject without doses is analysed: the dose-independent parameters
    # are computed on its times as they are, only the dose-dependent ones are
    # `NaN`, and the sample is not flagged as having no data
    result = nca(batch)
    cmax = result["cmax"].to_numpy()
    assert np.isfinite(cmax).all() and cmax[0] == pytest.approx(4.2)
    assert result["n_doses"].to_numpy().tolist() == [0.0, 2.0]
    cl_ss_f = result["cl_ss_f"].to_numpy()
    assert np.isnan(cl_ss_f[0]) and np.isfinite(cl_ss_f[1])
    assert not result.flag_table()["NO_DATA"].any()
    # the batch is analysed over the dosing intervals: no single dose quantities
    assert np.isnan(result["cl_f"].to_numpy()).all()


def test_read_pknca_infusion_duration() -> None:
    conc = pd.read_csv(DATA / "pknca_conc.csv")
    dose = pd.read_csv(DATA / "pknca_dose.csv").assign(duration=0.5)
    batch = read_pknca(
        conc,
        dose,
        time_unit="hr",
        unit="mg/l",
        dose_unit="mg",
        route=Route.IV_INFUSION,
        duration_col="duration",
    )
    assert batch.route is Route.IV_INFUSION
    d = batch.dosing_of(individual=2)
    assert d is not None and d.durations is not None
    assert d.durations.tolist() == [0.5, 0.5]


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


def test_read_adnca_cannot_read_an_infusion_protocol() -> None:
    # the dataset carries no duration: an infusion protocol is not readable
    df = pd.read_csv(DATA / "adnca.csv")
    with pytest.raises(ValueError, match="duration"):
        read_adnca(df, route=Route.IV_INFUSION)


def test_read_adnca() -> None:
    df = pd.read_csv(DATA / "adnca.csv")
    batch = read_adnca(df)
    assert batch.unit == "ng/mL"  # the unit as the 'AVALU' column spells it
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


def test_events_round_trip_keeps_the_uncertainty() -> None:
    # B6: `sd`, `se` and `n` were silently dropped by the only writer
    batch = Timecourses.from_arrays(
        np.array([1.0, 2.0, 4.0]),
        np.array([[1.0, 2.0, 1.0], [1.2, 2.4, 1.1]]),
        time_unit="hr",
        unit="ng/ml",
        sd=np.array([[0.1, 0.2, 0.1], [0.2, 0.3, 0.2]]),
        n=np.array([8.0, 6.0]),
        coords={"individual": [0, 1]},
        dose=Dose(amount=100, unit="mg"),
    )
    table = write_events(batch)
    assert list(table.columns) == [
        "ID",
        "TIME",
        "DV",
        "AMT",
        "EVID",
        "MDV",
        "RATE",
        "SD",
        "SE",
        "N",
    ]
    back = read_events(
        table, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL
    )
    assert back == batch


def test_write_events_without_uncertainty_has_no_extra_columns() -> None:
    batch = read_events(
        events(), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
    )
    assert "SD" not in write_events(batch).columns


def test_read_events_rejects_a_subject_without_two_observations() -> None:
    # B8: the reader built batches which `sel`/`isel`/`__iter__` then rejected
    df = pd.DataFrame(
        {
            "ID": ["A", "A", "A", "B", "B"],
            "TIME": [0.0, 1.0, 2.0, 0.0, 1.0],
            "DV": [np.nan, 5.0, 3.0, np.nan, 4.0],
            "AMT": [100.0, 0.0, 0.0, 100.0, 0.0],
            "EVID": [1, 0, 0, 1, 0],
        }
    )
    with pytest.raises(ValueError, match=r"subject B: .*2 time points"):
        read_events(
            df,
            time_unit="hr",
            unit="ng/ml",
            dose_unit="mg",
            route=Route.ORAL,
            keep_missing=False,
        )
    only_doses = df[df["EVID"] == 1]
    with pytest.raises(ValueError, match="subject A"):
        read_events(
            only_doses, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL
        )


def test_readers_reject_duplicate_sampling_times() -> None:
    # B9: duplicate times passed the readers, crashed `sel` and skewed the NCA
    conc = pd.DataFrame(
        {
            "subject": [1, 1, 1, 1],
            "time": [0.5, 0.5, 1.0, 2.0],
            "conc": [1.0, 1.0, 2.0, 1.5],
        }
    )
    dose = pd.DataFrame({"subject": [1], "dose": [100.0], "time": [0.0]})
    with pytest.raises(ValueError, match=r"subject 1: duplicate sampling time 0\.5"):
        read_pknca(
            conc, dose, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL
        )
    df = pd.DataFrame(
        {
            "ID": [1, 1, 1],
            "TIME": [1.0, 1.0, 2.0],
            "DV": [1.0, 9.0, 1.5],
            "AMT": [0.0, 0.0, 0.0],
            "EVID": [0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="subject 1: duplicate sampling time 1"):
        read_events(df, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL)


def test_read_events_infusion_without_a_duration_names_the_subject() -> None:
    # B30: a raw pydantic dump named neither the subject nor the column
    df = pd.DataFrame(
        {
            "ID": [1, 1, 1, 1],
            "TIME": [0.0, 1.0, 2.0, 4.0],
            "DV": [np.nan, 5.0, 3.0, 1.0],
            "AMT": [100.0, 0.0, 0.0, 0.0],
            "EVID": [1, 0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match=r"subject 1: .*duration"):
        read_events(
            df,
            time_unit="hr",
            unit="mg/l",
            dose_unit="mg",
            route=Route.IV_INFUSION,
        )


def test_read_pknca_nan_dose_time_names_the_subject() -> None:
    # B32: the error named the internal flat row, not the subject
    conc = pd.DataFrame({"subject": [1, 1], "time": [1.0, 2.0], "conc": [1.0, 2.0]})
    dose = pd.DataFrame({"subject": [1], "dose": [100.0], "time": [np.nan]})
    with pytest.raises(ValueError, match=r"subject 1: .*finite"):
        read_pknca(
            conc, dose, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL
        )


def test_events_round_trip_keeps_a_shared_grid_with_missing_values() -> None:
    # B33: BLQ points turned a shared grid into a ragged one on every round trip
    values = np.array([[1.0, 2.0, 1.5, 0.8, 0.2], [1.1, 2.2, 1.4, 0.7, 0.1]])
    values[0, 0] = np.nan
    values[1, -1] = np.nan
    batch = Timecourses.from_arrays(
        np.array([0.5, 1.0, 2.0, 4.0, 8.0]),
        values,
        time_unit="hr",
        unit="ng/ml",
        coords={"individual": [0, 1]},
        dose=Dose(amount=100, unit="mg"),
    )
    table = batch.to_events()
    back = Timecourses.from_events(
        table, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL
    )
    assert "times" not in back.ds
    assert back == batch
    dropped = read_events(
        table,
        time_unit="hr",
        unit="ng/ml",
        dose_unit="mg",
        route=Route.ORAL,
        keep_missing=False,
    )
    assert "times" in dropped.ds
    assert dropped.sel(individual=0).time.tolist() == [1.0, 2.0, 4.0, 8.0]


def test_readers_coerce_a_route_string() -> None:
    batch = read_events(
        events(),
        time_unit="hr",
        unit="mg/l",
        dose_unit="mg",
        route="oral",
    )
    assert batch.route is Route.ORAL

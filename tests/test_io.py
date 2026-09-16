from pathlib import Path
from typing import Any

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
        covariates=["treatment"],
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
            covariates=["treatment"],
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


def test_read_adnca_covariates_become_coordinates() -> None:
    """Every reader takes `covariates`, the columns kept along the sample dimension."""
    df = pd.read_csv(DATA / "adnca.csv")
    weights = {"S1": 70.0, "S2": 82.0}
    with_covariate = df.assign(WEIGHT=df["USUBJID"].map(weights))
    batch = read_adnca(with_covariate, covariates=["WEIGHT"])
    assert batch.ds["WEIGHT"].to_numpy().tolist() == [70.0, 82.0]
    with pytest.raises(ValueError, match="no 'AGE' column"):
        read_adnca(with_covariate, covariates=["AGE"])
    varying = with_covariate.copy()
    varying.loc[varying.index[0], "WEIGHT"] = 99.0
    with pytest.raises(ValueError, match="not constant"):
        read_adnca(varying, covariates=["WEIGHT"])


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


def test_write_events_of_a_count_per_time_point_writes_the_number_of_subjects() -> None:
    # the group curve of a ragged batch counts every time point on its own; the
    # event format has one `N` per subject, which is the size of the group
    batch = Timecourses.from_arrays(
        np.array([1.0, 2.0, 4.0]),
        np.array([[1.0, 2.0, 1.0]]),
        time_unit="hr",
        unit="ng/ml",
        sd=np.array([[0.1, 0.2, 0.1]]),
        n=np.array([[3.0, 3.0, 2.0]]),
        coords={"individual": ["group"]},
        dose=Dose(amount=100, unit="mg"),
    )
    assert batch.ds["n"].dims == ("individual", "time")
    table = write_events(batch)
    assert table["N"].dropna().unique().tolist() == [3.0]
    back = read_events(
        table, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL
    )
    assert back.n is not None
    np.testing.assert_allclose(back.n, [3.0])


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


def generated_events(n_subjects: int = 60) -> pd.DataFrame:
    """An event table with the four dose patterns and a missing observation.

    Subject `i` carries `i % 4 + 3` observations on a shifted grid and, by
    `i % 4`, a single dose, `ADDL` doses, a steady state dose or two dose rows
    given in the wrong order.
    """
    rows: list[dict[str, Any]] = []
    for i in range(1, n_subjects + 1):
        weight = 60.0 + i
        shared = {"ID": f"s{i}", "WT": weight, "N": float(4 + i % 5)}
        dose = {**shared, "DV": np.nan, "EVID": 1, "MDV": 1, "ADDL": 0, "II": np.nan}
        if i % 4 == 0:
            rows.append({**dose, "TIME": 0.0, "AMT": 100.0, "SS": 0})
        elif i % 4 == 1:
            rows.append(
                {**dose, "TIME": 0.0, "AMT": 50.0, "ADDL": 3, "II": 12.0, "SS": 0}
            )
        elif i % 4 == 2:
            rows.append({**dose, "TIME": 0.0, "AMT": 75.0, "II": 8.0, "SS": 1})
        else:
            rows.append({**dose, "TIME": 24.0, "AMT": 25.0, "SS": 0})
            rows.append({**dose, "TIME": 0.0, "AMT": 25.0, "SS": 0})
        times = 0.5 * np.arange(1, i % 4 + 4) + 0.25 * (i % 3)
        for k, t in enumerate(times):
            missing = (i + k) % 7 == 0
            rows.append(
                {
                    **shared,
                    "TIME": float(t),
                    "DV": np.nan if missing else round(10.0 - 0.1 * i + k, 3),
                    "AMT": 0.0,
                    "EVID": 0,
                    "MDV": 1 if missing else 0,
                    "ADDL": 0,
                    "II": np.nan,
                    "SS": 0,
                }
            )
    return pd.DataFrame(rows)


def test_read_events_expands_every_dose_record_of_the_table_at_once() -> None:
    # B3: the reader builds the per subject arrays and the dose expansion with
    # numpy instead of a scalar pandas lookup per row; the batch is the one the
    # per row implementation built
    df = generated_events()
    batch = read_events(
        df,
        time_unit="hr",
        unit="mg/l",
        dose_unit="mg",
        route=Route.ORAL,
        n_col="N",
    )
    assert batch.sample_shape == (60,)
    assert batch.ds["individual"].to_numpy()[:3].tolist() == ["s1", "s2", "s3"]
    assert batch.ds["WT"].to_numpy()[:3].tolist() == [61.0, 62.0, 63.0]
    assert batch.ds.attrs["steady_state_marker"] is True
    # the four dose patterns: ADDL, steady state, the unsorted pair, one dose
    assert batch.dosing_of(individual="s1") == Dosing(
        amounts=[50.0] * 4, times=[0.0, 12.0, 24.0, 36.0], unit="mg", route=Route.ORAL
    )
    assert batch.dosing_of(individual="s2") == Dosing(
        amounts=[75.0] * 6,
        times=[-40.0, -32.0, -24.0, -16.0, -8.0, 0.0],
        unit="mg",
        route=Route.ORAL,
    )
    assert batch.dosing_of(individual="s3") == Dosing(
        amounts=[25.0, 25.0], times=[0.0, 24.0], unit="mg", route=Route.ORAL
    )
    assert batch.dosing_of(individual="s4") == Dosing(
        amounts=[100.0], times=[0.0], unit="mg", route=Route.ORAL
    )
    # the observations of a subject, in time order and with the missing value
    tc = batch.sel(individual="s6")
    assert tc.time.tolist() == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert np.isnan(tc.value[1]) and tc.value[0] == pytest.approx(9.4)
    # the whole batch, against the values of the per row implementation
    assert batch.n_time == 6 and batch.n_dose == 6
    n = batch.n
    assert n is not None and n.tolist()[:5] == [5.0, 6.0, 7.0, 8.0, 4.0]
    assert int(np.isfinite(batch.values).sum()) == 233
    assert float(np.nansum(batch.values)) == pytest.approx(2059.9)
    assert float(np.nansum(batch.times)) == pytest.approx(457.5)
    amounts = batch.dose_amount
    times = batch.dose_time
    assert amounts is not None and times is not None
    assert float(np.nansum(amounts)) == pytest.approx(12000.0)
    assert float(np.nansum(times)) == pytest.approx(-360.0)


def test_read_events_reads_a_table_of_20000_rows() -> None:
    df = generated_events(3500)
    assert len(df) > 20_000
    batch = read_events(
        df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
    )
    assert batch.sample_shape == (3500,)
    assert int(np.isfinite(batch.values).sum()) == 13500
    assert float(np.nansum(batch.values)) == pytest.approx(-2202250.0)
    assert batch.dosing_of(individual="s3497") == Dosing(
        amounts=[50.0] * 4, times=[0.0, 12.0, 24.0, 36.0], unit="mg", route=Route.ORAL
    )


def test_read_events_reports_the_first_subject_of_every_check() -> None:
    df = generated_events(8)
    without_interval = df.copy()
    without_interval.loc[without_interval["ID"] == "s5", "II"] = 0.0
    with pytest.raises(ValueError, match=r"subject 's5' has the dose record at time 0"):
        read_events(
            without_interval, time_unit="hr", unit="mg/l", dose_unit="mg", route="oral"
        )
    varying = df.copy()
    varying.loc[varying.index[-1], "N"] = 99.0
    with pytest.raises(ValueError, match=r"subject s8: 'N' is not constant"):
        read_events(
            varying,
            time_unit="hr",
            unit="mg/l",
            dose_unit="mg",
            route="oral",
            n_col="N",
        )


def test_read_events_drops_the_rows_without_a_subject(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = generated_events(8)
    df.loc[df["ID"] == "s3", "ID"] = np.nan
    with caplog.at_level("WARNING", logger="pkpdutils.io"):
        batch = read_events(
            df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL
        )
    assert "Dropped 8 rows without a value in 'ID'" in caplog.text
    assert batch.ds["individual"].to_numpy().tolist() == [
        "s1",
        "s2",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
    ]


def two_dose_batch(route: Route = Route.ORAL) -> Timecourses:
    """Two subjects of two doses, with a covariate and a limit of quantification."""
    from pkpdutils import Timecourse

    time = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 13.0, 14.0, 16.0, 20.0, 24.0])
    value = np.array([2.0, 5.0, 4.0, 3.0, 1.5, 0.6, 3.0, 5.5, 4.2, 2.0, 1.0])
    durations = [0.5, 0.5] if route is Route.IV_INFUSION else None
    curves = [
        Timecourse(
            time=time,
            value=(1.0 + 0.1 * index) * value,
            time_unit="hr",
            unit="ng/ml",
            substance="drug",
            lloq=0.1,
            dosing=Dosing(
                amounts=[100.0, 100.0],
                times=[0.0, 12.0],
                durations=durations,
                unit="mg",
                route=route,
            ),
            label=f"S{index + 1}",
        )
        for index in range(2)
    ]
    batch = Timecourses.from_timecourses(curves)
    return Timecourses(batch.ds.assign_coords(WT=("individual", [70.0, 82.0])))


def assert_same_batch(back: Timecourses, batch: Timecourses) -> None:
    """The values, the times, the protocol and the labels of two batches agree."""
    np.testing.assert_allclose(back.times, batch.times)
    np.testing.assert_allclose(back.values, batch.values)
    np.testing.assert_allclose(back.dose_amount, batch.dose_amount)  # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(back.dose_time, batch.dose_time)  # ty: ignore[no-matching-overload]
    assert list(back.ds["individual"].to_numpy()) == list(
        batch.ds["individual"].to_numpy()
    )
    assert back.unit == batch.unit
    assert back.time_unit == batch.time_unit
    assert back.dose_unit == batch.dose_unit


@pytest.mark.parametrize("route", [Route.ORAL, Route.IV_INFUSION])
def test_write_pknca_round_trips_through_read_pknca(
    route: Route, tmp_path: Path
) -> None:
    """The two tables of a batch read back as the same batch."""
    from pkpdutils.io import write_pknca

    batch = two_dose_batch(route)
    conc, dose = write_pknca(batch, tmp_path / "conc.csv", tmp_path / "dose.csv")
    assert list(conc.columns) == ["subject", "lloq", "WT", "time", "conc"]
    assert (pd.read_csv(tmp_path / "dose.csv")["dose"] == 100.0).all()
    back = read_pknca(
        conc,
        dose,
        time_unit="hr",
        unit="ng/ml",
        dose_unit="mg",
        route=route,
        duration_col="duration",
        covariates=["WT", "lloq"],
        substance="drug",
    )
    assert_same_batch(back, batch)
    assert back.route is route
    np.testing.assert_allclose(back.dose_duration, batch.dose_duration)  # ty: ignore[no-matching-overload]
    assert back.lloq is not None
    np.testing.assert_allclose(back.lloq, [0.1, 0.1])
    assert back.ds["WT"].to_numpy().tolist() == [70.0, 82.0]


@pytest.mark.parametrize("route", [Route.ORAL, Route.IV_INFUSION])
def test_write_adnca_round_trips_through_read_adnca(
    route: Route, tmp_path: Path
) -> None:
    """The dataset of a batch reads back as the same batch, duration and all."""
    from pkpdutils.io import write_adnca

    batch = two_dose_batch(route)
    frame = write_adnca(batch, tmp_path / "adnca.csv")
    assert set(frame["ROUTE"]) == {"ORAL" if route is Route.ORAL else "IV INFUSION"}
    back = read_adnca(pd.read_csv(tmp_path / "adnca.csv"), covariates=["WT"])
    assert_same_batch(back, batch)
    assert back.route is route
    assert back.substance == "drug"
    np.testing.assert_allclose(back.dose_duration, batch.dose_duration)  # ty: ignore[no-matching-overload]
    assert back.lloq is not None
    np.testing.assert_allclose(back.lloq, [0.1, 0.1])
    assert back.ds["WT"].to_numpy().tolist() == [70.0, 82.0]


def test_read_adnca_without_a_duration_column_keeps_raising_for_an_infusion() -> None:
    """`ADUR` is optional; without it an infusion protocol cannot be read."""
    df = pd.read_csv(DATA / "adnca.csv")
    with pytest.raises(ValueError, match="infusion"):
        read_adnca(df, analyte="XAN", route=Route.IV_INFUSION)


def analyte_events() -> pd.DataFrame:
    """An event table of two analytes, the dose rows naming none."""
    rows = []
    for subject in (1, 2):
        rows.append({"ID": subject, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "EVID": 1})
        for time, value in zip([1.0, 2.0, 4.0, 8.0], [5.0, 4.0, 3.0, 1.0], strict=True):
            for analyte, factor in (("PARENT", 1.0), ("META", 0.5)):
                rows.append(
                    {
                        "ID": subject,
                        "TIME": time,
                        "DV": factor * value * subject,
                        "AMT": 0.0,
                        "EVID": 0,
                        "ANALYTE": analyte,
                    }
                )
    return pd.DataFrame(rows)


def test_read_events_reads_several_analytes_into_one_batch() -> None:
    """`analytes` gives the analyte dimension and the substance coordinate."""
    batch = read_events(
        analyte_events(),
        time_unit="hr",
        unit="ng/ml",
        dose_unit="mg",
        route=Route.ORAL,
        analyte_col="ANALYTE",
        analytes=["PARENT", "META"],
    )
    assert batch.sample_dims == ("analyte", "individual")
    assert list(batch.ds["analyte"].to_numpy()) == ["PARENT", "META"]
    substances = batch.substances
    assert substances is not None
    assert list(substances[:, 0]) == ["PARENT", "META"]
    # the dose rows name no analyte and belong to both of them
    np.testing.assert_allclose(batch.dose_amount, 100.0)  # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(batch.values[1], 0.5 * batch.values[0])
    assert "ANALYTE" not in batch.ds.coords


def test_read_pknca_reads_several_analytes_into_one_batch() -> None:
    """The analyte column of the concentration table gives the dimension."""
    conc = pd.DataFrame(
        [
            {"subject": subject, "analyte": analyte, "time": time, "conc": value}
            for subject in ("S1", "S2")
            for analyte, factor in (("parent", 1.0), ("metabolite", 0.5))
            for time, value in zip(
                [1.0, 2.0, 4.0, 8.0], [5.0, 4.0, 3.0, 1.0], strict=True
            )
            for value in [factor * value]
        ]
    )
    dose = pd.DataFrame(
        [{"subject": subject, "time": 0.0, "dose": 100.0} for subject in ("S1", "S2")]
    )
    batch = read_pknca(
        conc,
        dose,
        time_unit="hr",
        unit="ng/ml",
        dose_unit="mg",
        route=Route.ORAL,
        analyte_col="analyte",
        analytes=["parent", "metabolite"],
    )
    assert batch.sample_dims == ("analyte", "individual")
    np.testing.assert_allclose(batch.values[1], 0.5 * batch.values[0])
    np.testing.assert_allclose(batch.dose_amount, 100.0)  # ty: ignore[no-matching-overload]


def test_read_adnca_reads_several_analytes_and_an_analyte_which_is_not_there() -> None:
    """`analytes` of the ADNCA reader, and the error of an unknown analyte."""
    df = pd.read_csv(DATA / "adnca.csv")
    other = df.copy()
    other["PARAMCD"] = "MET"
    other["AVAL"] = other["AVAL"] * 0.5
    both = pd.concat([df, other], ignore_index=True)
    batch = read_adnca(both, analytes=["XAN", "MET"])
    assert batch.sample_dims == ("analyte", "individual")
    substances = batch.substances
    assert substances is not None
    assert list(substances[:, 0]) == ["XAN", "MET"]
    with pytest.raises(ValueError, match="either 'analyte' or 'analytes'"):
        read_adnca(both, analyte="XAN", analytes=["XAN"])
    with pytest.raises(ValueError, match="No record of the analyte"):
        read_adnca(both, analytes=["XAN", "NOPE"])


def test_write_pknca_needs_the_sample_dimensions_of_a_table() -> None:
    """A batch of two unrelated sample dimensions cannot be written as a table."""
    from pkpdutils.io import write_adnca, write_pknca

    batch = Timecourses.from_arrays(
        np.array([0.0, 1.0, 2.0]),
        np.ones((2, 2, 3)),
        time_unit="hr",
        unit="mg/l",
        dims=("group", "individual"),
        dose=Dose(amount=1.0, unit="mg"),
    )
    with pytest.raises(ValueError, match="one sample dimension"):
        write_pknca(batch)
    with pytest.raises(ValueError, match="one sample dimension"):
        write_adnca(batch)


def test_read_adnca_reads_the_nominal_time_and_writes_it_back(tmp_path: Path) -> None:
    """`NRRLT` becomes the variable `nominal_time` and round trips."""
    from pkpdutils.io import write_adnca

    frame = pd.DataFrame(
        [
            {
                "USUBJID": subject,
                "PARAMCD": "DRUG",
                "AVAL": value,
                "AVALU": "ng/mL",
                "AFRLT": actual,
                "ARRLT": actual,
                "NRRLT": nominal,
                "DOSEA": 100.0,
                "DOSEU": "mg",
                "ROUTE": "ORAL",
            }
            for subject in ("S1", "S2")
            for actual, nominal, value in (
                (0.55, 0.5, 2.0),
                (1.05, 1.0, 5.0),
                (2.1, 2.0, 4.0),
                (4.0, 4.0, 2.0),
            )
        ]
    )
    batch = read_adnca(frame)
    assert "nominal_time" in batch.ds.data_vars
    assert batch.ds["nominal_time"].dims == ("individual", "time")
    assert batch.ds["nominal_time"].attrs["units"] == "hr"
    np.testing.assert_allclose(
        batch.ds["nominal_time"].to_numpy(), [[0.5, 1.0, 2.0, 4.0]] * 2
    )
    np.testing.assert_allclose(batch.times, [[0.55, 1.05, 2.1, 4.0]] * 2)
    written = write_adnca(batch, tmp_path / "adnca.csv")
    np.testing.assert_allclose(written["NRRLT"], frame["NRRLT"])
    back = read_adnca(pd.read_csv(tmp_path / "adnca.csv"))
    np.testing.assert_allclose(
        back.ds["nominal_time"].to_numpy(), batch.ds["nominal_time"].to_numpy()
    )


def test_read_adnca_without_a_nominal_time_column_carries_none() -> None:
    """The column is optional; the fixture carries it and is read with it."""
    df = pd.read_csv(DATA / "adnca.csv")
    with_nominal = read_adnca(df, analyte="XAN")
    np.testing.assert_allclose(
        with_nominal.ds["nominal_time"].to_numpy()[0], [0.5, 1.0, 12.0, np.nan]
    )
    batch = read_adnca(df.drop(columns=["NFRLT", "NRRLT"]), analyte="XAN")
    assert "nominal_time" not in batch.ds.data_vars

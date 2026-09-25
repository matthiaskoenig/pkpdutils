"""The result tables of Phoenix WinNonlin, PKNCA and NonCompart."""

import logging

import numpy as np
import pandas as pd
import pytest

from pkpdutils import Dose, Route, Timecourse, Timecourses, nca
from pkpdutils.crosswalk import (
    PKNCA_COLUMNS,
    WINNONLIN_NAMES,
    read_noncompart,
    read_pknca_results,
    read_winnonlin,
    to_noncompart,
    to_winnonlin,
    write_noncompart,
    write_pknca_results,
    write_winnonlin,
)
from pkpdutils.nca.result import NCAResult

TIME = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])


def batch(route: Route) -> Timecourses:
    curves = [
        Timecourse(
            time=TIME,
            value=scale * (np.exp(-0.2 * TIME) - np.exp(-1.5 * TIME))
            if route is Route.ORAL
            else scale * np.exp(-0.2 * TIME),
            time_unit="hr",
            unit="mg/l",
            dose=Dose(amount=100, unit="mg", route=route),
            substance="drug",
            label=label,
        )
        for label, scale in (("s1", 10.0), ("s2", 8.0))
    ]
    return Timecourses.from_timecourses(curves, dim="subject")


def result(route: Route) -> NCAResult:
    return nca(batch(route))


def test_winnonlin_columns_follow_the_phoenix_order() -> None:
    table = to_winnonlin(result(Route.ORAL))
    assert table.columns[0] == "subject"
    order = list(WINNONLIN_NAMES.values())
    positions = [order.index(c) for c in table.columns[1:]]
    assert positions == sorted(positions)
    # an intravenous parameter is not written for an extravascular dose
    assert "Cl_F_obs" in table and "Cl_obs" not in table and "C0" not in table


def test_winnonlin_writes_percentages() -> None:
    r = result(Route.ORAL)
    table = to_winnonlin(r)
    fractions = r.ds["auc_extrap_fraction"].to_numpy()
    np.testing.assert_allclose(table["AUC_%Extrap_obs"], 100 * fractions)


@pytest.mark.parametrize("route", [Route.ORAL, Route.IV_BOLUS])
def test_winnonlin_round_trip(route: Route, tmp_path) -> None:
    r = result(route)
    written = write_winnonlin(r, tmp_path / "phoenix.csv")
    back = read_winnonlin(tmp_path / "phoenix.csv")
    assert list(back.index) == ["s1", "s2"]
    assert len(back.columns) == len(written.columns) - 1
    ours = r.to_dataframe().set_index("subject")
    for name in back.columns:
        np.testing.assert_allclose(back[name], ours[name], rtol=1e-12, err_msg=name)


def test_read_winnonlin_drops_the_correlation(caplog) -> None:
    frame = pd.DataFrame(
        {"Subject": [1], "Rsq": [0.99], "Corr_XY": [-0.995], "Extra": [1.0]}
    )
    with caplog.at_level(logging.INFO, logger="pkpdutils.crosswalk"):
        table = read_winnonlin(frame)
    assert list(table.columns) == ["lambda_z_r2"]
    assert "Extra" in caplog.text


def test_noncompart_names_the_mean_residence_time_by_route() -> None:
    oral = to_noncompart(result(Route.ORAL))
    bolus = to_noncompart(result(Route.IV_BOLUS))
    assert {"MRTEVLST", "MRTEVIFO", "MRTEVIFP", "CLFO", "VZFO"} <= set(oral.columns)
    assert {"MRTIVLST", "MRTIVIFO", "MRTIVIFP", "CLO", "VSSO", "C0"} <= set(
        bolus.columns
    )
    back = read_noncompart(bolus)
    ours = result(Route.IV_BOLUS).to_dataframe().set_index("subject")
    for name in back.columns:
        np.testing.assert_allclose(back[name], ours[name], rtol=1e-12, err_msg=name)


def test_noncompart_refuses_mixed_routes() -> None:
    mixed = Timecourses.from_timecourses(
        [
            *batch(Route.ORAL),
            Timecourse(
                time=TIME,
                value=np.exp(-0.2 * TIME),
                time_unit="hr",
                unit="mg/l",
                dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
                substance="drug",
                label="s3",
            ),
        ],
        dim="subject",
    )
    with pytest.raises(ValueError, match="mixes the routes"):
        to_noncompart(nca(mixed))


def test_pknca_long_table(tmp_path) -> None:
    r = result(Route.IV_BOLUS)
    table = write_pknca_results(r, tmp_path / "pknca.csv")
    assert list(table.columns) == ["subject", *PKNCA_COLUMNS]
    rows = table.set_index(["subject", "PPTESTCD"])
    assert rows.loc[("s1", "cl.obs"), "PPORRES"] == pytest.approx(
        float(r.ds["cl"].sel(subject="s1"))
    )
    # the mean residence time of an intravenous dose, the percentages
    assert ("s1", "mrt.iv.obs") in rows.index
    assert rows.loc[("s1", "aucpext.obs"), "PPORRES"] == pytest.approx(
        100 * float(r.ds["auc_extrap_fraction"].sel(subject="s1"))
    )
    assert (table["start"] == 0).all() and np.isinf(table["end"]).all()


@pytest.mark.parametrize("route", [Route.ORAL, Route.IV_BOLUS])
def test_pknca_round_trip(route: Route, tmp_path) -> None:
    r = result(route)
    write_pknca_results(r, tmp_path / "pknca.csv")
    back = read_pknca_results(tmp_path / "pknca.csv").droplevel(["start", "end"])
    ours = r.to_dataframe().set_index("subject")
    # `cl.obs` is read as the clearance of the route
    assert ("cl_f" in back) == (route is Route.ORAL)
    assert ("cl" in back) == (route is Route.IV_BOLUS)
    for name in back.columns:
        np.testing.assert_allclose(back[name], ours[name], rtol=1e-12, err_msg=name)


def test_read_pknca_results_drops_excluded_rows_and_needs_a_route() -> None:
    frame = pd.DataFrame(
        {
            "ID": [1, 1, 1],
            "start": [0.0, 0.0, 0.0],
            "end": [np.inf, np.inf, np.inf],
            "PPTESTCD": ["cmax", "cl.obs", "tlag"],
            "PPORRES": [2.0, 5.0, 0.5],
            "exclude": [np.nan, "too short", np.nan],
        }
    )
    table = read_pknca_results(frame)
    assert list(table.columns) == ["cmax", "tlag"]
    with pytest.raises(ValueError, match="route can not be recognized"):
        read_pknca_results(frame[frame["PPTESTCD"] == "cmax"])
    named = read_pknca_results(frame.iloc[[0, 1]].assign(exclude=None), route="oral")
    assert list(named.columns) == ["cmax", "cl_f"]
    with pytest.raises(ValueError, match="not a PKNCA result table"):
        read_pknca_results(frame.drop(columns="PPORRES"))


def test_write_noncompart(tmp_path) -> None:
    written = write_noncompart(result(Route.ORAL), tmp_path / "nc.csv")
    assert pd.read_csv(tmp_path / "nc.csv").shape == written.shape

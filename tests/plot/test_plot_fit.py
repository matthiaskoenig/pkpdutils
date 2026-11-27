from typing import Any

import numpy as np
import pytest
import xarray as xr
from sbmlsim_core.chart import Chart

from pkpdutils import Dose, Route, Timecourses
from pkpdutils.fit import (
    FitOptions,
    Weighting,
    fit,
    fit_table,
    fit_timecourses,
    proportionality_test,
)
from pkpdutils.fit.models import MonoExp, Power
from pkpdutils.plot import (
    new_chart,
    plot_bland_altman,
    plot_dose_proportionality,
    plot_fit,
    plot_goodness_of_fit,
)

T = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])


def monoexp_result(n: int = 1):
    rng = np.random.default_rng(0)
    ys = np.stack(
        [
            10 * np.exp(-(0.2 + 0.1 * i) * T) * rng.lognormal(0, 0.05, T.size)
            for i in range(n)
        ]
    )
    if n == 1:
        return fit(
            MonoExp(),
            T,
            ys[0],
            sd=0.05 * ys[0],
            options=FitOptions(weighting=Weighting.INV_SD),
            x_unit="hr",
            y_unit="mg/l",
        )
    return fit(
        MonoExp(),
        T,
        ys,
        dims=("individual",),
        coords={"individual": [f"s{i}" for i in range(n)]},
        x_unit="hr",
        y_unit="mg/l",
    )


def plots(chart: Chart) -> list[dict[str, Any]]:
    return chart.spec()["plots"]


def labels(plot: dict[str, Any]) -> list[str]:
    return [e["label"] for e in plot["elements"] if "label" in e]


def element(plot: dict[str, Any], label: str) -> dict[str, Any]:
    return next(e for e in plot["elements"] if e.get("label") == label)


def test_plot_fit_single_and_batch() -> None:
    chart = plot_fit(monoexp_result(), log_y=True)
    assert isinstance(chart, Chart)
    fit_panel, residuals = plots(chart)
    assert fit_panel["y"]["scale"] == "log"
    assert "monoexp" in fit_panel["title"] and "k" in fit_panel["title"]
    # the shared x axis is labelled once, on the residual panel, and the fit
    # panel is three times as high
    assert fit_panel["x"]["label_visible"] is False
    assert residuals["x"]["label"] == "x [hr]"
    assert (fit_panel["x"]["min"], fit_panel["x"]["max"]) == (
        residuals["x"]["min"],
        residuals["x"]["max"],
    )
    assert fit_panel["row_span"] == 3 and residuals["row"] == 3
    assert fit_panel["y"]["label"] == "y [mg/l]"
    assert "fit" in labels(fit_panel)
    # the data carry their sd as error bars
    assert "yerr" in element(fit_panel, "data")
    second = plots(plot_fit(monoexp_result(3), individual="s1", title="s1"))[0]
    assert "s1" in second["title"]


def test_plot_fit_draws_into_two_cells_of_a_given_chart() -> None:
    chart = new_chart(7, 6, rows=2, columns=2)
    assert plot_fit(monoexp_result(), chart=chart, column=1) is chart
    fit_panel, residuals = plots(chart)
    assert (fit_panel["row"], fit_panel["column"]) == (0, 1)
    assert (residuals["row"], residuals["column"]) == (1, 1)
    with pytest.raises(ValueError):
        plot_fit(monoexp_result(), chart=chart, row=1)


def test_plot_goodness_of_fit() -> None:
    plot = plots(plot_goodness_of_fit(monoexp_result(3), log_x=True, log_y=True))[0]
    assert plot["x"]["scale"] == "log" and plot["y"]["scale"] == "log"
    assert "identity" in labels(plot)
    points = [
        e
        for e in plot["elements"]
        if e["kind"] == "curve" and e["style"]["marker"]["type"] == "circle"
    ]
    assert len(points) == 3
    assert all("R² = " in e["label"] for e in points)


def test_plot_fit_log_x_masks_a_zero_time_point() -> None:
    # B14: a timecourse sampled at t=0 (kept by `fit_timecourses`) used to
    # crash `np.geomspace` with "Geometric sequence cannot include zero"
    t = np.array([0.0, 0.5, 1, 2, 4, 8, 12, 24])
    y = 5 * np.exp(-0.2 * t)
    result = fit(
        MonoExp(),
        t,
        y,
        options=FitOptions(n_starts=3, seed=0),
        x_unit="hr",
        y_unit="mg/l",
    )
    chart = plot_fit(result, log_x=True)
    chart.svg()
    plot = plots(chart)[0]
    assert plot["x"]["scale"] == "log"
    assert "fit" in labels(plot)
    assert np.all(np.asarray(element(plot, "fit")["x"], dtype=float) > 0)
    assert np.all(np.asarray(element(plot, "data")["x"], dtype=float) > 0)


def test_plot_goodness_of_fit_has_no_legend_entry_without_labels() -> None:
    # B27: an all-NaN batch fit has no labelled element (no identity line, no
    # per-sample label above 8 samples)
    t = np.array([0.25, 0.5, 1, 2, 4, 8, 12, 24])
    batch = Timecourses.from_arrays(
        t,
        np.full((10, t.size), np.nan),
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": [f"s{i}" for i in range(10)]},
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    )
    result = fit_timecourses(MonoExp(), batch, options=FitOptions(n_starts=3, seed=0))
    chart = plot_goodness_of_fit(result)
    assert "sbmlsim-legend-entry" not in chart.svg()
    assert labels(plots(chart)[0]) == []


def test_plot_dose_proportionality() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    auc = 2.0 * doses**1.02
    ds = xr.Dataset(
        {"auc": (("dose",), auc, {"units": "hr*mg/l"})},
        coords={"dose": ("dose", doses, {"units": "mg"})},
    )
    result = fit_table(Power(), ds, "dose", "auc", dim="dose")
    test = proportionality_test(result, dose_range=(25.0, 200.0))
    plot = plots(plot_dose_proportionality(result, test=test))[0]
    assert plot["x"]["scale"] == "log" and plot["y"]["scale"] == "log"
    assert "b =" in plot["title"]
    # the acceptance wedge
    assert [e["kind"] for e in plot["elements"]].count("band") == 1
    # the doses are the ticks of the x axis
    assert plot["x"]["ticks"] == list(doses)
    assert plot["x"]["tick_labels"] == ["25", "50", "100", "200"]


def test_plot_bland_altman() -> None:
    chart = plot_bland_altman(monoexp_result(3))
    assert isinstance(chart, Chart)
    plot = plots(chart)[0]
    assert plot["x"]["label"].startswith("mean")
    assert plot["y"]["label"].startswith("difference")
    # the mean difference and the two limits of agreement
    rules = [e for e in plot["elements"] if e["kind"] == "rule"]
    assert (
        sum(1 for e in rules if e["style"]["line"]["type"] in ("dashed", "dotted")) >= 3
    )
    log = plots(plot_bland_altman(monoexp_result(), log_ratio=True))[0]
    assert log["y"]["label"].startswith("log ratio")
    assert log["x"]["scale"] == "log"


def test_plot_fit_labels_the_axes_from_the_names_of_the_front_end() -> None:
    batch = Timecourses.from_arrays(
        T,
        (10 * np.exp(-0.2 * T))[None, :],
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["s1"]},
        substance="caffeine",
    )
    result = fit_timecourses(MonoExp(), batch)
    fit_panel, residuals = plots(plot_fit(result, individual="s1"))
    assert fit_panel["y"]["label"] == "caffeine [mg/l]"
    assert residuals["x"]["label"] == "time [hr]"


def test_plot_fit_labels_the_axes_with_the_names_given_to_fit() -> None:
    rng = np.random.default_rng(1)
    y = 10 * np.exp(-0.2 * T) * rng.lognormal(0, 0.05, T.size)
    result = fit(
        MonoExp(),
        T,
        y,
        x_unit="hr",
        y_unit="mg/l",
        x_name="concentration",
        y_name="effect",
        options=FitOptions(n_starts=2, seed=0),
    )
    assert result.ds.attrs["x_name"] == "concentration"
    assert result.ds.attrs["y_name"] == "effect"
    fit_panel, residuals = plots(plot_fit(result))
    assert fit_panel["y"]["label"] == "effect [mg/l]"
    assert residuals["x"]["label"] == "concentration [hr]"


def test_plot_fit_falls_back_to_x_and_y_without_the_names() -> None:
    result = monoexp_result()
    assert "x_name" not in result.ds.attrs
    fit_panel, residuals = plots(plot_fit(result))
    assert fit_panel["y"]["label"].startswith("y [")
    assert residuals["x"]["label"].startswith("x [")


def test_plot_dose_proportionality_labels_the_axes_from_the_table_columns() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    ds = xr.Dataset(
        {"auc_inf_obs": (("dose",), 2.0 * doses**1.05, {"units": "mg*hr/l"})},
        coords={"dose": doses},
    )
    ds["dose"].attrs["units"] = "mg"
    result = fit_table(Power(), ds, "dose", "auc_inf_obs", dim="dose")
    plot = plots(plot_dose_proportionality(result))[0]
    assert plot["x"]["label"] == "dose [mg]"
    assert plot["y"]["label"].startswith("auc_inf_obs [")


def test_plot_fit_leaves_out_a_dimensionless_unit() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    ds = xr.Dataset(
        {"auc": (("dose",), 2.0 * doses**1.05, {"units": "mg*hr/l"})},
        coords={"dose": doses},  # a coordinate without a unit
    )
    result = fit_table(Power(), ds, "dose", "auc", dim="dose")
    assert plots(plot_dose_proportionality(result))[0]["x"]["label"] == "dose"

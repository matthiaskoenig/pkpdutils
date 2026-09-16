import warnings

import matplotlib
import matplotlib.pyplot
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

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
    plot_bland_altman,
    plot_dose_proportionality,
    plot_fit,
    plot_goodness_of_fit,
)

matplotlib.use("Agg")
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


def test_plot_fit_single_and_batch() -> None:
    fig = plot_fit(monoexp_result(), log_y=True)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    assert fig.axes[0].get_yscale() == "log"
    assert "monoexp" in fig.axes[0].get_title() and "k" in fig.axes[0].get_title()
    # the shared x axis is labelled once, on the residual panel
    assert fig.axes[0].get_xlabel() == "" and fig.axes[1].get_xlabel() == "x [hr]"
    assert fig.axes[0].get_ylabel() == "y [mg/l]"
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "fit" in labels
    fig2 = plot_fit(monoexp_result(3), individual="s1", title="s1")
    assert "s1" in fig2.axes[0].get_title()
    matplotlib.pyplot.close("all")


def test_plot_goodness_of_fit() -> None:
    fig = plot_goodness_of_fit(monoexp_result(3), log_x=True, log_y=True)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert any("identity" in str(line.get_label()) for line in ax.get_lines())
    assert len(ax.collections) >= 3 or len(ax.get_lines()) >= 4
    matplotlib.pyplot.close(fig)


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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_fit(result, log_x=True)
        fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    labels = [line.get_label() for line in ax.get_lines()]
    assert "fit" in labels
    fit_line = next(line for line in ax.get_lines() if line.get_label() == "fit")
    assert np.all(np.asarray(fit_line.get_xdata()) > 0)
    matplotlib.pyplot.close(fig)


def test_plot_goodness_of_fit_no_legend_warning_without_labels() -> None:
    # B27: an all-NaN batch fit has no labelled artist (no identity line, no
    # per-sample label above 8 samples), an unconditional ax.legend() used to
    # raise "UserWarning: No artists with labels found to put in legend."
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_goodness_of_fit(result)
        fig.canvas.draw()
    assert fig.axes[0].get_legend() is None
    matplotlib.pyplot.close(fig)


def test_plot_dose_proportionality() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    auc = 2.0 * doses**1.02
    ds = xr.Dataset(
        {"auc": (("dose",), auc, {"units": "hr*mg/l"})},
        coords={"dose": ("dose", doses, {"units": "mg"})},
    )
    result = fit_table(Power(), ds, "dose", "auc", dim="dose")
    test = proportionality_test(result, dose_range=(25.0, 200.0))
    fig = plot_dose_proportionality(result, test=test)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert "b =" in ax.get_title()
    assert len(ax.collections) >= 1  # the acceptance wedge
    matplotlib.pyplot.close(fig)


def test_plot_bland_altman() -> None:
    fig = plot_bland_altman(monoexp_result(3))
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("mean") and ax.get_ylabel().startswith(
        "difference"
    )
    # the mean difference and the two limits of agreement
    assert sum(1 for line in ax.get_lines() if line.get_linestyle() in ("--", ":")) >= 3
    fig_log = plot_bland_altman(monoexp_result(), log_ratio=True)
    assert fig_log.axes[0].get_ylabel().startswith("log ratio")
    matplotlib.pyplot.close("all")


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
    fig = plot_fit(result, individual="s1")
    assert fig.axes[0].get_ylabel() == "caffeine [mg/l]"
    assert fig.axes[1].get_xlabel() == "time [hr]"
    matplotlib.pyplot.close(fig)


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
    fig = plot_fit(result)
    assert fig.axes[0].get_ylabel() == "effect [mg/l]"
    assert fig.axes[1].get_xlabel() == "concentration [hr]"
    matplotlib.pyplot.close(fig)


def test_plot_fit_falls_back_to_x_and_y_without_the_names() -> None:
    result = monoexp_result()
    assert "x_name" not in result.ds.attrs
    fig = plot_fit(result)
    assert fig.axes[0].get_ylabel().startswith("y [")
    assert fig.axes[1].get_xlabel().startswith("x [")
    matplotlib.pyplot.close(fig)


def test_plot_dose_proportionality_labels_the_axes_from_the_table_columns() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    ds = xr.Dataset(
        {"auc_inf_obs": (("dose",), 2.0 * doses**1.05, {"units": "mg*hr/l"})},
        coords={"dose": doses},
    )
    ds["dose"].attrs["units"] = "mg"
    result = fit_table(Power(), ds, "dose", "auc_inf_obs", dim="dose")
    fig = plot_dose_proportionality(result)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "dose [mg]"
    assert ax.get_ylabel().startswith("auc_inf_obs [")
    matplotlib.pyplot.close(fig)


def test_plot_fit_leaves_out_a_dimensionless_unit() -> None:
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    ds = xr.Dataset(
        {"auc": (("dose",), 2.0 * doses**1.05, {"units": "mg*hr/l"})},
        coords={"dose": doses},  # a coordinate without a unit
    )
    result = fit_table(Power(), ds, "dose", "auc", dim="dose")
    fig = plot_dose_proportionality(result)
    assert fig.axes[0].get_xlabel() == "dose"
    matplotlib.pyplot.close(fig)

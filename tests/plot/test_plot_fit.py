import matplotlib
import matplotlib.pyplot
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from pkpdutils.fit import FitOptions, Weighting, fit, fit_table, proportionality_test
from pkpdutils.fit.models import MonoExp, Power
from pkpdutils.plot import plot_dose_proportionality, plot_fit, plot_goodness_of_fit

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
    assert (
        fig.axes[0].get_xlabel() == "x [hr]" and fig.axes[0].get_ylabel() == "y [mg/l]"
    )
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "fit" in labels
    fig2 = plot_fit(monoexp_result(3), individual="s1", title="s1")
    assert "s1" in fig2.axes[0].get_title()
    matplotlib.pyplot.close("all")


def test_plot_goodness_of_fit() -> None:
    fig = plot_goodness_of_fit(monoexp_result(3), log=True)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert any("identity" in str(line.get_label()) for line in ax.get_lines())
    assert len(ax.collections) >= 3 or len(ax.get_lines()) >= 4
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

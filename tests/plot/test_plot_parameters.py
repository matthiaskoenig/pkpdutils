import matplotlib
import matplotlib.pyplot
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from pkpdutils import Route, Timecourses, nca
from pkpdutils.plot import plot_parameters
from pkpdutils.result import ParameterResult
from pkpdutils.stats import Scale

matplotlib.use("Agg")


def nca_result():
    time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
    rng = np.random.default_rng(2)
    n = 8
    curves = np.stack(
        [rng.lognormal(np.log(10), 0.2) * np.exp(-0.2 * time) for _ in range(n)]
    )
    batch = Timecourses.from_arrays(
        time,
        curves,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"s{i}" for i in range(n)],
            "sex": ("individual", ["F", "M"] * 4),
        },
        dose={"amount": np.full(n, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    return nca(batch)


def test_plot_parameters_groups() -> None:
    result = nca_result()
    fig = plot_parameters(result, "auc_inf_obs", "individual", by="sex", log=True)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["F", "M"]
    assert ax.get_ylabel().startswith("auc_inf_obs [")
    fig.canvas.draw()
    y_labels = [t.get_text() for t in ax.get_yticklabels(which="both")]
    assert not any("^" in label or "10^" in label for label in y_labels)
    matplotlib.pyplot.close("all")


def test_plot_parameters_single_group_linear() -> None:
    result = nca_result()
    fig, ax = matplotlib.pyplot.subplots()
    out = plot_parameters(result, "cmax", "individual", scale=Scale.LINEAR, ax=ax)
    assert out is fig
    assert [t.get_text() for t in ax.get_xticklabels()] == ["cmax"]
    assert len(ax.containers) >= 1  # the error bar of the mean
    matplotlib.pyplot.close("all")


def _result_with_values(values: np.ndarray) -> ParameterResult:
    n = values.size
    ds = xr.Dataset(
        {
            "value": (("individual",), values, {"units": "mg/l"}),
            "flags": (
                ("individual",),
                np.zeros(n, dtype=np.int64),
                {"units": "dimensionless"},
            ),
        },
        coords={
            "individual": [f"s{i}" for i in range(n)],
            "sex": ("individual", ["F", "M"] * (n // 2)),
        },
    )
    return ParameterResult(ds)


def test_plot_parameters_log_ticks_over_several_decades() -> None:
    values = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
    fig = plot_parameters(_result_with_values(values), "value", "individual", log=True)
    ax = fig.axes[0]
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels(which="both")]
    non_empty = [label for label in labels if label]
    assert len(non_empty) <= 12, non_empty
    assert not any("^" in label for label in non_empty)
    assert "1000" in non_empty
    matplotlib.pyplot.close("all")


def _result_with_nonpositive_value() -> ParameterResult:
    individual = [f"s{i}" for i in range(4)]
    values = np.array([1.0, 2.0, 3.0, -0.5])
    ds = xr.Dataset(
        {
            "value": (("individual",), values, {"units": "mg/l"}),
            "flags": (
                ("individual",),
                np.zeros(4, dtype=np.int64),
                {"units": "dimensionless"},
            ),
        },
        coords={
            "individual": individual,
            "sex": ("individual", ["F", "F", "M", "M"]),
        },
    )
    return ParameterResult(ds)


def test_plot_parameters_nonpositive_value_falls_back_to_linear_marker() -> None:
    result = _result_with_nonpositive_value()
    fig = plot_parameters(result, "value", "individual", by="sex", log=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_yscale() == "log"
    matplotlib.pyplot.close("all")

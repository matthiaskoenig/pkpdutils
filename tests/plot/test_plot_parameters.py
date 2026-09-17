import warnings

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
import xarray as xr
from matplotlib import cbook
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
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
    fig = plot_parameters(result, "auc_inf_obs", "individual", by="sex", log_y=True)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["F", "M"]
    assert ax.get_ylabel().startswith("auc_inf_obs [")
    fig.canvas.draw()
    y_labels = [t.get_text() for t in ax.get_yticklabels(which="both")]
    assert not any("^" in label or "10^" in label for label in y_labels)
    matplotlib.pyplot.close("all")


def test_plot_parameters_names_the_statistic_in_the_legend_once() -> None:
    result = nca_result()
    fig = plot_parameters(result, "auc_inf_obs", "individual", by="sex")
    legend = fig.axes[0].get_legend()
    assert legend is not None
    # the groups are the ticks of the x axis, only the marker is named
    assert [text.get_text() for text in legend.get_texts()] == [
        "geometric mean [95 % CI]"
    ]
    matplotlib.pyplot.close(fig)


def test_plot_parameters_legend_follows_the_scale_and_the_level() -> None:
    result = nca_result()
    fig = plot_parameters(
        result, "cmax", "individual", scale=Scale.LINEAR, ci_level=0.90
    )
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["mean [90 % CI]"]
    matplotlib.pyplot.close(fig)


def test_plot_parameters_takes_the_scale_as_a_string() -> None:
    # `scale="log"` is the analysis of `scale=Scale.LOG`, as in `pkpdutils.stats`
    result = nca_result()
    fig = plot_parameters(result, "cmax", "individual", scale="log")
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "geometric mean [95 % CI]"
    ]
    linear = plot_parameters(result, "cmax", "individual", scale="linear")
    linear_legend = linear.axes[0].get_legend()
    assert linear_legend is not None
    assert [text.get_text() for text in linear_legend.get_texts()] == ["mean [95 % CI]"]
    with pytest.raises(ValueError, match="not a valid Scale"):
        plot_parameters(result, "cmax", "individual", scale="geometric")
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
    fig = plot_parameters(
        _result_with_values(values), "value", "individual", log_y=True
    )
    ax = fig.axes[0]
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels(which="both")]
    non_empty = [label for label in labels if label]
    assert len(non_empty) <= 12, non_empty
    assert not any("^" in label for label in non_empty)
    assert "1000" in non_empty
    matplotlib.pyplot.close("all")


def _grouped_result(groups: dict[str, list[float]]) -> ParameterResult:
    """A result of one parameter with the individuals of every group in order."""
    values = np.array([v for group in groups.values() for v in group])
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
            "group": (
                "individual",
                [key for key, group in groups.items() for _ in group],
            ),
        },
    )
    return ParameterResult(ds)


def _box_values(ax: Axes, position: int) -> list[float]:
    """The distinct y values of the box, whiskers, caps and median at a position.

    The box plot is the only drawing of lines: the strip and the markers of
    the statistic are points without a line.
    """
    y: set[float] = set()
    for line in ax.lines:
        if line.get_linestyle() == "None":
            continue
        x = np.asarray(line.get_xdata(), dtype=float)
        if np.all(np.abs(x - position) < 0.5):
            y.update(np.asarray(line.get_ydata(), dtype=float).tolist())
    return sorted(y)


def _strip_values(ax: Axes, position: int) -> list[float]:
    """The y values of the jittered points of the individuals at a position."""
    y: list[float] = []
    for line in ax.lines:
        if line.get_marker() != "o":
            continue
        x = np.asarray(line.get_xdata(), dtype=float)
        at = np.abs(x - position) < 0.5
        y.extend(np.asarray(line.get_ydata(), dtype=float)[at].tolist())
    return sorted(y)


def _errorbars(ax: Axes) -> list[ErrorbarContainer]:
    """The error bars of the statistic, in the order of the groups."""
    return [c for c in ax.containers if isinstance(c, ErrorbarContainer)]


def _markers(ax: Axes) -> list[tuple[str, float]]:
    """Marker and value of the statistic of every group, in the order of the groups."""
    return [
        (
            str(container.lines[0].get_marker()),
            float(np.asarray(container.lines[0].get_ydata(), dtype=float)[0]),
        )
        for container in _errorbars(ax)
    ]


def test_plot_parameters_log_box_uses_the_values_of_the_strip() -> None:
    # issue #71: under `log_y=True` the strip dropped the non-positive values
    # but the box was still computed from every finite value, so the box of
    # the mixed group reached below the axis and did not describe its points
    positive = [1.0, 2.0, 4.0, 8.0]
    mixed = [2.0, 4.0, 8.0, -1.0]
    result = _grouped_result({"A": positive, "B": mixed})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_parameters(result, "value", "individual", by="group", log_y=True)
        fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    shown = [2.0, 4.0, 8.0]
    assert _strip_values(ax, 1) == positive
    assert _strip_values(ax, 2) == shown
    for position, values in ((1, positive), (2, shown)):
        stats = cbook.boxplot_stats(np.array(values))[0]
        expected = {stats[key] for key in ("whislo", "q1", "med", "q3", "whishi")}
        assert _box_values(ax, position) == pytest.approx(sorted(expected))
    # the statistic is computed from every finite value: the all-positive group
    # keeps its geometric mean, the mixed group falls back to the arithmetic
    # mean with a marker of its own, and the legend names both
    (geo_marker, geo_value), (mean_marker, mean_value) = _markers(ax)
    assert geo_value == pytest.approx(np.exp(np.mean(np.log(positive))))
    assert mean_value == pytest.approx(np.mean(mixed))
    assert geo_marker != mean_marker
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "geometric mean [95 % CI]",
        "mean [95 % CI]",
    ]
    matplotlib.pyplot.close(fig)


def test_plot_parameters_linear_box_keeps_the_nonpositive_values() -> None:
    mixed = [2.0, 4.0, 8.0, -1.0]
    result = _grouped_result({"A": [1.0, 2.0, 4.0, 8.0], "B": mixed})
    fig = plot_parameters(result, "value", "individual", by="group")
    ax = fig.axes[0]
    assert _strip_values(ax, 2) == sorted(mixed)
    assert min(_box_values(ax, 2)) == -1.0
    matplotlib.pyplot.close(fig)


def test_plot_parameters_log_group_without_a_positive_value() -> None:
    # a group without a positive value has nothing a logarithmic axis can
    # show: no points, no box and no marker, only its tick; a group of a
    # single value is a box of one line and a marker without an interval
    result = _grouped_result(
        {"A": [1.0, 2.0, 4.0, 8.0], "B": [0.0, -1.0, -2.0, 0.0], "C": [5.0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_parameters(result, "value", "individual", by="group", log_y=True)
        fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    assert _strip_values(ax, 2) == []
    assert _box_values(ax, 2) == []
    assert _strip_values(ax, 3) == [5.0]
    assert _box_values(ax, 3) == [5.0]
    # the arithmetic mean of B is negative: A and C are the only markers
    markers = _markers(ax)
    assert len(markers) == 2
    assert markers[1] == ("D", pytest.approx(5.0))
    assert not _errorbars(ax)[1].has_yerr
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "geometric mean [95 % CI]"
    ]
    matplotlib.pyplot.close(fig)


def test_plot_parameters_log_without_positive_values_stays_linear() -> None:
    # B28: `plot_parameters(log_y=True)` used to call `ax.set_yscale("log")`
    # directly, so an all-non-positive parameter (e.g. an all-zero `cmax`
    # batch) raised "UserWarning: Data has no positive values, and therefore
    # cannot be log-scaled" at draw time.
    values = np.zeros(4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_parameters(
            _result_with_values(values), "value", "individual", log_y=True
        )
        fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_yscale() == "linear"
    # the axis stays linear, so it shows the values a logarithmic one could not
    assert _strip_values(ax, 1) == [0.0, 0.0, 0.0, 0.0]
    assert _box_values(ax, 1) == [0.0]
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["mean [95 % CI]"]
    matplotlib.pyplot.close(fig)

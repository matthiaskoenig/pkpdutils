import logging
import warnings

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
import xarray as xr

from pkpdutils.plot._common import (
    axes_of,
    figure_of,
    log_scale,
    plain_log_ticks,
    sample_colors,
    sample_labels,
)

matplotlib.use("Agg")


def test_figure_of_creates_a_figure_with_constrained_layout() -> None:
    fig, ax = figure_of(None)
    assert fig.get_layout_engine() is not None
    assert ax.get_figure() is fig
    matplotlib.pyplot.close(fig)


def test_figure_of_never_touches_the_layout_engine_of_a_passed_ax() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    assert fig.get_layout_engine() is None
    fig2, ax2 = figure_of(ax)
    assert fig2 is fig and ax2 is ax
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_axes_of_creates_a_grid_with_constrained_layout() -> None:
    fig, axes = axes_of(None, nrows=2, ncols=3, figsize=(6, 4))
    assert fig.get_layout_engine() is not None
    assert axes.shape == (2, 3)
    matplotlib.pyplot.close(fig)


def test_axes_of_never_touches_the_layout_engine_of_passed_axes() -> None:
    fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2)
    assert fig.get_layout_engine() is None
    fig2, grid = axes_of(axes, nrows=1, ncols=2, figsize=(6, 4))
    assert fig2 is fig
    assert grid.shape == (1, 2)
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_axes_of_raises_for_a_mismatched_number_of_axes() -> None:
    _fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2)
    with pytest.raises(ValueError):
        axes_of(axes, nrows=2, ncols=2, figsize=(6, 4))
    matplotlib.pyplot.close("all")


def test_plain_log_ticks_formats_as_plain_numbers() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([1, 10, 100], [1, 2, 3])
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels
    assert all("^" not in label for label in labels)
    matplotlib.pyplot.close(fig)


def test_log_scale_sets_log_and_plain_ticks_when_data_is_positive() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0.0, 1.0, 2.0], [1.0, 2.0, 4.0])
    log_scale(ax, "y")
    assert ax.get_yscale() == "log"
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels(which="both") if t.get_text()]
    assert labels
    assert all("^" not in label for label in labels)
    matplotlib.pyplot.close(fig)


def test_log_scale_falls_back_to_linear_without_positive_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0.0, 1.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0])
    with (
        caplog.at_level(logging.DEBUG, logger="pkpdutils.plot._common"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        log_scale(ax, "y")
        fig.canvas.draw()
    assert ax.get_yscale() == "linear"
    assert any("no positive value" in message for message in caplog.messages)
    matplotlib.pyplot.close(fig)


def test_sample_labels_joins_coordinates_per_dimension() -> None:
    ds = xr.Dataset(
        {"value": (("individual", "occasion"), np.zeros((2, 3)))},
        coords={"individual": ["a", "b"], "occasion": [1, 2, 3]},
    )
    labels = sample_labels(ds, ("individual", "occasion"))
    assert labels == ["a|1", "a|2", "a|3", "b|1", "b|2", "b|3"]


def test_sample_labels_uses_the_flat_index_without_a_coordinate() -> None:
    ds = xr.Dataset({"value": (("individual",), np.zeros(3))})
    labels = sample_labels(ds, ("individual",))
    assert labels == ["0", "1", "2"]


def test_sample_labels_empty_for_no_sample_dimensions() -> None:
    ds = xr.Dataset({"value": (("time",), np.zeros(3))})
    assert sample_labels(ds, ()) == []


def test_sample_labels_by_broadcasts_a_coordinate_to_the_sample_dims() -> None:
    group = np.array([["g00", "g01"], ["g10", "g11"], ["g20", "g21"]])
    ds = xr.Dataset(
        {"value": (("individual", "dose"), np.zeros((3, 2)))},
        coords={
            "individual": [0, 1, 2],
            "dose": ["low", "high"],
            "group": (("individual", "dose"), group),
        },
    )
    labels = sample_labels(ds, ("individual", "dose"), by="group")
    assert labels == ["g00", "g01", "g10", "g11", "g20", "g21"]


def test_sample_colors_uses_the_colormap_start_for_a_single_sample() -> None:
    colors = sample_colors(1, "viridis")
    assert len(colors) == 1
    assert colors[0] == matplotlib.pyplot.get_cmap("viridis")(0.0)


def test_sample_colors_spreads_evenly_across_the_colormap() -> None:
    colors = sample_colors(3, "viridis")
    cmap = matplotlib.pyplot.get_cmap("viridis")
    assert colors == [cmap(0.0), cmap(0.5), cmap(1.0)]

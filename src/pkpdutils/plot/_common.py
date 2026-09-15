"""Helpers shared by the figures: axes and figure creation, ticks, labels and colors."""

import logging
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatter

logger = logging.getLogger(__name__)


def figure_of(
    ax: Axes | None, figsize: tuple[float, float] = (6, 4)
) -> tuple[Figure, Axes]:
    """The axes to draw on and its figure, a new figure without `ax`.

    A function drawing into a caller-supplied `ax` never touches its
    figure's layout engine or size; only a figure created here gets the
    constrained layout engine, so a caller-supplied `ax` keeps its figure's
    own layout, meaning long tick labels can clip unless the caller sets one
    (`fig.set_layout_engine("constrained")`).

    Args:
        ax: axes to draw on, `None` for a new figure.
        figsize: size of a new figure; ignored when `ax` is given.

    Returns:
        The figure and the axes to draw on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax


def axes_of(
    axes: Any | None, nrows: int, ncols: int, figsize: tuple[float, float]
) -> tuple[Figure, np.ndarray]:
    """The grid of axes to draw on and its figure, a new figure without `axes`.

    Mirrors `figure_of` for a multi-panel figure: a function drawing into
    caller-supplied axes never touches their figure's layout engine.

    Args:
        axes: existing axes to draw the grid into, `nrows * ncols` of them
            (any shape, flattened in C order), `None` for a new figure.
        nrows: number of rows of the grid.
        ncols: number of columns of the grid.
        figsize: size of a new figure; ignored when `axes` is given.

    Returns:
        The figure and the axes, always as an `(nrows, ncols)` array.

    Raises:
        ValueError: if `axes` does not have `nrows * ncols` entries.
    """
    if axes is None:
        fig, new_axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False
        )
        fig.set_layout_engine("constrained")
        return fig, new_axes
    flat = np.asarray(axes).reshape(-1)
    if flat.size != nrows * ncols:
        raise ValueError(
            f"{nrows * ncols} axes are needed for a {nrows}x{ncols} grid, got {flat.size}"
        )
    grid = flat.reshape(nrows, ncols)
    fig = flat[0].get_figure()
    assert isinstance(fig, Figure)
    return fig, grid


class _PlainLogFormatter(LogFormatter):
    """A `LogFormatter` which writes the labels as plain numbers, not as `10^n`.

    Only the rendering of a labelled tick is changed; which ticks carry a
    label stays with `LogFormatter`, which drops the labels between the
    decades once an axis spans more than a couple of them.
    """

    def _num_to_string(self, x: float, vmin: float, vmax: float) -> str:
        """The label of a tick value.

        Args:
            x: the tick value.
            vmin: lower bound of the view interval (unused).
            vmax: upper bound of the view interval (unused).

        Returns:
            The value as a plain number.
        """
        return f"{x:g}"


def plain_log_ticks(axis: Axis) -> None:
    """Format the major and minor ticks of a logarithmic axis as plain numbers.

    Args:
        axis: the axis (`ax.xaxis` or `ax.yaxis`) to format.
    """
    axis.set_major_formatter(_PlainLogFormatter())
    axis.set_minor_formatter(
        _PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5))
    )


def log_scale(ax: Axes, which: Literal["x", "y"]) -> None:
    """Set a logarithmic scale on one axis of `ax`, with plain tick labels.

    The axis is scaled from the extent of the data already drawn on `ax`
    (`ax.dataLim`), so this must run after the data is plotted. Without a
    positive value the axis stays linear and the fallback is logged at debug
    level, instead of the scale raising `UserWarning: Data has no positive
    values, and therefore cannot be log-scaled` at draw or save time.

    Args:
        ax: axes already carrying the data.
        which: `"x"` or `"y"`, the axis to scale.
    """
    interval = ax.dataLim.intervalx if which == "x" else ax.dataLim.intervaly
    if not np.isfinite(interval).any() or float(np.nanmax(interval)) <= 0.0:
        logger.debug("the %s axis has no positive value, staying linear", which)
        return
    if which == "x":
        ax.set_xscale("log")
        plain_log_ticks(ax.xaxis)
    else:
        ax.set_yscale("log")
        plain_log_ticks(ax.yaxis)


def sample_labels(
    ds: xr.Dataset, sample_dims: tuple[str, ...] | list[str], *, by: str | None = None
) -> list[str]:
    """One label per sample of the sample dimensions, in C order.

    Without `by`, the label of a sample is the coordinate value of every
    sample dimension joined by `"|"` (the flat index when a dimension
    carries no coordinate). With `by`, the label is the value of that
    coordinate broadcast to every sample dimension, used when the samples
    are labelled by one grouping coordinate instead of by their own index
    (`plot_timecourse(by=...)`).

    Args:
        ds: the dataset the coordinates are read from.
        sample_dims: the sample dimensions, in the order the samples are
            enumerated.
        by: name of a coordinate broadcast to the sample dimensions instead
            of the default per-dimension coordinates.

    Returns:
        One label per sample, in C order of `sample_dims`; empty when
        `sample_dims` is empty.
    """
    if not sample_dims:
        return []
    sizes = [ds.sizes[d] for d in sample_dims]
    if by is not None:
        template = xr.DataArray(np.zeros(sizes), dims=list(sample_dims))
        coord, _ = xr.broadcast(ds[by], template)
        values = coord.transpose(*sample_dims).to_numpy().reshape(-1)
        return [str(v) for v in values]
    labels = []
    for index in np.ndindex(*sizes):
        parts = [
            str(ds.coords[d].to_numpy()[i]) if d in ds.coords else str(i)
            for d, i in zip(sample_dims, index, strict=True)
        ]
        labels.append("|".join(parts))
    return labels


def sample_colors(n: int, cmap: str) -> list[Any]:
    """One color per sample from a colormap, evenly spaced across `n` samples.

    Args:
        n: number of samples; colors are evenly spaced from the start to the
            end of the colormap (`cmap(0.0)` for `n <= 1`).
        cmap: name of the colormap.

    Returns:
        One RGBA color per sample, `n` entries.
    """
    palette = plt.get_cmap(cmap)
    return [palette(i / max(n - 1, 1)) for i in range(n)]

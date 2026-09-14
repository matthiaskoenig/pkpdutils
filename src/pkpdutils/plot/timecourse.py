"""Figures of timecourses."""

from typing import Any

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import TIME_DIM, Timecourse, Timecourses


def _figure_of(ax: Axes | None) -> tuple[Figure, Axes]:
    """The axes to draw on and its figure, a new figure without `ax`.

    Args:
        ax: axes to draw on, `None` for a new figure.

    Returns:
        The figure and the axes to draw on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax


def _draw_curve(
    ax: Axes,
    tc: Timecourse,
    *,
    color: Any,
    label: str | None,
    errorbars: bool,
    style: PlotStyle,
) -> None:
    """One curve with optional error bars.

    Args:
        ax: axes to draw on
        tc: the curve
        color: color of the line and markers
        label: legend label, `None` for none
        errorbars: draw `se` (or `sd`) as error bars when present
        style: colors and markers
    """
    err = tc.se if tc.se is not None else tc.sd
    if errorbars and err is not None:
        ax.errorbar(
            tc.time,
            tc.value,
            yerr=err,
            marker=style.data_marker,
            linestyle="-",
            color=color,
            label=label,
            linewidth=style.linewidth,
            markersize=style.markersize,
            capsize=2,
        )
    else:
        ax.plot(
            tc.time,
            tc.value,
            marker=style.data_marker,
            linestyle="-",
            color=color,
            label=label,
            linewidth=style.linewidth,
            markersize=style.markersize,
        )


def plot_timecourse(
    timecourses: Timecourse | Timecourses,
    ax: Axes | None = None,
    *,
    log: bool = False,
    errorbars: bool = True,
    by: str | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Plot one timecourse or every timecourse of a batch.

    Args:
        timecourses: the curve or the batch
        ax: axes to draw on, a new figure by default
        log: logarithmic value axis
        errorbars: draw `se` (or `sd`) as error bars when present
        by: coordinate of the batch used as legend label, the sample label by default
        style: colors and markers

    Returns:
        The figure.
    """
    fig, ax = _figure_of(ax)
    curves: list[Timecourse]
    labels: list[str | None]
    if isinstance(timecourses, Timecourse):
        curves = [timecourses]
        labels = [timecourses.label]
        first = timecourses
    else:
        curves = list(timecourses)
        first = curves[0]
        if by is not None:
            template = timecourses.ds["value"].isel({TIME_DIM: 0}, drop=True)
            coord, _ = xr.broadcast(timecourses.ds[by], template)
            values = coord.transpose(*timecourses.sample_dims).to_numpy().reshape(-1)
            labels = [str(v) for v in values]
        else:
            labels = [tc.label for tc in curves]
    cmap = plt.get_cmap(style.cmap)
    for i, (tc, label) in enumerate(zip(curves, labels, strict=True)):
        color: Any = (
            style.data_color if len(curves) == 1 else cmap(i / max(len(curves) - 1, 1))
        )
        _draw_curve(ax, tc, color=color, label=label, errorbars=errorbars, style=style)
    ax.set_xlabel(f"time [{first.time_unit}]")
    ax.set_ylabel(f"{first.substance} [{first.unit}]")
    if log:
        ax.set_yscale("log")
    if any(label is not None for label in labels):
        ax.legend()
    return fig

"""Figures of timecourses."""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot._common import figure_of, log_scale, sample_colors, sample_labels
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Timecourse, Timecourses


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
    *,
    log_y: bool = False,
    errorbars: bool = True,
    by: str | None = None,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Plot one timecourse or every timecourse of a batch.

    A single curve whose protocol has more than one dose also gets one thin
    dotted vertical line per dose time (`style.dose_color`); a batch draws no
    dose lines, since its curves may carry different protocols.

    Args:
        timecourses: the curve or the batch

    Keyword Args:
        log_y: logarithmic value axis; a curve without a positive value stays
            linear (logged at debug level)
        errorbars: draw `se` (or `sd`) as error bars when present
        by: coordinate of the batch used as legend label, the sample label by default
        ax: axes to draw on, a new figure by default; a caller-supplied `ax`
            keeps its figure's own layout engine, so long tick labels can
            clip unless the caller sets one (`fig.set_layout_engine("constrained")`)
        style: colors and markers

    Returns:
        The figure.
    """
    fig, ax = figure_of(ax)
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
            labels = list(sample_labels(timecourses.ds, timecourses.sample_dims, by=by))
        else:
            labels = [tc.label for tc in curves]
    colors = sample_colors(len(curves), style.cmap)
    for i, (tc, label) in enumerate(zip(curves, labels, strict=True)):
        color: Any = style.data_color if len(curves) == 1 else colors[i]
        _draw_curve(ax, tc, color=color, label=label, errorbars=errorbars, style=style)
    if (
        isinstance(timecourses, Timecourse)
        and timecourses.dosing is not None
        and timecourses.dosing.n_doses >= 2
    ):
        for dose_time in timecourses.dosing.times:
            ax.axvline(
                float(dose_time), color=style.dose_color, linestyle=":", linewidth=1
            )
    ax.set_xlabel(f"time [{first.time_unit}]")
    ax.set_ylabel(f"{first.substance} [{first.unit}]")
    if log_y:
        log_scale(ax, "y")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig

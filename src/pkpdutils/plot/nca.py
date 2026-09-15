"""Diagnostic figures of the non-compartmental analysis."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from pkpdutils.nca.intervals import INTERVAL_DIM
from pkpdutils.nca.options import decode_flags
from pkpdutils.nca.result import NCAResult
from pkpdutils.plot._common import (
    axes_of,
    figure_of,
    log_scale,
    sample_colors,
    sample_labels,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Route, Timecourse, Timecourses


def _sample_values(
    result: NCAResult, indexers: dict[str, Any]
) -> tuple[dict[str, float], list[str]]:
    """Parameter magnitudes and flags of one sample of a result.

    Args:
        result: the result
        indexers: coordinate label per sample dimension, empty for a result
            without sample dimensions

    Returns:
        The parameter magnitudes by name and the names of the set flags.
    """
    quantities = result.to_quantities(**indexers)
    return {name: float(q.magnitude) for name, q in quantities.items()}, result.flags(
        **indexers
    )


def draw_nca_panel(
    timecourse: Timecourse,
    values: dict[str, float],
    flags: list[str],
    *,
    log_y: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Axes:
    """Draw the NCA diagnostics of one curve into one axes.

    A multiple dose result (carrying `auc_tau`) shades the analysed last
    dosing interval `[0, tau]` (relative to the last dose) instead of
    `[0, tlast]`, labelled `AUC(0-tau)`; a single dose result keeps shading
    `[0, tlast]` as `AUC(0-tlast)`.

    Args:
        timecourse: the curve (times relative to its dose)
        values: parameter magnitudes of the curve
        flags: flag names of the curve

    Keyword Args:
        log_y: logarithmic value axis; a curve without a positive value stays
            linear (logged at debug level)
        title: title, the label of the curve by default
        ax: axes to draw on, a new figure by default
        style: colors and markers

    Returns:
        The axes the panel was drawn on.
    """
    _, ax = figure_of(ax)
    tc = timecourse.relative_to_dose(which="last")
    t, c = tc.time, tc.value
    ok = np.isfinite(c)
    tlast, clast = values.get("tlast", np.nan), values.get("clast", np.nan)
    lambda_z = values.get("lambda_z", np.nan)
    intercept = values.get("lambda_z_intercept", np.nan)
    thalf = values.get("thalf", np.nan)

    steady_state = np.isfinite(values.get("auc_tau", np.nan))
    tau = values.get("tau", np.nan)
    auc_bound = tau if steady_state and np.isfinite(tau) else tlast
    auc_label = "AUC(0-tau)" if steady_state else "AUC(0-tlast)"
    if np.isfinite(auc_bound):
        area = ok & (t <= auc_bound)
        ax.fill_between(
            t[area],
            0.0,
            c[area],
            color=style.auc_color,
            alpha=style.alpha,
            label=auc_label,
        )
    if np.isfinite(lambda_z) and np.isfinite(tlast):
        t_ext = np.linspace(tlast, tlast + 3.0 * thalf, 50)
        c_ext = clast * np.exp(-lambda_z * (t_ext - tlast))
        ax.fill_between(
            t_ext,
            0.0,
            c_ext,
            color=style.extrapolation_color,
            alpha=style.alpha,
            label="extrapolated",
        )
    t_first = values.get("lambda_z_t_first", np.nan)
    if np.isfinite(lambda_z) and np.isfinite(tlast) and np.isfinite(t_first):
        t_fit = np.linspace(t_first, tlast + 3.0 * thalf, 50)
        ax.plot(
            t_fit,
            np.exp(intercept - lambda_z * t_fit),
            linestyle="-",
            color=style.fit_color,
            linewidth=style.linewidth,
            label=f"lambda_z = {lambda_z:.3g}",
        )
        used = ok & (t >= t_first) & (t <= tlast) & (c > 0)
        ax.plot(
            t[used],
            c[used],
            marker=style.terminal_marker,
            linestyle="none",
            color=style.fit_color,
            markersize=style.markersize + 3,
            markerfacecolor="none",
            label="regression points",
        )
    cmax, tmax = values.get("cmax", np.nan), values.get("tmax", np.nan)
    if np.isfinite(cmax):
        ax.plot([0, tmax], [cmax, cmax], linestyle="--", color="gray", linewidth=1)
        ax.plot([tmax, tmax], [0, cmax], linestyle="--", color="gray", linewidth=1)
    c0 = values.get("c0", np.nan)
    if np.isfinite(c0) and tc.dose is not None and tc.dose.route is Route.IV_BOLUS:
        ax.plot(
            [0.0],
            [c0],
            marker="D",
            linestyle="none",
            color=style.fit_color,
            markersize=style.markersize + 1,
            label="C0",
        )
    ax.plot(
        t,
        c,
        marker=style.data_marker,
        linestyle="-",
        color=style.data_color,
        linewidth=style.linewidth,
        markersize=style.markersize,
        label="data",
    )
    ax.set_xlabel(f"time [{tc.time_unit}]")
    ax.set_ylabel(f"{tc.substance} [{tc.unit}]")
    if log_y:
        log_scale(ax, "y")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    heading = title if title is not None else (tc.label or tc.substance)
    if flags:
        heading = f"{heading} [{', '.join(flags)}]"
    ax.set_title(heading)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize="small")
    return ax


def plot_nca(
    timecourse: Timecourse,
    result: NCAResult,
    *,
    title: str | None = None,
    axes: Sequence[Axes] | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Linear and logarithmic panel of one curve with its NCA diagnostics.

    Args:
        timecourse: the curve
        result: the result of its analysis (a batch result with `indexers`, or a single result)

    Keyword Args:
        title: title of the panels, the label of the curve by default; the flags are appended
        axes: the two axes to draw the linear and the logarithmic panel into,
            a new figure by default
        style: colors and markers
        **indexers: coordinate labels selecting the sample of a batch result

    Returns:
        The figure.
    """
    values, flags = _sample_values(result, indexers)
    fig, grid = axes_of(axes, nrows=1, ncols=2, figsize=(11, 4.5))
    ax1, ax2 = grid[0]
    if title is None and indexers:
        title = "|".join(str(v) for v in indexers.values())
    draw_nca_panel(
        timecourse, values, flags, log_y=False, title=title, ax=ax1, style=style
    )
    draw_nca_panel(
        timecourse, values, flags, log_y=True, title=title, ax=ax2, style=style
    )
    return fig


def plot_nca_grid(
    timecourses: Timecourses,
    result: NCAResult,
    *,
    ncols: int = 3,
    log_y: bool = True,
    axes: Sequence[Axes] | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """One NCA panel per sample of a batch.

    Args:
        timecourses: the batch
        result: its result

    Keyword Args:
        ncols: panels per row
        log_y: logarithmic value axes
        axes: the `nrows * ncols` axes to draw the panels into, a new figure by
            default
        style: colors and markers

    Returns:
        The figure.
    """
    curves = list(timecourses)
    n = len(curves)
    nrows = int(np.ceil(n / ncols))
    fig, grid = axes_of(axes, nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    flat_axes = grid.ravel()
    indices = list(np.ndindex(*timecourses.sample_shape))
    for k, (tc, index) in enumerate(zip(curves, indices, strict=True)):
        sample = result.ds.isel(
            dict(zip(timecourses.sample_dims, (int(i) for i in index), strict=True))
        )
        values = {name: float(sample[name].values) for name in result.parameters}
        flags = decode_flags(int(sample["flags"].values))
        draw_nca_panel(
            tc, values, flags, log_y=log_y, title=None, ax=flat_axes[k], style=style
        )
    for ax in flat_axes[n:]:
        ax.set_visible(False)
    return fig


def plot_intervals(
    result: NCAResult,
    name: str = "interval_auc",
    *,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Plot a per-interval parameter against the dosing interval.

    Without `indexers`, one line per sample of the result, labelled with the
    sample's coordinate values; with `indexers` selecting one sample, a single
    line. Non-finite values (an incomplete interval) are masked so the line
    breaks there instead of raising a matplotlib warning. The x axis carries
    the integer interval numbers only, never a fractional tick.

    Args:
        result: the result of a multiple dose analysis
        name: name of the per-interval variable (`interval_*`)

    Keyword Args:
        ax: axes to draw on, a new figure by default; a caller-supplied `ax`
            keeps its figure's own layout engine, so long tick labels can
            clip unless the caller sets one (`fig.set_layout_engine("constrained")`)
        style: colors and markers
        **indexers: coordinate label per sample dimension selecting one sample

    Returns:
        The figure.

    Raises:
        ValueError: if the result has no interval parameters, or if `name` is
            not one of them.
    """
    if not result.has_intervals:
        raise ValueError("The result has no interval parameters")
    if name not in result.ds.data_vars or INTERVAL_DIM not in result.ds[name].dims:
        raise ValueError(f"'{name}' is not an interval variable of the result")
    fig, ax = figure_of(ax)
    da = result[name]
    x = result.ds[INTERVAL_DIM].to_numpy()
    if indexers:
        y = np.ma.masked_invalid(da.sel(indexers).to_numpy().astype(float))
        ax.plot(
            x,
            y,
            marker=style.data_marker,
            linestyle="-",
            color=style.data_color,
            linewidth=style.linewidth,
            markersize=style.markersize,
        )
    else:
        sample_dims = list(result.sample_dims)
        sizes = [result.ds.sizes[d] for d in sample_dims]
        indices = list(np.ndindex(*sizes))
        n = len(indices)
        labels: list[str | None] = (
            list(sample_labels(result.ds, sample_dims)) if sample_dims else [None]
        )
        colors = sample_colors(n, style.cmap)
        for i, index in enumerate(indices):
            sel = dict(zip(sample_dims, (int(k) for k in index), strict=True))
            sample = da.isel(sel)
            color: Any = style.data_color if n <= 1 else colors[i]
            y = np.ma.masked_invalid(sample.to_numpy().astype(float))
            ax.plot(
                x,
                y,
                marker=style.data_marker,
                linestyle="-",
                color=color,
                label=labels[i],
                linewidth=style.linewidth,
                markersize=style.markersize,
            )
        if n > 1:
            ax.legend(fontsize="small")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(INTERVAL_DIM)
    ax.set_ylabel(f"{name} [{result.units(name)}]")
    return fig

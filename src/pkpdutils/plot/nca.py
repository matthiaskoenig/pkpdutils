"""Diagnostic figures of the non-compartmental analysis."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from pkpdutils.nca.intervals import INTERVAL_DIM, INTERVAL_PREFIX
from pkpdutils.nca.options import decode_flags
from pkpdutils.nca.result import NCAResult
from pkpdutils.plot._common import (
    axes_of,
    axis_label,
    dose_markers,
    figure_of,
    format_value,
    group_colors,
    group_order,
    log_scale,
    sample_colors,
    sample_labels,
    sample_title,
    unit_label,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Dosing, Route, Timecourse, Timecourses

#: start of the legend label of the terminal regression line of a panel, which
#: carries the `lambda_z` of that very curve
LAMBDA_Z_LABEL = "lambda_z = "

#: the label the regression line gets in a legend shared by several panels,
#: which cannot name one curve's `lambda_z`
TERMINAL_LABEL = "terminal regression"


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
    legend: bool = True,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Axes:
    """Draw the NCA diagnostics of one curve into one axes.

    A multiple dose result (carrying `auc_tau`) shades the analysed last
    dosing interval `[0, tau]` (relative to the last dose) instead of
    `[0, tlast]`, labelled `AUC(0-tau)`; a single dose result keeps shading
    `[0, tlast]` as `AUC(0-tlast)`.

    An infusion is marked by the shaded window from the dose time to the end
    of the infusion, so that the panel shows how long the dose went in. The
    panel starts at the dose it analyses (the last one of a multiple dose
    curve), so only that dose falls inside it and the earlier doses of a
    protocol are not marked.

    Args:
        timecourse: the curve (times relative to its dose)
        values: parameter magnitudes of the curve
        flags: flag names of the curve

    Keyword Args:
        log_y: logarithmic value axis; a curve without a positive value stays
            linear (logged at debug level)
        title: title, the label of the curve by default
        legend: draw the legend of the panel; `False` for a figure whose
            panels share one legend (`plot_nca_grid`)
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
            label=f"{LAMBDA_Z_LABEL}{lambda_z:.3g}",
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
    if tc.dosing is not None:
        dose_markers(
            ax,
            Dosing.single(tc.dosing.last),
            style=style,
            time_unit=tc.time_unit,
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
    # a panel with several flags has a long title, which the default size
    # runs over the width of the panel with
    ax.set_title(heading, fontsize="small")
    if legend and ax.get_legend_handles_labels()[0]:
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

    Both panels show the same curve and carry the same title; the legend is
    drawn once, on the linear panel.

    Args:
        timecourse: the curve
        result: the result of its analysis (a batch result with `indexers`, or a single result)

    Keyword Args:
        title: title of the panels, `name = value` per indexer (the label of
            the curve without indexers) by default; the flags are appended
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
        title = ", ".join(
            f"{name} = {format_value(value)}" for name, value in indexers.items()
        )
    draw_nca_panel(
        timecourse, values, flags, log_y=False, title=title, ax=ax1, style=style
    )
    draw_nca_panel(
        timecourse,
        values,
        flags,
        log_y=True,
        title=title,
        legend=False,
        ax=ax2,
        style=style,
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
    """One NCA panel per sample of a batch, with one legend for the figure.

    The title of a panel names the sample by its coordinates,
    `dose = 50 mg, individual = s1`, with the unit of a coordinate which
    carries one; the `dose` coordinate of a batch takes the dose unit of the
    batch. The panels all draw the same artists, so the legend is drawn once:
    on the figure when this function creates it, and into the first panel
    when the caller supplies `axes`, whose figure keeps its own layout.

    Args:
        timecourses: the batch
        result: its result

    Keyword Args:
        ncols: panels per row, at most one per sample
        log_y: logarithmic value axes
        axes: the `nrows * ncols` axes to draw the panels into, a new figure by
            default
        style: colors and markers

    Returns:
        The figure.
    """
    curves = list(timecourses)
    n = len(curves)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, grid = axes_of(axes, nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    flat_axes = grid.ravel()
    indices = list(np.ndindex(*timecourses.sample_shape))
    units = {"dose": timecourses.dose_unit} if "dose" in timecourses.ds.coords else {}
    for k, (tc, index) in enumerate(zip(curves, indices, strict=True)):
        sample = result.ds.isel(
            dict(zip(timecourses.sample_dims, (int(i) for i in index), strict=True))
        )
        values = {name: float(sample[name].values) for name in result.parameters}
        flags = decode_flags(int(sample["flags"].values))
        title = sample_title(
            timecourses.ds, index, timecourses.sample_dims, units=units
        )
        draw_nca_panel(
            tc,
            values,
            flags,
            log_y=log_y,
            title=title or None,
            legend=axes is not None and k == 0,
            ax=flat_axes[k],
            style=style,
        )
    for ax in flat_axes[n:]:
        ax.set_visible(False)
    if axes is None:
        # the union of the panels' artists, since a panel without a terminal
        # regression or without a C0 draws fewer of them than its neighbours
        seen: dict[str, Any] = {}
        for ax in flat_axes[:n]:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels, strict=True):
                shared = TERMINAL_LABEL if label.startswith(LAMBDA_Z_LABEL) else label
                seen.setdefault(shared, handle)
        if seen:
            fig.legend(
                list(seen.values()),
                list(seen),
                loc="outside upper center",
                ncols=min(len(seen), 6),
                fontsize="small",
            )
    return fig


def _mean_spread(
    values: np.ndarray, statistic: Literal["sd", "se"]
) -> tuple[np.ndarray, np.ndarray]:
    r"""Mean and spread over the rows of a `(n_samples, n_interval)` block.

    The reduction ignores the non-finite entries of a column (an interval
    which is incomplete for a subject) and gives `NaN` for a column without
    one, without the `RuntimeWarning` of `numpy.nanmean` on an empty slice.
    The spread is the standard deviation \(s\) of the column with
    \(n - 1\) degrees of freedom, or \(s / \sqrt{n}\) for `"se"`.

    Args:
        values: the block, one row per sample.
        statistic: the spread to compute.

    Returns:
        The mean and the spread per column, `NaN` where they are undefined.
    """
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    filled = np.where(finite, values, 0.0)
    total = filled.sum(axis=0)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    deviation = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=0)
    sd = np.where(count > 1, np.sqrt(deviation / np.maximum(count - 1, 1)), np.nan)
    spread = sd if statistic == "sd" else sd / np.sqrt(np.maximum(count, 1))
    return mean, spread


def plot_troughs(
    result: NCAResult,
    *,
    by: str | None = None,
    spread: Literal["sd", "se"] | None = "sd",
    x: Literal["time", "interval"] = "time",
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """The trough concentration of every dosing interval, the figure of steady state.

    The trough of an interval (`interval_ctrough`, the value at its end) and,
    when the analysis reports it, its minimum (`interval_cmin`) against the
    time the trough was taken (`x="time"`, the end of the interval
    `interval_end`, which for a regular regimen is the time of the next dose)
    or against the interval number (`x="interval"`). Steady state is where
    the troughs stop rising.

    Over a batch the samples are reduced to the mean of every interval with
    its spread as error bars, per group when `by` names a coordinate; a
    result of a single curve draws that curve's values.

    Args:
        result: the result of a multiple dose analysis.

    Keyword Args:
        by: a sample dimension of the result or a coordinate along one,
            grouping the samples; one group by default.
        spread: error bars of the mean, the standard deviation (`"sd"`), the
            standard error (`"se"`) or none (`None`).
        x: the x axis, the dose time of the interval or the interval number.
        ax: axes to draw on, a new figure by default; a caller-supplied `ax`
            keeps its figure's own layout engine, so long tick labels can
            clip unless the caller sets one (`fig.set_layout_engine("constrained")`)
        style: colors and markers.

    Returns:
        The figure.

    Raises:
        ValueError: if the result has no interval parameters or no
            `interval_ctrough`, or if `x` is neither `"time"` nor
            `"interval"`.
    """
    if not result.has_intervals or "interval_ctrough" not in result.ds.data_vars:
        raise ValueError("The result has no 'interval_ctrough' to plot")
    if x not in ("time", "interval"):
        raise ValueError(f"'x' must be 'time' or 'interval', got '{x}'")
    statistic: Literal["sd", "se"] = "sd" if spread is None else spread
    names = [
        name
        for name in ("interval_ctrough", "interval_cmin")
        if name in result.ds.data_vars
    ]
    sample_dims = list(result.sample_dims)
    n_interval = int(result.ds.sizes[INTERVAL_DIM])
    n_rows = max(int(result.ds["interval_ctrough"].size) // n_interval, 1)
    blocks = {
        name: result.ds[name].to_numpy().astype(float).reshape(n_rows, n_interval)
        for name in names
    }
    by_time = x == "time" and "interval_end" in result.ds.data_vars
    if by_time:
        ends = result.ds["interval_end"].to_numpy().astype(float)
        ends = ends.reshape(n_rows, n_interval)
        x_label = axis_label("time", unit_label(result.units("interval_end")))
    else:
        ends = None
        x_label = INTERVAL_DIM
    intervals = result.ds[INTERVAL_DIM].to_numpy().astype(float)
    labels = (
        sample_labels(result.ds, sample_dims, by=by)
        if by is not None
        else ["all"] * n_rows
    )
    groups = group_order(result.ds, sample_dims, by) if by is not None else ["all"]
    fig, ax = figure_of(ax)
    colors = group_colors(len(groups), style.cmap)
    rows = np.asarray(labels)
    for i, group in enumerate(groups):
        color: Any = style.data_color if len(groups) == 1 else colors[i]
        in_group = rows == group
        # the times of the intervals of this group: two groups on different
        # regimens have their intervals at different times
        x_values = (
            _mean_spread(ends[in_group], "sd")[0] if ends is not None else intervals
        )
        for k, name in enumerate(names):
            mean, error = _mean_spread(blocks[name][in_group], statistic)
            short = name.removeprefix(INTERVAL_PREFIX)
            label = short if by is None else f"{group}, {short}"
            ax.errorbar(
                x_values,
                np.ma.masked_invalid(mean),
                yerr=None if spread is None else np.ma.masked_invalid(error),
                marker=style.data_marker if k == 0 else style.terminal_marker,
                linestyle="-" if k == 0 else "--",
                markerfacecolor=color if k == 0 else "none",
                color=color,
                label=label,
                linewidth=style.linewidth,
                markersize=style.markersize,
                capsize=2,
            )
    if x == "interval":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(x_label)
    ax.set_ylabel(axis_label("trough", unit_label(result.units("interval_ctrough"))))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize="small", title=by, title_fontsize="small")
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
    ax.set_ylabel(axis_label(name, unit_label(result.units(name))))
    return fig

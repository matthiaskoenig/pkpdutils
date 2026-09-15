"""Figures of fits: data and curve with residuals, goodness of fit, dose proportionality."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from pkpdutils.fit.proportionality import ProportionalityResult
from pkpdutils.fit.result import FitResult
from pkpdutils.plot._common import (
    axes_of,
    figure_of,
    log_scale,
    sample_colors,
    sample_labels,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle


def _parameter_text(result: FitResult, sample: xr.Dataset) -> str:
    """`name = value +- se` per parameter of the model, comma separated.

    Args:
        result: the fit (for the parameter names of the model).
        sample: the dataset of one sample.

    Returns:
        The parameter text.
    """
    parts = []
    for name in result.model.parameter_names:
        value = float(sample[name].to_numpy())
        se = float(sample[f"{name}_se"].to_numpy())
        if np.isfinite(se):
            parts.append(f"{name} = {value:.3g} ± {se:.2g}")
        else:
            parts.append(f"{name} = {value:.3g}")
    return ", ".join(parts)


def plot_fit(
    result: FitResult,
    *,
    log_x: bool = False,
    log_y: bool = False,
    n_grid: int = 200,
    title: str | None = None,
    axes: Sequence[Axes] | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Data, fitted curve and weighted residuals of one sample.

    The upper panel draws the data points (`x_data`, `y_data`, with error
    bars from `sd_data` when the fit had `sd`), the fitted curve on a fine
    grid between the smallest and the largest finite, plotted `x` (log-spaced
    when `log_x`; the curve is skipped when fewer than two points remain),
    and a title with the model name, the parameters (`name = value +- se`)
    and the flags of the sample. The lower panel draws the weighted
    residuals against `x` with a zero line.

    Args:
        result: the fit.

    Keyword Args:
        log_x: logarithmic x axis (and a log-spaced curve grid); a point
            with `x <= 0` is left out of the plot and the curve grid.
        log_y: logarithmic y axis; a point with `y <= 0` is left out of the plot.
        n_grid: number of points of the curve grid.
        title: title, the model name by default; the parameters and the
            flags are appended.
        axes: the two axes to draw the fit and the residual panel into, a new
            figure by default; a caller-supplied pair is used as it is, so it
            gives up the `height_ratios=[3, 1]` and the shared x axis of the
            default panels unless the caller sets them itself.
        style: colors and markers.
        **indexers: coordinate label per sample dimension, none for a 0-D result.

    Returns:
        The figure with the fit panel and the residual panel.
    """
    sample = result._sample(indexers)
    x = sample["x_data"].to_numpy()
    y = sample["y_data"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x_ok = (x > 0) if log_x else np.ones_like(x, dtype=bool)
    y_ok = (y > 0) if log_y else np.ones_like(y, dtype=bool)
    ok_plot = ok & x_ok & y_ok
    if axes is None:
        fig, panels = plt.subplots(
            nrows=2, ncols=1, figsize=(7, 6.5), height_ratios=[3, 1], sharex=True
        )
        fig.set_layout_engine("constrained")
        ax, ax_res = panels
    else:
        fig, panels = axes_of(axes, nrows=2, ncols=1, figsize=(7, 6.5))
        ax, ax_res = panels[0][0], panels[1][0]
    if "sd_data" in sample and np.any(np.isfinite(sample["sd_data"].to_numpy())):
        sd = sample["sd_data"].to_numpy()
        ax.errorbar(
            x[ok_plot],
            y[ok_plot],
            yerr=sd[ok_plot],
            marker=style.data_marker,
            linestyle="none",
            color=style.data_color,
            markersize=style.markersize,
            capsize=2,
            label="data",
        )
    else:
        ax.plot(
            x[ok_plot],
            y[ok_plot],
            marker=style.data_marker,
            linestyle="none",
            color=style.data_color,
            markersize=style.markersize,
            label="data",
        )
    p = np.array(
        [float(sample[name].to_numpy()) for name in result.model.parameter_names]
    )
    if ok_plot.sum() >= 2 and np.all(np.isfinite(p)):
        lo, hi = float(x[ok_plot].min()), float(x[ok_plot].max())
        if lo < hi:
            grid = (
                np.geomspace(lo, hi, n_grid) if log_x else np.linspace(lo, hi, n_grid)
            )
            ax.plot(
                grid,
                result.model.predict(grid, p),
                "-",
                color=style.fit_color,
                linewidth=style.linewidth,
                label="fit",
            )
    heading = title if title is not None else result.model.name
    flags = result.decode_flags(int(sample["flags"].to_numpy()))
    text = f"{heading}: {_parameter_text(result, sample)}"
    if flags:
        text = f"{text} [{', '.join(flags)}]"
    ax.set_title(text, fontsize="small")
    # the x axis is shared with the residual panel below, which carries the label
    ax.set_ylabel(f"y [{result.units('y_data')}]")
    if log_x:
        log_scale(ax, "x")
    if log_y:
        log_scale(ax, "y")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize="small")
    res = sample["residuals"].to_numpy()
    ok_res = ok & np.isfinite(res) & x_ok
    ax_res.axhline(0.0, color="gray", linewidth=1)
    ax_res.plot(
        x[ok_res],
        res[ok_res],
        marker=style.data_marker,
        linestyle="none",
        color=style.data_color,
        markersize=style.markersize,
    )
    ax_res.set_xlabel(f"x [{result.units('x_data')}]")
    ax_res.set_ylabel("weighted residual")
    return fig


def plot_goodness_of_fit(
    result: FitResult,
    *,
    log_x: bool = False,
    log_y: bool = False,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Predicted against observed values of every sample, with the identity line.

    Args:
        result: the fit.

    Keyword Args:
        log_x: logarithmic axis of the observed values; non-positive points are
            masked out to avoid a warning from the log scale.
        log_y: logarithmic axis of the predicted values, masked the same way.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.

    Returns:
        The figure.
    """
    fig, ax = figure_of(ax, figsize=(5.5, 5.5))
    n_point = result.ds.sizes["point"]
    y = result["y_data"].to_numpy().reshape(-1, n_point)
    pred = result["y_pred"].to_numpy().reshape(-1, n_point)
    r2 = result["r2"].to_numpy().reshape(-1)
    labels = sample_labels(result.ds, result.sample_dims) or [result.model.name]
    n = y.shape[0]
    colors = sample_colors(n, style.cmap)
    for i in range(n):
        ok = np.isfinite(y[i]) & np.isfinite(pred[i])
        if log_x:
            ok &= y[i] > 0
        if log_y:
            ok &= pred[i] > 0
        label = f"{labels[i]} (R² = {r2[i]:.3f})" if n <= 8 else None
        color = colors[i] if n > 1 else style.data_color
        ax.scatter(
            y[i][ok], pred[i][ok], color=color, s=style.markersize**2 * 1.5, label=label
        )
    finite = np.concatenate([y[np.isfinite(y)], pred[np.isfinite(pred)]])
    if log_x or log_y:
        finite = finite[finite > 0]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="identity")
    unit = result.units("y_data")
    ax.set_xlabel(f"observed [{unit}]")
    ax.set_ylabel(f"predicted [{unit}]")
    if log_x:
        log_scale(ax, "x")
    if log_y:
        log_scale(ax, "y")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize="small")
    return fig


def plot_dose_proportionality(
    result: FitResult,
    *,
    test: ProportionalityResult | None = None,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Log-log exposure against dose with the power fit and the acceptance bounds.

    Args:
        result: fit of `Power` (parameters `a`, `b`).

    Keyword Args:
        test: the result of `proportionality_test`; when given, draws the
            acceptance wedge through the first data point and the verdict
            ("proportional", "inconclusive" or "not proportional") in the title.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.
        **indexers: coordinate label per sample dimension, none for a 0-D result.

    Returns:
        The figure.
    """
    sample = result._sample(indexers)
    x = sample["x_data"].to_numpy()
    y = sample["y_data"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    fig, ax = figure_of(ax, figsize=(6, 4.5))
    ax.plot(
        x[ok],
        y[ok],
        marker=style.data_marker,
        linestyle="none",
        color=style.data_color,
        markersize=style.markersize,
        label="data",
    )
    a, b = float(sample["a"].to_numpy()), float(sample["b"].to_numpy())
    ci_low, ci_high = (
        float(sample["b_ci_low"].to_numpy()),
        float(sample["b_ci_high"].to_numpy()),
    )
    heading = f"b = {b:.3f}"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        heading = f"{heading} [{ci_low:.3f}, {ci_high:.3f}]"
    t = None if test is None else test.sel(**indexers)
    if ok.any() and np.isfinite(a) and np.isfinite(b):
        lo, hi = float(x[ok].min()), float(x[ok].max())
        if lo < hi:
            grid = np.geomspace(lo, hi, 100)
            ax.plot(
                grid,
                a * grid**b,
                "-",
                color=style.fit_color,
                linewidth=style.linewidth,
                label=f"a·x^b, b = {b:.3f}",
            )
            if t is not None:
                bound_low, bound_high = t.bounds
                x0, y0 = grid[0], a * grid[0] ** b
                ax.fill_between(
                    grid,
                    y0 * (grid / x0) ** bound_low,
                    y0 * (grid / x0) ** bound_high,
                    color=style.auc_color,
                    alpha=style.alpha,
                    label=f"acceptance [{bound_low:.2f}, {bound_high:.2f}]",
                )
    if t is not None:
        if bool(t.proportional.to_numpy()):
            verdict = "proportional"
        elif bool(t.inconclusive.to_numpy()):
            verdict = "inconclusive"
        else:
            verdict = "not proportional"
        heading = f"{heading}: {verdict}"
    ax.set_title(heading)
    log_scale(ax, "x")
    log_scale(ax, "y")
    if ok.any():
        # a dose escalation has few, known doses: label those instead of the
        # decade ticks of the log scale, whose labels overlap over a range of
        # one or two decades
        doses = np.unique(x[ok])
        ax.set_xticks(doses, labels=[f"{dose:g}" for dose in doses])
        ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(f"x [{result.units('x_data')}]")
    ax.set_ylabel(f"y [{result.units('y_data')}]")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize="small")
    return fig


def plot_bland_altman(
    result: FitResult,
    *,
    log_ratio: bool = False,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Bland-Altman plot of the predictions against the data of every sample.

    The difference `y_pred - y_data` (the log ratio with `log_ratio`) against
    the mean of both per point, with the mean difference and the limits of
    agreement `mean +- 1.96 sd` as horizontal lines (Bland & Altman 1986).

    `log_ratio` selects the statistic the figure shows and not only the scale
    of an axis, which is why it is not one of the `log_x`/`log_y` keywords of
    the other figures: it replaces the difference by the log ratio and draws
    the mean on a logarithmic axis.

    Args:
        result: the fit.

    Keyword Args:
        log_ratio: use the log ratio and the log mean; non-positive points are
            masked out.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.

    Returns:
        The figure.
    """
    y = result.ds["y_data"].to_numpy().ravel()
    pred = result.ds["y_pred"].to_numpy().ravel()
    ok = np.isfinite(y) & np.isfinite(pred)
    if log_ratio:
        ok &= (y > 0) & (pred > 0)
        mean = np.exp((np.log(y[ok]) + np.log(pred[ok])) / 2.0)
        diff = np.log(pred[ok]) - np.log(y[ok])
    else:
        mean = (y[ok] + pred[ok]) / 2.0
        diff = pred[ok] - y[ok]
    fig, ax = figure_of(ax, figsize=(6, 4))
    ax.plot(
        mean,
        diff,
        linestyle="none",
        marker=style.data_marker,
        color=style.data_color,
        markersize=style.markersize,
    )
    if diff.size:
        center, sd = (
            float(diff.mean()),
            float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        )
        ax.axhline(
            center, color=style.fit_color, linestyle="--", linewidth=style.linewidth
        )
        ax.axhline(
            center + 1.96 * sd, color=style.limit_color, linestyle=":", linewidth=1.0
        )
        ax.axhline(
            center - 1.96 * sd, color=style.limit_color, linestyle=":", linewidth=1.0
        )
    ax.axhline(0.0, color="gray", linewidth=1.0)
    unit = result.units("y_data")
    if log_ratio:
        log_scale(ax, "x")
        ax.set_xlabel(f"mean of observed and predicted [{unit}]")
        ax.set_ylabel("log ratio predicted / observed")
    else:
        ax.set_xlabel(f"mean of observed and predicted [{unit}]")
        ax.set_ylabel(f"difference predicted - observed [{unit}]")
    return fig

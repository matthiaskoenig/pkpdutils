"""Figures of fits: data and curve with residuals, goodness of fit, dose proportionality."""

import math
from typing import Any

import numpy as np
import xarray as xr
from sbmlsim_core.chart import Chart, ChartAxis

from pkpdutils.fit.proportionality import ProportionalityResult
from pkpdutils.fit.result import FitResult
from pkpdutils.plot._common import (
    FONT_SIZES,
    axis_label,
    chart_of,
    data_limits,
    finite,
    log_axis,
    marker,
    sample_colors,
    sample_labels,
    unit_label,
    value_axis,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle


def _axis_labels(result: FitResult) -> tuple[str, str]:
    """The axis labels of a fit, `name [unit]` for `x` and for `y`.

    The names are `attrs["x_name"]` and `attrs["y_name"]`, which the front
    ends of the fit set from what they were given (`time` and the substance
    for a timecourse, the column names for a table); a result built without
    them falls back to `x` and `y`. The unit is written in the short symbols
    of the data (`unit_label`), and a variable without one is labelled with
    its name alone, since `[dimensionless]` says nothing about it.

    Args:
        result: the fit.

    Returns:
        The label of the x axis and the label of the y axis.
    """
    return (
        axis_label(
            str(result.ds.attrs.get("x_name", "x")), unit_label(result.units("x_data"))
        ),
        axis_label(
            str(result.ds.attrs.get("y_name", "y")), unit_label(result.units("y_data"))
        ),
    )


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
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Chart:
    """Data, fitted curve and weighted residuals of one sample.

    The upper panel draws the data points (`x_data`, `y_data`, with error
    bars from `sd_data` when the fit had `sd`), the fitted curve on a fine
    grid between the smallest and the largest finite, plotted `x` (log-spaced
    when `log_x`; the curve is skipped when fewer than two points remain),
    and a title with the model name, the parameters (`name = value +- se`)
    and the flags of the sample. The lower panel draws the weighted
    residuals against `x` with a zero line; both panels share the x axis,
    which the lower one labels.

    The axes are labelled with the names the front end of the fit stored in
    `attrs["x_name"]` and `attrs["y_name"]` (`time` and the substance for a
    timecourse, the column names for a table), `x` and `y` without them.

    Args:
        result: the fit.

    Keyword Args:
        log_x: logarithmic x axis (and a log-spaced curve grid); a point
            with `x <= 0` is left out of the plot and the curve grid.
        log_y: logarithmic y axis; a point with `y <= 0` is left out of the plot.
        n_grid: number of points of the curve grid.
        title: title, the model name by default; the parameters and the
            flags are appended.
        chart: the chart to draw into, a new one by default, in which the fit
            panel is three times as high as the residual panel; in a chart of
            the caller the fit panel goes into the cell `(row, column)` and
            the residual panel into the one below.
        row: the row of the fit panel in `chart`.
        column: the column of the panels in `chart`.
        style: colors and markers.
        **indexers: coordinate label per sample dimension, none for a 0-D result.

    Returns:
        The chart.

    Raises:
        ValueError: if the panels do not fit into the grid of `chart`.
    """
    sample = result._sample(indexers)
    x = sample["x_data"].to_numpy().astype(float)
    y = sample["y_data"].to_numpy().astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x_ok = (x > 0) if log_x else np.ones_like(x, dtype=bool)
    y_ok = (y > 0) if log_y else np.ones_like(y, dtype=bool)
    ok_plot = ok & x_ok & y_ok
    sd = None
    if "sd_data" in sample and np.any(np.isfinite(sample["sd_data"].to_numpy())):
        sd = sample["sd_data"].to_numpy().astype(float)[ok_plot]
    p = np.array(
        [float(sample[name].to_numpy()) for name in result.model.parameter_names]
    )
    curve = None
    if ok_plot.sum() >= 2 and np.all(np.isfinite(p)):
        lo, hi = float(x[ok_plot].min()), float(x[ok_plot].max())
        if lo < hi:
            grid = (
                np.geomspace(lo, hi, n_grid) if log_x else np.linspace(lo, hi, n_grid)
            )
            curve = (grid, np.asarray(result.model.predict(grid, p), dtype=float))
    res = sample["residuals"].to_numpy().astype(float)
    ok_res = ok & np.isfinite(res) & x_ok
    heading = title if title is not None else result.model.name
    flags = result.decode_flags(int(sample["flags"].to_numpy()))
    text = f"{heading}: {_parameter_text(result, sample)}"
    if flags:
        text = f"{text} [{', '.join(flags)}]"
    x_label, y_label = _axis_labels(result)
    y_values = [y[ok_plot]]
    if sd is not None:
        y_values += [y[ok_plot] - sd, y[ok_plot] + sd]
    if curve is not None:
        y_values.append(curve[1])
    logx = log_axis(log_x, "x", x[ok_plot], x[ok_res])
    logy = log_axis(log_y, "y", *y_values)
    # the panels share the x axis, which holds the data of both
    x_limits = data_limits(x[ok_plot], x[ok_res], log=logx)
    own = chart is None
    target = chart_of(
        chart,
        row,
        column,
        rows=4 if own else 2,
        columns=1,
        size=(7.0, 6.5),
        fonts={"plot_title": FONT_SIZES["small"]},
    )
    fit_panel = target.plot(
        row,
        column,
        row_span=3 if own else 1,
        title=text,
        # the x axis is shared with the residual panel below, which labels it
        x=value_axis(
            "",
            log=logx,
            limits=x_limits,
            label_visible=False,
            ticks_visible=False,
            tick_marks_visible=True,
        ),
        y=value_axis(y_label, log=logy, limits=data_limits(*y_values, log=logy)),
        # matplotlib lists the data with error bars, a container, after the fit
        legend_order=["fit", "data"] if sd is not None else ["data", "fit"],
    )
    fit_panel.curve(
        x[ok_plot],
        y[ok_plot],
        yerr=sd,
        error_cap=2.0 if sd is not None else 0.0,
        line="none",
        marker=marker(style.data_marker),
        size=style.markersize,
        color=style.data_color,
        label="data",
        legend="data",
    )
    if curve is not None:
        fit_panel.curve(
            *curve,
            width=style.linewidth,
            color=style.fit_color,
            label="fit",
            legend="fit",
        )
    residual_panel = target.plot(
        row + (3 if own else 1),
        column,
        x=value_axis(x_label, log=logx, limits=x_limits),
        y=ChartAxis(label="weighted residual"),
        legend=False,
    )
    residual_panel.rule(0.0, color="gray", width=1.0)
    residual_panel.curve(
        x[ok_res],
        res[ok_res],
        line="none",
        marker=marker(style.data_marker),
        size=style.markersize,
        color=style.data_color,
    )
    return target


def plot_goodness_of_fit(
    result: FitResult,
    *,
    log_x: bool = False,
    log_y: bool = False,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    """Predicted against observed values of every sample, with the identity line.

    Args:
        result: the fit.

    Keyword Args:
        log_x: logarithmic axis of the observed values; non-positive points
            are left out.
        log_y: logarithmic axis of the predicted values, likewise.
        chart: the chart to draw into, a new one by default.
        row: the row of the panel in `chart`.
        column: the column of the panel in `chart`.
        style: colors and markers.

    Returns:
        The chart.

    Raises:
        ValueError: if the cell is not in the grid of `chart`.
    """
    n_point = result.ds.sizes["point"]
    y = result["y_data"].to_numpy().astype(float).reshape(-1, n_point)
    pred = result["y_pred"].to_numpy().astype(float).reshape(-1, n_point)
    r2 = result["r2"].to_numpy().reshape(-1)
    labels = sample_labels(result.ds, result.sample_dims) or [result.model.name]
    n = y.shape[0]
    colors = sample_colors(n, style.cmap)
    points = []
    for i in range(n):
        ok = np.isfinite(y[i]) & np.isfinite(pred[i])
        if log_x:
            ok &= y[i] > 0
        if log_y:
            ok &= pred[i] > 0
        label = f"{labels[i]} (R² = {r2[i]:.3f})" if n <= 8 else None
        color = colors[i] if n > 1 else style.data_color
        points.append((y[i][ok], pred[i][ok], color, label))
    values = finite(y, pred)
    if log_x or log_y:
        values = values[values > 0]
    identity = None
    if values.size:
        lo, hi = float(values.min()), float(values.max())
        identity = ([lo, hi], [lo, hi])
    xs = [p[0] for p in points] + ([identity[0]] if identity else [])
    ys = [p[1] for p in points] + ([identity[1]] if identity else [])
    logx = log_axis(log_x, "x", *xs)
    logy = log_axis(log_y, "y", *ys)
    unit = unit_label(result.units("y_data"))
    target = chart_of(chart, row, column, rows=1, columns=1, size=(5.5, 5.5))
    plot = target.plot(
        row,
        column,
        x=value_axis(
            axis_label("observed", unit), log=logx, limits=data_limits(*xs, log=logx)
        ),
        y=value_axis(
            axis_label("predicted", unit), log=logy, limits=data_limits(*ys, log=logy)
        ),
    )
    # a scatter of matplotlib of the area `markersize**2 * 1.5`, whose
    # diameter is the root of it, with an edge of 1 point in its color
    size = math.sqrt(style.markersize**2 * 1.5)
    for observed, predicted, color, label in points:
        plot.curve(
            observed,
            predicted,
            line="none",
            marker="circle",
            size=size,
            color=color,
            label=label,
            legend=label,
        )
    if identity is not None:
        plot.curve(
            *identity,
            line="dashed",
            width=1.0,
            color="gray",
            label="identity",
            legend="identity",
        )
    return target


def plot_dose_proportionality(
    result: FitResult,
    *,
    test: ProportionalityResult | None = None,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Chart:
    """Log-log exposure against dose with the power fit and the acceptance bounds.

    The axes are labelled with the names the front end of the fit stored in
    `attrs["x_name"]` and `attrs["y_name"]` (the column names for
    `fit_table`), `x` and `y` without them. A dose escalation has few, known
    doses: the x axis ticks at those instead of at the decades.

    Args:
        result: fit of `Power` (parameters `a`, `b`).

    Keyword Args:
        test: the result of `proportionality_test`; when given, draws the
            acceptance wedge through the first data point and the verdict
            ("proportional", "inconclusive" or "not proportional") in the title.
        chart: the chart to draw into, a new one by default.
        row: the row of the panel in `chart`.
        column: the column of the panel in `chart`.
        style: colors and markers.
        **indexers: coordinate label per sample dimension, none for a 0-D result.

    Returns:
        The chart.

    Raises:
        ValueError: if the cell is not in the grid of `chart`.
    """
    sample = result._sample(indexers)
    x = sample["x_data"].to_numpy().astype(float)
    y = sample["y_data"].to_numpy().astype(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    a, b = float(sample["a"].to_numpy()), float(sample["b"].to_numpy())
    ci_low, ci_high = (
        float(sample["b_ci_low"].to_numpy()),
        float(sample["b_ci_high"].to_numpy()),
    )
    heading = f"b = {b:.3f}"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        heading = f"{heading} [{ci_low:.3f}, {ci_high:.3f}]"
    t = None if test is None else test.sel(**indexers)
    fit_curve = None
    wedge = None
    if ok.any() and np.isfinite(a) and np.isfinite(b):
        lo, hi = float(x[ok].min()), float(x[ok].max())
        if lo < hi:
            grid = np.geomspace(lo, hi, 100)
            fit_curve = (grid, a * grid**b)
            if t is not None:
                bound_low, bound_high = t.bounds
                x0, y0 = grid[0], a * grid[0] ** b
                wedge = (
                    grid,
                    y0 * (grid / x0) ** bound_low,
                    y0 * (grid / x0) ** bound_high,
                    f"acceptance [{bound_low:.2f}, {bound_high:.2f}]",
                )
    if t is not None:
        if bool(t.proportional.to_numpy()):
            verdict = "proportional"
        elif bool(t.inconclusive.to_numpy()):
            verdict = "inconclusive"
        else:
            verdict = "not proportional"
        heading = f"{heading}: {verdict}"
    xs = [x[ok]] + ([fit_curve[0]] if fit_curve else [])
    ys = [y[ok]]
    if fit_curve is not None:
        ys.append(fit_curve[1])
    if wedge is not None:
        ys += [wedge[1], wedge[2]]
    logx = log_axis(True, "x", *xs)
    logy = log_axis(True, "y", *ys)
    x_label, y_label = _axis_labels(result)
    x_limits = data_limits(*xs, log=logx)
    x_axis = value_axis(x_label, log=logx, limits=x_limits)
    if ok.any():
        # a dose escalation has few, known doses: label those instead of the
        # decade ticks of the log scale, whose labels overlap over a range of
        # one or two decades
        doses = np.unique(x[ok])
        x_axis = value_axis(
            x_label,
            log=logx,
            limits=x_limits,
            ticks=[float(d) for d in doses],
            tick_labels=[f"{d:g}" for d in doses],
        )
    target = chart_of(chart, row, column, rows=1, columns=1, size=(6.0, 4.5))
    plot = target.plot(
        row,
        column,
        title=heading,
        x=x_axis,
        y=value_axis(y_label, log=logy, limits=data_limits(*ys, log=logy)),
        legend_order=["data", *([] if fit_curve is None else ["fit"]), "acceptance"],
    )
    if wedge is not None:
        plot.band(
            wedge[0],
            wedge[1],
            wedge[2],
            line="none",
            color=style.auc_color,
            fill_opacity=style.alpha,
            label=wedge[3],
            legend="acceptance",
        )
    plot.curve(
        x[ok],
        y[ok],
        line="none",
        marker=marker(style.data_marker),
        size=style.markersize,
        color=style.data_color,
        label="data",
        legend="data",
    )
    if fit_curve is not None:
        plot.curve(
            *fit_curve,
            width=style.linewidth,
            color=style.fit_color,
            label=f"a·x^b, b = {b:.3f}",
            legend="fit",
        )
    return target


def plot_bland_altman(
    result: FitResult,
    *,
    log_ratio: bool = False,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
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
            left out.
        chart: the chart to draw into, a new one by default.
        row: the row of the panel in `chart`.
        column: the column of the panel in `chart`.
        style: colors and markers.

    Returns:
        The chart.

    Raises:
        ValueError: if the cell is not in the grid of `chart`.
    """
    y = result.ds["y_data"].to_numpy().astype(float).ravel()
    pred = result.ds["y_pred"].to_numpy().astype(float).ravel()
    ok = np.isfinite(y) & np.isfinite(pred)
    if log_ratio:
        ok &= (y > 0) & (pred > 0)
        mean = np.exp((np.log(y[ok]) + np.log(pred[ok])) / 2.0)
        diff = np.log(pred[ok]) - np.log(y[ok])
    else:
        mean = (y[ok] + pred[ok]) / 2.0
        diff = pred[ok] - y[ok]
    unit = unit_label(result.units("y_data"))
    mean_label = axis_label("mean of observed and predicted", unit)
    y_label = (
        "log ratio predicted / observed"
        if log_ratio
        else axis_label("difference predicted - observed", unit)
    )
    logx = log_axis(log_ratio, "x", mean)
    target = chart_of(chart, row, column, rows=1, columns=1, size=(6.0, 4.0))
    plot = target.plot(
        row,
        column,
        x=value_axis(mean_label, log=logx, limits=data_limits(mean, log=logx)),
        y=ChartAxis(label=y_label),
        legend=False,
    )
    plot.curve(
        mean,
        diff,
        line="none",
        marker=marker(style.data_marker),
        size=style.markersize,
        color=style.data_color,
    )
    # the zero line under the mean difference, which it would hide when the
    # mean is close to zero
    plot.rule(0.0, color="gray", width=1.0)
    if diff.size:
        center = float(diff.mean())
        sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
        plot.rule(center, color=style.fit_color, line="dashed", width=style.linewidth)
        plot.rule(center + 1.96 * sd, color=style.limit_color, line="dotted", width=1.0)
        plot.rule(center - 1.96 * sd, color=style.limit_color, line="dotted", width=1.0)
    return target

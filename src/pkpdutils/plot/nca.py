"""Diagnostic figures of the non-compartmental analysis."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import t as student_t

from pkpdutils.nca.intervals import INTERVAL_DIM, INTERVAL_PREFIX
from pkpdutils.nca.options import decode_flags
from pkpdutils.nca.result import NCAResult
from pkpdutils.plot._common import (
    axes_of,
    axis_label,
    dose_markers,
    estimate_text,
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


#: the parameters the box of the NCA panel lists, in this order, when the
#: result carries them; a clearance and a volume appear in the form of the
#: route (`cl` or `cl_f`)
PANEL_PARAMETERS: tuple[str, ...] = (
    "cmax",
    "tmax",
    "auc_last",
    "auc_inf_obs",
    "thalf",
    "cl",
    "cl_f",
    "vz",
    "vz_f",
    "mrt",
    "auc_tau",
    "cmin_ss",
    "accumulation_ratio",
)


def _quantity_text(
    values: Mapping[str, float],
    units: Mapping[str, str] | None,
    name: str,
    digits: int = 3,
) -> str:
    """`value unit` of one parameter, the interval of an uncertainty analysis included.

    Args:
        values: the parameter magnitudes of the curve.
        units: the unit per parameter, `None` for magnitudes without units.
        name: the parameter.
        digits: significant digits.

    Returns:
        `2.9 [2.7, 3.1] mg/l` or `2.9 mg/l`; empty for a missing parameter.
    """
    value = values.get(name, np.nan)
    if not np.isfinite(value):
        return ""
    text = estimate_text(
        value,
        values.get(f"{name}_ci_low", np.nan),
        values.get(f"{name}_ci_high", np.nan),
        digits,
    )
    unit = unit_label(units.get(name, "")) if units else ""
    return f"{text} {unit}".rstrip()


def _terminal_band(
    t: np.ndarray,
    c: np.ndarray,
    values: Mapping[str, float],
    t_grid: np.ndarray,
    ci_level: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""The confidence band of the terminal regression line on `t_grid`.

    The regression of \(\ln c\) on \(t\) over its \(n\) points has the
    standard error of the slope \(\mathrm{se}(\lambda_z)\) (`lambda_z_stderr`),
    from which the residual standard deviation follows as
    \(s = \mathrm{se}(\lambda_z) \sqrt{S_{tt}}\) with
    \(S_{tt} = \sum (t_i - \bar t)^2\). The band of the regression line at a
    time \(t\) is the usual one of a simple linear regression,
    \(\hat y(t) \pm t_{n-2, 1 - \alpha/2}\, s \sqrt{1/n + (t - \bar t)^2 / S_{tt}}\),
    exponentiated back to the concentration scale, so it widens away from
    the centre of the regression window and shows how uncertain the
    extrapolated tail is [@Draper1998].

    Args:
        t: the times of the curve (relative to the analysed dose).
        c: the values of the curve.
        values: the parameter magnitudes (`lambda_z`, `lambda_z_intercept`,
            `lambda_z_stderr`, `lambda_z_t_first`, `tlast`).
        t_grid: the times to evaluate the band at.
        ci_level: level of the band.

    Returns:
        The lower and the upper bound on `t_grid`, or `None` when the
        regression has fewer than three points or no standard error.
    """
    lambda_z = values.get("lambda_z", np.nan)
    intercept = values.get("lambda_z_intercept", np.nan)
    stderr = values.get("lambda_z_stderr", np.nan)
    t_first, tlast = values.get("lambda_z_t_first", np.nan), values.get("tlast", np.nan)
    if not all(np.isfinite([lambda_z, intercept, stderr, t_first, tlast])):
        return None
    used = np.isfinite(c) & (t >= t_first) & (t <= tlast) & (c > 0)
    n = int(used.sum())
    if n < 3 or stderr <= 0:
        return None
    x = t[used]
    x_mean = float(x.mean())
    s_tt = float(np.sum((x - x_mean) ** 2))
    if s_tt <= 0:
        return None
    residual_sd = stderr * np.sqrt(s_tt)
    quantile = student_t.ppf(0.5 + ci_level / 2.0, n - 2)
    half = quantile * residual_sd * np.sqrt(1.0 / n + (t_grid - x_mean) ** 2 / s_tt)
    predicted = intercept - lambda_z * t_grid
    return np.exp(predicted - half), np.exp(predicted + half)


def _thalf_interval(
    values: Mapping[str, float], ci_level: float
) -> tuple[float, float]:
    r"""The interval of the half-life, from the analysis or from the regression.

    An uncertainty analysis reports `thalf_ci_low`/`thalf_ci_high`; without one,
    the interval follows from the regression, \(\lambda_z \pm t_{n-2}\,
    \mathrm{se}(\lambda_z)\) mapped through \(t_{1/2} = \ln 2 / \lambda_z\).

    Args:
        values: the parameter magnitudes.
        ci_level: level of the interval.

    Returns:
        The bounds, `NaN` when neither source is available.
    """
    low, high = values.get("thalf_ci_low", np.nan), values.get("thalf_ci_high", np.nan)
    if np.isfinite(low) and np.isfinite(high):
        return low, high
    lambda_z, stderr = (
        values.get("lambda_z", np.nan),
        values.get("lambda_z_stderr", np.nan),
    )
    n = values.get("lambda_z_n_points", np.nan)
    if not (
        np.isfinite(lambda_z) and np.isfinite(stderr) and np.isfinite(n) and n >= 3
    ):
        return np.nan, np.nan
    quantile = student_t.ppf(0.5 + ci_level / 2.0, int(n) - 2)
    upper_rate = lambda_z + quantile * stderr
    lower_rate = lambda_z - quantile * stderr
    high = np.log(2.0) / lower_rate if lower_rate > 0 else np.inf
    return np.log(2.0) / upper_rate, high


def parameter_rows(
    values: Mapping[str, float],
    units: Mapping[str, str] | None,
    parameters: Sequence[str],
    ci_level: float,
) -> list[tuple[str, str, str, str]]:
    """The rows of the parameter table of `plot_nca`.

    Args:
        values: the parameter magnitudes.
        units: the unit per parameter, `None` for magnitudes without units.
        parameters: the parameters to list, in this order; a missing one is
            skipped.
        ci_level: level of the interval of the half-life derived from the
            regression when the analysis reports none.

    Returns:
        One `(name, value, interval, unit)` row per available parameter, the
        interval `[low, high]` or empty; `auc_inf_obs` is followed by the row
        `extrapolated`, its extrapolated share in percent.
    """
    rows: list[tuple[str, str, str, str]] = []
    for name in parameters:
        value = values.get(name, np.nan)
        if not np.isfinite(value):
            continue
        if name == "thalf":
            low, high = _thalf_interval(values, ci_level)
        else:
            low = values.get(f"{name}_ci_low", np.nan)
            high = values.get(f"{name}_ci_high", np.nan)
        interval = (
            f"[{low:.3g}, {high:.3g}]" if np.isfinite(low) and np.isfinite(high) else ""
        )
        unit = unit_label(units.get(name, "")) if units else ""
        rows.append((name, f"{value:.3g}", interval, unit))
        if name == "auc_inf_obs":
            fraction = values.get("auc_extrap_fraction", np.nan)
            if np.isfinite(fraction):
                rows.append(("extrapolated", f"{100.0 * fraction:.2g}", "", "%"))
    return rows


def _draw_parameter_table(
    ax: Axes,
    rows: Sequence[tuple[str, str, str, str]],
    ci_level: float,
    style: PlotStyle,
) -> None:
    """Write the parameter table into an axes of its own.

    Args:
        ax: the axes, turned off and used as a text panel.
        rows: the rows of `parameter_rows`.
        ci_level: level of the intervals, named in the heading when a row
            carries one.
        style: the font size of the annotations.
    """
    ax.set_axis_off()
    if not rows:
        return
    widths = [max(len(row[k]) for row in rows) for k in range(3)]
    lines = [
        f"{name:<{widths[0]}}  {value:>{widths[1]}}  {interval:<{widths[2]}}  {unit}".rstrip()
        for name, value, interval, unit in rows
    ]
    heading = "parameters"
    if any(row[2] for row in rows):
        heading = f"parameters, {100.0 * ci_level:g} % interval"
    ax.set_title(heading, fontsize="small")
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.annotation_fontsize,
        family="monospace",
        linespacing=1.6,
    )


def draw_nca_panel(
    timecourse: Timecourse,
    values: Mapping[str, float],
    flags: list[str],
    *,
    log_y: bool = False,
    title: str | None = None,
    legend: bool = True,
    annotate: bool = True,
    spread: Literal["sd", "se"] | None = "sd",
    ci_level: float = 0.95,
    units: Mapping[str, str] | None = None,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Axes:
    r"""Draw the NCA diagnostics of one curve into one axes.

    The panel shows the data with their spread, the area to \(t_\mathrm{last}\)
    (`AUC(0-tlast)`) or over the last dosing interval (`AUC(0-tau)` of a
    multiple dose result), the extrapolated tail, the terminal regression line
    with the points it used and its confidence band (`_terminal_band`), the
    peak \(C_\mathrm{max}\)/\(t_\mathrm{max}\) with its guide lines, \(C_0\)
    of a bolus, and the dose it analyses (an infusion as its window). With
    `annotate` the areas, the peak, the last point, \\(C_0\\) and the regression
    carry their values on the plot, the interval of the half-life from the uncertainty analysis
    (`thalf_ci_low`/`thalf_ci_high`) or from the regression
    (`_thalf_interval`); the table of every parameter is the third panel of
    `plot_nca`.

    The panel starts at the dose it analyses (the last one of a multiple dose
    curve), so only that dose falls inside it and the earlier doses of a
    protocol are not marked.

    Args:
        timecourse: the curve (times relative to its dose)
        values: parameter magnitudes of the curve, the uncertainty variables
            included when the result carries them
        flags: flag names of the curve

    Keyword Args:
        log_y: logarithmic value axis; a curve without a positive value stays
            linear (logged at debug level)
        title: title, the label of the curve by default
        legend: draw the legend of the panel; `False` for a figure whose
            panels share one legend (`plot_nca_grid`)
        annotate: write the values of the peak, the last point and the
            regression on the plot
        spread: the error bars of the data, `sd` or `se` of the curve when it
            carries them, `None` for none
        ci_level: level of the confidence band of the regression and of the
            interval of the half-life derived from it
        units: the unit per parameter, for the annotations
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
    unit_of = (
        (lambda name: unit_label(units.get(name, ""))) if units else (lambda name: "")
    )
    time_unit = unit_label(tc.time_unit) or tc.time_unit
    value_unit = unit_label(tc.unit) or tc.unit
    fontsize = style.annotation_fontsize

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
            alpha=style.alpha + 0.1,
            linewidth=0,
            label=auc_label,
        )
        if annotate:
            auc_name = "auc_tau" if steady_state else "auc_last"
            auc_text = _quantity_text(values, units, auc_name)
            if auc_text:
                # inside the area, at a third of its width and height
                x_area, c_area = t[area], c[area]
                x_text = 0.55 * auc_bound
                top = float(np.interp(x_text, x_area, c_area))
                bottom = ax.get_ylim()[0]
                y_text = (
                    float(np.sqrt(max(top, 1e-300) * max(bottom, top * 1e-3)))
                    if log_y
                    else 0.35 * top
                )
                ax.text(
                    x_text,
                    y_text,
                    f"{auc_label} = {auc_text}",
                    fontsize=fontsize,
                    color=style.data_color,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.7,
                    },
                )
    t_end = tlast + 3.0 * thalf if np.isfinite(thalf) else np.nan
    if np.isfinite(lambda_z) and np.isfinite(tlast) and np.isfinite(t_end):
        t_ext = np.linspace(tlast, t_end, 50)
        c_ext = clast * np.exp(-lambda_z * (t_ext - tlast))
        ax.fill_between(
            t_ext,
            0.0,
            c_ext,
            color=style.extrapolation_color,
            alpha=style.alpha + 0.1,
            linewidth=0,
            label="extrapolated",
        )
        auc_inf = values.get("auc_inf_obs", np.nan)
        auc_last = values.get("auc_last", np.nan)
        if annotate and np.isfinite(auc_inf) and np.isfinite(auc_last):
            tail = auc_inf - auc_last
            fraction = values.get("auc_extrap_fraction", np.nan)
            share = f" ({100.0 * fraction:.2g} %)" if np.isfinite(fraction) else ""
            unit = unit_of("auc_inf_obs")
            x_tail = tlast + thalf
            ax.annotate(
                f"AUC(tlast-inf) = {tail:.3g} {unit}{share}".rstrip(),
                xy=(x_tail, float(clast * np.exp(-lambda_z * thalf))),
                xytext=(14, 8),
                textcoords="offset points",
                fontsize=fontsize,
                color=style.extrapolation_color,
                ha="left",
                va="bottom",
            )
    t_first = values.get("lambda_z_t_first", np.nan)
    if np.isfinite(lambda_z) and np.isfinite(tlast) and np.isfinite(t_first):
        t_fit = np.linspace(t_first, t_end, 80)
        band = _terminal_band(t, c, values, t_fit, ci_level)
        if band is not None:
            ax.fill_between(
                t_fit,
                band[0],
                band[1],
                color=style.fit_color,
                alpha=style.band_alpha,
                linewidth=0,
                label=f"{100.0 * ci_level:g} % band of the regression",
            )
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
            markersize=style.markersize + 4,
            markerfacecolor="none",
            markeredgewidth=1.5,
            label="regression points",
        )
        if annotate:
            low, high = _thalf_interval(values, ci_level)
            half_life = estimate_text(thalf, low, high)
            n_points = values.get("lambda_z_n_points", np.nan)
            count = f", n = {int(n_points)}" if np.isfinite(n_points) else ""
            t_mid = 0.5 * (t_first + tlast)
            ax.annotate(
                f"lambda_z = {lambda_z:.3g} {unit_of('lambda_z') or f'1/{time_unit}'}"
                f"\nt1/2 = {half_life} {time_unit}{count}",
                xy=(t_mid, float(np.exp(intercept - lambda_z * t_mid))),
                xytext=(16, 14),
                textcoords="offset points",
                fontsize=fontsize,
                color=style.fit_color,
                ha="left",
                va="bottom",
            )
    cmax, tmax = values.get("cmax", np.nan), values.get("tmax", np.nan)
    if np.isfinite(cmax) and np.isfinite(tmax):
        ax.plot(
            [0, tmax],
            [cmax, cmax],
            linestyle="--",
            color=style.peak_color,
            linewidth=1,
        )
        ax.plot(
            [tmax, tmax],
            [0, cmax],
            linestyle="--",
            color=style.peak_color,
            linewidth=1,
        )
        ax.plot(
            [tmax],
            [cmax],
            marker="D",
            linestyle="none",
            color=style.peak_color,
            markersize=style.markersize + 2,
            label="Cmax at tmax",
        )
        if annotate:
            peak_text = (
                f"Cmax = {_quantity_text(values, units, 'cmax') or f'{cmax:.3g}'}"
                f"\ntmax = {tmax:.3g} {time_unit}"
            )
            if log_y:
                # the peak sits at the top of a logarithmic panel among the
                # first regression points: the text goes to the empty upper
                # right corner with a leader line to the marker
                ax.annotate(
                    peak_text,
                    xy=(tmax, cmax),
                    xytext=(0.3, 0.93),
                    textcoords="axes fraction",
                    fontsize=fontsize,
                    color=style.peak_color,
                    ha="left",
                    va="top",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": style.peak_color,
                        "linewidth": 0.8,
                        "alpha": 0.6,
                    },
                )
            else:
                # a bolus writes its C0 above the first point, so the peak
                # of a bolus (the same point, or the next) goes below it
                has_c0 = np.isfinite(values.get("c0", np.nan))
                ax.annotate(
                    peak_text,
                    xy=(tmax, cmax),
                    xytext=(12, -12) if has_c0 else (12, 4),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=style.peak_color,
                    ha="left",
                    va="top" if has_c0 else "bottom",
                )
    if annotate and np.isfinite(tlast) and np.isfinite(clast):
        ax.annotate(
            f"clast = {clast:.3g} {value_unit}\ntlast = {tlast:.3g} {time_unit}",
            xy=(tlast, clast),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=fontsize,
            color=style.data_color,
            ha="center",
            va="bottom",
        )
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
        if annotate:
            # C0 is the top of the panel: its text goes to the right of the
            # marker, the peak text (the next point) below it
            ax.annotate(
                f"C0 = {_quantity_text(values, units, 'c0') or f'{c0:.3g}'}",
                xy=(0.0, c0),
                xytext=(12, 0),
                textcoords="offset points",
                fontsize=fontsize,
                color=style.fit_color,
                ha="left",
                va="center",
            )
    error = None
    if spread == "sd" and tc.sd is not None:
        error = np.asarray(tc.sd, dtype=float)
    elif spread == "se" and tc.se is not None:
        error = np.asarray(tc.se, dtype=float)
    if error is not None and np.isfinite(error).any():
        ax.errorbar(
            t,
            c,
            yerr=np.where(np.isfinite(error), error, 0.0),
            fmt="none",
            ecolor=style.data_color,
            elinewidth=1,
            capsize=2,
            alpha=0.6,
            label=f"data ± {spread}",
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
    ax.set_xlabel(axis_label("time", tc.time_unit))
    ax.set_ylabel(axis_label(tc.substance, tc.unit))
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
    annotate: bool = True,
    parameters: Sequence[str] = PANEL_PARAMETERS,
    spread: Literal["sd", "se"] | None = "sd",
    ci_level: float = 0.95,
    axes: Sequence[Axes] | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Linear and logarithmic panel of one curve with its NCA diagnostics and its parameters.

    Both panels show the same curve (`draw_nca_panel`): the title of the
    figure names the sample and its flags once, the two panels name the scale
    they draw it on (`linear`, `semi-logarithmic`), and the legend is drawn
    once, on the linear panel. A third, narrow panel lists the `parameters`
    with their values, units and, where the result carries one, the interval
    of the uncertainty analysis (`x_ci_low`/`x_ci_high`; the half-life falls
    back to the interval of the regression), so that the figure reads as the
    report of the analysis. `axes` of the caller take two entries for the two
    curve panels alone, or three with the table.

    Args:
        timecourse: the curve
        result: the result of its analysis (a batch result with `indexers`, or a single result)

    Keyword Args:
        title: title of the figure, `name = value` per indexer (the label of
            the curve without indexers) by default; the flags are appended
        annotate: write the values of the peak, the last point and the
            regression on the plot and add the parameter table
        parameters: the parameters of the table, in this order (missing ones
            are skipped)
        spread: error bars of the data, `sd` or `se` of the curve when it
            carries them, `None` for none
        ci_level: level of the confidence band of the terminal regression and
            of the interval of the half-life derived from it
        axes: the two axes to draw the linear and the logarithmic panel into,
            or three with the parameter table, a new figure by default; a
            figure of the caller keeps its own title, so the heading goes on
            the first panel instead
        style: colors and markers
        **indexers: coordinate labels selecting the sample of a batch result

    Returns:
        The figure.
    """
    values, flags = _sample_values(result, indexers)
    units = {name: result.units(name) for name in values}
    own_figure = axes is None
    table_ax: Axes | None = None
    if own_figure and annotate:
        fig, grid = axes_of(
            None, nrows=1, ncols=3, figsize=(14.5, 4.5), width_ratios=(1, 1, 0.55)
        )
        ax1, ax2, table_ax = grid[0]
    elif own_figure or np.asarray(axes).size == 2:
        fig, grid = axes_of(axes, nrows=1, ncols=2, figsize=(11, 4.5))
        ax1, ax2 = grid[0]
    else:
        fig, grid = axes_of(axes, nrows=1, ncols=3, figsize=(14.5, 4.5))
        ax1, ax2, table_ax = grid[0]
    if title is None:
        title = (
            ", ".join(
                f"{name} = {format_value(value)}" for name, value in indexers.items()
            )
            if indexers
            else (timecourse.label or timecourse.substance)
        )
    heading = f"{title} [{', '.join(flags)}]" if flags else title
    # the panels carry the scale, the sample and its flags are the title of
    # the figure: the same heading over both panels says it twice. A figure
    # of the caller keeps its own title, so the heading goes on its first
    # panel instead
    if own_figure:
        fig.suptitle(heading, fontsize="medium")
    common: dict[str, Any] = {
        "annotate": annotate,
        "spread": spread,
        "ci_level": ci_level,
        "units": units,
        "style": style,
    }
    draw_nca_panel(
        timecourse,
        values,
        [],
        log_y=False,
        title="linear" if own_figure else heading,
        ax=ax1,
        **common,
    )
    draw_nca_panel(
        timecourse,
        values,
        [],
        log_y=True,
        title="semi-logarithmic",
        legend=False,
        ax=ax2,
        **common,
    )
    if table_ax is not None:
        _draw_parameter_table(
            table_ax,
            parameter_rows(values, units, parameters, ci_level),
            ci_level,
            style,
        )
    return fig


def plot_nca_grid(
    timecourses: Timecourses,
    result: NCAResult,
    *,
    ncols: int = 3,
    log_y: bool = True,
    annotate: bool = False,
    spread: Literal["sd", "se"] | None = "sd",
    ci_level: float = 0.95,
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
        annotate: the values and the parameter box in every panel, off by
            default since the panels of a grid are small; the confidence band
            of the regression is drawn either way
        spread: error bars of the data, `sd` or `se` when the batch carries
            them, `None` for none
        ci_level: level of the confidence band of the terminal regression
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
    parameter_units = {name: result.units(name) for name in result._variables}
    for k, (tc, index) in enumerate(zip(curves, indices, strict=True)):
        sample = result.ds.isel(
            dict(zip(timecourses.sample_dims, (int(i) for i in index), strict=True))
        )
        values = {name: float(sample[name].values) for name in result._variables}
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
            annotate=annotate,
            spread=spread,
            ci_level=ci_level,
            units=parameter_units,
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
    result of a single curve draws that curve's values. The title names the
    statistic of the markers and the dimension it was taken over (`mean ± sd
    over individual`), and stays empty where no reduction is drawn.

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
    # the statistic of the markers, named once for the figure and only where
    # there is one: a group of a single sample draws that sample's troughs
    reduced = [dim for dim in sample_dims if dim != by]
    if reduced and n_rows > len(groups):
        over = ", ".join(reduced)
        ax.set_title(
            f"mean over {over}" if spread is None else f"mean ± {statistic} over {over}"
        )
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

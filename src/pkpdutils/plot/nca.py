"""Diagnostic figures of the non-compartmental analysis."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sbmlsim_core.chart import Chart, ChartAxis, ChartPlot
from scipy.stats import t as student_t

from pkpdutils.nca.intervals import INTERVAL_DIM, INTERVAL_PREFIX
from pkpdutils.nca.options import NCAOptions, decode_flags
from pkpdutils.nca.result import NCAResult
from pkpdutils.nca.sparse import area_window
from pkpdutils.nca.urine import Excretion
from pkpdutils.plot._common import (
    FONT_SIZES,
    Frame,
    axis_label,
    chart_of,
    data_limits,
    dose_markers,
    estimate_text,
    format_value,
    group_colors,
    group_order,
    has_infusion,
    log_axis,
    marker,
    sample_colors,
    sample_labels,
    sample_title,
    unit_label,
    value_axis,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Dosing, Route, Timecourse, Timecourses

#: the fonts of the charts of the module: plot titles in matplotlib's
#: `small`, the title of a chart in its `medium`
_FONTS: dict[str, float | str] = {
    "title": FONT_SIZES["medium"],
    "plot_title": FONT_SIZES["small"],
}

#: the box behind an annotation inside a shaded area, matplotlib's
#: `bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.7}`
_TEXT_BOX: dict[str, Any] = {"fill": "white", "opacity": 0.7, "style": "round"}

LAMBDA_Z_LABEL = "lambda_z = "

#: the label the regression line gets in a legend shared by several panels,
#: which cannot name one curve's `lambda_z`
TERMINAL_LABEL = "terminal regression"

#: share the top of an axis is extended by so that the legend of
#: `plot_excretion` does not sit on the plateau of the recovered amount
LEGEND_HEADROOM = 1.6

#: the variables `plot_sparse` writes into its panel, which say whether a
#: result comes from `pkpdutils.nca.sparse.nca_sparse`
SPARSE_PANEL_VARIABLES: tuple[str, ...] = ("auc_last", "auc_last_se", "auc_last_df")

#: share of the visible x range at either end within which an annotation is
#: anchored at its own end rather than centred, so that it does not run over
#: the spine of the panel
EDGE_MARGIN = 0.15

#: the variables `plot_terminal_windows` reads, which a result carries when the
#: analysis kept the candidate windows (`TerminalPhase.keep_candidates`)
CANDIDATE_VARIABLES: tuple[str, ...] = (
    "candidate_t_first",
    "candidate_r2_adj",
    "candidate_n_points",
)


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
    extrapolated tail is (Draper & Smith 1998, ch. 1).

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


#: an axis which is not drawn, of a panel used for text alone
_HIDDEN = ChartAxis(
    label_visible=False, ticks_visible=False, line_visible=False, min=0.0, max=1.0
)

#: the spacing of the lines of the parameter table, in font sizes
_TABLE_LINE_SPACING = 1.6


def _draw_parameter_table(
    chart: Chart,
    row: int,
    column: int,
    rows: Sequence[tuple[str, str, str, str]],
    ci_level: float,
    style: PlotStyle,
    *,
    column_span: int = 1,
) -> ChartPlot:
    """Write the parameter table into a panel of its own.

    Args:
        chart: the chart.
        row: the row of the panel.
        column: the column of the panel.
        rows: the rows of `parameter_rows`.
        ci_level: level of the intervals, named in the heading when a row
            carries one.
        style: the font size of the annotations.

    Keyword Args:
        column_span: the columns the panel spans.

    Returns:
        The panel, without axes.
    """
    heading = None
    if rows:
        heading = "parameters"
        if any(r[2] for r in rows):
            heading = f"parameters, {100.0 * ci_level:g} % interval"
    plot = chart.plot(
        row,
        column,
        column_span=column_span,
        title=heading,
        x=_HIDDEN,
        y=_HIDDEN,
        legend=False,
    )
    if not rows:
        return plot
    widths = [max(len(r[k]) for r in rows) for k in range(3)]
    lines = [
        f"{name:<{widths[0]}}  {value:>{widths[1]}}  {interval:<{widths[2]}}  {unit}".rstrip()
        for name, value, interval, unit in rows
    ]
    # one text per line, so that the lines keep matplotlib's spacing of 1.6
    # font sizes, which a text of several lines does not take
    size = style.annotation_fontsize
    step = _TABLE_LINE_SPACING * size / max(Frame.of(plot).height, 1.0)
    for k, line in enumerate(lines):
        plot.text(
            0.0,
            1.0 - k * step,
            line,
            size=size,
            family="monospace",
            baseline="top",
            coordinates="axes",
            clip=False,
            color="black",
        )
    return plot


def partial_window(
    result: NCAResult, timecourse: Timecourse, partial: str
) -> tuple[float, float]:
    """The interval of a named partial area in the times of an NCA panel.

    The intervals of `NCAOptions.partial_aucs` are relative to the first dose
    of the protocol, while a panel starts at the dose it analyses (the last one
    of a multiple dose curve): the interval is shifted by the time between the
    two doses, so that a named area of a multiple dose curve lands where the
    analysis computed it.

    Args:
        result: the result of the analysis, which carries the intervals
            (`NCAResult.partial_aucs`)
        timecourse: the curve the panel draws, for its dosing protocol
        partial: name of the area

    Returns:
        The `(t_start, t_end)` of the area, relative to the analysed dose.

    Raises:
        ValueError: if the result carries no area of that name.
    """
    windows = result.partial_aucs
    if partial not in windows:
        known = ", ".join(sorted(windows)) or "none"
        raise ValueError(
            f"the result carries no partial area '{partial}' (it has: {known}); "
            "run the analysis with NCAOptions(partial_aucs={...})"
        )
    start, end = windows[partial]
    dosing = timecourse.dosing
    offset = 0.0 if dosing is None else float(dosing.last.time - dosing.first.time)
    return start - offset, end - offset


def _partial_area(
    t: np.ndarray,
    c: np.ndarray,
    values: Mapping[str, float],
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""The outline of a named partial area of the curve.

    The area lies between the two times of the interval: the values at the
    bounds are interpolated linearly, as the shading of the other areas
    follows the data. An interval which reaches beyond the last measurable
    value is closed with the terminal regression,
    \(\hat C_\mathrm{last} e^{-\lambda_z (t - t_\mathrm{last})}\), which is
    what the analysis integrates there
    (`pkpdutils.nca.nca.named_partial_aucs`).

    Args:
        t: the times of the curve, relative to the analysed dose
        c: the values of the curve
        values: the parameter magnitudes of the curve, for the terminal
            regression of its extrapolated part
        window: the bounds of the interval, in the times of the panel

    Returns:
        The times and the values of the upper edge of the area, `None` when
        the curve or the interval has no finite value.
    """
    ok = np.isfinite(c) & np.isfinite(t)
    start, end = window
    if not ok.any() or not np.isfinite([start, end]).all():
        return None
    inside = ok & (t >= start) & (t <= end)
    tlast, clast = values.get("tlast", np.nan), values.get("clast", np.nan)
    clast_pred = values.get("clast_pred", clast)
    lambda_z = values.get("lambda_z", np.nan)
    extrapolated = np.isfinite([tlast, clast_pred, lambda_z]).all() and end > tlast
    tail = np.linspace(max(start, tlast), end, 40) if extrapolated else np.empty(0)
    x = np.concatenate([[start], t[inside], tail, [end]])
    x.sort()
    y = np.interp(x, t[ok], c[ok])
    if extrapolated:
        beyond = x > tlast
        y[beyond] = clast_pred * np.exp(-lambda_z * (x[beyond] - tlast))
    return x, y


@dataclass
class _Annotation:
    """A text of a panel, placed some points from a point of the data.

    Attributes:
        text: the text
        xy: the point of the data it explains
        offset: the offset of the text from the point in points, right and up
        color: the color of the text
        anchor: the horizontal anchor, `start`, `middle` or `end`
        baseline: the vertical anchor, `bottom`, `center` or `top`
        box: whether a white box is drawn behind the text
        axes: the position of the text in fractions of the panel instead of
            an offset from the point, with a line from the text to the point
            (matplotlib's annotation with `arrowprops`)
        inside: the share of the height of the point the text is written at,
            inside a shaded area under the point; on a logarithmic panel the
            text goes to the middle of the decades between the bottom of the
            panel and the point
    """

    text: str
    xy: tuple[float, float]
    offset: tuple[float, float] = (0.0, 0.0)
    color: str = "black"
    anchor: Literal["start", "middle", "end"] = "start"
    baseline: Literal["bottom", "center", "top"] = "bottom"
    box: bool = False
    axes: tuple[float, float] | None = None
    inside: float | None = None


@dataclass
class _Panel:
    """The elements of an NCA panel before they are drawn, which its axes are scaled to.

    Attributes:
        x: the x values of the elements, which the x axis holds
        y: the y values of the elements, which the y axis holds
        annotations: the texts, drawn last
    """

    x: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)
    annotations: list[_Annotation] = field(default_factory=list)

    def hold(self, x: Any, y: Any) -> None:
        """Add the values of an element to the axes."""
        self.x.append(np.asarray(x, dtype=float).ravel())
        self.y.append(np.asarray(y, dtype=float).ravel())


def _write(plot: ChartPlot, annotations: Sequence[_Annotation], size: float) -> None:
    """Write the annotations of a panel, their offsets converted with its frame.

    Args:
        plot: the panel, with every other element drawn
        annotations: the texts
        size: their font size
    """
    if not annotations:
        return
    frame = Frame.of(plot)
    for a in annotations:
        if a.axes is not None:
            x, y = frame.at(*a.axes)
            plot.arrow(
                a.xy[0],
                a.xy[1],
                x,
                y,
                head="none",
                width=0.8,
                opacity=0.6,
                color=a.color,
                clip=False,
            )
        elif a.inside is not None:
            x, top = a.xy
            y = (
                float(np.sqrt(max(top, 1e-300) * frame.y[0]))
                if frame.log_y
                else a.inside * top
            )
        else:
            x, y = frame.offset(a.xy[0], a.xy[1], *a.offset)
        plot.text(
            x,
            y,
            a.text,
            size=size,
            color=a.color,
            anchor=a.anchor,
            baseline=a.baseline,
            box={**_TEXT_BOX, "padding": 0.2 * size} if a.box else None,
            clip=False,
        )


def _value_at_dose(values: Mapping[str, float], route: Route) -> float:
    """The value the analysis inserts at the dose of a single dose curve.

    The back-extrapolated `c0` after a bolus, 0 after an infusion and after an
    extravascular dose (`pkpdutils.nca.nca.compute_parameters`).

    Args:
        values: the parameters of the sample
        route: route of the dose

    Returns:
        The value, `NaN` when the analysis estimated none.
    """
    if route is Route.IV_BOLUS:
        return float(values.get("c0", np.nan))
    return 0.0


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
    partial: str | None = None,
    partial_range: tuple[float, float] | None = None,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> ChartPlot:
    r"""Draw the NCA diagnostics of one curve into one panel.

    The panel shows the data with their spread, the area to \(t_\mathrm{last}\)
    (`AUC(0-tlast)`) or over the last dosing interval (`AUC(0-tau)` of a
    multiple dose result), the extrapolated tail, the terminal regression line
    with the points it used and its confidence band (`_terminal_band`), the
    peak \(C_\mathrm{max}\)/\(t_\mathrm{max}\) with its guide lines, \(C_0\)
    of a bolus, the named partial area `partial` over the area to
    \(t_\mathrm{last}\), and the dose it analyses (an infusion as its window). With
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
        partial: name of a named partial area of the result
            (`NCAOptions.partial_aucs`, a variable of `values`), shaded over
            the area to the last measurable point in `style.partial_color`
            with its value written into it; `None` for none
        partial_range: the bounds `(t_start, t_end)` of that area in the times
            of the panel, i.e. relative to the analysed dose. The intervals of
            `NCAOptions.partial_aucs` are relative to the first dose of the
            protocol, so a multiple dose curve needs them shifted by the time
            between the first and the analysed dose; `plot_nca` and
            `plot_nca_grid` read the interval from the result
            (`NCAResult.partial_aucs`) and shift it. An interval which starts
            before the first sample of the curve is shaded flat from that
            sample on, while the analysis may add the segment from the dose to
            it (`pkpdutils.nca.nca.area_between`), so the shading of such an
            area can cover a little less than the number it carries
        chart: the chart to draw into, a new one by default
        row: the row of the panel in `chart`
        column: the column of the panel in `chart`
        style: colors and markers

    Returns:
        The panel; its chart is `panel.chart`.

    Raises:
        ValueError: if `partial` is given without `partial_range`, or if the
            cell is not in the grid of `chart`.
    """
    target = chart_of(
        chart, row, column, rows=1, columns=1, size=(6.0, 4.5), fonts=_FONTS
    )
    return _nca_panel(
        target,
        row,
        column,
        timecourse,
        values,
        flags,
        log_y=log_y,
        title=title,
        legend=legend,
        annotate=annotate,
        spread=spread,
        ci_level=ci_level,
        units=units,
        partial=partial,
        partial_range=partial_range,
        style=style,
    )


def _nca_panel(
    chart: Chart,
    row: int,
    column: int,
    timecourse: Timecourse,
    values: Mapping[str, float],
    flags: list[str],
    *,
    log_y: bool,
    title: str | None,
    legend: bool,
    annotate: bool,
    spread: Literal["sd", "se"] | None,
    ci_level: float,
    units: Mapping[str, str] | None,
    partial: str | None,
    partial_range: tuple[float, float] | None,
    style: PlotStyle,
    column_span: int = 1,
    shared: bool = False,
    windows: Sequence[float] = (),
) -> ChartPlot:
    """The panel of `draw_nca_panel` in a cell of a chart.

    Args:
        chart: the chart.
        row: the row of the panel.
        column: the column of the panel.
        timecourse: the curve.
        values: the parameter magnitudes of the curve.
        flags: the flag names of the curve.

    Keyword Args:
        log_y: logarithmic value axis.
        title: the title.
        legend: draw the legend of the panel.
        annotate: write the values on the plot.
        spread: the error bars of the data.
        ci_level: level of the band and of the interval of the half-life.
        units: the unit per parameter.
        partial: name of a named partial area.
        partial_range: the bounds of that area in the times of the panel.
        style: colors and markers.
        column_span: the columns the panel spans.
        shared: whether the legend is shared by several panels, which labels
            the regression line `TERMINAL_LABEL` instead of its `lambda_z`.
        windows: the bounds of the chosen window of the terminal regression,
            drawn as dashed lines (`plot_terminal_windows`).

    Returns:
        The panel.
    """
    if partial is not None and partial_range is None:
        raise ValueError(
            f"the partial area '{partial}' needs 'partial_range', its bounds "
            "in the times of the panel"
        )
    tc = timecourse.relative_to_dose(which="last")
    t = np.asarray(tc.time, dtype=float)
    c = np.asarray(tc.value, dtype=float)
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
    panel = _Panel()
    panel.hold(t, c)
    notes = panel.annotations

    error = None
    if spread == "sd" and tc.sd is not None:
        error = np.asarray(tc.sd, dtype=float)
    elif spread == "se" and tc.se is not None:
        error = np.asarray(tc.se, dtype=float)
    if error is not None and not np.isfinite(error).any():
        error = None
    if error is not None:
        error = np.where(np.isfinite(error), error, 0.0)
        panel.hold(t, c - error)
        panel.hold(t, c + error)

    steady_state = np.isfinite(values.get("auc_tau", np.nan))
    tau = values.get("tau", np.nan)
    auc_bound = tau if steady_state and np.isfinite(tau) else tlast
    auc_label = "AUC(0-tau)" if steady_state else "AUC(0-tlast)"
    area: tuple[np.ndarray, np.ndarray] | None = None
    to_dose: tuple[list[float], list[float]] | None = None
    if np.isfinite(auc_bound):
        inside = ok & (t <= auc_bound)
        x_area, c_area = t[inside], c[inside]
        at_dose = (
            _value_at_dose(values, tc.dose.route)
            if not steady_state and tc.dose is not None and len(x_area)
            else np.nan
        )
        if np.isfinite(at_dose) and x_area[0] > 0.0:
            # the area starts at the dose with the value the analysis inserted
            # there, which the data does not show
            to_dose = ([0.0, float(x_area[0])], [float(at_dose), float(c_area[0])])
            x_area = np.concatenate([[0.0], x_area])
            c_area = np.concatenate([[at_dose], c_area])
        area = (x_area, c_area)
        panel.hold(x_area, c_area)
        panel.hold(x_area, np.zeros_like(x_area))
        if annotate:
            auc_name = "auc_tau" if steady_state else "auc_last"
            auc_text = _quantity_text(values, units, auc_name)
            if auc_text and len(x_area):
                # inside the area, at a third of its width and height
                x_text = 0.55 * auc_bound
                top = float(np.interp(x_text, x_area, c_area))
                notes.append(
                    _Annotation(
                        f"{auc_label} = {auc_text}",
                        (x_text, top),
                        color=style.data_color,
                        anchor="middle",
                        baseline="center",
                        box=True,
                        inside=0.35,
                    )
                )
    t_end = tlast + 3.0 * thalf if np.isfinite(thalf) else np.nan
    extrapolated: tuple[np.ndarray, np.ndarray] | None = None
    if np.isfinite(lambda_z) and np.isfinite(tlast) and np.isfinite(t_end):
        t_ext = np.linspace(tlast, t_end, 50)
        c_ext = clast * np.exp(-lambda_z * (t_ext - tlast))
        extrapolated = (t_ext, c_ext)
        panel.hold(t_ext, c_ext)
        auc_inf = values.get("auc_inf_obs", np.nan)
        auc_last = values.get("auc_last", np.nan)
        if annotate and np.isfinite(auc_inf) and np.isfinite(auc_last):
            tail = auc_inf - auc_last
            fraction = values.get("auc_extrap_fraction", np.nan)
            share = f" ({100.0 * fraction:.2g} %)" if np.isfinite(fraction) else ""
            notes.append(
                _Annotation(
                    f"AUC(tlast-inf) = {tail:.3g} {unit_of('auc_inf_obs')}{share}".rstrip(),
                    (tlast + thalf, float(clast * np.exp(-lambda_z * thalf))),
                    offset=(14.0, 8.0),
                    color=style.extrapolation_color,
                )
            )
    t_first = values.get("lambda_z_t_first", np.nan)
    regression: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    band: tuple[np.ndarray, np.ndarray] | None = None
    line_label = TERMINAL_LABEL if shared else f"{LAMBDA_Z_LABEL}{lambda_z:.3g}"
    band_label = f"{100.0 * ci_level:g} % band of the regression"
    if np.isfinite(lambda_z) and np.isfinite(tlast) and np.isfinite(t_first):
        t_fit = np.linspace(t_first, t_end, 80)
        band = _terminal_band(t, c, values, t_fit, ci_level)
        if band is not None:
            panel.hold(t_fit, band[0])
            panel.hold(t_fit, band[1])
        used = ok & (t >= t_first) & (t <= tlast) & (c > 0)
        regression = (t_fit, np.exp(intercept - lambda_z * t_fit), used)
        panel.hold(t_fit, regression[1])
        if annotate:
            low, high = _thalf_interval(values, ci_level)
            half_life = estimate_text(thalf, low, high)
            n_points = values.get("lambda_z_n_points", np.nan)
            count = f", n = {int(n_points)}" if np.isfinite(n_points) else ""
            t_mid = 0.5 * (t_first + tlast)
            notes.append(
                _Annotation(
                    f"lambda_z = {lambda_z:.3g} {unit_of('lambda_z') or f'1/{time_unit}'}"
                    f"\nt1/2 = {half_life} {time_unit}{count}",
                    (t_mid, float(np.exp(intercept - lambda_z * t_mid))),
                    offset=(16.0, 14.0),
                    color=style.fit_color,
                )
            )
    cmax, tmax = values.get("cmax", np.nan), values.get("tmax", np.nan)
    peak = bool(np.isfinite(cmax) and np.isfinite(tmax))
    if peak:
        panel.hold([0.0, tmax], [cmax, cmax])
        if annotate:
            peak_text = (
                f"Cmax = {_quantity_text(values, units, 'cmax') or f'{cmax:.3g}'}"
                f"\ntmax = {tmax:.3g} {time_unit}"
            )
            if log_y:
                # the peak sits at the top of a logarithmic panel among the
                # first regression points: the text goes to the empty upper
                # right corner with a leader line to the marker
                notes.append(
                    _Annotation(
                        peak_text,
                        (tmax, cmax),
                        color=style.peak_color,
                        baseline="top",
                        axes=(0.3, 0.93),
                    )
                )
            else:
                # a bolus writes its C0 above the first point, so the peak
                # of a bolus (the same point, or the next) goes below it
                has_c0 = np.isfinite(values.get("c0", np.nan))
                notes.append(
                    _Annotation(
                        peak_text,
                        (tmax, cmax),
                        offset=(12.0, -12.0) if has_c0 else (12.0, 4.0),
                        color=style.peak_color,
                        baseline="top" if has_c0 else "bottom",
                    )
                )
    if annotate and np.isfinite(tlast) and np.isfinite(clast):
        notes.append(
            _Annotation(
                f"clast = {clast:.3g} {value_unit}\ntlast = {tlast:.3g} {time_unit}",
                (tlast, clast),
                offset=(0.0, 12.0),
                color=style.data_color,
                anchor="middle",
            )
        )
    c0 = values.get("c0", np.nan)
    bolus = bool(
        np.isfinite(c0) and tc.dose is not None and tc.dose.route is Route.IV_BOLUS
    )
    if bolus:
        panel.hold([0.0], [c0])
        if annotate:
            # C0 is the top of the panel: its text goes to the right of the
            # marker, the peak text (the next point) below it
            notes.append(
                _Annotation(
                    f"C0 = {_quantity_text(values, units, 'c0') or f'{c0:.3g}'}",
                    (0.0, c0),
                    offset=(12.0, 0.0),
                    color=style.fit_color,
                    baseline="center",
                )
            )
    dosing = None if tc.dosing is None else Dosing.single(tc.dosing.last)
    infusion_label = _infusion_label(tc)
    if infusion_label is not None:
        assert dosing is not None and dosing.durations is not None
        start = float(dosing.times[0])
        duration = float(dosing.durations[0])
        panel.hold([start, start + duration], [np.nan, np.nan])
    shaded: tuple[np.ndarray, np.ndarray] | None = None
    if partial is not None and partial_range is not None:
        shaded = _partial_area(t, c, values, partial_range)
        if shaded is not None:
            panel.hold(*shaded)
    panel.hold(np.asarray(windows, dtype=float), np.full(len(windows), np.nan))

    log = log_axis(log_y, "y", *panel.y)
    x_limits = data_limits(*panel.x, low=0.0)
    y_limits = data_limits(*panel.y, log=log, low=None if log else 0.0)
    bottom = y_limits[0] if log else 0.0
    heading = title if title is not None else (tc.label or tc.substance)
    if flags:
        heading = f"{heading} [{', '.join(flags)}]"
    order = [
        auc_label,
        "extrapolated",
        band_label,
        line_label,
        "regression points",
        "Cmax at tmax",
        "C0",
        "data",
        *([infusion_label] if infusion_label else []),
        *([partial] if partial else []),
        *(["chosen window"] if windows else []),
        f"data ± {spread}",
    ]
    plot = chart.plot(
        row,
        column,
        column_span=column_span,
        # a panel with several flags has a long title, which the font size of
        # the plot titles (matplotlib's `small`) keeps within its width
        title=heading,
        x=value_axis(axis_label("time", tc.time_unit), limits=x_limits),
        y=value_axis(axis_label(tc.substance, tc.unit), log=log, limits=y_limits),
        legend=legend,
        legend_order=order,
    )
    dose_markers(plot, dosing, y_limits=y_limits, style=style, time_unit=tc.time_unit)
    if area is not None:
        x_area, c_area = area
        plot.area(
            x_area,
            np.full_like(x_area, bottom),
            c_area,
            color=style.auc_color,
            fill_opacity=style.alpha + 0.1,
            label=auc_label,
            legend=auc_label,
        )
    if extrapolated is not None:
        t_ext, c_ext = extrapolated
        plot.area(
            t_ext,
            np.full_like(t_ext, bottom),
            c_ext,
            color=style.extrapolation_color,
            fill_opacity=style.alpha + 0.1,
            label="extrapolated",
            legend="extrapolated",
        )
    if regression is not None and band is not None:
        plot.band(
            regression[0],
            band[0],
            band[1],
            line="none",
            color=style.fit_color,
            fill_opacity=style.band_alpha,
            label=band_label,
            legend=band_label,
        )
    if shaded is not None:
        assert partial is not None
        x_shaded, y_shaded = shaded
        plot.area(
            x_shaded,
            np.full_like(x_shaded, bottom),
            y_shaded,
            color=style.partial_color,
            fill_opacity=style.alpha + 0.2,
            label=partial,
            legend=partial,
        )
    if to_dose is not None:
        plot.curve(*to_dose, color=style.data_color, line="dotted", width=1.0)
    if regression is not None:
        t_fit, c_fit, used = regression
        plot.curve(
            t_fit,
            c_fit,
            color=style.fit_color,
            width=style.linewidth,
            label=line_label,
            legend=line_label,
        )
        plot.curve(
            t[used],
            c[used],
            line="none",
            marker=marker(style.terminal_marker),
            size=style.markersize + 4,
            marker_fill="none",
            marker_edge_width=1.5,
            color=style.fit_color,
            label="regression points",
            legend="regression points",
        )
    for bound in windows:
        # the bounds of the window as two dashed lines rather than a shaded
        # span: the panel already shades the areas, and a second fill over
        # them would be read as a third area
        plot.rule(
            bound,
            orientation="vertical",
            line="dashed",
            width=1.0,
            color=style.fit_color,
            label="chosen window",
            legend="chosen window",
        )
    if peak:
        plot.curve(
            [0.0, tmax], [cmax, cmax], line="dashed", width=1.0, color=style.peak_color
        )
        plot.curve(
            [tmax, tmax],
            [bottom, cmax],
            line="dashed",
            width=1.0,
            color=style.peak_color,
        )
        plot.curve(
            [tmax],
            [cmax],
            line="none",
            marker="diamond",
            size=style.markersize + 2,
            color=style.peak_color,
            label="Cmax at tmax",
            legend="Cmax at tmax",
        )
    if bolus:
        plot.curve(
            [0.0],
            [c0],
            line="none",
            marker="diamond",
            size=style.markersize + 1,
            color=style.fit_color,
            label="C0",
            legend="C0",
        )
    if error is not None:
        plot.curve(
            t,
            c,
            yerr=error,
            error_cap=2.0,
            error_width=1.0,
            line="none",
            color=style.data_color,
            opacity=0.6,
            label=f"data ± {spread}",
            legend=f"data ± {spread}",
        )
    plot.curve(
        t,
        c,
        marker=marker(style.data_marker),
        size=style.markersize,
        width=style.linewidth,
        color=style.data_color,
        label="data",
        legend="data",
    )
    if shaded is not None and annotate:
        assert partial is not None
        value = values.get(partial, np.nan)
        x_shaded, y_shaded = shaded
        if np.isfinite(value):
            # the text goes where the shaded area is tallest and therefore
            # has room, which for an area covering the peak is the peak and
            # for a late interval its beginning
            tallest = int(np.nanargmax(y_shaded))
            x_text, top = float(x_shaded[tallest]), float(y_shaded[tallest])
            # a text centred close to an edge of the panel runs over its
            # spine, so there it is anchored at its own left or right end
            left, right = x_limits
            margin = EDGE_MARGIN * (right - left)
            anchor: Literal["start", "middle", "end"] = (
                "start"
                if x_text < left + margin
                else "end"
                if x_text > right - margin
                else "middle"
            )
            notes.append(
                _Annotation(
                    f"{partial} = {value:.3g} {unit_of(partial)}".rstrip(),
                    (x_text, top),
                    color=style.partial_color,
                    anchor=anchor,
                    baseline="center",
                    box=True,
                    inside=0.5,
                )
            )
    _write(plot, notes, style.annotation_fontsize)
    return plot


def plot_nca(
    timecourse: Timecourse,
    result: NCAResult,
    *,
    title: str | None = None,
    annotate: bool = True,
    parameters: Sequence[str] = PANEL_PARAMETERS,
    spread: Literal["sd", "se"] | None = "sd",
    ci_level: float = 0.95,
    partial: str | None = None,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Chart:
    r"""Linear and logarithmic panel of one curve with its NCA diagnostics and its parameters.

    Both panels show the same curve (`draw_nca_panel`): the title of the
    figure names the sample and its flags once, the two panels name the scale
    they draw it on (`linear`, `semi-logarithmic`), and the legend is drawn
    once, on the linear panel. With `annotate` a third, narrow panel lists the
    `parameters` with their values, units and, where the result carries one,
    the interval of the uncertainty analysis (`x_ci_low`/`x_ci_high`; the
    half-life falls back to the interval of the regression), so that the
    figure reads as the report of the analysis.

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
        partial: name of a named partial area of the result
            (`NCAOptions.partial_aucs`) to shade over the area to
            \(t_\mathrm{last}\), its interval read from
            `NCAResult.partial_aucs`; `None` for none
        chart: the chart to draw into, a new one by default; the linear and
            the logarithmic panel go into two cells of its grid from
            `(row, column)` on, the parameter table into the third. A chart
            of the caller keeps its own title, so the heading goes on the
            first panel instead
        row: the row of the panels in `chart`
        column: the column of the first panel in `chart`
        style: colors and markers
        **indexers: coordinate labels selecting the sample of a batch result

    Returns:
        The chart.

    Raises:
        ValueError: if the panels do not fit into the grid of `chart`.
    """
    values, flags = _sample_values(result, indexers)
    units = {name: result.units(name) for name in values}
    own = chart is None
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
    # the figure: the same heading over both panels says it twice. A chart of
    # the caller keeps its own title, so the heading goes on its first panel
    # instead. A chart of its own gives the table three quarters of the width
    # of a curve panel, which the widest table of the monospace font fills
    # (matplotlib's `width_ratios=(1, 1, 0.55)` let the table run into the
    # margin of the figure)
    span, table_span = (4, 3) if own and annotate else (1, 1)
    target = chart_of(
        chart,
        row,
        column,
        rows=1,
        columns=2 * span + (table_span if annotate else 0),
        size=(14.5, 4.5) if annotate else (11.0, 4.5),
        title=heading,
        fonts=_FONTS,
    )
    common: dict[str, Any] = {
        "annotate": annotate,
        "spread": spread,
        "ci_level": ci_level,
        "units": units,
        "partial": partial,
        "partial_range": (
            None if partial is None else partial_window(result, timecourse, partial)
        ),
        "style": style,
        "column_span": span,
    }
    _nca_panel(
        target,
        row,
        column,
        timecourse,
        values,
        [],
        log_y=False,
        title="linear" if own else heading,
        legend=True,
        **common,
    )
    _nca_panel(
        target,
        row,
        column + span,
        timecourse,
        values,
        [],
        log_y=True,
        title="semi-logarithmic",
        legend=False,
        **common,
    )
    if annotate:
        _draw_parameter_table(
            target,
            row,
            column + 2 * span,
            parameter_rows(values, units, parameters, ci_level),
            ci_level,
            style,
            column_span=table_span,
        )
    return target


def plot_terminal_windows(
    timecourse: Timecourse,
    result: NCAResult,
    *,
    options: NCAOptions | None = None,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Chart:
    r"""The candidate windows of the terminal regression next to the chosen one.

    The selection of the terminal phase is the judgement call a reviewer
    questions, and every interactive tool shows it (the Slopes Selector of
    Phoenix WinNonlin, the "Check lambda_z" tab of PKanalix). The figure has
    two panels: the curve on a logarithmic value axis with the regression line,
    the points it used and the chosen window between two dashed lines
    (`draw_nca_panel`), and the adjusted \(R^2\) of every candidate window
    against the time its first point was taken at, the chosen window marked and
    the number of points of every window written above its marker. A window
    starting later has fewer points, so the second panel reads from left
    (many points) to right (three): where the curve is flat over several
    windows the choice hardly matters, where it drops the terminal phase is
    where the last points sit.

    The result must carry the candidate windows, which the analysis keeps only
    when it is asked for them and only for a single curve:

    ```python
    # not executed
    options = NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    result = nca_single(tc, options=options)
    chart = plot_terminal_windows(tc, result, options=options)
    ```

    Args:
        timecourse: the curve
        result: the result of its analysis, carrying `candidate_t_first`,
            `candidate_r2_adj` and `candidate_n_points`
            (`TerminalPhase.keep_candidates`)

    Keyword Args:
        options: the options of the analysis; its
            `NCAOptions.acceptance.r2_adj_min` is drawn as the acceptance
            threshold of the second panel when it is set
        chart: the chart to draw into, a new one by default; the two panels
            go into two cells of its grid from `(row, column)` on
        row: the row of the panels in `chart`
        column: the column of the first panel in `chart`
        style: colors and markers
        **indexers: coordinate labels selecting the sample of a batch result

    Returns:
        The chart.

    Raises:
        ValueError: if the result carries no candidate windows; a result of
            several samples carries none, since the analysis keeps them for a
            single curve only.
    """
    values, flags = _sample_values(result, indexers)
    ds = result.ds.sel(**indexers) if indexers else result.ds
    missing = [name for name in CANDIDATE_VARIABLES if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"the result carries no candidate windows ({', '.join(missing)} "
            "missing); run the analysis of the single curve with "
            "NCAOptions(terminal=TerminalPhase(keep_candidates=True))"
        )
    starts = ds["candidate_t_first"].to_numpy().astype(float)
    r2_adj = ds["candidate_r2_adj"].to_numpy().astype(float)
    n_points = ds["candidate_n_points"].to_numpy().astype(float)
    chosen = values.get("lambda_z_t_first", np.nan)
    t_last = values.get("lambda_z_t_last", np.nan)
    target = chart_of(
        chart, row, column, rows=1, columns=2, size=(11.0, 4.5), fonts=_FONTS
    )
    _nca_panel(
        target,
        row,
        column,
        timecourse,
        values,
        flags,
        log_y=True,
        title="the curve with the chosen window",
        legend=True,
        annotate=False,
        spread="sd",
        ci_level=0.95,
        units={name: result.units(name) for name in values},
        partial=None,
        partial_range=None,
        style=style,
        windows=(
            (chosen, t_last) if np.isfinite(chosen) and np.isfinite(t_last) else ()
        ),
    )
    time_unit = unit_label(timecourse.time_unit) or timecourse.time_unit
    threshold = None if options is None else options.acceptance.r2_adj_min
    plot = target.plot(
        row,
        column + 1,
        title="candidate windows, points per window above the marker",
        x=ChartAxis(label=axis_label("first point of the window", time_unit)),
        y=ChartAxis(label="adjusted R²"),
    )
    plot.curve(
        starts,
        r2_adj,
        marker=marker(style.data_marker),
        size=style.markersize,
        line="dotted",
        width=style.linewidth,
        color=style.data_color,
        label="candidate window",
        legend="candidate window",
    )
    picked = np.isclose(starts, chosen) if np.isfinite(chosen) else np.zeros(0, bool)
    if picked.any():
        plot.curve(
            starts[picked],
            r2_adj[picked],
            line="none",
            marker=marker(style.terminal_marker),
            size=style.markersize + 4,
            marker_fill="none",
            marker_edge_width=1.5,
            color=style.fit_color,
            label="chosen window",
            legend="chosen window",
        )
    if threshold is not None:
        label = f"acceptance, adjusted R² ≥ {threshold:g}"
        plot.rule(
            threshold,
            line="dashed",
            width=1.0,
            color=style.limit_color,
            label=label,
            legend=label,
        )
    _write(
        plot,
        [
            _Annotation(
                f"{int(count)}",
                (start, value),
                offset=(0.0, 7.0),
                color=style.data_color,
                anchor="middle",
            )
            for start, value, count in zip(starts, r2_adj, n_points, strict=True)
            if np.isfinite(value)
        ],
        style.annotation_fontsize,
    )
    return target


def _infusion_label(tc: Timecourse) -> str | None:
    """The legend label of the infusion window of an NCA panel, `infusion (1 hr)`.

    Args:
        tc: the curve, relative to the dose the panel analyses.

    Returns:
        The label, `None` when the analysed dose is no infusion.
    """
    dosing = None if tc.dosing is None else Dosing.single(tc.dosing.last)
    if not has_infusion(dosing):
        return None
    assert dosing is not None and dosing.durations is not None
    return f"infusion ({float(dosing.durations[0]):g} {tc.time_unit})"


def _grid_legend_groups(
    curves: Sequence[Timecourse],
    ci_level: float,
    partial: str | None,
    spread: Literal["sd", "se"] | None,
) -> list[str]:
    """The legend groups an NCA panel of a grid may draw, in the order of its legend.

    Args:
        curves: the curves of the panels.
        ci_level: the level of the band of the regression.
        partial: the name of the partial area, `None` for none.
        spread: the error bars of the data.

    Returns:
        The groups, every infusion of the curves once.
    """
    infusions = dict.fromkeys(
        label
        for label in (
            _infusion_label(tc.relative_to_dose(which="last")) for tc in curves
        )
        if label is not None
    )
    return [
        "AUC(0-tlast)",
        "AUC(0-tau)",
        "extrapolated",
        f"{100.0 * ci_level:g} % band of the regression",
        TERMINAL_LABEL,
        "regression points",
        "Cmax at tmax",
        "C0",
        "data",
        *infusions,
        *([partial] if partial else []),
        f"data ± {spread}",
    ]


def plot_nca_grid(
    timecourses: Timecourses,
    result: NCAResult,
    *,
    ncols: int = 3,
    log_y: bool = True,
    annotate: bool = False,
    spread: Literal["sd", "se"] | None = "sd",
    ci_level: float = 0.95,
    partial: str | None = None,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    """One NCA panel per sample of a batch, with one legend for the figure.

    The title of a panel names the sample by its coordinates,
    `dose = 50 mg, individual = s1`, with the unit of a coordinate which
    carries one; the `dose` coordinate of a batch takes the dose unit of the
    batch. The panels all draw the same elements, so the legend is drawn
    once: for the figure, above the grid, when this function creates the
    chart, and into the first panel when the caller supplies the chart,
    which keeps its own legends.

    Args:
        timecourses: the batch
        result: its result

    Keyword Args:
        ncols: panels per row, at most one per sample
        log_y: logarithmic value axes
        annotate: the values in every panel, off by default since the panels
            of a grid are small; the confidence band of the regression is
            drawn either way
        spread: error bars of the data, `sd` or `se` when the batch carries
            them, `None` for none
        ci_level: level of the confidence band of the terminal regression
        partial: name of a named partial area of the result
            (`NCAOptions.partial_aucs`) to shade in every panel, `None` for none
        chart: the chart to draw into, a new one by default; the panels go
            into the cells of its grid from `(row, column)` on, `ncols` per
            row
        row: the row of the first panel in `chart`
        column: the column of the first panel in `chart`
        style: colors and markers

    Returns:
        The chart.

    Raises:
        ValueError: if the panels do not fit into the grid of `chart`.
    """
    curves = list(timecourses)
    n = len(curves)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    own = chart is None
    target = chart_of(
        chart,
        row,
        column,
        rows=nrows,
        columns=ncols,
        size=(4.5 * ncols, 3.5 * nrows),
        fonts=_FONTS,
        # the panels draw the same elements: one legend of their union above
        # the grid, matplotlib's `fig.legend(loc="outside upper center",
        # ncols=6)`; a group no panel draws has no entry
        legends=[
            {
                "groups": _grid_legend_groups(curves, ci_level, partial, spread),
                "position": "top",
                "columns": 6,
            }
        ],
    )
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
        _nca_panel(
            target,
            row + k // ncols,
            column + k % ncols,
            tc,
            values,
            flags,
            log_y=log_y,
            title=title or None,
            # a chart of its own has the legend of the figure, which takes the
            # entries of every plot with a legend
            legend=own or k == 0,
            annotate=annotate,
            spread=spread,
            ci_level=ci_level,
            units=parameter_units,
            partial=partial,
            partial_range=(
                None if partial is None else partial_window(result, tc, partial)
            ),
            style=style,
            # a legend of the figure names the regression of no single panel
            shared=own,
        )
    return target


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


def _integer_ticks(values: np.ndarray) -> list[float] | None:
    """Integer ticks over the range of integer positions, matplotlib's `MaxNLocator(integer=True)`.

    Args:
        values: the positions, the numbers of the dosing intervals.

    Returns:
        At most ten integer ticks from the smallest to the largest position,
        `None` without a finite position.
    """
    data = values[np.isfinite(values)]
    if not data.size:
        return None
    low, high = int(np.floor(data.min())), int(np.ceil(data.max()))
    step = max(1, int(np.ceil((high - low + 1) / 10)))
    return [float(v) for v in range(low, high + 1, step)]


def plot_troughs(
    result: NCAResult,
    *,
    by: str | None = None,
    spread: Literal["sd", "se"] | None = "sd",
    x: Literal["time", "interval"] = "time",
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
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
        chart: the chart to draw into, a new one by default.
        row: the row of the panel in `chart`.
        column: the column of the panel in `chart`.
        style: colors and markers.

    Returns:
        The chart.

    Raises:
        ValueError: if the result has no interval parameters or no
            `interval_ctrough`, if `x` is neither `"time"` nor `"interval"`,
            or if the cell is not in the grid of `chart`.
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
    colors = group_colors(len(groups), style.cmap)
    rows = np.asarray(labels)
    series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, str]] = []
    for i, group in enumerate(groups):
        color = style.data_color if len(groups) == 1 else colors[i]
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
            series.append((label, x_values, mean, error, k, color))
    # the statistic of the markers, named once for the figure and only where
    # there is one: a group of a single sample draws that sample's troughs
    reduced = [dim for dim in sample_dims if dim != by]
    title = None
    if reduced and n_rows > len(groups):
        over = ", ".join(reduced)
        title = (
            f"mean over {over}" if spread is None else f"mean ± {statistic} over {over}"
        )
    target = chart_of(
        chart, row, column, rows=1, columns=1, size=(6.0, 4.0), fonts=_FONTS
    )
    plot = target.plot(
        row,
        column,
        title=title,
        x=ChartAxis(
            label=x_label,
            ticks=_integer_ticks(intervals) if x == "interval" else None,
        ),
        y=ChartAxis(
            label=axis_label("trough", unit_label(result.units("interval_ctrough")))
        ),
        legend_title=by,
    )
    for label, x_values, mean, error, k, color in series:
        plot.curve(
            x_values,
            mean,
            yerr=None if spread is None else error,
            error_cap=2.0,
            marker=marker(style.data_marker if k == 0 else style.terminal_marker),
            size=style.markersize,
            line="solid" if k == 0 else "dashed",
            width=style.linewidth,
            marker_fill=None if k == 0 else "none",
            color=color,
            label=label,
            legend=label,
        )
    return target


def plot_intervals(
    result: NCAResult,
    name: str = "interval_auc",
    *,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Chart:
    """Plot a per-interval parameter against the dosing interval.

    Without `indexers`, one line per sample of the result, labelled with the
    sample's coordinate values; with `indexers` selecting one sample, a single
    line. Non-finite values (an incomplete interval) are gaps of the line.
    The x axis carries the integer interval numbers only, never a fractional
    tick.

    Args:
        result: the result of a multiple dose analysis
        name: name of the per-interval variable (`interval_*`)

    Keyword Args:
        chart: the chart to draw into, a new one by default
        row: the row of the panel in `chart`
        column: the column of the panel in `chart`
        style: colors and markers
        **indexers: coordinate label per sample dimension selecting one sample

    Returns:
        The chart.

    Raises:
        ValueError: if the result has no interval parameters, if `name` is
            not one of them, or if the cell is not in the grid of `chart`.
    """
    if not result.has_intervals:
        raise ValueError("The result has no interval parameters")
    if name not in result.ds.data_vars or INTERVAL_DIM not in result.ds[name].dims:
        raise ValueError(f"'{name}' is not an interval variable of the result")
    da = result[name]
    x = result.ds[INTERVAL_DIM].to_numpy().astype(float)
    lines: list[tuple[np.ndarray, str, str | None]] = []
    if indexers:
        lines.append(
            (da.sel(indexers).to_numpy().astype(float), style.data_color, None)
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
            color = style.data_color if n <= 1 else colors[i]
            label = labels[i] if n > 1 else None
            lines.append((da.isel(sel).to_numpy().astype(float), color, label))
    target = chart_of(
        chart, row, column, rows=1, columns=1, size=(6.0, 4.0), fonts=_FONTS
    )
    plot = target.plot(
        row,
        column,
        x=ChartAxis(label=INTERVAL_DIM, ticks=_integer_ticks(x)),
        y=ChartAxis(label=axis_label(name, unit_label(result.units(name)))),
        legend=len(lines) > 1,
    )
    for y, color, label in lines:
        plot.curve(
            x,
            y,
            marker=marker(style.data_marker),
            size=style.markersize,
            width=style.linewidth,
            color=color,
            label=label,
            legend=label,
        )
    return target


def plot_excretion(
    result: NCAResult,
    excretion: Excretion,
    *,
    cumulative: bool = True,
    log_rate: bool = True,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    r"""Plot the excretion rate curve of a urine analysis and the amount recovered.

    The figure a mass balance study is read from: the excretion rate
    \(\dot A_k\) of every collection against the midpoint \(\bar t_k\) of its
    interval on a logarithmic axis, with the terminal regression
    \(e^{b - \lambda_z t}\) over its window, and the cumulative amount
    recovered against the end of every collection on a second axis, which
    flattens out as the excretion stops.

    Args:
        result: the result of `pkpdutils.nca.urine.nca_urine`
        excretion: the collections the result was computed from, which carry
            the amount and the end of every interval

    Keyword Args:
        cumulative: whether the amount recovered is drawn on a second axis
        log_rate: whether the rate axis is logarithmic, where the terminal
            phase is a straight line
        chart: the chart to draw into, a new one by default
        row: the row of the panel in `chart`
        column: the column of the panel in `chart`
        style: colors and markers

    Returns:
        The chart.

    Raises:
        ValueError: if the result carries no excretion rate curve (it is not
            the result of `nca_urine`), or if the cell is not in the grid of
            `chart`.
    """
    if "rate" not in result.ds.data_vars or "midpoint" not in result.ds.data_vars:
        raise ValueError(
            "the result carries no excretion rate curve; plot_excretion draws "
            "the result of nca_urine"
        )
    midpoint = np.asarray(result["midpoint"].to_numpy(), dtype=float)
    rate = np.asarray(result["rate"].to_numpy(), dtype=float)
    values, _ = _sample_values(result, {})
    lambda_z = values.get("lambda_z", np.nan)
    intercept = values.get("lambda_z_intercept", np.nan)
    t_first = values.get("lambda_z_t_first", np.nan)
    t_last = values.get("lambda_z_t_last", np.nan)
    regression = None
    if np.isfinite([lambda_z, intercept, t_first, t_last]).all():
        # the rate needs no unit conversion (an amount per time is reported as
        # it is measured), so the intercept is in the unit of the drawn curve
        grid = np.linspace(t_first, t_last, 50)
        regression = (grid, np.exp(intercept - lambda_z * grid))
    log = log_axis(log_rate, "y", rate, None if regression is None else regression[1])
    limits = data_limits(rate, None if regression is None else regression[1], log=log)
    recovered = cumulative and excretion.amount is not None
    y2 = None
    if recovered:
        # the plateau of the recovery sits where the legend goes, so the axis
        # keeps room above it
        top = float(np.nanmax(excretion.cumulative))
        y2 = ChartAxis(
            label=axis_label(
                "amount recovered", unit_label(result.units("amount_recovered"))
            ),
            min=0.0,
            max=top * LEGEND_HEADROOM if top > 0.0 else None,
        )
    target = chart_of(
        chart, row, column, rows=1, columns=1, size=(6.0, 4.0), fonts=_FONTS
    )
    plot = target.plot(
        row,
        column,
        x=ChartAxis(label=axis_label("time", unit_label(result.units("midpoint")))),
        y=value_axis(
            axis_label("excretion rate", unit_label(result.units("rate"))),
            log=log,
            limits=limits,
        ),
        y2=y2,
        legend_loc="upper right",
    )
    plot.curve(
        midpoint,
        rate,
        marker=marker(style.data_marker),
        size=style.markersize,
        width=style.linewidth,
        color=style.data_color,
        label="excretion rate",
        legend="excretion rate",
    )
    if regression is not None:
        label = (
            f"{LAMBDA_Z_LABEL}{lambda_z:.3g} 1/{unit_label(result.units('midpoint'))}"
        )
        plot.curve(
            *regression,
            line="dashed",
            width=style.linewidth,
            color=style.fit_color,
            label=label,
            legend=label,
        )
    if recovered:
        plot.curve(
            excretion.end,
            excretion.cumulative,
            marker=marker(style.terminal_marker),
            size=style.markersize,
            width=style.linewidth,
            color=style.auc_color,
            label="amount recovered",
            legend="amount recovered",
            axis="y2",
        )
    return target


def plot_sparse(
    mean_curve: Timecourse | Timecourses,
    result: NCAResult,
    *,
    log_y: bool = False,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    r"""Plot the mean curve of a sparse design with its area and the Bailer error.

    The mean of the animals sampled at every nominal time with the standard
    error \(s_j/\sqrt{n_j}\) of that mean as an error bar, `auc_last` shaded
    under the polygon the linear trapezoid rule integrates (from the first
    nominal time to the last measurable one, which is the window the estimator
    covers, `pkpdutils.nca.sparse.area_window`), and the estimate with its
    standard error and `n_animals` per time point written into the panel.

    Args:
        mean_curve: the mean curve, a `Timecourse` or a batch of one sample
            (`pkpdutils.nca.sparse.sparse_mean`)
        result: the result of `pkpdutils.nca.sparse.nca_sparse` of the same
            design

    Keyword Args:
        log_y: whether the value axis is logarithmic
        chart: the chart to draw into, a new one by default
        row: the row of the panel in `chart`
        column: the column of the panel in `chart`
        style: colors and markers

    Returns:
        The chart.

    Raises:
        ValueError: if the batch holds more than one sample, if the result
            is not the one of a sparse analysis, or if the cell is not in the
            grid of `chart`.
    """
    missing = [
        name for name in SPARSE_PANEL_VARIABLES if name not in result.ds.data_vars
    ]
    if missing:
        raise ValueError(
            f"the result carries no {missing}; plot_sparse draws the result of "
            "nca_sparse"
        )
    if isinstance(mean_curve, Timecourses):
        if mean_curve.n_samples != 1:
            raise ValueError(
                f"the batch holds {mean_curve.n_samples} samples, plot_sparse "
                "draws the one mean curve of a design"
            )
        curve = next(iter(mean_curve))
    else:
        curve = mean_curve
    time = np.asarray(curve.time, dtype=float)
    value = np.asarray(curve.value, dtype=float)
    se = None if curve.se is None else np.asarray(curve.se, dtype=float)
    # the same window the estimator weights (`pkpdutils.nca.sparse`)
    through_last = area_window(np.isfinite(value), value)
    spread = [value] if se is None else [value, value - se, value + se]
    log = log_axis(log_y, "y", *spread)
    limits = data_limits(*spread, *([0.0] if through_last.any() else []), log=log)
    target = chart_of(
        chart, row, column, rows=1, columns=1, size=(6.0, 4.0), fonts=_FONTS
    )
    plot = target.plot(
        row,
        column,
        x=ChartAxis(label=axis_label("time", unit_label(curve.time_unit))),
        y=value_axis(
            axis_label(curve.substance, unit_label(curve.unit)), log=log, limits=limits
        ),
        legend_loc="center right",
    )
    if through_last.any():
        plot.area(
            time[through_last],
            np.full(int(through_last.sum()), limits[0] if log else 0.0),
            value[through_last],
            color=style.auc_color,
            fill_opacity=style.alpha,
            label="auc_last",
            legend="auc_last",
        )
    plot.curve(
        time,
        value,
        yerr=se,
        error_cap=3.0,
        marker=marker(style.data_marker),
        size=style.markersize,
        width=style.linewidth,
        color=style.data_color,
        label="mean of the animals",
        legend="mean of the animals",
    )
    counts = (
        np.asarray(result["n_animals"].to_numpy(), dtype=int)
        if "n_animals" in result.ds.data_vars
        else np.array([], dtype=int)
    )
    lines = [
        f"auc_last = {estimate_text(float(result['auc_last']), np.nan, np.nan)}"
        f" ± {float(result['auc_last_se']):.3g} {unit_label(result.units('auc_last'))}",
        f"df = {float(result['auc_last_df']):.3g}",
    ]
    if counts.size:
        lines.append(
            f"n animals per time point: {', '.join(str(int(c)) for c in counts)}"
        )
    plot.text(
        0.98,
        0.98,
        "\n".join(lines),
        anchor="end",
        baseline="top",
        size=style.annotation_fontsize,
        color="black",
        coordinates="axes",
        clip=False,
    )
    return target

"""Figures of timecourses: the curves of a batch and the mean curve of a group."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from sbmlsim_core.chart import Chart, ChartAxis, ChartPlot

from pkpdutils.plot._common import (
    axis_label,
    chart_of,
    data_limits,
    dose_markers,
    format_value,
    group_colors,
    group_order,
    has_infusion,
    log_axis,
    marker,
    sample_colors,
    sample_labels,
    unit_label,
    value_axis,
    value_with_unit,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import (
    NOMINAL_TIMES_VAR,
    TIME_DIM,
    TIMES_VAR,
    Timecourse,
    Timecourses,
)

#: the panels of `plot_mean_timecourse` and their titles
_PANEL_TITLES = {"linear": "linear", "log": "semi-logarithmic"}


@dataclass(frozen=True)
class _Curve:
    """One curve of a panel with the way it is drawn.

    Attributes:
        tc: the curve
        color: color of the line and the markers
        label: legend label, `None` for none
        errorbars: draw `se` (or `sd`) as error bars when present
        opacity: opacity of the line and the markers
        markers: draw a marker per point (a short curve) or the line alone
    """

    tc: Timecourse
    color: str
    label: str | None
    errorbars: bool
    opacity: float = 1.0
    markers: bool = True

    @property
    def error(self) -> np.ndarray | None:
        """The error bars of the curve, `None` for none."""
        if not self.errorbars:
            return None
        err = self.tc.se if self.tc.se is not None else self.tc.sd
        return None if err is None else np.asarray(err, dtype=float)

    def values(self) -> list[np.ndarray]:
        """The values the curve puts on the y axis, its error bars included."""
        value = np.asarray(self.tc.value, dtype=float)
        error = self.error
        if error is None:
            return [value]
        return [value, value - error, value + error]


def _draw_curve(plot: ChartPlot, curve: _Curve, style: PlotStyle) -> None:
    """One curve with optional error bars.

    A curve of more than `style.marker_max_points` points is drawn as a line
    without markers: a simulated or densely sampled curve otherwise becomes a
    solid band of markers which hides its shape.

    Args:
        plot: the plot to draw into
        curve: the curve
        style: colors and markers
    """
    tc = curve.tc
    shown = curve.markers and tc.time.size <= style.marker_max_points
    error = curve.error
    plot.curve(
        tc.time,
        tc.value,
        yerr=error,
        error_cap=2.0 if error is not None else 0.0,
        marker=marker(style.data_marker) if shown else "none",
        size=style.markersize,
        width=style.linewidth,
        color=curve.color,
        opacity=curve.opacity,
        label=curve.label,
        legend=curve.label,
    )


def _group_legend_title(batch: Timecourses, by: str | None) -> str | None:
    """The title of the legend of a grouped figure, `dose [mg]`.

    The unit of the grouping coordinate is written once into the title of the
    legend, rather than into every entry of it. A coordinate whose values are
    labels (`dose` as `"low"` and `"high"`) carries no unit.

    Args:
        batch: the batch the coordinate belongs to.
        by: name of the grouping coordinate, `None` for no legend title.

    Returns:
        The name of the coordinate with its unit, `None` without `by`.
    """
    if by is None:
        return None
    unit = ""
    if by in batch.ds.coords:
        numeric = np.issubdtype(batch.ds[by].dtype, np.number)
        unit = (
            str(batch.ds[by].attrs.get("units", ""))
            or (batch.dose_unit if by == "dose" and numeric else "")
            or ""
        )
    return axis_label(by, unit_label(unit) if unit else "")


def _facet_value(batch: Timecourses, facet: str, value: Any) -> str:
    """The value of a facet coordinate for the title of its panel, with its unit.

    Args:
        batch: the batch the coordinate belongs to.
        facet: name of the coordinate the panels come from.
        value: the value of the panel.

    Returns:
        The value, followed by the unit of the coordinate when the value is a
        number and the coordinate has one (the dose unit of the batch for
        `dose`); a label (`"low"`) is written as it is.
    """
    units = {"dose": batch.dose_unit} if "dose" in batch.ds.coords else {}
    return value_with_unit(batch.ds, facet, value, units=units)


def _panel(
    chart: Chart,
    row: int,
    column: int,
    curves: Sequence[_Curve],
    reference: Timecourse,
    *,
    log_y: bool,
    style: PlotStyle,
    title: str | None = None,
    max_legend: int = 12,
    legend_title: str | None = None,
    proxies: Mapping[str, str] | None = None,
    dosing: Timecourse | None = None,
    bands: Sequence[tuple[Timecourse, np.ndarray, str]] = (),
    floor: float | None = None,
) -> ChartPlot:
    """One panel of timecourses: its axes, its curves and its legend.

    The y axis holds every curve with its error bars and every band, with
    matplotlib's margins; a logarithmic axis without a positive value stays
    linear (logged at debug level). A legend of more than `max_legend`
    entries is left out, since it would cover the figure rather than explain
    it.

    Args:
        chart: the chart to draw into
        row: the row of the panel
        column: the column of the panel
        curves: the curves, in the order they are drawn (the last on top)
        reference: a curve of the data, for the substance and the units
        log_y: logarithmic value axis
        style: colors and markers
        title: title of the panel
        max_legend: most entries the legend may have, `0` for no legend
        legend_title: title of the legend
        proxies: color per legend label of the groups the panel names in its
            legend whether or not it draws them (a faceted figure, whose
            legend on the first panel names every group of the batch)
        dosing: the curve whose dosing protocol is marked, `None` for none
        bands: the bands drawn behind the curves, each the curve of the mean,
            the spread around it and the color
        floor: the lowest value of a logarithmic axis, which a band is clipped
            at, `None` for the lowest value of the data

    Returns:
        The plot.
    """
    values: list[np.ndarray] = []
    for curve in curves:
        values.extend(curve.values())
    for mean, spread, _ in bands:
        values.extend([mean.value - spread, mean.value + spread])
    log = log_axis(log_y, "y", *values)
    if log and floor is not None:
        values = [np.where(v < floor, np.nan, v) for v in values]
    limits = data_limits(*values, log=log)
    if log and floor is not None:
        limits = (floor, limits[1])
    labels: list[str] = []
    for curve in curves:
        if curve.label is not None and curve.label not in labels:
            labels.append(curve.label)
    missing = [label for label in proxies or {} if label not in labels]
    infusion = dosing is not None and has_infusion(dosing.dosing)
    n_entries = len(labels) + len(missing) + int(infusion)
    plot = chart.plot(
        row,
        column,
        title=title,
        x=ChartAxis(label=f"time [{reference.time_unit}]"),
        y=value_axis(
            f"{reference.substance} [{reference.unit}]", log=log, limits=limits
        ),
        legend=0 < n_entries <= max_legend,
        legend_title=legend_title,
    )
    if dosing is not None:
        dose_markers(
            plot,
            dosing.dosing,
            y_limits=limits,
            style=style,
            time_unit=dosing.time_unit,
            taken=set(labels),
        )
    for mean, spread, color in bands:
        lower = mean.value - spread
        if log and floor is not None:
            lower = np.maximum(lower, floor)
        plot.band(
            mean.time,
            lower,
            mean.value + spread,
            line="none",
            color=color,
            fill_opacity=style.alpha,
        )
    for curve in curves:
        _draw_curve(plot, curve, style)
    for label in missing:
        assert proxies is not None
        plot.curve(
            [np.nan],
            [np.nan],
            marker=marker(style.data_marker),
            size=style.markersize,
            width=style.linewidth,
            color=proxies[label],
            label=label,
            legend=label,
        )
    return plot


def plot_timecourse(
    timecourses: Timecourse | Timecourses,
    *,
    log_y: bool = False,
    errorbars: bool = True,
    by: str | None = None,
    facet: str | None = None,
    max_legend: int = 12,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    """Plot one timecourse or every timecourse of a batch.

    With `by` the curves are colored by group: every sample carrying the same
    value of the coordinate gets the same color and the group contributes one
    legend entry, instead of one color and one entry per sample. With `facet`
    the batch is split into one panel per value of a coordinate, the colors
    staying the same across the panels and the legend, on the first panel,
    naming every group. A legend of more than `max_legend` entries is left
    out, since it would cover the figure rather than explain it.

    A single curve whose protocol has more than one dose also gets one thin
    dotted vertical line per dose time (`style.dose_color`) and, for an
    infusion, the window from the dose time to the end of the infusion as a
    shaded span; a batch draws no dose markers, since its curves may carry
    different protocols (`plot_mean_timecourse` draws them for the group).

    Args:
        timecourses: the curve or the batch

    Keyword Args:
        log_y: logarithmic value axis; a curve without a positive value stays
            linear (logged at debug level)
        errorbars: draw `se` (or `sd`) as error bars when present
        by: coordinate of the batch grouping the curves: one color and one
            legend entry per group, the sample label per curve by default
        facet: coordinate of the batch drawn as one panel per value, a single
            panel by default
        max_legend: most entries the legend of a panel may have
        chart: the chart to draw into, a new one by default; the panels go
            into the cells of its grid from `(row, column)` on, one per value
            of `facet` along the row
        row: the row of the first panel in `chart`
        column: the column of the first panel in `chart`
        style: colors and markers

    Returns:
        The chart.

    Raises:
        ValueError: if `by` or `facet` is given for a single curve, or if the
            panels do not fit into the grid of `chart`.
    """
    if isinstance(timecourses, Timecourse):
        if facet is not None or by is not None:
            raise ValueError("'by' and 'facet' need a batch of timecourses")
        target = chart_of(chart, row, column, rows=1, columns=1, size=(6.0, 4.0))
        _panel(
            target,
            row,
            column,
            [_Curve(timecourses, style.data_color, timecourses.label, errorbars)],
            timecourses,
            log_y=log_y,
            style=style,
            max_legend=max_legend,
            dosing=timecourses,
        )
        return target

    batch = timecourses
    first = batch.isel(**dict.fromkeys(batch.sample_dims, 0))
    color_of: dict[str, str] | None = None
    if by is not None:
        groups = group_order(batch.ds, batch.sample_dims, by)
        color_of = dict(zip(groups, group_colors(len(groups), style.cmap), strict=True))
    if facet is None:
        target = chart_of(chart, row, column, rows=1, columns=1, size=(6.0, 4.0))
        _panel(
            target,
            row,
            column,
            _batch_curves(
                batch, by=by, color_of=color_of, errorbars=errorbars, style=style
            ),
            first,
            log_y=log_y,
            style=style,
            max_legend=max_legend,
            legend_title=_group_legend_title(batch, by),
        )
        return target

    panels = list(batch.groupby(facet))
    target = chart_of(
        chart,
        row,
        column,
        rows=1,
        columns=len(panels),
        size=(5.0 * len(panels), 4.0),
    )
    # the legend of a grouped figure names every group of the batch, also the
    # ones no sample of the first panel carries
    for k, (value, group) in enumerate(panels):
        _panel(
            target,
            row,
            column + k,
            _batch_curves(
                group, by=by, color_of=color_of, errorbars=errorbars, style=style
            ),
            first,
            log_y=log_y,
            style=style,
            title=f"{facet} = {_facet_value(batch, facet, value)}",
            max_legend=max_legend if k == 0 else 0,
            legend_title=_group_legend_title(batch, by),
            proxies=color_of if k == 0 else None,
        )
    return target


def _batch_curves(
    batch: Timecourses,
    *,
    by: str | None,
    color_of: Mapping[str, str] | None,
    errorbars: bool,
    style: PlotStyle,
    names: Mapping[str, str] | None = None,
) -> list[_Curve]:
    """Every curve of a batch, colored by sample or by group.

    Args:
        batch: the batch
        by: grouping coordinate, `None` for one color per sample
        color_of: color per group label, `None` without `by`
        errorbars: draw `se` (or `sd`) as error bars when present
        style: colors and markers
        names: the legend label per group label, the group label by default

    Returns:
        One curve per sample; the curves of a group share their legend label.
    """
    curves = list(batch)
    if by is not None and color_of is not None:
        labels = sample_labels(batch.ds, batch.sample_dims, by=by)
        return [
            _Curve(tc, color_of[label], (names or {}).get(label, label), errorbars)
            for tc, label in zip(curves, labels, strict=True)
        ]
    palette = sample_colors(len(curves), style.cmap)
    return [
        _Curve(
            tc,
            style.data_color if len(curves) == 1 else palette[i],
            tc.label,
            errorbars,
        )
        for i, tc in enumerate(curves)
    ]


def _mean_curve(
    batch: Timecourses, *, group_dim: str | None, spread: Literal["sd", "se"]
) -> Timecourse:
    """The mean curve of a batch over its sample dimensions.

    The dimension the groups live on is kept (it holds one group here) and
    every other sample dimension is reduced by `Timecourses.mean`, one after
    the other; a batch whose only sample dimension is the group dimension is
    reduced over it, which is the case of a grouping coordinate along the
    dimension of the individuals.

    Args:
        batch: the batch of one group
        group_dim: the dimension the grouping coordinate lives on, `None`
            without grouping
        spread: the statistic `Timecourses.mean` computes from the curves

    Returns:
        The mean curve with its `sd`, `se` and `n`.
    """
    reduced = batch
    dims = [d for d in reduced.sample_dims if d != group_dim] or list(
        reduced.sample_dims
    )
    for dim in dims:
        reduced = reduced.mean(dim, spread=spread)
    return next(iter(reduced))


def plot_mean_timecourse(
    batch: Timecourses,
    *,
    by: str | None = None,
    spread: Literal["sd", "se"] | None = "sd",
    individuals: bool = True,
    individual_alpha: float = 0.25,
    panels: Sequence[str] = ("linear", "log"),
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    r"""The mean curve of every group with its spread, on a linear and a semi-log panel.

    The concentration-time figure of a study report: the arithmetic mean of
    the samples at every time point, the band \(\bar c \pm s\) of its spread
    around it and, faint behind both, the individual curves. With `by` the
    batch is split into groups (a dose level, a treatment, an arm) and every
    group gets its own color; without it the whole batch is one group. The
    reduction is `Timecourses.groupby` and `Timecourses.mean`, so the spread
    is the scatter of the curves and not an uncertainty carried by them.

    Both panels show the same curves, the second one on a logarithmic value
    axis with plain tick labels, and only the first one carries the legend.
    The dose markers of the first group are drawn on every panel: a dotted
    line per dose of a protocol of several doses, an infusion as the shaded
    window from the dose time to the end of the infusion.

    Args:
        batch: the batch, with at least one sample dimension.

    Keyword Args:
        by: coordinate of the batch grouping the curves (a sample dimension or
            a coordinate along one), one group by default.
        spread: the statistic of the band, the standard deviation (`"sd"`),
            the standard error (`"se"`) or no band (`None`).
        individuals: draw the individual curves behind the mean.
        individual_alpha: transparency of the individual curves.
        panels: the panels, `"linear"` and `"log"` in the order they are drawn.
        chart: the chart to draw into, a new one by default; the panels go
            into the cells of its grid from `(row, column)` on, along the row.
        row: the row of the first panel in `chart`.
        column: the column of the first panel in `chart`.
        style: colors and markers.

    Returns:
        The chart.

    Raises:
        ValueError: if `panels` is empty or names an unknown panel, if
            `spread` is not `"sd"`, `"se"` or `None`, if the batch has no
            sample dimension to reduce, or if the panels do not fit into the
            grid of `chart`.
    """
    return _mean_panels(
        batch,
        by=by,
        spread=spread,
        individuals=individuals,
        individual_alpha=individual_alpha,
        panels=panels,
        chart=chart,
        row=row,
        column=column,
        style=style,
    )


def _mean_panels(
    batch: Timecourses,
    *,
    by: str | None,
    spread: Literal["sd", "se"] | None,
    individuals: bool,
    individual_alpha: float,
    panels: Sequence[str],
    chart: Chart | None,
    row: int,
    column: int,
    style: PlotStyle,
    title_prefix: str = "",
) -> Chart:
    """The panels of `plot_mean_timecourse`, their titles after `title_prefix`.

    Args:
        batch: the batch.

    Keyword Args:
        by: the grouping coordinate.
        spread: the statistic of the band.
        individuals: draw the individual curves.
        individual_alpha: transparency of the individual curves.
        panels: the panels.
        chart: the chart to draw into, a new one by default.
        row: the row of the first panel.
        column: the column of the first panel.
        style: colors and markers.
        title_prefix: the start of the title of every panel.

    Returns:
        The chart.
    """
    names = list(panels)
    unknown = [name for name in names if name not in _PANEL_TITLES]
    if not names or unknown:
        raise ValueError(
            f"'panels' must name {tuple(_PANEL_TITLES)}, got {tuple(panels)}"
        )
    if spread not in ("sd", "se", None):
        raise ValueError(f"'spread' must be 'sd', 'se' or None, got {spread!r}")
    if not batch.sample_dims:
        raise ValueError("a mean curve needs a batch with a sample dimension")
    statistic: Literal["sd", "se"] = "sd" if spread is None else spread
    groups: list[tuple[Any, Timecourses]]
    group_dim: str | None
    if by is None:
        groups, group_dim = [(None, batch)], None
    else:
        groups = list(batch.groupby(by))
        group_dim = by if by in batch.sample_dims else str(batch.ds[by].dims[0])
    colors = group_colors(len(groups), style.cmap)
    target = chart_of(
        chart,
        row,
        column,
        rows=1,
        columns=len(names),
        size=(5.5 * len(names), 4.2),
    )
    first = batch.isel(**dict.fromkeys(batch.sample_dims, 0))
    # the reduction of every group runs once, not once per panel: the panels
    # draw the same curves on two scales
    curves: list[_Curve] = []
    means: list[_Curve] = []
    bands: list[tuple[Timecourse, np.ndarray, str]] = []
    smallest = np.inf
    for i, (value, group) in enumerate(groups):
        color = style.data_color if len(groups) == 1 else colors[i]
        mean = _mean_curve(group, group_dim=group_dim, spread=statistic)
        members = list(group) if individuals else []
        # the number of curves the group was reduced from, which is what the
        # `n` of a study figure means; a group is named by its coordinate
        # value with the unit of that coordinate (`50 mg`), as the panel
        # titles of `plot_nca_grid` are
        name_of = (
            "mean" if value is None or by is None else _facet_value(batch, by, value)
        )
        curves.extend(
            _Curve(tc, color, None, False, opacity=individual_alpha, markers=False)
            for tc in members
        )
        means.append(_Curve(mean, color, f"{name_of} (n = {group.n_samples})", False))
        if spread is not None:
            bands.append(
                (mean, np.asarray(getattr(mean, statistic), dtype=float), color)
            )
        for values in (mean.value, *(tc.value for tc in members)):
            positive = values[np.isfinite(values) & (values > 0.0)]
            if positive.size:
                smallest = min(smallest, float(positive.min()))
    # the lower edge of a band reaches towards zero wherever the spread is as
    # large as the mean, which would stretch a logarithmic axis over decades
    # the data never reaches: the axis is bounded by the curves and the band
    # is clipped at that bound
    floor = 0.5 * smallest if np.isfinite(smallest) else None
    for k, name in enumerate(names):
        heading = _PANEL_TITLES[name]
        if k == 0 and spread is not None:
            heading = f"{heading}, band = mean ± {statistic}"
        _panel(
            target,
            row,
            column + k,
            [*curves, *means],
            first,
            log_y=name == "log",
            style=style,
            title=f"{title_prefix}{heading}",
            max_legend=len(groups) + 2 if k == 0 else 0,
            legend_title=by,
            dosing=means[0].tc,
            bands=bands,
            floor=floor,
        )
    return target


def nominal_grid(
    batch: Timecourses, nominal_times: npt.ArrayLike | None = None
) -> np.ndarray:
    """The nominal (scheduled) time of every point of a batch.

    The variable `nominal_time` of the batch when it carries one
    (`Timecourses.from_arrays(nominal_time=...)`,
    `Timecourses.from_dataframe(nominal_time=...)`), else the nearest entry of
    `nominal_times` for every actual time, else the actual times themselves,
    which is the right answer for data recorded on its schedule (a simulation,
    a mean curve of a publication).

    Args:
        batch: the batch
        nominal_times: the sampling schedule of the study, mapped to every
            actual time by nearest neighbour; `None` to use the variable of the
            batch or the actual times

    Returns:
        The nominal time of every sample and point, `(*sample_shape, n_time)`,
        `NaN` where the batch has no point.
    """
    actual = batch.times
    stored = batch.nominal_times
    if stored is not None:
        return stored
    if nominal_times is None:
        return actual
    grid = np.asarray(nominal_times, dtype=np.float64).ravel()
    if grid.size == 0:
        return actual
    nearest = grid[np.abs(actual[..., None] - grid[None, :]).argmin(axis=-1)]
    return np.where(np.isfinite(actual), nearest, np.nan)


def on_nominal_times(batch: Timecourses, nominal: np.ndarray) -> Timecourses:
    """A copy of the batch whose samples sit on the shared grid of their nominal times.

    Every point moves from the time it was taken at to the time it was
    scheduled for, and the batch gets the sorted union of those times as its
    grid: the samples of a study then share their time points and can be
    averaged, which is what the mean profile of a study report is taken on
    (ICH M13A 2.2.2.1). Two points of one sample with the same nominal time
    would land on the same place; the later one wins.

    Args:
        batch: the batch
        nominal: the nominal time of every sample and point, `(*sample_shape,
            n_time)` (`nominal_grid`)

    Returns:
        The batch on the nominal grid, the doses and the coordinates unchanged.
    """
    sample_dims = batch.sample_dims
    n_rows, n_time = batch.n_samples, batch.n_time
    flat = np.asarray(nominal, dtype=np.float64).reshape(n_rows, n_time)
    finite = np.isfinite(flat)
    if not finite.any():
        return batch
    grid = np.unique(flat[finite])
    rows = np.repeat(np.arange(n_rows), finite.sum(axis=1))
    columns = np.searchsorted(grid, flat[finite])
    moved = [
        str(name)
        for name, da in batch.ds.data_vars.items()
        if TIME_DIM in da.dims and str(name) not in (TIMES_VAR, NOMINAL_TIMES_VAR)
    ]
    placed: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, Any]]] = {}
    for name in moved:
        values = (
            batch.ds[name]
            .transpose(*sample_dims, TIME_DIM)
            .to_numpy()
            .astype(np.float64)
            .reshape(n_rows, n_time)
        )
        block = np.full((n_rows, grid.size), np.nan)
        block[rows, columns] = values[finite]
        placed[name] = (
            (*sample_dims, TIME_DIM),
            block.reshape(*batch.sample_shape, grid.size),
            dict(batch.ds[name].attrs),
        )
    time_attrs = dict(
        (batch.ds[TIMES_VAR] if TIMES_VAR in batch.ds else batch.ds[TIME_DIM]).attrs
    )
    ds = batch.ds.drop_vars(
        [*moved, *(name for name in (TIMES_VAR, NOMINAL_TIMES_VAR) if name in batch.ds)]
    )
    ds = ds.drop_dims(TIME_DIM) if TIME_DIM in ds.dims else ds
    ds = ds.assign_coords({TIME_DIM: grid})
    ds[TIME_DIM].attrs.update(time_attrs)
    for name, (dims, values, attrs) in placed.items():
        ds[name] = xr.DataArray(values, dims=dims, attrs=attrs)
    return Timecourses(ds.transpose(*sample_dims, TIME_DIM, ...))


def plot_study_curves(
    batch: Timecourses,
    *,
    by: str | None = None,
    nominal_times: npt.ArrayLike | None = None,
    log_y_panels: bool = True,
    max_legend: int = 12,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    """The concentration-time figures of a study report, individuals and means.

    ICH M13A (2024, 2.2.2.1) asks for the concentration-time profile of every
    subject on a linear and on a semi-logarithmic scale, and for the mean
    profile of every treatment on both scales as well, the individual figures
    drawn against the actual sampling times and the mean figures against the
    nominal ones: a mean over the subjects only exists on the schedule the
    study sampled by. This function draws the four panels in one call, the
    individual curves as `plot_timecourse` draws them in the first row and the
    mean curves of `plot_mean_timecourse` (mean, spread band and the
    individuals behind it) in the second. A group keeps its color and its name
    over all four panels: the legend of the individual row names the groups
    as the one of the mean row does (`50 mg`, which there carries the number
    of subjects behind the mean as well).

    The nominal times come from the variable `nominal_time` of the batch when
    it carries one, else from `nominal_times` by nearest neighbour, else the
    actual times are taken as nominal, which is the right answer for data
    recorded on its schedule (`nominal_grid`, `on_nominal_times`).

    Args:
        batch: the batch of the study, with at least one sample dimension

    Keyword Args:
        by: coordinate of the batch grouping the curves (a treatment, an arm, a
            dose level), one group and one color by default; the groups keep
            their colors across the four panels
        nominal_times: the sampling schedule of the study, used when the batch
            carries no `nominal_time` variable: every actual time is mapped to
            the nearest of them
        log_y_panels: draw the two semi-logarithmic panels; `False` gives the
            two linear panels alone
        max_legend: most entries the legend of the individual row may have;
            above it none is drawn, as in `plot_timecourse`
        chart: the chart to draw into, a new one by default; the panels go
            into two rows of its grid from `(row, column)` on
        row: the row of the individual panels in `chart`, the mean panels
            are in the next one
        column: the column of the first panel in `chart`
        style: colors and markers

    Returns:
        The chart.

    Raises:
        ValueError: if the batch has no sample dimension, or if the panels do
            not fit into the grid of `chart`.
    """
    if not batch.sample_dims:
        raise ValueError("a study figure needs a batch with a sample dimension")
    scales = (False, True) if log_y_panels else (False,)
    target = chart_of(
        chart,
        row,
        column,
        rows=2,
        columns=len(scales),
        size=(5.5 * len(scales), 8.4),
    )
    nominal = on_nominal_times(batch, nominal_grid(batch, nominal_times))
    first = batch.isel(**dict.fromkeys(batch.sample_dims, 0))
    # the groups are named as the mean panels name them (`50 mg`), so that one
    # reading of the legend holds for the whole figure
    color_of: dict[str, str] | None = None
    names: dict[str, str] | None = None
    proxies: dict[str, str] | None = None
    if by is not None:
        groups = group_order(batch.ds, batch.sample_dims, by)
        colors = group_colors(len(groups), style.cmap)
        color_of = dict(zip(groups, colors, strict=True))
        names = {
            format_value(value): _facet_value(batch, by, value)
            for value, _ in batch.groupby(by)
        }
        proxies = {names.get(label, label): color for label, color in color_of.items()}
    curves = _batch_curves(
        batch, by=by, color_of=color_of, errorbars=True, style=style, names=names
    )
    for k, log_y in enumerate(scales):
        _panel(
            target,
            row,
            column + k,
            curves,
            first,
            log_y=log_y,
            style=style,
            title=f"individuals, {_PANEL_TITLES['log' if log_y else 'linear']}",
            max_legend=max_legend if k == 0 else 0,
            legend_title=by if by is not None else None,
            proxies=proxies if k == 0 else None,
        )
    _mean_panels(
        nominal,
        by=by,
        spread="sd",
        individuals=True,
        individual_alpha=0.25,
        panels=tuple("log" if log_y else "linear" for log_y in scales),
        chart=target,
        row=row + 1,
        column=column,
        style=style,
        title_prefix="mean on the nominal times, ",
    )
    return target

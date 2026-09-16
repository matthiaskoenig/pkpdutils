"""Figures of timecourses: the curves of a batch and the mean curve of a group."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pkpdutils.plot._common import (
    axes_of,
    axis_label,
    dose_markers,
    figure_of,
    group_colors,
    group_order,
    log_scale,
    sample_colors,
    sample_labels,
    unit_label,
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
class _Group:
    """One reduced group of `plot_mean_timecourse`, drawn on every panel.

    Attributes:
        label: legend label of the mean curve
        color: color of the group
        mean: the mean curve with its spread
        curves: the individual curves, empty when they are not drawn
        band: the spread drawn around the mean, `None` for no band
    """

    label: str
    color: Any
    mean: Timecourse
    curves: list[Timecourse]
    band: np.ndarray | None


def _draw_curve(
    ax: Axes,
    tc: Timecourse,
    *,
    color: Any,
    label: str | None,
    errorbars: bool,
    style: PlotStyle,
    alpha: float = 1.0,
) -> None:
    """One curve with optional error bars.

    A curve of more than `style.marker_max_points` points is drawn as a line
    without markers: a simulated or densely sampled curve otherwise becomes a
    solid band of markers which hides its shape.

    Args:
        ax: axes to draw on
        tc: the curve
        color: color of the line and markers
        label: legend label, `None` for none
        errorbars: draw `se` (or `sd`) as error bars when present
        style: colors and markers
        alpha: transparency of the line and the markers
    """
    marker = style.data_marker if tc.time.size <= style.marker_max_points else "none"
    err = tc.se if tc.se is not None else tc.sd
    if errorbars and err is not None:
        ax.errorbar(
            tc.time,
            tc.value,
            yerr=err,
            marker=marker,
            linestyle="-",
            color=color,
            label=label,
            linewidth=style.linewidth,
            markersize=style.markersize,
            capsize=2,
            alpha=alpha,
        )
    else:
        ax.plot(
            tc.time,
            tc.value,
            marker=marker,
            linestyle="-",
            color=color,
            label=label,
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=alpha,
        )


def _legend(
    ax: Axes,
    max_legend: int,
    title: str | None = None,
    handles: Sequence[Artist] | None = None,
) -> None:
    """Draw the legend of an axes unless it would have too many entries.

    Args:
        ax: the axes
        max_legend: most entries a legend may have; above it none is drawn
        title: title of the legend, the name of the grouping coordinate
        handles: the entries of the legend, the labelled artists of `ax` by
            default; a faceted figure passes one proxy per group, since the
            panel the legend is drawn on need not carry every group
    """
    entries = list(ax.get_legend_handles_labels()[0] if handles is None else handles)
    labels = [str(handle.get_label()) for handle in entries]
    if not entries or len(labels) > max_legend:
        return
    ax.legend(entries, labels, fontsize="small", title=title, title_fontsize="small")


def _group_handles(color_of: Mapping[str, Any], style: PlotStyle) -> list[Artist]:
    """One legend proxy per group, in the order of the groups.

    Args:
        color_of: color per group label
        style: colors and markers

    Returns:
        A `Line2D` per group, drawn like the curves of that group.
    """
    return [
        Line2D(
            [],
            [],
            color=color,
            marker=style.data_marker,
            linestyle="-",
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=label,
        )
        for label, color in color_of.items()
    ]


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


def plot_timecourse(
    timecourses: Timecourse | Timecourses,
    *,
    log_y: bool = False,
    errorbars: bool = True,
    by: str | None = None,
    facet: str | None = None,
    max_legend: int = 12,
    ax: Axes | None = None,
    axes: Sequence[Axes] | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Plot one timecourse or every timecourse of a batch.

    With `by` the curves are colored by group: every sample carrying the same
    value of the coordinate gets the same color and the group contributes one
    legend entry, instead of one color and one entry per sample. With `facet`
    the batch is split into one panel per value of a coordinate, the colors
    staying the same across the panels. A legend of more than `max_legend`
    entries is left out, since it would cover the figure rather than explain
    it.

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
        ax: axes to draw on, a new figure by default; a caller-supplied `ax`
            keeps its figure's own layout engine, so long tick labels can
            clip unless the caller sets one (`fig.set_layout_engine("constrained")`)
        axes: one axes per value of `facet`, a new figure by default
        style: colors and markers

    Returns:
        The figure.

    Raises:
        ValueError: if `facet` is given for a single curve, if `ax` and `axes`
            are given together, or if `ax` is given with `facet`.
    """
    if ax is not None and axes is not None:
        raise ValueError("pass either 'ax' or 'axes', not both")
    if facet is not None and ax is not None:
        raise ValueError("a faceted figure takes 'axes', not 'ax'")
    if isinstance(timecourses, Timecourse):
        if facet is not None or by is not None:
            raise ValueError("'by' and 'facet' need a batch of timecourses")
        fig, panel = figure_of(ax)
        _draw_curve(
            panel,
            timecourses,
            color=style.data_color,
            label=timecourses.label,
            errorbars=errorbars,
            style=style,
        )
        dose_markers(
            panel, timecourses.dosing, style=style, time_unit=timecourses.time_unit
        )
        _finish_panel(panel, timecourses, log_y=log_y, max_legend=max_legend)
        return fig

    batch = timecourses
    first = batch.isel(**dict.fromkeys(batch.sample_dims, 0))
    color_of: dict[str, Any] | None = None
    if by is not None:
        groups = group_order(batch.ds, batch.sample_dims, by)
        palette = group_colors(len(groups), style.cmap)
        color_of = dict(zip(groups, palette, strict=True))
    if facet is None:
        fig, panel = figure_of(ax)
        _draw_batch(
            panel,
            batch,
            by=by,
            color_of=color_of,
            errorbars=errorbars,
            style=style,
            drawn=set(),
        )
        _finish_panel(
            panel,
            first,
            log_y=log_y,
            max_legend=max_legend,
            legend_title=_group_legend_title(batch, by),
        )
        return fig

    panels = list(batch.groupby(facet))
    fig, grid = axes_of(axes, 1, len(panels), figsize=(5.0 * len(panels), 4.0))
    # the legend of a grouped figure names every group of the batch, also the
    # ones no sample of the first panel carries, so it is built from proxies
    # of the colors rather than from the artists of that panel
    handles = None if color_of is None else _group_handles(color_of, style)
    drawn: set[str] = set() if color_of is None else set(color_of)
    for k, (value, group) in enumerate(panels):
        panel = grid[0][k]
        _draw_batch(
            panel,
            group,
            by=by,
            color_of=color_of,
            errorbars=errorbars,
            style=style,
            drawn=drawn,
        )
        panel.set_title(f"{facet} = {_facet_value(batch, facet, value)}")
        _finish_panel(
            panel,
            first,
            log_y=log_y,
            max_legend=max_legend if k == 0 else 0,
            legend_title=_group_legend_title(batch, by),
            handles=handles,
        )
    return fig


def _draw_batch(
    ax: Axes,
    batch: Timecourses,
    *,
    by: str | None,
    color_of: dict[str, Any] | None,
    errorbars: bool,
    style: PlotStyle,
    drawn: set[str],
) -> None:
    """Every curve of a batch into one panel, colored by sample or by group.

    Args:
        ax: axes to draw on
        batch: the batch
        by: grouping coordinate, `None` for one color per sample
        color_of: color per group label, `None` without `by`
        errorbars: draw `se` (or `sd`) as error bars when present
        style: colors and markers
        drawn: group labels which already carry a legend entry, extended here
            so that a group appears once in the legend of a faceted figure
    """
    curves = list(batch)
    if by is not None and color_of is not None:
        labels = sample_labels(batch.ds, batch.sample_dims, by=by)
        for tc, label in zip(curves, labels, strict=True):
            legend_label = None if label in drawn else label
            drawn.add(label)
            _draw_curve(
                ax,
                tc,
                color=color_of[label],
                label=legend_label,
                errorbars=errorbars,
                style=style,
            )
        return
    palette = sample_colors(len(curves), style.cmap)
    for i, tc in enumerate(curves):
        color: Any = style.data_color if len(curves) == 1 else palette[i]
        _draw_curve(
            ax, tc, color=color, label=tc.label, errorbars=errorbars, style=style
        )


def _finish_panel(
    ax: Axes,
    reference: Timecourse,
    *,
    log_y: bool,
    max_legend: int,
    legend_title: str | None = None,
    handles: Sequence[Artist] | None = None,
) -> None:
    """Axis labels, the scale and the legend of one panel.

    Args:
        ax: the panel
        reference: a curve of the data, for the substance and the units
        log_y: logarithmic value axis
        max_legend: most entries the legend may have, `0` for no legend
        legend_title: title of the legend
        handles: the entries of the legend, the labelled artists by default
    """
    ax.set_xlabel(f"time [{reference.time_unit}]")
    ax.set_ylabel(f"{reference.substance} [{reference.unit}]")
    if log_y:
        log_scale(ax, "y")
    _legend(ax, max_legend, legend_title, handles)


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
    axes: Sequence[Axes] | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
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
        axes: one axes per panel, a new figure by default.
        style: colors and markers.

    Returns:
        The figure.

    Raises:
        ValueError: if `panels` is empty or names an unknown panel, if
            `spread` is not `"sd"`, `"se"` or `None`, or if the batch has no
            sample dimension to reduce.
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
    fig, grid = axes_of(axes, 1, len(names), figsize=(5.5 * len(names), 4.2))
    first = batch.isel(**dict.fromkeys(batch.sample_dims, 0))
    # the reduction of every group runs once, not once per panel: the panels
    # draw the same curves on two scales
    reduced: list[_Group] = []
    smallest = np.inf
    for i, (value, group) in enumerate(groups):
        color: Any = style.data_color if len(groups) == 1 else colors[i]
        mean = _mean_curve(group, group_dim=group_dim, spread=statistic)
        curves = list(group) if individuals else []
        # the number of curves the group was reduced from, which is what the
        # `n` of a study figure means; a time point some of them are missing
        # from carries fewer, in the `n` of the mean curve
        # a group is named by its coordinate value with the unit of that
        # coordinate (`50 mg`), as the panel titles of `plot_nca_grid` are
        name_of = (
            "mean" if value is None or by is None else _facet_value(batch, by, value)
        )
        reduced.append(
            _Group(
                label=f"{name_of} (n = {group.n_samples})",
                color=color,
                mean=mean,
                curves=curves,
                band=getattr(mean, statistic) if spread is not None else None,
            )
        )
        for values in (mean.value, *(tc.value for tc in curves)):
            positive = values[np.isfinite(values) & (values > 0.0)]
            if positive.size:
                smallest = min(smallest, float(positive.min()))
    # the lower edge of a band reaches towards zero wherever the spread is as
    # large as the mean, which would stretch a logarithmic axis over decades
    # the data never reaches: the axis is bounded by the curves and the band
    # is clipped at that bound
    floor = 0.5 * smallest if np.isfinite(smallest) else None
    for k, name in enumerate(names):
        ax = grid[0][k]
        for group_index, entry in enumerate(reduced):
            for tc in entry.curves:
                ax.plot(
                    tc.time,
                    tc.value,
                    linestyle="-",
                    color=entry.color,
                    linewidth=style.linewidth,
                    alpha=individual_alpha,
                )
            if entry.band is not None:
                lower = entry.mean.value - entry.band
                if name == "log" and floor is not None:
                    lower = np.maximum(lower, floor)
                ax.fill_between(
                    entry.mean.time,
                    lower,
                    entry.mean.value + entry.band,
                    color=entry.color,
                    alpha=style.alpha,
                    linewidth=0.0,
                )
            _draw_curve(
                ax,
                entry.mean,
                color=entry.color,
                label=entry.label,
                errorbars=False,
                style=style,
            )
            if group_index == 0:
                dose_markers(
                    ax, entry.mean.dosing, style=style, time_unit=first.time_unit
                )
        heading = _PANEL_TITLES[name]
        if k == 0 and spread is not None:
            heading = f"{heading}, band = mean ± {statistic}"
        ax.set_title(heading)
        _finish_panel(
            ax,
            first,
            log_y=name == "log",
            max_legend=len(groups) + 2 if k == 0 else 0,
            legend_title=by,
        )
        if name == "log" and ax.get_yscale() == "log" and floor is not None:
            ax.set_ylim(bottom=floor)
    return fig


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
    axes: Sequence[Axes] | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """The concentration-time figures of a study report, individuals and means.

    ICH M13A (2024, 2.2.2.1) asks for the concentration-time profile of every
    subject on a linear and on a semi-logarithmic scale, and for the mean
    profile of every treatment on both scales as well, the individual figures
    drawn against the actual sampling times and the mean figures against the
    nominal ones: a mean over the subjects only exists on the schedule the
    study sampled by. This function draws the four panels in one call, the
    individual curves with `plot_timecourse` in the first row and the mean
    curves with `plot_mean_timecourse` (mean, spread band and the individuals
    behind it) in the second.

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
        axes: the four axes to draw into (two with `log_y_panels=False`), in
            reading order, a new figure by default
        style: colors and markers

    Returns:
        The figure.

    Raises:
        ValueError: if the batch has no sample dimension.
    """
    if not batch.sample_dims:
        raise ValueError("a study figure needs a batch with a sample dimension")
    scales = (False, True) if log_y_panels else (False,)
    fig, grid = axes_of(axes, 2, len(scales), figsize=(5.5 * len(scales), 8.4))
    nominal = on_nominal_times(batch, nominal_grid(batch, nominal_times))
    for k, log_y in enumerate(scales):
        panel = grid[0][k]
        plot_timecourse(
            batch,
            log_y=log_y,
            by=by,
            max_legend=12 if k == 0 else 0,
            ax=panel,
            style=style,
        )
        panel.set_title(f"individuals, {_PANEL_TITLES['log' if log_y else 'linear']}")
    plot_mean_timecourse(
        nominal,
        by=by,
        panels=tuple("log" if log_y else "linear" for log_y in scales),
        axes=list(grid[1]),
        style=style,
    )
    for panel in grid[1]:
        panel.set_title(f"mean on the nominal times, {panel.get_title()}")
    return fig

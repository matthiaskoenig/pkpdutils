"""Figures of timecourses and analyses (matplotlib).

Every function takes its data first and every option as a keyword,
`f(data, *, <options>, ax=None, style=DEFAULT_STYLE)`, returns the
`matplotlib.figure.Figure` it drew on and never shows it; pass `ax` to draw
into an existing axes and `axes` to a multi-panel figure (`plot_nca`,
`plot_nca_grid`, `plot_fit`). A logarithmic axis is `log_x` or `log_y`, and
every one of them carries plain tick labels (`10`, `100`) instead of powers
of ten. `draw_nca_panel` draws one NCA panel and returns the
`matplotlib.axes.Axes`, for a figure the caller lays out.
"""

from pkpdutils.plot.fit import (
    plot_bland_altman,
    plot_dose_proportionality,
    plot_fit,
    plot_goodness_of_fit,
)
from pkpdutils.plot.meta import plot_forest
from pkpdutils.plot.nca import (
    draw_nca_panel,
    plot_intervals,
    plot_nca,
    plot_nca_grid,
)
from pkpdutils.plot.parameters import plot_parameters
from pkpdutils.plot.ratio import plot_ratio
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import plot_timecourse

__all__ = [
    "DEFAULT_STYLE",
    "PlotStyle",
    "draw_nca_panel",
    "plot_bland_altman",
    "plot_dose_proportionality",
    "plot_fit",
    "plot_forest",
    "plot_goodness_of_fit",
    "plot_intervals",
    "plot_nca",
    "plot_nca_grid",
    "plot_parameters",
    "plot_ratio",
    "plot_timecourse",
]

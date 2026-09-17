"""Figures of timecourses and analyses (matplotlib).

Every function takes its data first and every option as a keyword,
`f(data, *, <options>, ax=None, style=DEFAULT_STYLE)`, returns the
`matplotlib.figure.Figure` it drew on and never shows it; pass `ax` to draw
into an existing axes and `axes` to a multi-panel figure (`plot_nca`,
`plot_nca_grid`, `plot_fit`). A logarithmic axis is `log_x` or `log_y`, and
every one of them carries plain tick labels (`10`, `100`) instead of powers
of ten. `draw_nca_panel` draws one NCA panel and returns the
`matplotlib.axes.Axes`, for a figure the caller lays out.

`plot_mean_timecourse` is the group figure of a study report (the mean of
every group with its spread, the individuals faint behind it, on a linear and
a semi-logarithmic panel), `plot_timecourse` takes `by` to color the curves by
group and `facet` for one panel per value of a coordinate, and `plot_troughs`
shows the trough of every dosing interval, the figure of steady state.
`plot_study_curves` is the figure pair of a study report (the individual curves
on the actual times and the mean curves on the nominal times, linear and
semi-logarithmic) and `plot_terminal_windows` the diagnostic of the terminal
phase (the adjusted R² of every candidate window against its start time with
the chosen one marked). `plot_excretion` is the figure of a urine study (the
excretion rate curve with its terminal regression and the amount recovered) and
`plot_sparse` the figure of a sparse design (the mean curve with the Bailer
standard errors and the shaded area).
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
    plot_excretion,
    plot_intervals,
    plot_nca,
    plot_nca_grid,
    plot_sparse,
    plot_terminal_windows,
    plot_troughs,
)
from pkpdutils.plot.parameters import plot_parameters
from pkpdutils.plot.ratio import plot_ratio
from pkpdutils.plot.save import save_figure
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import (
    plot_mean_timecourse,
    plot_study_curves,
    plot_timecourse,
)

__all__ = [
    "DEFAULT_STYLE",
    "PlotStyle",
    "draw_nca_panel",
    "plot_bland_altman",
    "plot_dose_proportionality",
    "plot_excretion",
    "plot_fit",
    "plot_forest",
    "plot_goodness_of_fit",
    "plot_intervals",
    "plot_mean_timecourse",
    "plot_nca",
    "plot_nca_grid",
    "plot_parameters",
    "plot_ratio",
    "plot_sparse",
    "plot_study_curves",
    "plot_terminal_windows",
    "plot_timecourse",
    "plot_troughs",
    "save_figure",
]

# API reference

The API reference is generated from the docstrings of the package.

## pkpdutils

| module | description |
| --- | --- |
| [units](units.md) | the unit registry of the package and unit helpers |
| [timecourse](timecourse.md) | `Timecourse`, `Timecourses`, `Dose`, `Route` and `DosingRegimen`, the data model |
| [console](console.md) | shared rich console |
| [log](log.md) | logging of the package |

## pkpdutils.nca

Non-compartmental analysis, see [Non-compartmental analysis](../nca.md).

| module | description |
| --- | --- |
| [nca.nca](nca.md) | `nca`, `nca_single`, `compute_parameters`: the analysis |
| [nca.options](nca.options.md) | `NCAOptions`, `TerminalPhase`, the method enumerations and `NCAFlag` |
| [nca.result](nca.result.md) | `NCAResult` and the units of the parameters |
| [nca.auc](nca.auc.md) | vectorized trapezoid areas, interpolation |
| [nca.terminal](nca.terminal.md) | vectorized terminal phase regression |
| [nca.steady_state](nca.steady_state.md) | steady state parameters, accumulation ratio, superposition |

## pkpdutils.plot

Figures, see [Plotting](../plotting.md).

| module | description |
| --- | --- |
| [plot](plot.md) | `PlotStyle`, `plot_timecourse`, `plot_nca`, `plot_nca_grid` |

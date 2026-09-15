# API reference

The API reference is generated from the docstrings of the package.

## pkpdutils

| module | description |
| --- | --- |
| [units](units.md) | the unit registry of the package and unit helpers |
| [timecourse](timecourse.md) | `Timecourse`, `Timecourses`, `Dose`, `Route` and `DosingRegimen`, the data model |
| [result](result.md) | `ParameterResult`, the shared container of `NCAResult` and `FitResult` |
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
| [nca.uncertainty](nca.uncertainty.md) | bootstrap and delta method of the parameters of group timecourses |

## pkpdutils.fit

Curve fitting, see [Curve fitting](../fitting.md) and [Pharmacodynamics](../pd.md).

| module | description |
| --- | --- |
| [fit](fit.md) | `fit`, `fit_timecourse`, `fit_timecourses`, `fit_table`, `FitOptions`, `FitResult`, `Model`: the engine, the result, the front ends and the options |
| [fit.models](fit.models.md) | the model library: exponentials, the Emax family, linear, power and allometric models |
| [fit.compare](fit.compare.md) | `compare_models` and `ModelComparison`: the ranking by AICc and the Akaike weights |
| [fit.proportionality](fit.proportionality.md) | `proportionality_test`: the confidence interval criterion of dose proportionality |

## pkpdutils.plot

Figures, see [Plotting](../plotting.md).

| module | description |
| --- | --- |
| [plot](plot.md) | `PlotStyle`, `plot_timecourse`, `plot_nca`, `plot_nca_grid`, `plot_fit`, `plot_goodness_of_fit`, `plot_dose_proportionality` |

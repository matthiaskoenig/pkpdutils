# API reference

The API reference is generated from the docstrings of the package.

## pkpdutils

| module | description |
| --- | --- |
| [units](units.md) | the unit registry of the package and unit helpers |
| [timecourse](timecourse.md) | `Timecourse`, `Timecourses`, `Dose`, `Dosing`, `Route` and `DosingRegimen`, the data model |
| [result](result.md) | `ParameterResult`, the shared container of `NCAResult` and `FitResult`; `sample` gives a `ParameterSample`, `summary_table` the parameter table of a publication |
| [io](io.md) | exchange formats, see [Data formats](../formats.md): `read_events`/`write_events`, `read_pknca`, `read_adnca` |
| [parallel](parallel.md) | the shared worker pools: `executor`, `resolve_workers`, `split_rows` |
| [console](console.md) | shared rich console, `rich_table` and `print_table` for the tables of the package |
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
| [nca.intervals](nca.intervals.md) | parameters of every dosing interval of a multiple dose curve |
| [nca.steady_state](nca.steady_state.md) | steady state parameters of the last interval, accumulation ratio, superposition |
| [nca.uncertainty](nca.uncertainty.md) | bootstrap and delta method of the parameters of group timecourses |
| [nca.report](nca.report.md) | `M13A_STATISTICS`, `acceptability_table`, `methods_line`: the tables of a regulatory report |

## pkpdutils.fit

Curve fitting, see [Curve fitting](../fitting.md) and [Pharmacodynamics](../pd.md).

| module | description |
| --- | --- |
| [fit](fit.md) | `fit`, `fit_timecourse`, `fit_timecourses`, `fit_table`, `FitOptions`, `FitResult`, `Model`: the engine, the result, the front ends and the options |
| [fit.models](fit.models.md) | the model library: exponentials, the Emax family, linear, power and allometric models |
| [fit.compare](fit.compare.md) | `compare_models` and `ModelComparison`: the ranking by AICc and the Akaike weights |
| [fit.proportionality](fit.proportionality.md) | `proportionality_test` and `proportionality_table`: the confidence interval criterion of dose proportionality |

## pkpdutils.stats

Statistics on parameters, see [Statistics](../statistics.md).

| module | description |
| --- | --- |
| [stats](stats.md) | `ParameterSample`, `Scale`, `summarize`, `compare`, `multiple_comparison`, `ratio`, `ratio_table`: samples, tests and the geometric mean ratio |
| [stats.bioequivalence](stats.bioequivalence.md) | `bioequivalence`, `tost`, `Design`: the two one-sided tests, paired, parallel and 2x2 crossover designs |
| [stats.ddi](stats.ddi.md) | `ddi_classification`, `ddi_table`, `substrate_sensitivity`, `DDIThresholds`: the FDA and EMA classification of interactions |
| [stats.meta](stats.meta.md) | `effect_size`, `fixed_effect`, `random_effects`, `heterogeneity`, `meta_analysis`: the meta-analysis |

## pkpdutils.plot

Figures, see [Plotting](../plotting.md).

| module | description |
| --- | --- |
| [plot](plot.md) | `PlotStyle`, `plot_timecourse`, `plot_nca`, `plot_nca_grid`, `plot_intervals`, `plot_fit`, `plot_goodness_of_fit`, `plot_dose_proportionality`, `plot_parameters`, `plot_ratio`, `plot_forest`, `plot_bland_altman` |

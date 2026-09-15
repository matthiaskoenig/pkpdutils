# pkpdutils 1.0.0: redesign of pkdb_analysis

Date: 2026-09-14. Status: approved design, basis of the implementation plan.

## Summary

`pkdb_analysis` (0.3.1) is a client and analysis toolbox for PK-DB. It is replaced by `pkpdutils`, a library for the pharmacokinetic and pharmacodynamic analysis of timecourses and parameters, with no PK-DB dependency: non-compartmental analysis (NCA) of concentration and effect timecourses, curve fitting of exponential, Emax, dose proportionality and covariate models, uncertainty propagation for group data, significance tests, bioequivalence, drug–drug interaction (DDI) classification and meta-analysis on PK parameters, and matplotlib figures for all of it. Data structures are xarray based so many timecourses (individuals, groups, studies, sbmlsim scans) are analysed in one vectorized call, with an optional process pool.

The rewrite is a clean slate in the existing repository on the branch `redesign`: the old package, its documentation and tests are deleted, the NCA algorithms and the meta-analysis math are ported with tests against the old numeric results, and the repository takes over the tooling, branching, documentation and release conventions of `sbmlsim` and `sbmlutils`. The result is released as `pkpdutils` 1.0.0; the PyPI project `pkdb-analysis` stays at 0.3.1.

## Decisions

| topic | decision |
| --- | --- |
| package and PyPI name | `pkpdutils`; GitHub repository renamed to `matthiaskoenig/pkpdutils` (redirect from the old name) |
| old PyPI name | `pkdb-analysis` left at 0.3.1, no tombstone release; README of `pkpdutils` mentions the former name |
| python | 3.14 for development (`.python-version`), 3.13 and 3.14 supported and tested |
| tooling | uv, hatchling, ruff, ty, tox, pre-commit, bump-my-version, Zensical + mkdocstrings, as in sbmlsim |
| runtime dependencies | `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib`, `rich` |
| removed dependencies | `requests`, `depinfo`, `coloredlogs`, `openpyxl`, `XlsxWriter`, `pyyaml`, `gspread-pandas`, `altair`, `seaborn`, `scikit-learn`, `IPython` |
| matplotlib | core dependency, not an extra |
| compartmental models | out of scope (no 1-/2-compartment PK models, no ODE fitting, no NLME); sbmlsim and PEtab cover mechanistic fitting |
| approach | clean slate on branch `redesign`, history preserved, one series of pull requests into `develop`, tag `1.0.0` at the end |

## Scope of 1.0.0

In scope:

1. NCA of single dose and steady state concentration timecourses, iv bolus, iv infusion and extravascular, unit aware
2. NCA of effect timecourses (PD descriptive: AUEC, observed Emax, TEmax, time above threshold)
3. uncertainty propagation for group timecourses (mean ± SD, n) by bootstrap or delta method; summary statistics over individuals
4. curve fitting engine with exponential, Bateman, Emax family, power (dose proportionality) and covariate (allometric, linear, log-linear) models, with standard errors, confidence intervals, bootstrap and model comparison
5. statistics on PK parameters: significance tests, geometric mean ratios, bioequivalence (TOST, 2×2 crossover), DDI classification (FDA and EMA thresholds), meta-analysis (fixed and random effects)
6. matplotlib figures: timecourses, NCA diagnostics, fits, parameter distributions, dose proportionality, forest plots, ratio plots
7. documentation with concepts, math, parameter tables, examples and references for every part

Out of scope, documented as non-goals: compartmental and ODE based models, population (mixed effects) modelling, covariate model building, PK-DB access, report generation (LaTeX, Jekyll), Gaussian process kernels, circos plots.

## 1. Package layout and data model

```
src/pkpdutils/
  __init__.py       __version__, top level exports
  units.py          shared UnitRegistry `ureg`, `Q_`, `Quantity` alias, unit helpers
  timecourse.py     Timecourse, Dose, Route, DosingRegimen, Timecourses
  nca/              non-compartmental analysis
    __init__.py
    options.py      NCAOptions, TerminalPhase, enumerations
    nca.py          nca, nca_single, NCAResult
    auc.py          vectorized AUC, AUMC, partial AUC
    terminal.py     vectorized terminal phase regression and window selection
    steady_state.py steady state parameters, superposition
    uncertainty.py  bootstrap and delta method
    summary.py      summarize over a sample dimension
  fit/              curve fitting
    __init__.py
    engine.py       fit, FitOptions, FitResult, compare_models
    model.py        Model protocol, ModelParameter
    models.py       MonoExp, BiExp, TriExp, Bateman, Emax family, PowerModel, covariate models
    proportionality.py  proportionality_test
  stats/            statistics on parameters
    __init__.py
    sample.py       ParameterSample, summarize
    tests.py        compare, TestResult, multiple_comparison
    ratio.py        ratio, RatioResult
    bioequivalence.py
    ddi.py          ddi_classification, DDIThresholds, DDIResult
    meta.py         effect_size, fixed_effect, random_effects, heterogeneity, meta_analysis
  plot/             matplotlib figures
    __init__.py
    style.py        PlotStyle
    timecourse.py, nca.py, fit.py, parameters.py, meta.py, ratio.py
  console.py        rich console for scripts and examples
  log.py            opt-in rich logging (`enable_rich_logging`)
```

Modules log through `logging.getLogger(__name__)` with lazy `%s` formatting; the package never configures logging and never prints.

### Units

One `pint.UnitRegistry` per process in `units.py` (`ureg`), with the custom definitions the old package needed (`percent`, `IU`, dimensionless `none`). `Quantity` is the type alias used in annotations. Numerics run on plain `float64` arrays in the units of the input; pint is used at the boundaries: parsing unit strings, deriving result units symbolically (`auc` has unit `unit·time_unit`, `cl` has `dose_unit/(unit·time_unit)`), converting (`vz` to `liter` or `liter/kg`, `cl` to `liter/hour` or `liter/hour/kg`) and building quantities on request.

### `Timecourse`

Pydantic model, frozen, for one curve:

- `time`, `value`: 1-D `numpy` arrays of equal length; `time` strictly increasing after validation (unsorted input is sorted with a warning, duplicate times raise `ValueError`); `NaN` allowed in `value`
- `time_unit`, `unit`: unit strings parsed by `ureg`; `value` is the generic name because effect timecourses are not concentrations
- optional `sd`, `se` (arrays, same length) and `n` (scalar or array); `se = sd/sqrt(n)` is derived when one of the two and `n` are given
- `dose: Dose | None` with `Dose(amount, unit, route, time=0.0, duration=None)`; `Route` is `IV_BOLUS`, `IV_INFUSION` (requires `duration`) or `ORAL` (any extravascular route); the dose unit must reduce to `[mass]`, `[substance]`, `[mass]/[mass]` or `[substance]/[mass]`, otherwise `ValueError`
- `substance: str`, `label: str | None`, `tissue: str | None`
- `time_q`, `value_q`: quantities; `relative_to_dose()`: copy with `time` shifted by `dose.time`
- `from_dataframe(df, time="time", value="value", ...)`, `to_dataframe()`

### `Timecourses`

Batch container around an `xarray.Dataset`, the form every analysis function works on:

- dimensions: `time` plus any number of sample dimensions (`individual`, `dose`, `study`, the scan dimensions of an sbmlsim result)
- data variables: `value` (required), optional `sd`, `se`, `n`; `time` is a coordinate shared by all samples, or a 2-D variable over `(sample dims, time)` when samples have different sampling times, in which case shorter curves are padded with `NaN`
- dose as variables `dose_amount`, `dose_time`, `dose_duration` over the sample dimensions (a scalar broadcasts), `route` as an attribute or a coordinate when routes differ
- units as `attrs["units"]` on every variable (pint strings), `substance` and `route` as dataset attributes; metadata such as sex, weight or study as coordinates on the sample dimensions
- constructors: `from_timecourses(list[Timecourse], dim="individual")`, `from_dataframe(long_df, sample=[...], time=..., value=...)`, `from_arrays(time, values, ...)`, `from_xresult(xres, selection, time_unit=..., unit=...)` for an sbmlsim `XResult` (`_time` dimension, units from `uinfo`; sbmlsim is not a dependency, the function only reads the dataset and the units dictionary)
- `__iter__`, `sel(...)`, `isel(...)` return `Timecourse` objects for single samples; `to_dataframe()`

### Results

`NCAResult`, `FitResult`, and the summary tables are `xarray.Dataset` objects with one data variable per parameter over the sample dimensions of the input, `attrs["units"]` per variable, and diagnostic variables next to the parameters. Common methods: `to_dataframe()`, `to_quantities()` (single sample), `sample(parameter, dim)` (a `ParameterSample` for the statistics), `summarize(dim)`.

### Performance

- AUC, AUMC, `cmax`, `tmax`, `cmin`, `clast` are vectorized numpy operations along the `time` axis, `NaN` aware
- the terminal phase regression evaluates every candidate window of at least `min_points` points as one batched least squares on masked arrays and picks the window per sample without a python loop over samples
- `n_workers` on `nca`, `fit` and the bootstrap: chunks along the sample dimensions run in a `concurrent.futures.ProcessPoolExecutor`; every worker function is a pure module level function with picklable arguments, so the `spawn` and `forkserver` start methods of python 3.14 work
- bootstrap adds a `boot` dimension and reuses the vectorized code; memory is `samples × n_boot × n_time` floats, chunked over the samples above a configurable limit

## 2. Non-compartmental analysis (`pkpdutils.nca`)

Entry points: `nca(timecourses, options=NCAOptions()) -> NCAResult` and `nca_single(timecourse, options) -> NCAResult` (one sample, same result type). Steady state analysis is requested with `NCAOptions(regimen=DosingRegimen(interval=tau, n_doses=None))`.

### `NCAOptions`

- `kind`: `CONCENTRATION` (default) or `EFFECT`; `EFFECT` renames the parameters (`auec`, `emax_obs`, `temax`, `e0`) and disables the dose dependent parameters
- `auc_method`: `LINEAR`, `LINEAR_LOG` (linear up, logarithmic down; default) or `LOG`
- `terminal: TerminalPhase(method=BEST_FIT | LAST_N | ALL_AFTER_TMAX | MANUAL, min_points=3, exclude_cmax=True, n_points=None, points=None, min_adj_r2=None)`. `BEST_FIT` is the Phoenix WinNonlin rule: among all windows of consecutive points ending at `tlast`, with at least `min_points` points and starting after `tmax` when `exclude_cmax`, take the largest adjusted R²; a window with more points wins when its adjusted R² is within `1e-4` of the best. `LAST_N` uses the last `n_points`, `ALL_AFTER_TMAX` every point after `tmax` (the rule of the old package), `MANUAL` the given indices
- `lloq: float | None` and `blq: NAN | ZERO_BEFORE_TMAX`: values below the lower limit of quantification become `NaN`, or `0` before `tmax` and `NaN` after
- `c0_method` for iv bolus: `LOG_BACK_EXTRAPOLATION` from the first two positive points (default) or `FIRST_VALUE`
- `extrapolation_warning = 0.2`: flag when `auc_extrap_fraction` exceeds it
- `regimen: DosingRegimen | None`
- `uncertainty: NONE | BOOTSTRAP | DELTA`, `n_boot = 1000`, `seed`, `ci_level = 0.95`, `bootstrap_spread: SE | SD` (resample from `N(mean, se)`, the uncertainty of the mean, or from `N(mean, sd)`, the spread of individuals). Default is `BOOTSTRAP` when the input carries `sd` or `se`, `NONE` otherwise
- `n_workers: int | None`

### Parameters, single dose

Observed: `cmax`, `tmax`, `clast`, `tlast`, `c0` (iv bolus only), `cmax_half`, `tmax_half` (time to half maximum during absorption, kept from the old package, `NaN` without absorption phase), `cmin`, `tmin`.

Exposure: `auc_last` (0 to `tlast`), `auc_inf_obs` and `auc_inf_pred` (extrapolated with `clast_obs` or `clast_pred = exp(intercept - lambda_z·tlast)`), `auc_extrap_fraction`, `aumc_last`, `aumc_inf`, `mrt` (`aumc_inf/auc_inf`, minus `duration/2` for an infusion; for extravascular dosing it includes the absorption time), partial `auc(t1, t2)` on request. `auc_inf` is `auc_inf_obs`.

Terminal phase: `lambda_z` (also exported as `kel`), `thalf = ln 2/lambda_z`, `lambda_z_n_points`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`, `lambda_z_t_first`, `lambda_z_se`.

Dose dependent, when a dose is present: `cl_f = dose/auc_inf` (`cl` for iv), `vz_f = cl_f/lambda_z` (`vz` for iv), `vss = cl·mrt` (iv only), `auc_inf_dn = auc_inf/dose`, `cmax_dn = cmax/dose`. With a dose per body weight the units follow (`liter/kg`, `liter/hour/kg`).

Flags per sample in the integer variable `flags` and as a decoded list: `POSITIVE_SLOPE` (terminal regression slope not negative, `lambda_z` and dependents `NaN`), `TOO_FEW_POINTS`, `EXTRAPOLATION_HIGH`, `NO_MAX` (maximum at the last point), `NO_ABSORPTION` (maximum at the first point), `BLQ_TRUNCATED`, `UNSORTED_INPUT`. The library logs one summary line per call instead of one warning per curve.

### Steady state

With a `regimen`: `auc_tau`, `cmin_ss`, `ctrough`, `cavg = auc_tau/tau`, `fluctuation = (cmax - cmin)/cavg`, `swing = (cmax - cmin)/cmin`, `cl_ss = dose/auc_tau`, `accumulation_ratio` (`auc_tau_ss/auc_tau_single` when a single dose curve is given as `single_dose`, else predicted as `1/(1 - exp(-lambda_z·tau))`). `superposition(timecourse, regimen) -> Timecourse` predicts the multiple dose curve from a single dose curve by linear superposition, on the union of the shifted time grids with `NaN` aware interpolation on the log scale after `tmax`.

### Uncertainty

Group timecourses carry `sd`/`se` and `n`:

- `BOOTSTRAP`: resample every time point, run the vectorized NCA on the `boot` dimension, reduce to `x_sd`, `x_se`, `x_ci_low`, `x_ci_high` per parameter `x` (percentile interval at `ci_level`), with geometric mean and geometric CV for the log normal parameters (`auc*`, `cmax`, `cl*`, `vz*`, `thalf`); `seed` makes the result reproducible
- `DELTA`: first order propagation. AUC and AUMC are linear in the values (`var = Σ wᵢ² seᵢ²` with the trapezoid weights); `cmax` takes `se` at `tmax`; `lambda_z` takes the regression standard error; `thalf`, `cl`, `vz`, `auc_inf`, `mrt` propagate through a numerical Jacobian. Cheap, symmetric, less exact for ratios
- `NONE`

The result variables `x`, `x_sd`, `x_se`, `x_ci_low`, `x_ci_high` and `n` are the same for group data and for `summarize(dim="individual")` of individual data (arithmetic mean, sd, se, geometric mean, geometric CV, median, IQR, CI, `n` per parameter), so the statistics accept both.

## 3. Curve fitting (`pkpdutils.fit`)

### Engine

`fit(x, y, model, options=FitOptions(), sd=None) -> FitResult`; the batch form takes `Timecourses` (x is `time`) or an xarray table over sample dimensions with `x` and `y` variables.

- `scipy.optimize.least_squares` (`trf`, bounds); `parameter_scale = LOG10 | LOG | LINEAR` (default `LOG10`, bounds, start values and results stay on the linear scale); `weighting = NONE | INV_Y | INV_Y2 | INV_SD`; `loss` as in scipy
- `n_starts` Latin hypercube start points in the bounds (seed); `n_workers` process pool over samples × starts; the best start by cost wins
- per fit: standard errors from the Jacobian (`(JᵀJ)⁻¹·s²`), CV%, t based confidence intervals at `ci_level` (delta method to the linear scale), correlation matrix, residuals, `R²`, `RMSE`, `AIC`, `AICc`, `BIC`, convergence flag; `bootstrap = n` residual bootstrap confidence intervals
- `FitResult`: `xarray.Dataset` with the parameters and `x_se`, `x_ci_low`, `x_ci_high`, `x_cv`, the derived parameters, the statistics and the flags over the sample dimensions; `predict(x)` returns values on a grid, `residuals`, `to_dataframe()`
- `compare_models(x, y, models, options) -> pandas.DataFrame` with `AICc` and Akaike weights per model and sample, and the best model per sample

`Model` protocol: `name`, `parameters: list[ModelParameter(name, unit_expr, lower, upper, description)]`, `predict(x, p)`, `derived(p) -> dict`, `initial_guess(x, y) -> array`. `unit_expr` derives the parameter unit from the units of `x` and `y` (`"1/[x]"`, `"[y]"`, `"[y]/[x]"`), so a fit result carries units like an NCA result.

### Model library

- exponential: `MonoExp`, `BiExp`, `TriExp` (`Σ Aᵢ·exp(-λᵢ·t)`, λ sorted descending (the fastest phase first, as implemented in 1.0.0; the spec originally said ascending), derived `thalf_i`, `lambda_z = min λ`), `Bateman` (`A·ka/(ka - ke)·(exp(-ke(t - tlag)) - exp(-ka(t - tlag)))` with optional `tlag`, `flip_flop` flag when `ka < ke`); initial guesses by the method of residuals and from the NCA. No compartmental interpretation of the coefficients
- Emax family: `Emax(e0, emax, ec50)`, `SigmoidEmax(+ hill)`, `Imax`, `SigmoidImax`, `Linear`, `LogLinear`; derived `ec90`/`ic90`. Used for effect against concentration and for a parameter against an inhibitor dose or concentration
- dose proportionality: `PowerModel` (`a·xᵇ`), `LinearModel` (with intercept); `proportionality_test(result, criterion=(0.8, 1.25), dose_range) -> ProportionalityResult` on the confidence interval of `b` (Smith et al. 2000)
- covariate: `Allometric(a, b)` with `b` free or fixed (0.75 for clearance, 1 for volumes), `LinearCovariate`, `LogLinearCovariate`

### PD descriptive

Effect timecourses run through `nca` with `NCAOptions(kind=EFFECT)`: `auec`, `emax_obs`, `temax`, `e0` (baseline, first value or given), `time_above(threshold)`, baseline corrected `auec_baseline` and `emax_baseline`.

## 4. Statistics on PK parameters (`pkpdutils.stats`)

`ParameterSample` is the input of every function: individual values over one dimension, or summary statistics (`mean`, `sd`, `n`, optional `geomean`, `geocv`). It is built with `result.sample("auc_inf", dim="individual")` from an `NCAResult` or `FitResult`, or directly from numbers taken from a publication. Log normal parameters are analysed on the log scale by default (`scale = LOG`), `scale = LINEAR` is available.

- `compare(a, b, test=AUTO, scale=LOG, paired=False) -> TestResult(statistic, p_value, effect, ci, test_name, n_a, n_b)`. Individual data: Student t, Welch t, paired t, Mann–Whitney U, Wilcoxon signed rank, permutation test (`n_perm`, seed). Summary data: Welch t from `mean, sd, n`, or from `geomean, geocv` on the log scale. `AUTO` picks Welch t (unpaired) or paired t. Effect sizes: difference of means or ratio with CI, Cohen's d and Hedges' g. `multiple_comparison(p_values, method=HOLM | BONFERRONI | BH)`
- `ratio(test, reference, ci_level=0.90, paired=False) -> RatioResult(gmr, ci_low, ci_high, n_test, n_reference)`: geometric mean ratio with a t interval on the log scale, paired for crossover designs, Welch for parallel groups, and a summary statistics variant on the log scale
- `bioequivalence(test, reference, parameters=["auc_inf", "cmax"], limits=(0.8, 1.25), ci_level=0.90) -> BEResult`: per parameter `gmr`, CI, `bioequivalent`, the two one sided tests (Schuirmann 1987) p values; when `period` and `sequence` coordinates exist a 2×2 crossover ANOVA with sequence, period and subject effects on the log scale
- `ddi_classification(auc_ratio, cmax_ratio=None, ci=None, thresholds=DDIThresholds.fda()) -> DDIResult(kind=INHIBITOR | INDUCER | NONE, strength=STRONG | MODERATE | WEAK | NONE, auc_ratio, ci, uncertain)`. FDA (2020): inhibitor strong `≥ 5`, moderate `2–5`, weak `1.25–2`; inducer by AUC decrease strong `≥ 80 %`, moderate `50–80 %`, weak `20–50 %`. With a CI the classification uses the bound closer to 1 and sets `uncertain` when the interval spans a boundary. `substrate_sensitivity(auc_ratio)` gives `SENSITIVE` (`≥ 5`) or `MODERATELY_SENSITIVE` (`2–5`). `DDIThresholds.ema()` is the EMA variant
- meta-analysis, ported from `effect_analysis.py`: `effect_size(control, treatment, kind=HEDGES_G | MEAN_DIFF | LOG_RATIO)`, `fixed_effect`, `random_effects` (DerSimonian–Laird), `heterogeneity` (`Q`, `I²`, `τ²`), `meta_analysis(pairs, by="category") -> MetaResult` with the per study and pooled effects, weights and CIs; `LOG_RATIO` (`ln GMR`) is the effect size native to PK
- `summarize(values, scale) -> Summary`: `n`, mean, sd, se, CV, geometric mean, geometric CV, median, IQR, min, max, CI

## 5. Plotting (`pkpdutils.plot`)

Every function returns a `matplotlib.figure.Figure` or draws on the given `ax`; nothing calls `plt.show()`. Styles come from one `PlotStyle` dataclass. No seaborn.

- `plot_timecourse(timecourse | timecourses, ax=None, log=False, errorbars=True, by=None)`: curves with `sd`/`se` bands or bars, one line per sample, legend from the coordinates
- `plot_nca(timecourse, result)`: the diagnostic figure of the old `TimecoursePK.figure()`: linear and logarithmic panel, the `auc_last` area, the extrapolated area, the terminal regression line with the points it used, `cmax`/`tmax`, `clast`, the flags in the title; `plot_nca_grid(timecourses, result, ncols)` for many samples
- `plot_fit(fit_result, x, y)`: data, fitted curve on a fine grid, bootstrap band when present, residual panel (raw and standardized)
- `plot_parameters(result, parameter, by=None)`: strip or box plot of a parameter over a sample dimension or a grouping coordinate, geometric mean with CI
- `plot_dose_proportionality(fit_result)`: log–log parameter against dose with the power fit and the CI of the slope
- `plot_forest(meta_result)`: per study effect with CI, pooled fixed and random effects as diamonds, weights
- `plot_ratio(ratio_results, limits=(0.8, 1.25))`: GMRs with CI against the BE or DDI limits, class annotations
- `plot_goodness_of_fit`, `plot_bland_altman`: predicted against observed of a fit, in the style of sbmlsim

## 6. Repository, tooling, documentation, tests, release

### Removed

`src/pkdb_analysis/` (all of it), `docs_builder/`, `tests/` except `tests/data/pk/*.tsv` and `*.csv` (reference timecourses, moved to `tests/data/`), `MANIFEST.in`, `RELEASE.md` (the content moves to `docs/development.md`), `.github/workflows/main.yml`, `.github/CONTRIBUTING.rst`, the mypy configuration in `tox.ini`. The dependabot bundler branches on `origin` are deleted. `release-notes/0.*.md` stay as history.

### Tooling

Copied from sbmlsim and adapted:

- `pyproject.toml`: `name = "pkpdutils"`, `requires-python = ">=3.13"`, classifiers 3.13 and 3.14, `Development Status :: 5 - Production/Stable`, the runtime dependencies above, `dev` extra with `bump-my-version`, `ruff`, `pre-commit`, `ty`, `tox`, `pytest`, `pytest-cov`, `pytest-xdist`, `zensical`, `mkdocstrings-python`; `[tool.ty]` with `root = ["./src", "."]`, `include = ["src", "tests", "examples", "scripts"]`, `error-on-warning = true`; `[tool.pytest.ini_options]` with `addopts = "-n auto"`; `[project.urls]` on `matthiaskoenig.github.io/pkpdutils`
- `.python-version` = `3.14`; `tox.ini` envs `ty, py3.{13,14}`; `.ruff.toml` with the sbmlsim rule set (`E4 E7 E9 F W I D UP B C4 SIM RET G PIE RUF`, google docstrings, `examples/**` and `tests/**` exempt from `D`); `.pre-commit-config.yaml` with the hooks, ruff and ty; `.bumpversion.toml` with `tag = false` updating `src/pkpdutils/__init__.py` and `CITATION.cff`; `CITATION.cff`; `.zenodo.json` updated to the new name; `CLAUDE.md` in the sbmlsim structure (project, commands, architecture, conventions); `conftest.py` selecting `Agg` and putting the repository on `sys.path`; `scripts/llms_txt.py`

### GitHub

- workflows: `ci-cd.yml` (test matrix linux 3.13 and 3.14, windows and macos 3.14, aggregated `tests` job, release on tag with trusted publishing to the PyPI project `pkpdutils`, GitHub release from `release-notes/<tag>.md`, `sync-main`), `ruff.yml`, `ty.yml`, `docs.yml` (Zensical build, agent files, publish from `develop`)
- `.github/rulesets/develop.json`, `main.json`, `tags.json` and `apply.sh`; `CODEOWNERS`, `pull_request_template.md`, `dependabot.yml`; the issue templates stay
- branches: `develop` default, every change through a pull request with the checks `tests`, `ruff`, `ty`, `docs`; `main` tracks the latest release and is fast-forwarded by `sync-main`
- manual steps of the maintainer: rename the repository to `pkpdutils`, create the PyPI project `pkpdutils` with a trusted publisher and the `pypi` environment, apply the rulesets, enable GitHub Pages for the documentation workflow

### Documentation

Zensical site from `docs/`, configured in `zensical.toml` with `pymdownx.arithmatex` and MathJax for formulas, API reference by mkdocstrings (`docs/api/<module>.md` contains `::: pkpdutils.<module>`), `scripts/llms_txt.py` for `llms.txt`, `llms-full.txt` and the markdown of every page.

Pages: `index.md`, `installation.md`, user guide `timecourses.md`, `nca.md`, `uncertainty.md`, `fitting.md`, `pd.md`, `statistics.md`, `plotting.md`, `units.md`, `glossary.md`, `references.md`, `development.md`, `api/`.

Every user guide page has the same structure:

1. **Concepts**: what the quantities mean, when they apply (route, single dose or steady state), assumptions and pitfalls
2. **Math**: the formulas in LaTeX and short derivations (trapezoid rules, terminal regression and the adjusted R² rule, extrapolation, MRT, delta method, bootstrap, Hedges' g, DerSimonian–Laird, TOST, GMR interval)
3. **Parameter table**: symbol, name, definition, formula, unit dimension, dose dependence, route restrictions
4. **API and examples**: runnable code from `examples/` with output and figures
5. **References**: numbered citations collected in `references.md` (Gabrielsson & Weiner; Rowland & Tozer; Gibaldi & Perrier; the Phoenix WinNonlin NCA documentation; FDA clinical DDI guidance 2020 and bioequivalence guidance; EMA DDI guideline 2012; Smith et al. 2000; Schuirmann 1987; Hedges 1981; DerSimonian & Laird 1986; Efron & Tibshirani)

Docstrings are google style, carry the formula and a citation key, so the API pages and the guide agree. `glossary.md` lists every parameter name of the result datasets with its symbol and its page. Markdown has no hard line wraps.

### Examples

`examples/` is a package at the root of the repository, not part of the wheel, run as modules: `nca_single.py`, `nca_batch.py`, `nca_from_sbmlsim.py` (optional sbmlsim import with `# ty: ignore[unresolved-import]`), `group_uncertainty.py`, `steady_state.py`, `fitting_exponential.py`, `emax.py`, `dose_proportionality.py`, `bioequivalence.py`, `ddi.py`, `meta_analysis.py`. Examples write into the working directory and never open a window; `tests/examples/test_examples.py` runs them.

### Tests

- `tests/nca/`: analytic curves with known `kel`, `auc` and `auc_inf` in closed form; the reference values of the old `TimecoursePK` on the fixtures of `tests/data/`, frozen as JSON before the old package is deleted, reproduced with `auc_method=LINEAR` and `terminal=TerminalPhase(method=ALL_AFTER_TMAX)`; `NaN` values, positive slope, too few points, unsorted input, iv against oral, infusion, steady state against superposition, `EFFECT` kind; bootstrap reproducible with a seed and the delta method against the bootstrap within tolerance; the batch call equal to the loop over `nca_single`
- `tests/fit/`: parameters recovered from synthetic noisy data for every model, CI coverage on repeated synthetic data, `compare_models` picks the true model, bootstrap and Jacobian intervals agree, multi start reaches the global optimum from a bad start, process pool equal to serial
- `tests/stats/`: tests against `scipy.stats` references, the bioequivalence textbook example (Chow & Liu), DDI thresholds at the boundaries and with intervals, meta-analysis against reference numbers from the R packages `esc` and `metafor` stored as fixtures, summary statistics variants against the individual variants
- `tests/plot/`: every function creates a figure under `Agg` without a window
- `tests/test_timecourse.py`, `tests/test_units.py`: validation, constructors, round trips, `from_xresult` on a synthetic dataset

### Release

`1.0.0` is released from `develop` by the sbmlsim steps: release branch, `release-notes/1.0.0.md` (Breaking: the rename, the removal of everything PK-DB specific, the new API; Features; a migration note from `pkdb_analysis.pk.pharmacokinetics`), `tox run-parallel`, `ruff check`, `tox r -e ty`, `uvx bump-my-version bump major`, pull request, tag on `develop`, PyPI, GitHub release, `main` fast-forward, installation test with `uv venv --python 3.14`, Zenodo citation update.

## Implementation phases

Each phase is a pull request into `develop` and leaves the checks green:

1. **Scaffold**: freeze the reference values of the old NCA, delete the old package and docs, new `pyproject.toml`, tooling, workflows, rulesets, `CLAUDE.md`, empty `pkpdutils` with `units.py`, `console.py`, `log.py`, docs skeleton with the development page
2. **Data model**: `timecourse.py` with `Timecourse`, `Dose`, `DosingRegimen`, `Timecourses` and the constructors, tests, `timecourses.md`, `units.md`
3. **NCA**: options, vectorized AUC and terminal regression, parameters, flags, steady state and superposition, `nca.md`, `glossary.md`, `plot_timecourse`, `plot_nca`, examples
4. **Uncertainty**: bootstrap, delta method, `summarize`, `uncertainty.md`, example
5. **Fitting**: engine, models, proportionality, PD descriptive, `plot_fit`, `plot_dose_proportionality`, `fitting.md`, `pd.md`, examples
6. **Statistics**: samples, tests, ratios, bioequivalence, DDI, meta-analysis, `plot_parameters`, `plot_forest`, `plot_ratio`, `statistics.md`, examples
7. **Release**: `index.md`, README, release notes, citation, 1.0.0

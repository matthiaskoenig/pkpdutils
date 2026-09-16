# Glossary

The names used for the variables of the result datasets, with their symbols and the page that defines them. Units are derived from the units of the input; `value` is the unit of the measured values, `time` the unit of the times, `dose` the unit of the doses.

| name | symbol | meaning | unit | page |
| --- | --- | --- | --- | --- |
| `auc_last` | \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\) | area under the curve to the last positive value | value·time | [NCA](nca.md) |
| `auc_inf_obs`, `auc_inf_pred` | \(\mathrm{AUC}_{0\text{-}\infty}\) | area extrapolated to infinity, observed or predicted last value | value·time | [NCA](nca.md) |
| `auc_extrap_fraction` | | extrapolated fraction of \(\mathrm{AUC}_{0\text{-}\infty}\) | – | [NCA](nca.md) |
| `aumc_last`, `aumc_inf` | \(\mathrm{AUMC}\) | area under the first moment curve | value·time² | [NCA](nca.md) |
| `mrt` | \(\mathrm{MRT}\) | mean residence time | time | [NCA](nca.md) |
| `cmax`, `tmax` | \(C_\mathrm{max}\), \(t_\mathrm{max}\) | maximum and its time | value, time | [NCA](nca.md) |
| `cmin`, `tmin` | \(C_\mathrm{min}\), \(t_\mathrm{min}\) | minimum and its time | value, time | [NCA](nca.md) |
| `clast`, `tlast` | \(C_\mathrm{last}\), \(t_\mathrm{last}\) | last positive value and its time | value, time | [NCA](nca.md) |
| `c0` | \(C_0\) | back-extrapolated value at time 0 (bolus) | value | [NCA](nca.md) |
| `cmax_half`, `tmax_half` | | half maximum during absorption | value, time | [NCA](nca.md) |
| `lambda_z` | \(\lambda_z\) | terminal rate constant | 1/time | [NCA](nca.md) |
| `thalf` | \(t_{1/2}\) | terminal half-life | time | [NCA](nca.md) |
| `lambda_z_n_points`, `lambda_z_t_first`, `lambda_z_t_last`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`, `lambda_z_stderr` | | regression diagnostics (`lambda_z_t_first`, `lambda_z_t_last`: first and last point of the terminal window; `lambda_z_stderr`: standard error of the slope of the terminal regression) | | [NCA](nca.md) |
| `lambda_z_span` | | half-lives the terminal phase covers, \((t_\mathrm{last} - t_\mathrm{first}) / t_{1/2}\); below 2 the sample is flagged `SPAN_LOW` | – | [NCA](nca.md) |
| `cl`, `cl_f` | \(\mathrm{CL}\), \(\mathrm{CL}/F\) | clearance, relative to the fraction absorbed | l/h | [NCA](nca.md) |
| `vz`, `vz_f` | \(V_z\), \(V_z/F\) | terminal volume of distribution | l | [NCA](nca.md) |
| `vss` | \(V_\mathrm{ss}\) | steady state volume of distribution | l | [NCA](nca.md) |
| `auc_inf_dn`, `cmax_dn` | | dose normalized exposure and maximum | value·time/dose, value/dose | [NCA](nca.md) |
| `auc_tau` | \(\mathrm{AUC}_{0\text{-}\tau}\) | area over a dosing interval | value·time | [NCA](nca.md) |
| `cmin_ss`, `cmax_ss`, `ctrough`, `cavg` | \(C_\mathrm{min,ss}\), \(C_\mathrm{max,ss}\), \(C_\mathrm{trough}\), \(C_\mathrm{avg}\) | minimum, maximum, trough and average over the interval | value | [NCA](nca.md) |
| `fluctuation`, `swing` | | peak-trough fluctuation and swing over the interval | – | [NCA](nca.md) |
| `accumulation_ratio` | \(R_\mathrm{pred}\) | accumulation at steady state, predicted from \(\lambda_z\) | – | [NCA](nca.md) |
| `accumulation_ratio_obs` | \(R_\mathrm{obs}\) | observed accumulation, last over first dosing interval of a protocol | – | [NCA](nca.md) |
| `cl_ss`, `cl_ss_f` | \(\mathrm{CL}_\mathrm{ss}\), \(\mathrm{CL}_\mathrm{ss}/F\) | clearance at steady state (`_f`: extravascular) | l/h | [NCA](nca.md) |
| `n_doses`, `tau` | \(K\), \(\tau\) | number of doses of the protocol, length of the last dosing interval | –, time | [NCA](nca.md) |
| `interval_auc`, `interval_cmax`, `interval_tmax`, `interval_cmin`, `interval_ctrough`, `interval_c_start`, `interval_cavg`, `interval_fluctuation`, `interval_swing`, `interval_n_points` | | parameters of every single dosing interval, over the extra dimension `interval` | value·time, value, time, value, value, value, value, –, –, – | [NCA](nca.md) |
| `interval_start`, `interval_end`, `interval_dose` | | bounds and dose amount of a dosing interval (columns of `NCAResult.intervals()`) | time, time, dose | [NCA](nca.md) |
| `e0`, `emax_obs`, `temax` | | baseline, maximum effect and its time | value, value, time | [NCA](nca.md) |
| `auec_last`, `auec_baseline` | \(\mathrm{AUEC}\) | area under the effect curve, raw and baseline corrected | value·time | [NCA](nca.md) |
| `emax_baseline`, `time_above` | | baseline corrected maximum, time above a threshold | value, time | [NCA](nca.md) |
| `auec_tau`, `emin_ss`, `emax_ss`, `eavg`, `time_above_tau` | | steady state effect parameters of the last dosing interval, and `interval_auec`, `interval_emax`, `interval_temax`, `interval_emin`, `interval_eavg`, `interval_time_above` per interval | value·time, value, value, value, time | [Pharmacodynamics](pd.md) |
| `flags` | | `NCAFlag` bits, including `SPAN_LOW` of the terminal phase and `INCOMPLETE_INTERVAL`, `EXTRAPOLATED_TROUGH` of a multiple dose analysis | – | [NCA](nca.md) |
| `x_sd`, `x_se` | | standard deviation over subjects and standard error of the mean of a parameter `x` | unit of `x` | [Uncertainty](uncertainty.md) |
| `x_ci_low`, `x_ci_high` | | confidence interval of the estimate of a parameter `x` at `ci_level` | unit of `x` | [Uncertainty](uncertainty.md) |
| `x_pi_low`, `x_pi_high` | | percentile interval of individual curves of a parameter `x`, `BootstrapSpread.SD` draws only | unit of `x` | [Uncertainty](uncertainty.md) |
| `x_geomean`, `x_geocv` | | geometric mean and geometric coefficient of variation over subjects of a parameter `x` (log-normal parameters) | unit of `x`, - | [Uncertainty](uncertainty.md) |
| `x_median`, `x_q25`, `x_q75`, `x_min`, `x_max`, `x_n` | | median, quartiles, smallest and largest value and count of finite values of a parameter `x` (summary only) | unit of `x`, - | [Uncertainty](uncertainty.md) |
| `x_cv` | \(\mathrm{CV}\) | coefficient of variation of a parameter `x` over the samples, \(\mathrm{sd}/\lvert \bar x \rvert\), a fraction (summary only) | 1 | [Uncertainty](uncertainty.md) |
| `n` | | number of subjects (group data) or of samples along the reduced dimension (summary) | – | [Uncertainty](uncertainty.md) |
| `auc_partial` | \(\mathrm{AUC}_{t_1\text{-}t_2}\) | area under the curve between two times, from `partial_auc` | value·time | [Uncertainty](uncertainty.md) |
| `p_se` | | standard error of a fitted parameter `p`, from the Jacobian or the residual bootstrap | unit of `p` | [Curve fitting](fitting.md) |
| `p_ci_low`, `p_ci_high` | | confidence interval of a fitted parameter `p` at `ci_level` | unit of `p` | [Curve fitting](fitting.md) |
| `p_cv` | | relative standard error of a fitted parameter, \(\mathrm{se}(p) / \lvert p \rvert\), a fraction | 1 | [Curve fitting](fitting.md) |
| `cost` | | the objective of the fit at the optimum, \(\tfrac12 \sum \rho(r^2)\) | – | [Curve fitting](fitting.md) |
| `r2`, `rmse` | \(R^2\), \(\mathrm{RMSE}\) | goodness of fit on the unweighted residuals (`rmse` carries the unit of the values, it is reported as dimensionless) | –, value | [Curve fitting](fitting.md) |
| `aic`, `aicc`, `bic` | | information criteria of the fit, with \(K = k + 1\) estimated parameters | – | [Curve fitting](fitting.md) |
| `n_points`, `n_parameters` | \(n\), \(k\) | points used in the fit and free model parameters | – | [Curve fitting](fitting.md) |
| `n_starts_converged`, `n_bootstrap` | | starts which converged and converged bootstrap replicates | – | [Curve fitting](fitting.md) |
| `x_data`, `y_data`, `sd_data` | | the data of the fit, per point | unit of `x`, of `y` | [Curve fitting](fitting.md) |
| `y_pred`, `residuals` | | prediction and weighted residual, per point | unit of `y`, – | [Curve fitting](fitting.md) |
| `correlation` | | correlation matrix of the fitted parameters | – | [Curve fitting](fitting.md) |
| `akaike_weight` | \(w_i\) | probability that a model is the best of the compared set | – | [Curve fitting](fitting.md) |
| `bound_low`, `bound_high` | | acceptance bounds of the exponent in the dose proportionality criterion | – | [Curve fitting](fitting.md) |
| `gmr`, `log_ratio`, `se_log` | \(\mathrm{GMR}\) | geometric mean ratio test / reference, its logarithm and the standard error of the logarithm | –, – | [Statistics](statistics.md) |
| `effect`, `cohen_d`, `hedges_g` | \(d\), \(g\) | effect of a comparison and the standardized effect sizes | unit of the parameter (ratio: –), – | [Statistics](statistics.md) |
| `bioequivalent`, `p_lower`, `p_upper`, `cv_intra`, `p_period`, `p_sequence` | | verdict and the two one-sided p values of the bioequivalence test, within-subject CV, period and carryover p values of a crossover | – | [Statistics](statistics.md) |
| `kind`, `strength`, `uncertain` | | class of an interaction (inhibitor, inducer), its strength and whether the interval spans a boundary | – | [Statistics](statistics.md) |
| `estimate`, `variance`, `weight_fixed`, `weight_random` | \(\theta_i\), \(v_i\), \(w_i\) | effect of a study, its variance and its normalized weights in the pooling | – | [Statistics](statistics.md) |
| `q`, `i2`, `h2`, `tau2` | \(Q\), \(I^2\), \(H^2\), \(\tau^2\) | heterogeneity statistics of a meta-analysis | –, %, –, – | [Statistics](statistics.md) |

## Statistics of the parameter tables

The columns `summary_table` writes, every one of them read from the summary of `ParameterResult.summarize(dim)` and formatted with `digits` significant digits as a string. `stats=` selects them and their order; the default is `n`, `mean`, `sd`, `cv`, `geomean`, `geocv`, `median`, `min`, `max`. A statistic a parameter does not carry, such as the standard deviation of a parameter read from the sampling grid, is an empty cell.

| statistic | variable | meaning |
| --- | --- | --- |
| `n` | `x_n` | number of samples at which the parameter is finite |
| `mean` | `x` | arithmetic mean over the samples |
| `sd` | `x_sd` | standard deviation over the samples |
| `se` | `x_se` | standard error of the mean, \(\mathrm{sd}/\sqrt{n}\) |
| `cv` | `x_cv` | coefficient of variation, a fraction in the result and a percentage in the table |
| `geomean` | `x_geomean` | geometric mean (log-normal parameters only) |
| `geocv` | `x_geocv` | geometric coefficient of variation, a percentage in the table |
| `median`, `q25`, `q75` | `x_median`, `x_q25`, `x_q75` | median and quartiles |
| `min`, `max` | `x_min`, `x_max` | smallest and largest value |
| `range` | `x_min`, `x_max` | `min - max` in one cell |

The columns of the other tables: `ratio_table` writes `parameter`, `unit`, `n_test`, `n_reference`, `gmr`, `ci_low`, `ci_high`, `ci_level` and, for a bioequivalence result, `cv_intra`, `limits` and `bioequivalent`; `ddi_table` writes `ratio`, `ci_low`, `ci_high`, `kind`, `strength`, `uncertain` and `source` per parameter; `proportionality_table` writes `slope`, `ci_low`, `ci_high`, `bound_low`, `bound_high`, `dose_low`, `dose_high` and `verdict`.

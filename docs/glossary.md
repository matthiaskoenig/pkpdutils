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
| `lambda_z_n_points`, `lambda_z_t_first`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`, `lambda_z_se` | | regression diagnostics | | [NCA](nca.md) |
| `cl`, `cl_f` | \(\mathrm{CL}\), \(\mathrm{CL}/F\) | clearance, relative to the fraction absorbed | l/h | [NCA](nca.md) |
| `vz`, `vz_f` | \(V_z\), \(V_z/F\) | terminal volume of distribution | l | [NCA](nca.md) |
| `vss` | \(V_\mathrm{ss}\) | steady state volume of distribution | l | [NCA](nca.md) |
| `auc_inf_dn`, `cmax_dn` | | dose normalized exposure and maximum | value·time/dose, value/dose | [NCA](nca.md) |
| `auc_tau` | \(\mathrm{AUC}_{0\text{-}\tau}\) | area over a dosing interval | value·time | [NCA](nca.md) |
| `cmin_ss`, `ctrough`, `cavg` | | minimum, trough and average over the interval | value | [NCA](nca.md) |
| `fluctuation`, `swing` | | peak–trough fluctuation and swing | – | [NCA](nca.md) |
| `accumulation_ratio` | \(R\) | accumulation at steady state | – | [NCA](nca.md) |
| `cl_ss` | \(\mathrm{CL}_\mathrm{ss}\) | clearance at steady state | l/h | [NCA](nca.md) |
| `e0`, `emax_obs`, `temax` | | baseline, maximum effect and its time | value, value, time | [NCA](nca.md) |
| `auec_last`, `auec_baseline` | \(\mathrm{AUEC}\) | area under the effect curve, raw and baseline corrected | value·time | [NCA](nca.md) |
| `emax_baseline`, `time_above` | | baseline corrected maximum, time above a threshold | value, time | [NCA](nca.md) |
| `flags` | | `NCAFlag` bits | – | [NCA](nca.md) |

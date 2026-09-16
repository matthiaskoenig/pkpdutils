# Validation table

The comparison of `pkpdutils.nca` against the published reference values of `tests/data/validation/reference.json`, written by `scripts/validation.py`. The table is part of [Validation](validation.md), which explains the datasets, the option mapping and the known differences.

| dataset | comparator | rule | parameter | n subjects | max relative deviation | tolerance | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| theoph | Phoenix WinNonlin | linear | `cmax` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `tmax` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `tlast` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `clast` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `auc_last` | 12 | < 1e-12 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `auc_all` | 12 | < 1e-12 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `auc_inf_obs` | 12 | 4.3e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `auc_inf_pred` | 12 | 4.3e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `auc_extrap_fraction` | 12 | 3.5e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `aumc_last` | 12 | 4.9e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `aumc_inf` | 12 | 3.1e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `mrt` | 12 | 3.9e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `lambda_z` | 12 | 4.7e-09 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `lambda_z_r2` | 12 | 4.3e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `lambda_z_r2_adj` | 12 | 4.8e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `lambda_z_n_points` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `lambda_z_t_first` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `lambda_z_t_last` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `thalf` | 12 | 7.7e-11 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `cmax_dn` | 12 | < 1e-12 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `auc_inf_dn` | 12 | 1.5e-09 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `tlag` | 9 (3 known differences) | 0 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `cl_f` | 12 | 2.1e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear | `vz_f` | 12 | 1.8e-10 | 1e-06 | winnonlin-theoph-linear |
| theoph | Phoenix WinNonlin | linear_log | `cmax` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `tmax` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `tlast` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `clast` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `auc_last` | 12 | 4.0e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `auc_all` | 12 | 4.0e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `auc_inf_obs` | 12 | 3.3e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `auc_inf_pred` | 12 | 4.4e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `auc_extrap_fraction` | 12 | 3.8e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `aumc_last` | 12 | 4.1e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `aumc_inf` | 12 | 2.5e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `mrt` | 12 | 3.7e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `lambda_z` | 12 | 4.7e-09 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `lambda_z_r2` | 12 | 4.3e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `lambda_z_r2_adj` | 12 | 4.8e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `lambda_z_n_points` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `lambda_z_t_first` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `lambda_z_t_last` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `thalf` | 12 | 7.7e-11 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `cmax_dn` | 12 | < 1e-12 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `auc_inf_dn` | 12 | 1.5e-09 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `tlag` | 9 (3 known differences) | 0 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `cl_f` | 12 | 2.8e-10 | 1e-06 | winnonlin-theoph-linear-log |
| theoph | Phoenix WinNonlin | linear_log | `vz_f` | 12 | 1.9e-10 | 1e-06 | winnonlin-theoph-linear-log |
| indometh | Phoenix WinNonlin | linear | `cmax` | 6 | 0 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `tmax` | 6 | 0 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `tlast` | 6 | 0 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `clast` | 6 | 0 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `auc_last` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `auc_all` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `auc_inf_obs` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `auc_inf_pred` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `auc_extrap_fraction` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `aumc_last` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `aumc_inf` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `mrt` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `lambda_z` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `lambda_z_r2` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `lambda_z_r2_adj` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `lambda_z_n_points` | 6 | 0 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `lambda_z_t_first` | 6 | 0 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `lambda_z_t_last` | 6 | 0 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `thalf` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `cmax_dn` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `auc_inf_dn` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `c0` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `auc_back_extrap_fraction` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `cl` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `vz` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear | `vss` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear |
| indometh | Phoenix WinNonlin | linear_log | `cmax` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `tmax` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `tlast` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `clast` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `auc_last` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `auc_all` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `auc_inf_obs` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `auc_inf_pred` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `auc_extrap_fraction` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `aumc_last` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `aumc_inf` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `mrt` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `lambda_z` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `lambda_z_r2` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `lambda_z_r2_adj` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `lambda_z_n_points` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `lambda_z_t_first` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `lambda_z_t_last` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `thalf` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `cmax_dn` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `auc_inf_dn` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `c0` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `auc_back_extrap_fraction` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `cl` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `vz` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| indometh | Phoenix WinNonlin | linear_log | `vss` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-log |
| theoph | PKNCA | linear_log | `clast` | 2 | 0 | 1e-04 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `cmax` | 2 | 0 | 1e-04 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `lambda_z` | 2 | 4.6e-07 | 1e-04 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `tlast` | 2 | 0 | 1e-04 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `tmax` | 2 | 0 | 1e-04 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `auc_last` | 1 | 7.8e-11 | 1e-04 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `auc_partial` | 1 | 4.8e-09 | 1e-04 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `cmax_geomean` | 1 | 4.4e-04 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `cmax_geocv` | 1 | 1.3e-03 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `auc_last_geomean` | 1 | 5.0e-04 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `auc_last_geocv` | 1 | 1.7e-03 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `auc_inf_obs_geomean` | 1 | 1.6e-03 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `auc_inf_obs_geocv` | 1 | 9.0e-04 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `thalf` | 1 | 5.8e-05 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `thalf_sd` | 1 | 2.3e-03 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `tmax_median` | 1 | 4.4e-03 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `tmax_min` | 1 | 0 | 1e-02 | pknca-theoph-vignette |
| theoph | PKNCA | linear_log | `tmax_max` | 1 | 0 | 1e-02 | pknca-theoph-vignette |

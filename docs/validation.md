# Validation

`pkpdutils` computes the same non-compartmental parameters that Phoenix WinNonlin, PKNCA, NonCompart and PKanalix compute, and an analysis is only worth as much as the evidence that it agrees with them. This page is that evidence: two public datasets, the published per-subject results of two other tools, the mapping of their settings onto [`NCAOptions`](api/nca.options.md), and the deviation of every parameter of every subject.

The comparison runs in the test suite (`tests/nca/test_validation.py`) and as a script (`scripts/validation.py`), both over the same reference file, so the numbers below are reproduced on every commit rather than transcribed once.

## The datasets

| dataset | file | subjects | route | dose | concentrations | times |
| --- | --- | --- | --- | --- | --- | --- |
| theophylline | `tests/data/validation/theoph.csv` | 12 | oral | 320 mg | mg/L | 0 to 24.65 h, 11 samples |
| indomethacin | `tests/data/validation/indometh.csv` | 6 | intravenous bolus, and the same profiles as a 0.25 h infusion | 25 mg | µg/mL | 0.25 to 8 h, 11 samples |

Both are the datasets `datasets::Theoph` and `datasets::Indometh` of R, copied verbatim from the [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) mirror; `tests/data/validation/README.md` names the source, the license (GPL-2 / GPL-3, as all of R) and the original studies. They are the two datasets the other tools publish their own validation against, which is the only reason to pick a theophylline study from 1994 and an indomethacin study from 1976.

**The dose of the theophylline dataset.** The dataset carries the dose as `Dose` in mg/kg together with the body weight `Wt` in kg. The dose in mg is `Dose * Wt`, which is 267.8 mg for subject 9 and between 318.6 and 320.7 mg for the other eleven. The published WinNonlin analysis used a flat 320 mg for every subject, so the validation does the same. Only `cl_f`, `vz_f`, `cmax_dn` and `auc_inf_dn` depend on the dose at all; every area, concentration and time parameter is unaffected by the choice.

## The reference values

No R installation was available, so every number is transcribed from a published source rather than computed here. Two sources are used.

**Phoenix WinNonlin 6.3 and 7.0**, through the validation report of the NonCompart R package (Han 2018). The report compares NonCompart against WinNonlin on exactly these two datasets and publishes the raw WinNonlin output as CSV files, one per dataset and trapezoidal rule, with 8 to 15 significant digits per number. Those CSV files are the reference of the five WinNonlin cases: `Final_Parameters_Pivoted_Theoph_Linear.csv`, `..._Theoph_Log.csv`, `..._Indometh_Linear.csv`, `..._Indometh_Log.csv` and `..._Indometh_Linear_Infusion.csv`, the last one the indomethacin profiles analysed as a 0.25 h infusion. They cover 24 parameters per subject of the theophylline dataset and 26 of the indomethacin one, which carries `c0`, the back extrapolated fraction and the clearance and volumes of an intravenous dose on top; the infusion case carries 24, since an infusion has no \(C_0\) to back-extrapolate.

**PKNCA**, through its theophylline vignette. The vignette prints the per-subject results only for the subjects 1 and 6 (`cmax`, `tmax`, `tlast`, `clast`, `lambda_z` and, for subject 6, `auc_last`) and a summary over all twelve subjects (the geometric mean and geometric coefficient of variation of `cmax`, `auc_last` and `auc_inf_obs`, the arithmetic mean and standard deviation of `thalf`, and the median with the range of `tmax`). The vignette also prints the `auclast` of subject 1 over the automatic interval 0 to 24 h, which is compared through `partial_auc` over the window PKNCA truncates that interval to, see "The known differences". Those are the numbers the reference file holds; the vignette prints nothing per subject for `cl`, `vz` or `mrt`, so nothing is recorded for them, and none is invented.

Every number sits in `tests/data/validation/reference.json` with its value, its unit, the identifier of its source and the tolerance of its comparison; the `sources` section of that file carries the URL, the retrieval date and the settings of every source.

## The option mapping

The comparators expose the same two decisions `pkpdutils` exposes, under different names.

| decision | WinNonlin / NonCompart | PKNCA | `pkpdutils` |
| --- | --- | --- | --- |
| trapezoidal rule, linear | "Linear Trapezoidal Linear Interpolation", `down="Linear"` | `auc.method="linear"` | `AUCMethod.LINEAR` |
| trapezoidal rule, mixed | "Linear Up Log Down", `down="Log"` | `auc.method="lin up/log down"` (the default) | `AUCMethod.LINEAR_LOG` |
| terminal phase | "Best Fit", largest adjusted R², at least 3 points, a window with more points wins within 0.0001 | largest adjusted R², `min.hl.points=3`, the same tolerance | `TerminalPhase(method=TerminalMethod.BEST_FIT, min_points=3, tie_tolerance=1e-4)` |
| the peak in the terminal phase | excluded for an extravascular dose, included for an intravenous bolus | `allow.tmax.in.half.life=FALSE` | `exclude_cmax=True` / `exclude_cmax=False` |
| C(0) of a bolus | log-linear back extrapolation of the first two values | `c0` by back extrapolation | `C0Method.LOG_BACK_EXTRAPOLATION` (the default) |

The one setting which is not the same for both datasets is the last one: Phoenix lets the point of the maximum enter the terminal regression for an intravenous bolus, where the maximum is the first sample, and keeps it out for an extravascular dose. So the theophylline analysis runs with `exclude_cmax=True` and the indomethacin analysis with `exclude_cmax=False`. This matters for exactly one subject of the indomethacin dataset (subject 4, whose regression then uses all eleven points instead of ten), and getting it wrong moves that subject's `thalf` and `vz` by about 6 %, which is what makes the mapping worth writing down.

```python
# not executed
import pandas as pd

from pkpdutils import (
    AUCMethod,
    NCAOptions,
    Route,
    TerminalMethod,
    TerminalPhase,
    Timecourses,
    nca,
)

frame = pd.read_csv("tests/data/validation/theoph.csv")
frame["dose_amount"] = 320.0
batch = Timecourses.from_dataframe(
    frame,
    sample=["Subject"],
    time_unit="hr",
    unit="mg/L",
    time="Time",
    value="conc",
    dose_amount="dose_amount",
    dose_unit="mg",
    route=Route.ORAL,
    substance="theophylline",
)
options = NCAOptions(
    auc_method=AUCMethod.LINEAR_LOG,
    terminal=TerminalPhase(
        method=TerminalMethod.BEST_FIT, min_points=3, exclude_cmax=True
    ),
)
result = nca(batch, options=options)
```

The indomethacin analysis is the same call with `route=Route.IV_BOLUS`, `unit="ug/mL"`, a dose of 25 mg and `exclude_cmax=False`; the infusion case is that call with `route=Route.IV_INFUSION` and a `dose_duration` of 0.25 h, the run the report makes with `adm="Infusion", dur=0.25`.

The infusion case is what validates two conventions of an infusion which the bolus case cannot: the zero inserted at the dose time of a curve whose first sample comes later, and the mean residence time corrected by half the duration. The indomethacin profiles start at 0.25 h, so `auc_last` of subject 1 is 1.741 with the inserted zero and 1.554 without it, and its `mrt` is 3.663 h with the correction and 3.788 h without; both agree with WinNonlin to machine precision.

## The comparison

One row per case and parameter, the largest relative deviation over the subjects of the case. The table is written by `scripts/validation.py`.

<!-- validation-table:start -->

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
| theoph | Phoenix WinNonlin | linear | `tlag` | 12 | 0 | 1e-06 | winnonlin-theoph-linear |
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
| theoph | Phoenix WinNonlin | linear_log | `tlag` | 12 | 0 | 1e-06 | winnonlin-theoph-linear-log |
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
| indometh | Phoenix WinNonlin | linear | `cmax` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `tmax` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `tlast` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `clast` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `auc_last` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `auc_all` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `auc_inf_obs` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `auc_inf_pred` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `auc_extrap_fraction` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `aumc_last` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `aumc_inf` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `mrt` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `lambda_z` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `lambda_z_r2` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `lambda_z_r2_adj` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `lambda_z_n_points` | 5 (1 known difference) | 0 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `lambda_z_t_first` | 5 (1 known difference) | 0 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `lambda_z_t_last` | 6 | 0 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `thalf` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `cmax_dn` | 6 | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `auc_inf_dn` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `cl` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `vz` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
| indometh | Phoenix WinNonlin | linear | `vss` | 5 (1 known difference) | < 1e-12 | 1e-06 | winnonlin-indometh-linear-infusion |
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

<!-- validation-table:end -->

## The tolerances

| reference | printed precision | tolerance | largest deviation observed |
| --- | --- | --- | --- |
| WinNonlin, indomethacin | full double precision | 1e-6 | below 1e-12 (the last bits of the platform, 3.7e-15 on linux) |
| WinNonlin, theophylline | 8 to 10 significant digits | 1e-6 | 4.7e-9 |
| PKNCA, per subject | 5 to 9 significant digits | 1e-4 | 4.6e-7 |
| PKNCA, summary over 12 subjects | 3 significant digits | 1e-2 | 4.4e-3 |

The tolerance of a WinNonlin number is the machine precision tolerance of 1e-6; the indomethacin comparison reaches double precision because the report published the full mantissa, and the theophylline comparison is limited by the eight digits the report printed, not by the analysis. The tolerance of a transcribed number is the rounding bound of the digits it was printed with, rounded up: a value printed with three significant digits carries a relative rounding error of up to 5e-3, so the summary comparison uses 1e-2.

## The known differences

**The terminal window of the infusion run.** WinNonlin starts the terminal regression of subject 4 of the infusion case after the end of the infusion, at 0.5 h with ten points, where it starts at the observed maximum at 0.25 h with eleven points for the very same profile analysed as a bolus. `pkpdutils` has no rule which keeps the samples taken during an infusion out of the regression, so it regresses the same eleven points in both runs; every parameter which reads the terminal window follows (`lambda_z` by 6 %, `thalf` by 6 %, `vz` by 5 %, `auc_inf_obs` and `cl` by 0.4 %). The other five subjects and every parameter which does not read the window agree to machine precision. The fifteen entries of subject 4 are marked `xfail(strict=True)` in `tests/nca/test_validation.py`, which means the test suite fails if they ever start to agree without this page being updated.

**The end of a partial interval.** The PKNCA vignette prints `auclast` of 92.365442 for subject 1 over the interval 0 to 24 h, where `partial_auc(batch, 0.0, 24.0)` returns 146.01. The difference is the treatment of the end of the interval, not the arithmetic: PKNCA sums the trapezoids between the observations which fall inside the interval and stops at the last of them, which is at 12.12 h for this subject, while `partial_auc` interpolates the concentration at 24 h with the trapezoidal rule of the analysis and integrates to there. Over the window PKNCA actually integrated, `partial_auc(batch, 0.0, 12.12)` reproduces its number to 4.8e-9, and that is the comparison the reference file holds (the case `theoph-pknca-partial`). An analyst who wants the PKNCA convention passes the last observation inside the interval as `t_end`.

## What is not covered

- **Extravascular indomethacin.** The report also publishes a run of the indomethacin dataset as an extravascular dose. It is not compared: the bolus and the infusion run already cover the dataset, and the extravascular run adds no rule which the theophylline dataset does not exercise.
- **The `pred` variants of the clearance, the volume and the mean residence time** (`Cl_pred`, `Vz_pred`, `Vss_pred`, `MRTINF_pred`, `AUMC_%Extrap_pred`) and `MRTlast`, which WinNonlin reports and `pkpdutils` does not. `auc_inf_pred` is reported and compared.
- **Multiple dosing, steady state, urine and sparse sampling.** Both datasets are single dose plasma profiles with dense sampling. The steady state parameters are covered by `tests/nca/test_steady_state.py` and by the regression reference of `pkdb_analysis` 0.3.1 in `tests/data/reference/nca_reference.json`, not by a comparison against another tool.
- **Values below the limit of quantification.** Neither dataset carries a limit of quantification, so no BLQ rule is exercised here; `tests/nca/test_blq.py` covers them against the written rules of the tools.

## Reproducing this page

```bash
uv run pytest tests/nca/test_validation.py
uv run python scripts/validation.py
```

The first runs the comparison as a test, one test per dataset, subject and parameter. The second rewrites `docs/validation_table.md` and the table above, and exits non-zero on a deviation beyond its tolerance.

The sources are cited on [References](references.md): Phoenix WinNonlin, PKNCA, NonCompart, the NonCompart validation report and the two original studies behind the datasets.

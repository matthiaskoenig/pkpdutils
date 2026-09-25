# Benchmark datasets

`pkpdutils` is compared here against three other tools for non-compartmental analysis on the public benchmark suite of the NonCompart validation report: Phoenix WinNonlin, the commercial reference, and PKNCA and NonCompart, two independent open source implementations in R. The suite has two datasets, three routes and two trapezoidal rules, eight scenarios in all. On top of it, analytical profiles check the numbers against the exact answer, since no tool is the oracle there.

Every table on this page is the printed output of the code above it. `scripts/benchmark_datasets.py` runs the code of the page and pastes what it prints, and the test suite checks that the page is current. The code reads the files of `docs/data/benchmarks/` and needs nothing else. It uses the exporters and readers of [`pkpdutils.crosswalk`](api/crosswalk.md), which translate between the parameter names of `pkpdutils` and the result tables of the three tools.

[Validation](validation.md) is the curated version of the same comparison: one tolerance per parameter, the known differences between the tools, the option mapping and the PKNCA vignette values. It runs as a test on every commit.

## The scenarios

`docs/data/benchmarks/scenarios.csv` records every setting of a scenario next to its data, since NCA is not uniquely defined and a difference in a setting looks like a difference in the result:

| scenario | data | route | trapezoid | peak in \(\lambda_z\) |
| --- | --- | --- | --- | --- |
| theoph-linear | theophylline, 12 subjects, 320 mg | oral | linear | excluded |
| theoph-linear-log | theophylline, 12 subjects, 320 mg | oral | linear up / log down | excluded |
| indometh-bolus-linear | indomethacin, 6 subjects, 25 mg | bolus | linear | included |
| indometh-bolus-linear-log | indomethacin, 6 subjects, 25 mg | bolus | linear up / log down | included |
| indometh-infusion-linear | indomethacin, 6 subjects, 25 mg | 0.25 h infusion | linear | included |
| indometh-infusion-linear-log | indomethacin, 6 subjects, 25 mg | 0.25 h infusion | linear up / log down | included |
| indometh-oral-linear | indomethacin, 6 subjects, 25 mg | oral | linear | excluded |
| indometh-oral-linear-log | indomethacin, 6 subjects, 25 mg | oral | linear up / log down | excluded |

In every scenario the terminal phase is the best fit: the window of at least three points with the largest adjusted \(R^2\), where a window with more points wins within 0.0001. The theophylline profiles start with a sample at the dose. The indomethacin profiles start at 0.25 h, so every tool has to decide what the curve does before its first sample. After a bolus it back-extrapolates \(C_0\) from the first two samples. After an infusion or an extravascular dose it inserts a zero at the dose. The last two scenarios treat the intravenous indomethacin data as an extravascular dose. That makes no sense pharmacologically, but it is the one scenario of the suite where an extravascular curve starts after its dose.

## The reference results

- **Phoenix WinNonlin 6.3 and 7.0.** The "Final Parameters Pivoted" tables of the validation report of NonCompart (Han 2018), one per scenario, in `docs/data/benchmarks/winnonlin/`, copied unchanged from <https://github.com/asancpt/NonCompart-tests> (commit `bb77168`). The last two are named `..._Wrong_Extravascular` in the report.
- **PKNCA 0.12.1** and **NonCompart 0.8.4**, run by `scripts/benchmark_datasets.R` under R 4.5.3 (`docs/data/benchmarks/versions.csv`), in `docs/data/benchmarks/pknca/` (the long table of `as.data.frame(pk.nca(...))`) and `docs/data/benchmarks/noncompart/` (the table of `tblNCA`). NonCompart is called exactly as the report calls it. PKNCA runs with `min.hl.points = 3`, `adj.r.squared.factor = 1e-4` and `allow.tmax.in.half.life` set as in the table above. PKNCA only computes an area from the dose when there is a value at the dose, so the script inserts one where the curve has no sample there, as the other tools do: 0 after an extravascular dose and an infusion, and PKNCA's own `pk.calc.c0(method = "logslope")` after a bolus. PKNCA reads the peak after such an imputation, so `cmax` and `tmax` come from an interval without it.

The datasets are `datasets::Theoph` and `datasets::Indometh` of R. The data and the sources behind the reference values are described in `tests/data/validation/README.md`.

## The comparison

The code below is all of the comparison. `analyse` runs `pkpdutils` on a scenario with the settings of the manifest. `references` reads the three reference tables into the variable names of `pkpdutils`, with the percentages as fractions. `compare` puts the largest relative deviation over the subjects of every parameter into one column per tool.

```python
import math

import numpy as np
import pandas as pd

from pkpdutils import (
    AUCMethod,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    Timecourses,
    nca,
)
from pkpdutils.crosswalk import read_noncompart, read_pknca_results, read_winnonlin

scenarios = pd.read_csv("benchmarks/scenarios.csv", index_col="scenario")


def analyse(scenario):
    """The analysis of a scenario, under the settings of the manifest."""
    s = scenarios.loc[scenario]
    data = pd.read_csv(f"benchmarks/{s.dataset}")
    data["dose"] = s.dose
    data["duration"] = s.duration
    batch = Timecourses.from_dataframe(
        data,
        sample=[s.subject],
        time=s.time,
        value=s.conc,
        time_unit=s.time_unit,
        unit=s.unit,
        dose_amount="dose",
        dose_unit=s.dose_unit,
        dose_duration="duration" if s.route == "iv_infusion" else None,
        route=s.route,
    )
    options = NCAOptions(
        auc_method=AUCMethod(s.auc_method),
        terminal=TerminalPhase(
            method=TerminalMethod.BEST_FIT,
            min_points=3,
            exclude_cmax=bool(s.exclude_cmax),
        ),
    )
    return nca(batch, options=options)


def references(scenario):
    """The result tables of the three tools, one row per subject."""
    s = scenarios.loc[scenario]
    pknca = read_pknca_results(f"benchmarks/pknca/{scenario}.csv")
    return {
        "WinNonlin": read_winnonlin(f"benchmarks/winnonlin/{s.winnonlin}"),
        "PKNCA": pknca.droplevel(["start", "end"]),
        "NonCompart": read_noncompart(f"benchmarks/noncompart/{scenario}.csv"),
    }


def deviation(ours, reference):
    """The largest relative deviation of every parameter over the subjects."""
    # a column the tool left empty is a parameter it does not report for the route
    reference = reference.dropna(axis="columns", how="all")
    ours = ours.reindex(index=reference.index, columns=reference.columns)
    scale = reference.abs().where(reference != 0, 1.0)
    return ((ours - reference).abs() / scale).max()


def compare(scenario):
    """One row per parameter, one column per tool."""
    ours = (
        analyse(scenario).to_dataframe().set_index(scenarios.loc[scenario, "subject"])
    )
    return pd.DataFrame(
        {
            tool: deviation(
                ours, reference.set_axis(reference.index.astype(ours.index.dtype))
            )
            for tool, reference in references(scenario).items()
        }
    )


def fmt(value):
    """A deviation for the table: '-' where the tool does not report the parameter."""
    if math.isnan(value):
        return "-"
    if value < 1e-12:
        # the last bits of floating point arithmetic differ between platforms
        return "< 1e-12"
    return f"{value:.1e}"
```

The summary over all scenarios: for every tool, the number of parameters it reports and, after the colon, the largest relative deviation of `pkpdutils` from it over all of them and all subjects.

```python
summary = {}
for scenario in scenarios.index:
    table = compare(scenario)
    summary[scenario] = {
        tool: f"{table[tool].count()}: {fmt(table[tool].max())}" for tool in table
    }
print(pd.DataFrame(summary).T.to_string())
```

<!-- output:start -->
```text
                                WinNonlin        PKNCA   NonCompart
theoph-linear                 33: 4.7e-09  33: < 1e-12  34: < 1e-12
theoph-linear-log             33: 4.7e-09  33: < 1e-12  34: < 1e-12
indometh-bolus-linear         37: < 1e-12  35: < 1e-12  38: < 1e-12
indometh-bolus-linear-log     37: < 1e-12  35: < 1e-12  38: < 1e-12
indometh-infusion-linear      34: < 1e-12  34: < 1e-12  35: < 1e-12
indometh-infusion-linear-log  34: < 1e-12  34: < 1e-12  35: < 1e-12
indometh-oral-linear          33: 4.0e-09  33: < 1e-12  34: < 1e-12
indometh-oral-linear-log      33: 5.9e-09  33: < 1e-12  34: < 1e-12
```
<!-- output:end -->

Every parameter of every subject agrees with all three tools, to the last digits the reference tables carry. The WinNonlin tables of the report carry 8 to 15 significant digits, so a deviation around \(10^{-9}\) is the rounding of the reference. The only WinNonlin column without a `pkpdutils` variable is `Corr_XY`, the correlation of the terminal regression, which is \(-\sqrt{R^2}\).

## Parameter by parameter

The largest relative deviation over the subjects, per parameter and tool. A `-` marks a parameter one side does not report. WinNonlin writes no `Clast_pred` into the pivoted table. PKNCA has no extrapolated fraction of the AUMC and no back-extrapolated fraction, and it is the only tool which reports the span of the terminal phase (`lambda_z_span`). NonCompart writes a `TLAG` of 0 after an infusion, where `pkpdutils` and WinNonlin report no lag time for an intravenous dose.

### Theophylline, oral

```python
print(compare("theoph-linear").map(fmt).to_string())
```

<!-- output:start -->
```text
                          WinNonlin    PKNCA NonCompart
auc_all                     < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction         3.5e-10  < 1e-12    < 1e-12
auc_extrap_fraction_pred    3.7e-10  < 1e-12    < 1e-12
auc_inf_dn                  1.5e-09  < 1e-12    < 1e-12
auc_inf_obs                 4.3e-10  < 1e-12    < 1e-12
auc_inf_pred                4.3e-10  < 1e-12    < 1e-12
auc_inf_pred_dn             1.4e-09  < 1e-12    < 1e-12
auc_last                    < 1e-12  < 1e-12    < 1e-12
aumc_extrap_fraction        1.6e-10        -    < 1e-12
aumc_extrap_fraction_pred   1.4e-10        -    < 1e-12
aumc_inf                    3.1e-10  < 1e-12    < 1e-12
aumc_inf_pred               2.3e-10  < 1e-12    < 1e-12
aumc_last                   4.9e-10  < 1e-12    < 1e-12
cl_f                        2.1e-10  < 1e-12    < 1e-12
cl_f_pred                   2.1e-10  < 1e-12    < 1e-12
clast                       < 1e-12  < 1e-12    < 1e-12
clast_pred                        -  < 1e-12    < 1e-12
cmax                        < 1e-12  < 1e-12    < 1e-12
cmax_dn                     < 1e-12  < 1e-12    < 1e-12
lambda_z                    4.7e-09  < 1e-12    < 1e-12
lambda_z_n_points           < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                 4.3e-10  < 1e-12    < 1e-12
lambda_z_r2_adj             4.8e-10  < 1e-12    < 1e-12
lambda_z_span                     -  < 1e-12          -
lambda_z_t_first            < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last             < 1e-12  < 1e-12    < 1e-12
mrt                         3.9e-10  < 1e-12    < 1e-12
mrt_last                    6.5e-11  < 1e-12    < 1e-12
mrt_pred                    4.0e-10  < 1e-12    < 1e-12
thalf                       7.7e-11  < 1e-12    < 1e-12
tlag                        < 1e-12  < 1e-12    < 1e-12
tlast                       < 1e-12  < 1e-12    < 1e-12
tmax                        < 1e-12  < 1e-12    < 1e-12
vz_f                        1.8e-10  < 1e-12    < 1e-12
vz_f_pred                   1.7e-10  < 1e-12    < 1e-12
```
<!-- output:end -->

```python
print(compare("theoph-linear-log").map(fmt).to_string())
```

<!-- output:start -->
```text
                          WinNonlin    PKNCA NonCompart
auc_all                     4.0e-10  < 1e-12    < 1e-12
auc_extrap_fraction         3.8e-10  < 1e-12    < 1e-12
auc_extrap_fraction_pred    4.3e-10  < 1e-12    < 1e-12
auc_inf_dn                  1.5e-09  < 1e-12    < 1e-12
auc_inf_obs                 3.3e-10  < 1e-12    < 1e-12
auc_inf_pred                4.4e-10  < 1e-12    < 1e-12
auc_inf_pred_dn             1.6e-09  < 1e-12    < 1e-12
auc_last                    4.0e-10  < 1e-12    < 1e-12
aumc_extrap_fraction        1.4e-10        -    < 1e-12
aumc_extrap_fraction_pred   1.1e-10        -    < 1e-12
aumc_inf                    2.5e-10  < 1e-12    < 1e-12
aumc_inf_pred               4.1e-10  < 1e-12    < 1e-12
aumc_last                   4.1e-10  < 1e-12    < 1e-12
cl_f                        2.8e-10  < 1e-12    < 1e-12
cl_f_pred                   2.8e-10  < 1e-12    < 1e-12
clast                       < 1e-12  < 1e-12    < 1e-12
clast_pred                        -  < 1e-12    < 1e-12
cmax                        < 1e-12  < 1e-12    < 1e-12
cmax_dn                     < 1e-12  < 1e-12    < 1e-12
lambda_z                    4.7e-09  < 1e-12    < 1e-12
lambda_z_n_points           < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                 4.3e-10  < 1e-12    < 1e-12
lambda_z_r2_adj             4.8e-10  < 1e-12    < 1e-12
lambda_z_span                     -  < 1e-12          -
lambda_z_t_first            < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last             < 1e-12  < 1e-12    < 1e-12
mrt                         3.7e-10  < 1e-12    < 1e-12
mrt_last                    3.3e-10  < 1e-12    < 1e-12
mrt_pred                    4.5e-10  < 1e-12    < 1e-12
thalf                       7.7e-11  < 1e-12    < 1e-12
tlag                        < 1e-12  < 1e-12    < 1e-12
tlast                       < 1e-12  < 1e-12    < 1e-12
tmax                        < 1e-12  < 1e-12    < 1e-12
vz_f                        1.9e-10  < 1e-12    < 1e-12
vz_f_pred                   1.7e-10  < 1e-12    < 1e-12
```
<!-- output:end -->

### Indomethacin, intravenous bolus

```python
print(compare("indometh-bolus-linear").map(fmt).to_string())
```

<!-- output:start -->
```text
                              WinNonlin    PKNCA NonCompart
auc_all                         < 1e-12  < 1e-12    < 1e-12
auc_back_extrap_fraction        < 1e-12        -    < 1e-12
auc_back_extrap_fraction_pred   < 1e-12        -    < 1e-12
auc_extrap_fraction             < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction_pred        < 1e-12  < 1e-12    < 1e-12
auc_inf_dn                      < 1e-12  < 1e-12    < 1e-12
auc_inf_obs                     < 1e-12  < 1e-12    < 1e-12
auc_inf_pred                    < 1e-12  < 1e-12    < 1e-12
auc_inf_pred_dn                 < 1e-12  < 1e-12    < 1e-12
auc_last                        < 1e-12  < 1e-12    < 1e-12
aumc_extrap_fraction            < 1e-12        -    < 1e-12
aumc_extrap_fraction_pred       < 1e-12        -    < 1e-12
aumc_inf                        < 1e-12  < 1e-12    < 1e-12
aumc_inf_pred                   < 1e-12  < 1e-12    < 1e-12
aumc_last                       < 1e-12  < 1e-12    < 1e-12
c0                              < 1e-12  < 1e-12    < 1e-12
cl                              < 1e-12  < 1e-12    < 1e-12
cl_pred                         < 1e-12  < 1e-12    < 1e-12
clast                           < 1e-12  < 1e-12    < 1e-12
clast_pred                            -  < 1e-12    < 1e-12
cmax                            < 1e-12  < 1e-12    < 1e-12
cmax_dn                         < 1e-12  < 1e-12    < 1e-12
lambda_z                        < 1e-12  < 1e-12    < 1e-12
lambda_z_n_points               < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                     < 1e-12  < 1e-12    < 1e-12
lambda_z_r2_adj                 < 1e-12  < 1e-12    < 1e-12
lambda_z_span                         -  < 1e-12          -
lambda_z_t_first                < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last                 < 1e-12  < 1e-12    < 1e-12
mrt                             < 1e-12  < 1e-12    < 1e-12
mrt_last                        < 1e-12  < 1e-12    < 1e-12
mrt_pred                        < 1e-12  < 1e-12    < 1e-12
thalf                           < 1e-12  < 1e-12    < 1e-12
tlast                           < 1e-12  < 1e-12    < 1e-12
tmax                            < 1e-12  < 1e-12    < 1e-12
vss                             < 1e-12  < 1e-12    < 1e-12
vss_pred                        < 1e-12  < 1e-12    < 1e-12
vz                              < 1e-12  < 1e-12    < 1e-12
vz_pred                         < 1e-12  < 1e-12    < 1e-12
```
<!-- output:end -->

```python
print(compare("indometh-bolus-linear-log").map(fmt).to_string())
```

<!-- output:start -->
```text
                              WinNonlin    PKNCA NonCompart
auc_all                         < 1e-12  < 1e-12    < 1e-12
auc_back_extrap_fraction        < 1e-12        -    < 1e-12
auc_back_extrap_fraction_pred   < 1e-12        -    < 1e-12
auc_extrap_fraction             < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction_pred        < 1e-12  < 1e-12    < 1e-12
auc_inf_dn                      < 1e-12  < 1e-12    < 1e-12
auc_inf_obs                     < 1e-12  < 1e-12    < 1e-12
auc_inf_pred                    < 1e-12  < 1e-12    < 1e-12
auc_inf_pred_dn                 < 1e-12  < 1e-12    < 1e-12
auc_last                        < 1e-12  < 1e-12    < 1e-12
aumc_extrap_fraction            < 1e-12        -    < 1e-12
aumc_extrap_fraction_pred       < 1e-12        -    < 1e-12
aumc_inf                        < 1e-12  < 1e-12    < 1e-12
aumc_inf_pred                   < 1e-12  < 1e-12    < 1e-12
aumc_last                       < 1e-12  < 1e-12    < 1e-12
c0                              < 1e-12  < 1e-12    < 1e-12
cl                              < 1e-12  < 1e-12    < 1e-12
cl_pred                         < 1e-12  < 1e-12    < 1e-12
clast                           < 1e-12  < 1e-12    < 1e-12
clast_pred                            -  < 1e-12    < 1e-12
cmax                            < 1e-12  < 1e-12    < 1e-12
cmax_dn                         < 1e-12  < 1e-12    < 1e-12
lambda_z                        < 1e-12  < 1e-12    < 1e-12
lambda_z_n_points               < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                     < 1e-12  < 1e-12    < 1e-12
lambda_z_r2_adj                 < 1e-12  < 1e-12    < 1e-12
lambda_z_span                         -  < 1e-12          -
lambda_z_t_first                < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last                 < 1e-12  < 1e-12    < 1e-12
mrt                             < 1e-12  < 1e-12    < 1e-12
mrt_last                        < 1e-12  < 1e-12    < 1e-12
mrt_pred                        < 1e-12  < 1e-12    < 1e-12
thalf                           < 1e-12  < 1e-12    < 1e-12
tlast                           < 1e-12  < 1e-12    < 1e-12
tmax                            < 1e-12  < 1e-12    < 1e-12
vss                             < 1e-12  < 1e-12    < 1e-12
vss_pred                        < 1e-12  < 1e-12    < 1e-12
vz                              < 1e-12  < 1e-12    < 1e-12
vz_pred                         < 1e-12  < 1e-12    < 1e-12
```
<!-- output:end -->

### Indomethacin, 0.25 h infusion

```python
print(compare("indometh-infusion-linear").map(fmt).to_string())
```

<!-- output:start -->
```text
                          WinNonlin    PKNCA NonCompart
auc_all                     < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction         < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction_pred    < 1e-12  < 1e-12    < 1e-12
auc_inf_dn                  < 1e-12  < 1e-12    < 1e-12
auc_inf_obs                 < 1e-12  < 1e-12    < 1e-12
auc_inf_pred                < 1e-12  < 1e-12    < 1e-12
auc_inf_pred_dn             < 1e-12  < 1e-12    < 1e-12
auc_last                    < 1e-12  < 1e-12    < 1e-12
aumc_extrap_fraction        < 1e-12        -    < 1e-12
aumc_extrap_fraction_pred   < 1e-12        -    < 1e-12
aumc_inf                    < 1e-12  < 1e-12    < 1e-12
aumc_inf_pred               < 1e-12  < 1e-12    < 1e-12
aumc_last                   < 1e-12  < 1e-12    < 1e-12
cl                          < 1e-12  < 1e-12    < 1e-12
cl_pred                     < 1e-12  < 1e-12    < 1e-12
clast                       < 1e-12  < 1e-12    < 1e-12
clast_pred                        -  < 1e-12    < 1e-12
cmax                        < 1e-12  < 1e-12    < 1e-12
cmax_dn                     < 1e-12  < 1e-12    < 1e-12
lambda_z                    < 1e-12  < 1e-12    < 1e-12
lambda_z_n_points           < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                 < 1e-12  < 1e-12    < 1e-12
lambda_z_r2_adj             < 1e-12  < 1e-12    < 1e-12
lambda_z_span                     -  < 1e-12          -
lambda_z_t_first            < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last             < 1e-12  < 1e-12    < 1e-12
mrt                         < 1e-12  < 1e-12    < 1e-12
mrt_last                    < 1e-12  < 1e-12    < 1e-12
mrt_pred                    < 1e-12  < 1e-12    < 1e-12
thalf                       < 1e-12  < 1e-12    < 1e-12
tlag                              -        -          -
tlast                       < 1e-12  < 1e-12    < 1e-12
tmax                        < 1e-12  < 1e-12    < 1e-12
vss                         < 1e-12  < 1e-12    < 1e-12
vss_pred                    < 1e-12  < 1e-12    < 1e-12
vz                          < 1e-12  < 1e-12    < 1e-12
vz_pred                     < 1e-12  < 1e-12    < 1e-12
```
<!-- output:end -->

```python
print(compare("indometh-infusion-linear-log").map(fmt).to_string())
```

<!-- output:start -->
```text
                          WinNonlin    PKNCA NonCompart
auc_all                     < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction         < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction_pred    < 1e-12  < 1e-12    < 1e-12
auc_inf_dn                  < 1e-12  < 1e-12    < 1e-12
auc_inf_obs                 < 1e-12  < 1e-12    < 1e-12
auc_inf_pred                < 1e-12  < 1e-12    < 1e-12
auc_inf_pred_dn             < 1e-12  < 1e-12    < 1e-12
auc_last                    < 1e-12  < 1e-12    < 1e-12
aumc_extrap_fraction        < 1e-12        -    < 1e-12
aumc_extrap_fraction_pred   < 1e-12        -    < 1e-12
aumc_inf                    < 1e-12  < 1e-12    < 1e-12
aumc_inf_pred               < 1e-12  < 1e-12    < 1e-12
aumc_last                   < 1e-12  < 1e-12    < 1e-12
cl                          < 1e-12  < 1e-12    < 1e-12
cl_pred                     < 1e-12  < 1e-12    < 1e-12
clast                       < 1e-12  < 1e-12    < 1e-12
clast_pred                        -  < 1e-12    < 1e-12
cmax                        < 1e-12  < 1e-12    < 1e-12
cmax_dn                     < 1e-12  < 1e-12    < 1e-12
lambda_z                    < 1e-12  < 1e-12    < 1e-12
lambda_z_n_points           < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                 < 1e-12  < 1e-12    < 1e-12
lambda_z_r2_adj             < 1e-12  < 1e-12    < 1e-12
lambda_z_span                     -  < 1e-12          -
lambda_z_t_first            < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last             < 1e-12  < 1e-12    < 1e-12
mrt                         < 1e-12  < 1e-12    < 1e-12
mrt_last                    < 1e-12  < 1e-12    < 1e-12
mrt_pred                    < 1e-12  < 1e-12    < 1e-12
thalf                       < 1e-12  < 1e-12    < 1e-12
tlag                              -        -          -
tlast                       < 1e-12  < 1e-12    < 1e-12
tmax                        < 1e-12  < 1e-12    < 1e-12
vss                         < 1e-12  < 1e-12    < 1e-12
vss_pred                    < 1e-12  < 1e-12    < 1e-12
vz                          < 1e-12  < 1e-12    < 1e-12
vz_pred                     < 1e-12  < 1e-12    < 1e-12
```
<!-- output:end -->

### Indomethacin, extravascular

```python
print(compare("indometh-oral-linear").map(fmt).to_string())
```

<!-- output:start -->
```text
                          WinNonlin    PKNCA NonCompart
auc_all                     < 1e-12  < 1e-12    < 1e-12
auc_extrap_fraction         3.0e-10  < 1e-12    < 1e-12
auc_extrap_fraction_pred    2.8e-10  < 1e-12    < 1e-12
auc_inf_dn                  3.1e-09  < 1e-12    < 1e-12
auc_inf_obs                 2.1e-10  < 1e-12    < 1e-12
auc_inf_pred                2.0e-10  < 1e-12    < 1e-12
auc_inf_pred_dn             4.0e-09  < 1e-12    < 1e-12
auc_last                    < 1e-12  < 1e-12    < 1e-12
aumc_extrap_fraction        1.6e-10        -    < 1e-12
aumc_extrap_fraction_pred   3.0e-10        -    < 1e-12
aumc_inf                    6.8e-11  < 1e-12    < 1e-12
aumc_inf_pred               5.2e-11  < 1e-12    < 1e-12
aumc_last                   < 1e-12  < 1e-12    < 1e-12
cl_f                        4.2e-10  < 1e-12    < 1e-12
cl_f_pred                   5.3e-11  < 1e-12    < 1e-12
clast                       < 1e-12  < 1e-12    < 1e-12
clast_pred                        -  < 1e-12    < 1e-12
cmax                        < 1e-12  < 1e-12    < 1e-12
cmax_dn                     < 1e-12  < 1e-12    < 1e-12
lambda_z                    2.5e-09  < 1e-12    < 1e-12
lambda_z_n_points           < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                 4.6e-10  < 1e-12    < 1e-12
lambda_z_r2_adj             5.8e-10  < 1e-12    < 1e-12
lambda_z_span                     -  < 1e-12          -
lambda_z_t_first            < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last             < 1e-12  < 1e-12    < 1e-12
mrt                         2.0e-10  < 1e-12    < 1e-12
mrt_last                    2.5e-10  < 1e-12    < 1e-12
mrt_pred                    1.6e-10  < 1e-12    < 1e-12
thalf                       2.0e-10  < 1e-12    < 1e-12
tlag                        < 1e-12  < 1e-12    < 1e-12
tlast                       < 1e-12  < 1e-12    < 1e-12
tmax                        < 1e-12  < 1e-12    < 1e-12
vz_f                        1.0e-10  < 1e-12    < 1e-12
vz_f_pred                   1.9e-10  < 1e-12    < 1e-12
```
<!-- output:end -->

```python
print(compare("indometh-oral-linear-log").map(fmt).to_string())
```

<!-- output:start -->
```text
                          WinNonlin    PKNCA NonCompart
auc_all                     2.0e-10  < 1e-12    < 1e-12
auc_extrap_fraction         3.1e-10  < 1e-12    < 1e-12
auc_extrap_fraction_pred    2.4e-10  < 1e-12    < 1e-12
auc_inf_dn                  5.0e-09  < 1e-12    < 1e-12
auc_inf_obs                 2.0e-10  < 1e-12    < 1e-12
auc_inf_pred                1.3e-10  < 1e-12    < 1e-12
auc_inf_pred_dn             5.9e-09  < 1e-12    < 1e-12
auc_last                    2.0e-10  < 1e-12    < 1e-12
aumc_extrap_fraction        1.4e-10        -    < 1e-12
aumc_extrap_fraction_pred   2.7e-10        -    < 1e-12
aumc_inf                    5.8e-11  < 1e-12    < 1e-12
aumc_inf_pred               6.8e-11  < 1e-12    < 1e-12
aumc_last                   6.7e-11  < 1e-12    < 1e-12
cl_f                        2.8e-10  < 1e-12    < 1e-12
cl_f_pred                   3.1e-10  < 1e-12    < 1e-12
clast                       < 1e-12  < 1e-12    < 1e-12
clast_pred                        -  < 1e-12    < 1e-12
cmax                        < 1e-12  < 1e-12    < 1e-12
cmax_dn                     < 1e-12  < 1e-12    < 1e-12
lambda_z                    2.5e-09  < 1e-12    < 1e-12
lambda_z_n_points           < 1e-12  < 1e-12    < 1e-12
lambda_z_r2                 4.6e-10  < 1e-12    < 1e-12
lambda_z_r2_adj             5.8e-10  < 1e-12    < 1e-12
lambda_z_span                     -  < 1e-12          -
lambda_z_t_first            < 1e-12  < 1e-12    < 1e-12
lambda_z_t_last             < 1e-12  < 1e-12    < 1e-12
mrt                         1.6e-10  < 1e-12    < 1e-12
mrt_last                    2.5e-10  < 1e-12    < 1e-12
mrt_pred                    1.7e-10  < 1e-12    < 1e-12
thalf                       2.0e-10  < 1e-12    < 1e-12
tlag                        < 1e-12  < 1e-12    < 1e-12
tlast                       < 1e-12  < 1e-12    < 1e-12
tmax                        < 1e-12  < 1e-12    < 1e-12
vz_f                        2.1e-10  < 1e-12    < 1e-12
vz_f_pred                   1.7e-10  < 1e-12    < 1e-12
```
<!-- output:end -->

These two scenarios found a bug. Up to version 1.2.0, `pkpdutils` started the area of an extravascular single dose at its first sample. WinNonlin, PKNCA and NonCompart all insert a zero at the dose, so the areas, the clearance and the volume of these profiles were up to 13 % off. The theophylline profiles could not show it, since every one of them starts with a sample at the dose.

## The exported tables

The exporters write a result in the table of each tool, so a result of `pkpdutils` can go wherever a result of the other tool is expected, and a published table can be checked column by column. `to_winnonlin` writes the "Final Parameters Pivoted" layout, with the columns in the order Phoenix writes them and the extrapolated fractions in percent. Here are the first columns next to the table of Phoenix itself:

```python
from pkpdutils.crosswalk import to_noncompart, to_pknca_results, to_winnonlin

result = analyse("theoph-linear")
exported = to_winnonlin(result)
phoenix = pd.read_csv("benchmarks/winnonlin/Final_Parameters_Pivoted_Theoph_Linear.csv")
columns = ["Subject", "Lambda_z", "HL_Lambda_z", "Cmax", "AUClast", "AUC_%Extrap_obs"]
print("to_winnonlin(result)")
print(exported[columns].head(3).to_string(index=False))
print("\nPhoenix WinNonlin, Final_Parameters_Pivoted_Theoph_Linear.csv")
print(phoenix[columns].head(3).to_string(index=False))
```

<!-- output:start -->
```text
to_winnonlin(result)
 Subject  Lambda_z  HL_Lambda_z  Cmax   AUClast  AUC_%Extrap_obs
       1  0.048457    14.304378 10.50 148.92305        31.248917
       2  0.104086     6.659342  8.33  91.52680         8.631687
       3  0.102444     6.766087  8.20  99.28650         9.357173

Phoenix WinNonlin, Final_Parameters_Pivoted_Theoph_Linear.csv
 Subject  Lambda_z  HL_Lambda_z  Cmax   AUClast  AUC_%Extrap_obs
       1  0.048457    14.304378 10.50 148.92305        31.248917
       2  0.104086     6.659342  8.33  91.52680         8.631687
       3  0.102444     6.766087  8.20  99.28650         9.357173
```
<!-- output:end -->

`to_pknca_results` writes the long table of PKNCA, one row per subject and parameter, and `to_noncompart` the wide table of NonCompart, whose columns are the CDISC codes:

```python
print("to_pknca_results(result)")
print(to_pknca_results(result).head(4).to_string(index=False))
print("\nto_noncompart(result)")
print(to_noncompart(result).iloc[:3, :6].to_string(index=False))
```

<!-- output:start -->
```text
to_pknca_results(result)
 Subject  start  end PPTESTCD  PPORRES exclude
       1    0.0  inf     cmax    10.50    <NA>
       1    0.0  inf     tmax     1.12    <NA>
       1    0.0  inf     cmin     0.74    <NA>
       1    0.0  inf    tlast    24.37    <NA>

to_noncompart(result)
 Subject  CMAX    CMAXD  TMAX  TLAG  CLST
       1 10.50 0.032813  1.12   0.0  3.28
       2  8.33 0.026031  1.92   0.0  0.90
       3  8.20 0.025625  1.02   0.0  1.05
```
<!-- output:end -->

`write_winnonlin`, `write_pknca_results` and `write_noncompart` write the same tables as csv files. `read_winnonlin`, `read_pknca_results` and `read_noncompart` read them back into the variables of `pkpdutils`. The CDISC `PP` domain of a submission is [`pkpdutils.cdisc.to_pp`](formats.md#the-pp-domain).

## Analytical profiles

When two tools agree, it may only be because they share a convention or a bug. An analytical profile has an exact answer instead. After an intravenous bolus, a one-compartment curve \(C(t) = C_0 e^{-kt}\) has

\[
\lambda_z = k, \qquad t_{1/2} = \frac{\ln 2}{k}, \qquad \mathrm{AUC}_{0\text{-}\infty} = \frac{C_0}{k}, \qquad \mathrm{AUMC}_{0\text{-}\infty} = \frac{C_0}{k^2}, \qquad \mathrm{MRT} = \frac{1}{k}, \qquad \mathrm{CL} = \frac{D k}{C_0}, \qquad V_z = V_\mathrm{ss} = \frac{D}{C_0}.
\]

The log trapezoid integrates an exponential exactly, and so does the back extrapolation of \(C_0\) from the first two samples. With the linear-up / log-down rule every one of these parameters is therefore exact, whatever the sampling:

```python
from pkpdutils import Dose, Route, Timecourse, nca_single

c0, k, dose = 12.0, 0.23, 100.0  # mg/L, 1/h, mg
t = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24])
bolus = Timecourse(
    time=t,
    value=c0 * np.exp(-k * t),
    time_unit="h",
    unit="mg/L",
    dose=Dose(amount=dose, unit="mg", route=Route.IV_BOLUS),
)
log_down = NCAOptions(
    auc_method=AUCMethod.LINEAR_LOG,
    terminal=TerminalPhase(method=TerminalMethod.BEST_FIT, min_points=3),
)
exact = {
    "c0": c0,
    "lambda_z": k,
    "thalf": math.log(2) / k,
    "auc_inf_obs": c0 / k,
    "aumc_inf": c0 / k**2,
    "mrt": 1 / k,
    "cl": dose * k / c0,
    "vz": dose / c0,
    "vss": dose / c0,
}
result = nca_single(bolus, options=log_down)
for name, value in exact.items():
    print(
        f"{name:12} exact {value:10.5f}   deviation {fmt(abs(float(result[name]) / value - 1))}"
    )
```

<!-- output:start -->
```text
c0           exact   12.00000   deviation < 1e-12
lambda_z     exact    0.23000   deviation < 1e-12
thalf        exact    3.01368   deviation < 1e-12
auc_inf_obs  exact   52.17391   deviation < 1e-12
aumc_inf     exact  226.84310   deviation < 1e-12
mrt          exact    4.34783   deviation < 1e-12
cl           exact    1.91667   deviation < 1e-12
vz           exact    8.33333   deviation < 1e-12
vss          exact    8.33333   deviation < 1e-12
```
<!-- output:end -->

An extravascular dose has no such shortcut: the rise is not an exponential, and the trapezoid over it is an approximation. The Bateman function of a first order absorption,

\[
C(t) = \frac{F D\, k_a}{V (k_a - k)} \left(e^{-kt} - e^{-k_a t}\right), \qquad \mathrm{AUC}_{0\text{-}\infty} = \frac{F D}{V k}, \qquad \mathrm{MRT} = \frac{1}{k} + \frac{1}{k_a},
\]

shows how that error shrinks as the sampling of the rise gets denser. The terminal rate constant shows something else: the best fit takes the longest window whose adjusted \(R^2\) is within 0.0001 of the best one, so with dense early samples it reaches back into the end of the absorption, where \(e^{-k_a t}\) has not quite died out. The same profile with the regression fixed on its last four samples (`TerminalMethod.LAST_N`) recovers \(k\) exactly:

```python
ka, k, v = 3.0, 0.2, 20.0  # 1/h, 1/h, L; F = 1
last_four = NCAOptions(
    auc_method=AUCMethod.LINEAR_LOG,
    terminal=TerminalPhase(method=TerminalMethod.LAST_N, n_points=4),
)
rows = []
for step in (1.0, 0.5, 0.25, 0.1):
    t = np.concatenate([np.arange(0, 4, step), [4, 6, 8, 12, 16, 24, 36]])
    value = dose * ka / (v * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))
    oral = Timecourse(
        time=t,
        value=value,
        time_unit="h",
        unit="mg/L",
        dose=Dose(amount=dose, unit="mg", route=Route.ORAL),
    )
    best = nca_single(oral, options=log_down)
    fixed = nca_single(oral, options=last_four)
    rows.append(
        {
            "step [h]": step,
            "auc_inf_obs": fmt(abs(float(best["auc_inf_obs"]) / (dose / (v * k)) - 1)),
            "mrt": fmt(abs(float(best["mrt"]) / (1 / k + 1 / ka) - 1)),
            "lambda_z best fit": fmt(abs(float(best["lambda_z"]) / k - 1)),
            "window from [h]": float(best["lambda_z_t_first"]),
            "lambda_z last four": fmt(abs(float(fixed["lambda_z"]) / k - 1)),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
```

<!-- output:start -->
```text
 step [h] auc_inf_obs     mrt lambda_z best fit  window from [h] lambda_z last four
     1.00     4.6e-02 4.9e-02           2.0e-04             2.00            < 1e-12
     0.50     1.3e-02 1.3e-02           6.6e-04             1.50            < 1e-12
     0.25     3.3e-03 3.4e-03           1.2e-03             1.25            < 1e-12
     0.10     5.2e-04 5.6e-04           2.0e-03             1.10            < 1e-12
```
<!-- output:end -->

## Reproducing this page

```bash
Rscript scripts/benchmark_datasets.R         # needs R, PKNCA, NonCompart
uv run python scripts/benchmark_datasets.py  # pastes the output
uv run pytest tests/docs/test_benchmark_datasets.py
```

The R script rewrites the PKNCA and NonCompart tables and `versions.csv`. The python script reruns every snippet of this page in a temporary directory with `docs/data/` next to it and writes what each one prints into the page. The test fails when the page is not current.

# Non-compartmental analysis

Non-compartmental analysis (NCA) describes a concentration timecourse by parameters computed directly from the measured points, without a model of the body: the exposure as the area under the curve, the peak, the terminal half-life, and, with the dose, the clearance and the volume of distribution. `pkpdutils.nca` computes these parameters for one curve or for a whole batch of curves in one vectorized call; every parameter carries its unit and every sample carries flags for the conditions that limit its interpretation. The definitions follow Gabrielsson & Weiner [^gw] and the NCA of Phoenix WinNonlin [^phoenix].

## Concepts

**Exposure.** The area under the concentration–time curve, \(\mathrm{AUC}\), is proportional to the amount of drug that reached the systemic circulation. \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\) is measured up to the last quantifiable concentration \(C_\mathrm{last}\); \(\mathrm{AUC}_{0\text{-}\infty}\) adds the tail after \(t_\mathrm{last}\), predicted from the terminal phase. The fraction of \(\mathrm{AUC}_{0\text{-}\infty}\) that is extrapolated tells how much of the exposure was not observed; above 20 % the estimate is considered unreliable (flag `EXTRAPOLATION_HIGH`).

**Peak.** \(C_\mathrm{max}\) and \(t_\mathrm{max}\) are read from the observed points. After an extravascular dose they reflect the balance of absorption and elimination; after an intravenous bolus the concentration at time zero, \(C_0\), is not observed and is back-extrapolated from the first two points.

**Terminal phase.** When absorption and distribution are over, the concentration declines mono-exponentially, \(C(t) = C_\mathrm{last}\, e^{-\lambda_z (t - t_\mathrm{last})}\). The terminal rate constant \(\lambda_z\) is the slope of \(\ln C\) against \(t\) over the terminal points; the half-life is \(t_{1/2} = \ln 2 / \lambda_z\). Which points belong to the terminal phase is a judgement: the default `BEST_FIT` rule takes the window with the largest adjusted \(R^2\) among all windows of at least three points that end at \(t_\mathrm{last}\) and start after \(t_\mathrm{max}\), preferring more points when the adjusted \(R^2\) is equal within a tolerance, as Phoenix does. `LAST_N`, `ALL_AFTER_TMAX` (the rule of pkdb_analysis 0.3.1) and `MANUAL` are the alternatives. `TerminalPhase.exclude_cmax` (default `True`) restricts every window to start after \(t_\mathrm{max}\); with `exclude_cmax=False` the window start is unrestricted and every window of at least `min_points` points ending at \(t_\mathrm{last}\) is a candidate, including windows that begin before or at the maximum, which is how a curve that only rises reports `POSITIVE_SLOPE` instead of `TOO_FEW_POINTS`.

**Clearance and volume.** With the dose \(D\), the clearance \(\mathrm{CL} = D / \mathrm{AUC}_{0\text{-}\infty}\) is the volume of plasma cleared of drug per time and the volume of distribution \(V_z = \mathrm{CL} / \lambda_z\) is the apparent volume the dose would occupy at the plasma concentration. After an extravascular dose the fraction absorbed \(F\) is unknown and both are reported relative to it as \(\mathrm{CL}/F\) (`cl_f`) and \(V_z/F\) (`vz_f`). The mean residence time \(\mathrm{MRT} = \mathrm{AUMC}_{0\text{-}\infty} / \mathrm{AUC}_{0\text{-}\infty}\) is the average time a molecule stays in the body (after an infusion of duration \(T\), minus \(T/2\)); the steady state volume \(V_\mathrm{ss} = \mathrm{CL} \cdot \mathrm{MRT}\) is reported for intravenous doses. A dose of 0, the encoding of a placebo arm, makes none of them a quantity: \(\mathrm{CL}\), \(V_z\), \(V_\mathrm{ss}\), `auc_inf_dn` and `cmax_dn` are `NaN` there, which the analysis reports in a debug log and not with a flag, since a zero dose is a property of the data and not a finding of the analysis.

**Multiple dosing.** A curve accompanied by a dosing protocol of more than one dose, or analysed with `NCAOptions.tau`, is split into its dosing intervals \([t_k, t_{k+1}]\) and the last interval \([t_K, t_K + \tau]\), whose length \(\tau\) comes from the protocol (the distance of the last two doses) or from `tau` when it is given [^rt]. Every interval is described the same way as the classic steady state interval below: the value at its start and at its end are interpolated and inserted, so a sample outside the interval adds no area and does not enter its \(C_\mathrm{max}\) or \(C_\mathrm{min}\); after an intravenous bolus an interval that starts before the first sample of the curve starts at the back-extrapolated \(C_0\). An interval whose end is not covered by the data is incomplete: its parameters, and the steady state parameters when it is the last interval, are `NaN` and the row is flagged `INCOMPLETE_INTERVAL`. A sample recorded exactly at the end of a bolus interval may already be the post-dose value of the next dose; when it lies above the last sample inside the interval, which no decline can do, the trough is instead the log-linear regression of the last (up to three) samples of the interval and the row is flagged `EXTRAPOLATED_TROUGH`.

**Steady state.** The steady state parameters describe the last complete interval, under the assumption that repeated dosing has reached a state where every interval looks the same: with linear kinetics \(\mathrm{AUC}_{0\text{-}\tau}\) at steady state equals the single dose \(\mathrm{AUC}_{0\text{-}\infty}\). The interval is described by the average concentration \(C_\mathrm{avg} = \mathrm{AUC}_{0\text{-}\tau} / \tau\), the trough \(C_\mathrm{trough} = C(\tau)\), the fluctuation, the swing, the clearance at steady state \(\mathrm{CL}_\mathrm{ss}\), and the accumulation ratio, predicted from the terminal phase or observed as the ratio of the exposure of the last and the first interval of the protocol (`accumulation_ratio_obs`, `NaN` for a single dose protocol or when the first interval is incomplete).

**The reference dose.** With more than one dose the point parameters (\(C_\mathrm{max}\), \(t_\mathrm{max}\), \(C_\mathrm{last}\), \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\), the extrapolated areas, the terminal phase, \(\mathrm{MRT}\)) are computed from the last dose on: the values before it are dropped and the times are relative to it, the same analysis a single dose curve given with its last dose only would get. With one dose this is the whole curve, as today.

**No single dose quantities.** That slice is not a single dose curve: it carries the exposure of every earlier dose as well, so dividing the dose by its area would report a clearance that is too low and a volume that is too small. A multiple dose analysis therefore reports \(\mathrm{CL}\), \(\mathrm{CL}/F\), \(V_z\), \(V_z/F\), \(V_\mathrm{ss}\), `auc_inf_dn` and `cmax_dn` as `NaN`; the clearance of such an analysis is \(\mathrm{CL}_\mathrm{ss} = D_K / \mathrm{AUC}_{0\text{-}\tau}\) (`cl_ss`, `cl_ss_f` after an extravascular dose) over the dosing interval. \(\mathrm{AUC}_{0\text{-}\infty}\), \(\mathrm{AUMC}_{0\text{-}\infty}\) and \(\mathrm{MRT}\) are reported and describe the exposure and the decline after the last dose, extrapolated with its terminal phase, not the single dose exposure of the substance. A single dose curve analysed with `tau` is a multiple dose analysis as well, so the same holds for it. The decision is taken per sample and not per batch, so a batch mixing single dose and multiple dose subjects reports \(\mathrm{CL}\)/\(\mathrm{CL}/F\) for its single dose samples and \(\mathrm{CL}_\mathrm{ss}\)/\(\mathrm{CL}_\mathrm{ss}/F\) for its multiple dose ones; it carries the union of the two sets of variables and every sample is `NaN` in the variables of the other path.

**Routes.** A batch has one route. `IV_BOLUS` reports \(C_0\), \(\mathrm{CL}\), \(V_z\), \(V_\mathrm{ss}\); `IV_INFUSION` corrects the \(\mathrm{MRT}\) by half the duration; `ORAL` (any extravascular route) reports \(\mathrm{CL}/F\), \(V_z/F\) and the half maximum during absorption (`cmax_half`, `tmax_half`).

**Missing values and the limit of quantification.** `NaN` values are dropped; values below `lloq` become `NaN` (or 0 before the maximum with `BLQHandling.ZERO_BEFORE_TMAX`), and the sample is flagged `BLQ_TRUNCATED`.

## Math

Trapezoid rules on a segment from \((t_1, C_1)\) to \((t_2, C_2)\) with \(\Delta t = t_2 - t_1\) and \(L = \ln(C_1 / C_2)\):

\[
\mathrm{AUC}^\mathrm{lin} = \frac{\Delta t\,(C_1 + C_2)}{2}, \qquad
\mathrm{AUMC}^\mathrm{lin} = \frac{\Delta t\,(t_1 C_1 + t_2 C_2)}{2}
\]

\[
\mathrm{AUC}^\mathrm{log} = \frac{\Delta t\,(C_1 - C_2)}{L}, \qquad
\mathrm{AUMC}^\mathrm{log} = \frac{\Delta t\,(t_1 C_1 - t_2 C_2)}{L} + \frac{\Delta t^2\,(C_1 - C_2)}{L^2}
\]

The logarithmic rule is exact for a mono-exponential segment. `AUCMethod.LINEAR_LOG` (the default, "linear up/log down") uses the linear rule on rising and the logarithmic rule on falling segments; `LINEAR` uses the linear rule everywhere (the rule of pkdb_analysis 0.3.1); `LOG` the logarithmic rule wherever both values are positive.

Terminal regression of \(y = \ln C\) on \(t\) over \(n\) points:

\[
\lambda_z = -\frac{n \sum t y - \sum t \sum y}{n \sum t^2 - (\sum t)^2}, \qquad
R^2_\mathrm{adj} = 1 - (1 - R^2)\,\frac{n - 1}{n - 2}, \qquad
t_{1/2} = \frac{\ln 2}{\lambda_z}
\]

Extrapolation and moments, with the observed \(C_\mathrm{last}\) (`auc_inf_obs`) or the value of the regression line at \(t_\mathrm{last}\), \(\hat C_\mathrm{last} = e^{b - \lambda_z t_\mathrm{last}}\) (`auc_inf_pred`):

\[
\mathrm{AUC}_{0\text{-}\infty} = \mathrm{AUC}_{0\text{-}t_\mathrm{last}} + \frac{C_\mathrm{last}}{\lambda_z}, \qquad
\mathrm{AUMC}_{0\text{-}\infty} = \mathrm{AUMC}_{0\text{-}t_\mathrm{last}} + \frac{C_\mathrm{last}\, t_\mathrm{last}}{\lambda_z} + \frac{C_\mathrm{last}}{\lambda_z^2}
\]

\[
\mathrm{MRT} = \frac{\mathrm{AUMC}_{0\text{-}\infty}}{\mathrm{AUC}_{0\text{-}\infty}} - \frac{T_\mathrm{inf}}{2}, \qquad
\mathrm{CL} = \frac{D}{\mathrm{AUC}_{0\text{-}\infty}}, \qquad
V_z = \frac{\mathrm{CL}}{\lambda_z}, \qquad
V_\mathrm{ss} = \mathrm{CL} \cdot \mathrm{MRT}
\]

\(C_0\) after a bolus by log-linear back-extrapolation of the first two points \((t_1, C_1)\), \((t_2, C_2)\): \(C_0 = \exp\!\left(\ln C_1 - t_1 \frac{\ln C_2 - \ln C_1}{t_2 - t_1}\right)\); the point \((0, C_0)\) enters the areas.

Steady state over the last complete interval \([t_K, t_K + \tau]\) (the values at its bounds are interpolated, or, after a bolus, back-extrapolated at the start and log-linearly regressed at the end when a post-dose sample lies at it):

\[
C_\mathrm{avg} = \frac{\mathrm{AUC}_{0\text{-}\tau}}{\tau}, \quad
\mathrm{fluctuation} = \frac{C_\mathrm{max,ss} - C_\mathrm{min,ss}}{C_\mathrm{avg}}, \quad
\mathrm{swing} = \frac{C_\mathrm{max,ss} - C_\mathrm{min,ss}}{C_\mathrm{min,ss}}, \quad
\mathrm{CL}_\mathrm{ss} = \frac{D_K}{\mathrm{AUC}_{0\text{-}\tau}}
\]

Accumulation ratio, predicted from the terminal phase or observed within one protocol as the ratio of the exposure of the last and the first dosing interval:

\[
R_\mathrm{pred} = \frac{1}{1 - e^{-\lambda_z \tau}}, \qquad
R_\mathrm{obs} = \frac{\mathrm{AUC}_{0\text{-}\tau}\text{ of the last interval}}{\mathrm{AUC}_{0\text{-}\tau}\text{ of the first interval}}
\]

`accumulation_ratio` (`pkpdutils.nca.steady_state`) compares two separate results the same way, the steady state and the single dose analysis of the same dosing interval: \(R = \mathrm{AUC}_{0\text{-}\tau}^\mathrm{ss} / \mathrm{AUC}_{0\text{-}\tau}^\mathrm{single}\).

Superposition predicts the multiple dose curve as the sum of the single dose curve shifted to every dose time, scaled by the dose ratio, interpolated inside the observed range and extrapolated with \(\lambda_z\) beyond \(t_\mathrm{last}\); it assumes linear kinetics.

## Parameters

| name | symbol | definition | unit | needs |
| --- | --- | --- | --- | --- |
| `cmax`, `tmax` | \(C_\mathrm{max}\), \(t_\mathrm{max}\) | maximum observed value and its time | value, time | |
| `cmin`, `tmin` | \(C_\mathrm{min}\), \(t_\mathrm{min}\) | minimum observed value and its time | value, time | |
| `clast`, `tlast` | \(C_\mathrm{last}\), \(t_\mathrm{last}\) | last positive value and its time | value, time | |
| `c0` | \(C_0\) | back-extrapolated value at time 0 | value | `IV_BOLUS` |
| `cmax_half`, `tmax_half` | | value closest to \(C_\mathrm{max}/2\) before the maximum and its time | value, time | `ORAL` |
| `auc_last` | \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\) | area to the last positive value | value·time | |
| `auc_inf_obs`, `auc_inf_pred` | \(\mathrm{AUC}_{0\text{-}\infty}\) | area extrapolated with the observed or predicted \(C_\mathrm{last}\) | value·time | \(\lambda_z\) |
| `auc_extrap_fraction` | | \((\mathrm{AUC}_{0\text{-}\infty} - \mathrm{AUC}_{0\text{-}t_\mathrm{last}}) / \mathrm{AUC}_{0\text{-}\infty}\) | – | \(\lambda_z\) |
| `aumc_last`, `aumc_inf` | \(\mathrm{AUMC}\) | first moment of the curve | value·time² | \(\lambda_z\) for `_inf` |
| `mrt` | \(\mathrm{MRT}\) | mean residence time | time | \(\lambda_z\) |
| `lambda_z` | \(\lambda_z\) | terminal rate constant | 1/time | ≥ 3 terminal points |
| `thalf` | \(t_{1/2}\) | terminal half-life | time | \(\lambda_z\) |
| `lambda_z_n_points`, `lambda_z_t_first`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`, `lambda_z_stderr` | | diagnostics of the regression (`lambda_z_stderr` is the standard error of the slope of the terminal regression) | –, time, –, –, – (\(\ln C\)), 1/time | \(\lambda_z\) |
| `cl`, `cl_f` | \(\mathrm{CL}\), \(\mathrm{CL}/F\) | clearance (`_f`: extravascular) | dose/(value·time) → l/h | dose, \(\lambda_z\), single dose analysis |
| `vz`, `vz_f` | \(V_z\), \(V_z/F\) | terminal volume of distribution | dose/value → l | dose, \(\lambda_z\), single dose analysis |
| `vss` | \(V_\mathrm{ss}\) | steady state volume of distribution | dose/value → l | intravenous dose, single dose analysis |
| `auc_inf_dn`, `cmax_dn` | | dose normalized exposure and peak | value·time/dose, value/dose | dose, single dose analysis |
| `auc_tau` | \(\mathrm{AUC}_{0\text{-}\tau}\) | area over the last complete dosing interval | value·time | protocol (≥ 2 doses) or `tau` |
| `cmin_ss`, `cmax_ss`, `ctrough`, `cavg` | \(C_\mathrm{min,ss}\), \(C_\mathrm{max,ss}\), \(C_\mathrm{trough}\), \(C_\mathrm{avg}\) | minimum, maximum, value at the end, average over the last interval | value | protocol (≥ 2 doses) or `tau` |
| `fluctuation`, `swing`, `accumulation_ratio` | | see Math | – | protocol (≥ 2 doses) or `tau` |
| `accumulation_ratio_obs` | \(R_\mathrm{obs}\) | observed accumulation, last over first interval | – | protocol of ≥ 2 doses, first interval complete |
| `cl_ss`, `cl_ss_f` | \(\mathrm{CL}_\mathrm{ss}\), \(\mathrm{CL}_\mathrm{ss}/F\) | \(D_K / \mathrm{AUC}_{0\text{-}\tau}\) (`_f`: extravascular) | → l/h | protocol (≥ 2 doses) or `tau`, dose |
| `n_doses`, `tau` | \(K\), \(\tau\) | number of doses of the protocol and the length of the last interval | –, time | protocol (≥ 2 doses) or `tau` |
| `flags` | | `NCAFlag` bits, see below | – | |

Volumes are reported in liter (per kilogram for doses per body weight), clearances in liter per hour; every other unit is derived from the units of the input. Effect timecourses (`Kind.EFFECT`) report `e0`, `emax_obs`, `temax`, `tlast`, `auec_last`, `auec_baseline`, `emax_baseline` and `time_above` instead, see [Pharmacodynamics](pd.md); with a protocol they additionally report `auec_tau`, `emin_ss`, `emax_ss`, `eavg`, `time_above_tau`, `accumulation_ratio_obs`, `n_doses` and `tau`, the effect analogues of the row above.

### Per-interval parameters

A protocol of more than one dose additionally reports the parameters of every single dosing interval, over the extra dimension `interval` (`NCAOptions.intervals`, default `True`); `NCAResult.intervals()` returns them as one row per sample and interval, with `interval_start`, `interval_end` and, with dose amounts, `interval_dose` as columns.

| name | symbol | definition | unit |
| --- | --- | --- | --- |
| `interval_auc` | \(\mathrm{AUC}_{0\text{-}\tau,k}\) | area over the interval | value·time |
| `interval_cmax`, `interval_tmax` | \(C_\mathrm{max,k}\), \(t_\mathrm{max,k}\) | maximum of the interval and its time relative to the interval start | value, time |
| `interval_cmin` | \(C_\mathrm{min,k}\) | minimum of the interval | value |
| `interval_ctrough` | \(C_\mathrm{trough,k}\) | value at the end of the interval | value |
| `interval_c_start` | \(C_\mathrm{start,k}\) | value at the start of the interval, interpolated or observed (after a bolus the post-dose value; the pre-dose value of interval \(k\) is `interval_ctrough` of interval \(k-1\)) | value |
| `interval_cavg`, `interval_fluctuation`, `interval_swing` | | average, fluctuation and swing of the interval | value, –, – |
| `interval_n_points` | | number of samples the interval uses (a boundary sample counts for both neighbours) | – |

For effect timecourses the same interval carries `interval_auec`, `interval_emax`, `interval_temax`, `interval_emin`, `interval_eavg` and `interval_time_above` instead. The interval variables are point variables (an extra dimension) and are excluded from `to_dataframe`.

Flags: `POSITIVE_SLOPE` (the terminal regression does not decline; \(\lambda_z\) and everything derived from it is `NaN`), `TOO_FEW_POINTS` (no window with the minimal number of points), `EXTRAPOLATION_HIGH`, `NO_MAX` (the maximum is the last point), `NO_ABSORPTION` (the maximum is the first point of an extravascular curve), `BLQ_TRUNCATED`, `NO_DATA` (fewer than two points), `DELTA_WINDOW_CHANGE` (the delta method skipped points at which the terminal window moved, see [Uncertainty](uncertainty.md)), `INCOMPLETE_INTERVAL` (the last dosing interval is not covered by the data; its parameters and the steady state parameters are `NaN`), `EXTRAPOLATED_TROUGH` (the trough of at least one dosing interval of a bolus was regressed because the sample at the dose time carries the post-dose value).

## API

One curve:

```python
from pkpdutils import Dose, Route, Timecourse, nca_single

tc = Timecourse(
    time=[0.25, 0.5, 1, 2, 4, 6, 8, 12, 24],
    value=[0.9, 1.7, 2.6, 2.8, 2.2, 1.6, 1.2, 0.6, 0.1],
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
result = nca_single(tc)
result.to_quantities()["auc_inf_obs"]  # pint quantity in hour * milligram / liter
result.to_quantities()["cl_f"]  # liter / hour
result.flags()  # e.g. []
```

Options select the methods; the analysis of a batch returns the parameters over its sample dimensions:

```python
from pkpdutils import NCAOptions, TerminalPhase, Timecourses, nca
from pkpdutils.nca import AUCMethod, TerminalMethod

options = NCAOptions(
    auc_method=AUCMethod.LINEAR_LOG,
    terminal=TerminalPhase(method=TerminalMethod.BEST_FIT, min_points=3),
    lloq=0.05,
    extrapolation_warning=0.2,
)
result = nca(batch, options=options)  # batch: Timecourses over (study, individual)
result.ds  # xarray.Dataset, one variable per parameter
result["thalf"]  # DataArray over (study, individual), attrs["units"]
result.to_dataframe()  # one row per sample, flags decoded
result.flag_table()  # one boolean column per flag
```

Steady state, with the dosing interval:

```python
from pkpdutils import Dosing
from pkpdutils.nca import accumulation_ratio, superposition

dose = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
# the dosing interval of a curve given with its last dose only
ss = nca(batch_ss, options=NCAOptions(tau=12))  # auc_tau, cavg, fluctuation, ...
sd = nca(batch_single, options=NCAOptions(tau=12))
ratio = accumulation_ratio(ss, sd)  # observed accumulation
predicted = superposition(tc_single, Dosing.regimen(dose, interval=12, n_doses=10))
```

A curve carrying a dosing protocol of more than one dose is analysed over its dosing intervals without `tau`, `nca_single` and `nca` the same way:

```python
from pkpdutils import Dosing, NCAOptions, nca_single
from pkpdutils.nca import AUCMethod

protocol = Dosing.regimen(dose, interval=12, n_doses=4)
tc_protocol = Timecourse(
    time=..., value=..., dosing=protocol, time_unit="hr", unit="mg/l"
)
result = nca_single(tc_protocol, options=NCAOptions(auc_method=AUCMethod.LOG))
result.intervals()  # one row per dosing interval: interval_auc, interval_cmax, ...
result.to_quantities()["auc_tau"]  # the last, complete interval
result.to_quantities()["accumulation_ratio_obs"]  # last interval over first interval
result.flags()  # e.g. ['INCOMPLETE_INTERVAL'] when the last interval is not covered
```

The per-interval parameters (`interval_*`, `NCAResult.intervals()`), the steady state parameters of the last interval and the point parameters from the last dose on are all part of the one result.

Large batches are analysed in chunks of at most `NCAOptions(chunk_rows=5000)` rows, which bounds the memory of the vectorized core, and the chunks are mapped in order over the workers of `NCAOptions(n_workers=...)`; both apply to the steady state path as well. The core is vectorized numpy and releases the GIL, so the workers are threads of the calling process (no `if __name__ == "__main__":` guard, no copy of the batch, a pool that starts in half a millisecond and is shared with every later call). The default `n_workers=None` decides by size: the calling thread up to 20 000 rows (`pkpdutils.parallel.NCA_WORKER_THRESHOLD`), where the analysis is faster than the pool, and one thread per usable core, at most 8, above it; `n_workers=1` forces the serial run and `n_workers=n` uses that many threads. The rows are cut into about one chunk per worker, so a large batch keeps every thread busy, and the temporaries of the core live for as many chunks as run at once: a parallel run holds `min(n_workers, n_chunks) * chunk_rows` rows of them, not `chunk_rows`, which is what a large batch pays for its speed. Group timecourses with `sd`/`se` get uncertainty variables per parameter, individual results are summarized with `NCAResult.summarize`, see [Uncertainty](uncertainty.md); partial areas come from `partial_auc`, whose interval may start before the first sample of a curve but not before its dose: the value at the dose is then 0 for an extravascular dose, the back-extrapolated \(C_0\) for a bolus and `NaN` for an infusion. The figures are described in [Plotting](plotting.md), the examples are `examples/nca_single.py`, `examples/nca_batch.py`, `examples/steady_state.py` and `examples/nca_from_sbmlsim.py`, the reference of the modules is in [API: nca](api/nca.md).

## References

[^gw]: Gabrielsson J, Weiner D. *Pharmacokinetic and Pharmacodynamic Data Analysis: Concepts and Applications*. 5th ed. Swedish Pharmaceutical Press; 2016. See [References](references.md#textbooks).
[^phoenix]: Certara. *Phoenix WinNonlin User's Guide: Noncompartmental Analysis*. See [References](references.md#non-compartmental-analysis).
[^rt]: Rowland M, Tozer TN. *Clinical Pharmacokinetics and Pharmacodynamics*. 4th ed. 2011, ch. 11. See [References](references.md#textbooks).

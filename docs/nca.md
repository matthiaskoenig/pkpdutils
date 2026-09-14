# Non-compartmental analysis

Non-compartmental analysis (NCA) describes a concentration timecourse by parameters computed directly from the measured points, without a model of the body: the exposure as the area under the curve, the peak, the terminal half-life, and, with the dose, the clearance and the volume of distribution. `pkpdutils.nca` computes these parameters for one curve or for a whole batch of curves in one vectorized call; every parameter carries its unit and every sample carries flags for the conditions that limit its interpretation. The definitions follow Gabrielsson & Weiner [^gw] and the NCA of Phoenix WinNonlin [^phoenix].

## Concepts

**Exposure.** The area under the concentration–time curve, \(\mathrm{AUC}\), is proportional to the amount of drug that reached the systemic circulation. \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\) is measured up to the last quantifiable concentration \(C_\mathrm{last}\); \(\mathrm{AUC}_{0\text{-}\infty}\) adds the tail after \(t_\mathrm{last}\), predicted from the terminal phase. The fraction of \(\mathrm{AUC}_{0\text{-}\infty}\) that is extrapolated tells how much of the exposure was not observed; above 20 % the estimate is considered unreliable (flag `EXTRAPOLATION_HIGH`).

**Peak.** \(C_\mathrm{max}\) and \(t_\mathrm{max}\) are read from the observed points. After an extravascular dose they reflect the balance of absorption and elimination; after an intravenous bolus the concentration at time zero, \(C_0\), is not observed and is back-extrapolated from the first two points.

**Terminal phase.** When absorption and distribution are over, the concentration declines mono-exponentially, \(C(t) = C_\mathrm{last}\, e^{-\lambda_z (t - t_\mathrm{last})}\). The terminal rate constant \(\lambda_z\) is the slope of \(\ln C\) against \(t\) over the terminal points; the half-life is \(t_{1/2} = \ln 2 / \lambda_z\). Which points belong to the terminal phase is a judgement: the default `BEST_FIT` rule takes the window with the largest adjusted \(R^2\) among all windows of at least three points that end at \(t_\mathrm{last}\) and start after \(t_\mathrm{max}\), preferring more points when the adjusted \(R^2\) is equal within a tolerance, as Phoenix does. `LAST_N`, `ALL_AFTER_TMAX` (the rule of pkdb_analysis 0.3.1) and `MANUAL` are the alternatives. `TerminalPhase.exclude_cmax` (default `True`) restricts every window to start after \(t_\mathrm{max}\); with `exclude_cmax=False` the window start is unrestricted and every window of at least `min_points` points ending at \(t_\mathrm{last}\) is a candidate, including windows that begin before or at the maximum, which is how a curve that only rises reports `POSITIVE_SLOPE` instead of `TOO_FEW_POINTS`.

**Clearance and volume.** With the dose \(D\), the clearance \(\mathrm{CL} = D / \mathrm{AUC}_{0\text{-}\infty}\) is the volume of plasma cleared of drug per time and the volume of distribution \(V_z = \mathrm{CL} / \lambda_z\) is the apparent volume the dose would occupy at the plasma concentration. After an extravascular dose the fraction absorbed \(F\) is unknown and both are reported relative to it as \(\mathrm{CL}/F\) (`cl_f`) and \(V_z/F\) (`vz_f`). The mean residence time \(\mathrm{MRT} = \mathrm{AUMC}_{0\text{-}\infty} / \mathrm{AUC}_{0\text{-}\infty}\) is the average time a molecule stays in the body (after an infusion of duration \(T\), minus \(T/2\)); the steady state volume \(V_\mathrm{ss} = \mathrm{CL} \cdot \mathrm{MRT}\) is reported for intravenous doses.

**Steady state.** Under repeated dosing every dosing interval \(\tau\) looks the same once steady state is reached, and with linear kinetics \(\mathrm{AUC}_{0\text{-}\tau}\) at steady state equals the single dose \(\mathrm{AUC}_{0\text{-}\infty}\). The interval is described by the average concentration \(C_\mathrm{avg} = \mathrm{AUC}_{0\text{-}\tau} / \tau\), the trough \(C_\mathrm{trough} = C(\tau)\), the fluctuation and the swing, and the accumulation ratio [^rt]. The interval starts at the dose: the value at the dose and the value at \(\tau\) are interpolated and inserted, so a pre-dose sample adds no area and a sample after \(\tau\) does not enter \(C_\mathrm{max,ss}\) or \(C_\mathrm{min,ss}\). After an intravenous bolus the interval starts at the back-extrapolated \(C_0\); for any other route the area from the dose to the first sample after it is not counted, so a curve whose first sample is well after the dose underestimates \(\mathrm{AUC}_{0\text{-}\tau}\).

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

Steady state over the interval \([0, \tau]\) (the value at \(\tau\) is interpolated):

\[
C_\mathrm{avg} = \frac{\mathrm{AUC}_{0\text{-}\tau}}{\tau}, \quad
\mathrm{fluctuation} = \frac{C_\mathrm{max,ss} - C_\mathrm{min,ss}}{C_\mathrm{avg}}, \quad
\mathrm{swing} = \frac{C_\mathrm{max,ss} - C_\mathrm{min,ss}}{C_\mathrm{min,ss}}, \quad
R_\mathrm{pred} = \frac{1}{1 - e^{-\lambda_z \tau}}, \quad
R_\mathrm{obs} = \frac{\mathrm{AUC}_{0\text{-}\tau}^\mathrm{ss}}{\mathrm{AUC}_{0\text{-}\tau}^\mathrm{single}}
\]

Superposition predicts the multiple dose curve as the sum of the single dose curve shifted to every dose time, interpolated inside the observed range and extrapolated with \(\lambda_z\) beyond \(t_\mathrm{last}\); it assumes linear kinetics.

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
| `cl`, `cl_f` | \(\mathrm{CL}\), \(\mathrm{CL}/F\) | clearance (`_f`: extravascular) | dose/(value·time) → l/h | dose, \(\lambda_z\) |
| `vz`, `vz_f` | \(V_z\), \(V_z/F\) | terminal volume of distribution | dose/value → l | dose, \(\lambda_z\) |
| `vss` | \(V_\mathrm{ss}\) | steady state volume of distribution | dose/value → l | intravenous dose |
| `auc_inf_dn`, `cmax_dn` | | dose normalized exposure and peak | value·time/dose, value/dose | dose |
| `auc_tau` | \(\mathrm{AUC}_{0\text{-}\tau}\) | area over the dosing interval | value·time | regimen |
| `cmin_ss`, `cmax_ss`, `ctrough`, `cavg` | \(C_\mathrm{min,ss}\), \(C_\mathrm{max,ss}\), \(C_\mathrm{trough}\), \(C_\mathrm{avg}\) | minimum, maximum, value at \(\tau\), average over the interval | value | regimen |
| `fluctuation`, `swing`, `accumulation_ratio` | | see Math | – | regimen |
| `cl_ss` | \(\mathrm{CL}_\mathrm{ss}\) | \(D / \mathrm{AUC}_{0\text{-}\tau}\) | → l/h | regimen, dose |
| `flags` | | `NCAFlag` bits, see below | – | |

Volumes are reported in liter (per kilogram for doses per body weight), clearances in liter per hour; every other unit is derived from the units of the input. Effect timecourses (`Kind.EFFECT`) report `e0`, `emax_obs`, `temax`, `auec_last`, `auec_baseline`, `emax_baseline` and `time_above` instead, see [Pharmacodynamics](pd.md).

Flags: `POSITIVE_SLOPE` (the terminal regression does not decline; \(\lambda_z\) and everything derived from it is `NaN`), `TOO_FEW_POINTS` (no window with the minimal number of points), `EXTRAPOLATION_HIGH`, `NO_MAX` (the maximum is the last point), `NO_ABSORPTION` (the maximum is the first point of an extravascular curve), `BLQ_TRUNCATED`, `NO_DATA` (fewer than two points), `DELTA_WINDOW_CHANGE` (the delta method skipped points at which the terminal window moved, see [Uncertainty](uncertainty.md)).

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
result = nca(batch, options)  # batch: Timecourses over (study, individual)
result.ds  # xarray.Dataset, one variable per parameter
result["thalf"]  # DataArray over (study, individual), attrs["units"]
result.to_dataframe()  # one row per sample, flags decoded
result.flag_table()  # one boolean column per flag
```

Steady state, with the dosing interval:

```python
from pkpdutils import DosingRegimen
from pkpdutils.nca import accumulation_ratio, superposition

regimen = DosingRegimen(
    dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=12
)
ss = nca(batch_ss, NCAOptions(regimen=regimen))  # auc_tau, cavg, fluctuation, ...
sd = nca(batch_single, NCAOptions(regimen=regimen))
ratio = accumulation_ratio(ss, sd)  # observed accumulation
predicted = superposition(
    tc_single, DosingRegimen(dose=regimen.dose, interval=12, n_doses=10)
)
```

Large batches are analysed in chunks of `NCAOptions(chunk_rows=5000)` rows, which bounds the memory of the vectorized core, and run in worker processes with `NCAOptions(n_workers=4)`, which map the chunks in order; both apply to the steady state path as well. The analysis itself is vectorized, so the workers only pay off for many thousands of curves. Group timecourses with `sd`/`se` get uncertainty variables per parameter, individual results are summarized with `NCAResult.summarize`, see [Uncertainty](uncertainty.md); partial areas come from `partial_auc`. The figures are described in [Plotting](plotting.md), the examples are `examples/nca_single.py`, `examples/nca_batch.py`, `examples/steady_state.py` and `examples/nca_from_sbmlsim.py`, the reference of the modules is in [API: nca](api/nca.md).

## References

[^gw]: Gabrielsson J, Weiner D. *Pharmacokinetic and Pharmacodynamic Data Analysis: Concepts and Applications*. 5th ed. Swedish Pharmaceutical Press; 2016. See [References](references.md#textbooks).
[^phoenix]: Certara. *Phoenix WinNonlin User's Guide: Noncompartmental Analysis*. See [References](references.md#non-compartmental-analysis).
[^rt]: Rowland M, Tozer TN. *Clinical Pharmacokinetics and Pharmacodynamics*. 4th ed. 2011, ch. 11. See [References](references.md#textbooks).

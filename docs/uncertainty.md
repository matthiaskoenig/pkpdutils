# Uncertainty

Published pharmacokinetic data are mostly group data: the mean concentration of a group at every sampling time with its standard deviation or standard error and the number of subjects. The parameters of the mean curve are point estimates; how uncertain they are depends on the uncertainty of the points and on how the parameters depend on them. `pkpdutils` propagates the uncertainty of a group timecourse to every parameter of the [non-compartmental analysis](nca.md) by a parametric bootstrap or by the delta method, and it summarizes the parameters of individual curves over the individuals with the same set of variables, so that the statistics of the next pages accept both.

## Concepts

**Spread of the mean and spread of the individuals.** The standard error \(\mathrm{se}_i = \mathrm{sd}_i / \sqrt{n}\) of a time point is the uncertainty of the group mean; the standard deviation \(\mathrm{sd}_i\) is the spread of the individual subjects. Which one to propagate depends on the question: the uncertainty of the parameters of the mean curve (`BootstrapSpread.SE`, the default) or the spread of the parameters over subjects (`BootstrapSpread.SD`). A `Timecourse` carries `sd`, `se` and `n` and derives the missing one; the result reports both `x_se` and `x_sd`, converting with \(\sqrt n\).

**Bootstrap.** The parametric bootstrap [^efron] draws every time point of the curve from a distribution with the observed mean and spread, analyses every replicate curve as if it were observed, and reads the uncertainty of a parameter from the spread of its replicates. Normal draws (the default) are set to 0 when they fall below 0; log-normal draws with the same mean and spread are positive by construction and suit concentrations with a large relative spread. `n_boot` replicates of every curve run through the same vectorized code as the curves themselves, so the bootstrap of a batch is one call, chunked by `NCAOptions.chunk_rows`.

**Delta method.** The delta method [^efron] linearizes a parameter around the observed curve: every time point is perturbed by a small step, the numerical derivative of every parameter with respect to every point is formed, and the variances of the points add through the squared derivatives. It costs one analysis per time point instead of one per replicate, gives symmetric normal intervals (log-normal parameters on the logarithmic scale) and is exact for the linear trapezoid area; it cannot follow a change of the terminal window.

**Discrete parameters.** `tmax`, `tlast`, `tmin`, `tmax_half`, `temax` and the counts of the terminal regression are read from the observed points; they carry no uncertainty variables. The regression diagnostics (`lambda_z_stderr`, `lambda_z_r2`, `lambda_z_r2_adj`, `lambda_z_intercept`) carry no uncertainty variables either.

**Individuals.** When every subject has its own curve the parameters of the subjects are a sample: `NCAResult.summarize(dim)` reduces the result over a sample dimension to the mean, standard deviation, standard error, a t-based confidence interval of the mean, median and quartiles, the number of values and, for log-normal parameters, the geometric mean and geometric CV. The variables have the same names as the bootstrap output, so a group result and a summary look alike.

## Math

Parametric bootstrap with \(B\) replicates of a point \(\bar C_i\) with spread \(s_i\):

\[
C_i^{(b)} \sim \mathcal N(\bar C_i, s_i^2) \quad\text{or}\quad C_i^{(b)} \sim \mathrm{LogNormal}\!\left(\ln \bar C_i - \tfrac{\sigma_i^2}{2},\ \sigma_i^2\right),\ \sigma_i^2 = \ln\!\left(1 + \frac{s_i^2}{\bar C_i^2}\right)
\]

\[
\mathrm{se}(x) = \sqrt{\frac{1}{B-1}\sum_b \left(x^{(b)} - \bar x^{(\cdot)}\right)^2}\ \ (s_i = \mathrm{se}_i), \qquad
\mathrm{CI} = \left[x^{(\alpha/2)},\ x^{(1-\alpha/2)}\right], \qquad
\mathrm{GM} = \exp\!\left(\overline{\ln x^{(b)}}\right), \quad
\mathrm{GCV} = \sqrt{e^{\mathrm{Var}(\ln x^{(b)})} - 1}
\]

Delta method with the step \(h_i = \delta\, \mathrm{se}_i\):

\[
\frac{\partial x}{\partial C_i} \approx \frac{x(C + h_i e_i) - x(C)}{h_i}, \qquad
\mathrm{Var}(x) = \sum_i \left(\frac{\partial x}{\partial C_i}\right)^2 \mathrm{se}_i^2, \qquad
\mathrm{CI} = x \pm z_{1-\alpha/2}\,\mathrm{se}(x) \ \text{ or } \ x\, e^{\pm z_{1-\alpha/2}\,\mathrm{se}(x)/x}
\]

For the linear trapezoid rule \(\mathrm{AUC} = \sum_i w_i C_i\) is linear in the points and the delta method is exact: \(\mathrm{Var}(\mathrm{AUC}) = \sum_i w_i^2 \mathrm{se}_i^2\).

Summary of \(n\) individual values \(x_j\): mean \(\bar x\), \(\mathrm{sd}\) with \(n-1\), \(\mathrm{se} = \mathrm{sd}/\sqrt n\), \(\mathrm{CI} = \bar x \pm t_{n-1,\,1-\alpha/2}\,\mathrm{se}\), geometric mean and CV from \(\ln x_j\).

## Variables

| variable | meaning | unit |
| --- | --- | --- |
| `x` | the parameter of the mean curve (bootstrap, delta) or the mean over the individuals (summary) | unit of `x` |
| `x_sd`, `x_se` | standard deviation over subjects and standard error of the mean | unit of `x` |
| `x_ci_low`, `x_ci_high` | confidence interval at `ci_level` (percentile, normal or t-based) | unit of `x` |
| `x_geomean`, `x_geocv` | geometric mean and geometric coefficient of variation (log-normal parameters) | unit of `x`, - |
| `x_median`, `x_q25`, `x_q75`, `x_n` | median, quartiles and count of finite values (summary only) | unit of `x`, - |
| `n` | number of subjects (group data) or of samples (summary) | - |

## API

Group data: the bootstrap is the default as soon as the timecourse carries `sd` or `se`:

```python
from pkpdutils import Dose, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.nca import UncertaintyMethod
from pkpdutils.nca.options import BootstrapDistribution, BootstrapSpread

group = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12, 24],
    value=[1.9, 2.6, 2.4, 1.8, 0.95, 0.5, 0.12],
    sd=[0.4, 0.5, 0.4, 0.3, 0.2, 0.12, 0.04],
    n=10,
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
result = nca_single(group, NCAOptions(seed=1, n_boot=2000))
q = result.to_quantities()
q["auc_inf_obs"], q["auc_inf_obs_se"], q["auc_inf_obs_ci_low"], q["auc_inf_obs_ci_high"]
result = nca_single(
    group,
    NCAOptions(
        bootstrap_spread=BootstrapSpread.SD,
        bootstrap_distribution=BootstrapDistribution.LOGNORMAL,
    ),
)
result = nca_single(group, NCAOptions(uncertainty=UncertaintyMethod.DELTA))
```

Individual curves:

```python
result = nca(individuals)  # individuals: Timecourses over "individual"
summary = result.summarize("individual")
summary.to_dataframe()  # auc_inf_obs, auc_inf_obs_sd, ..., auc_inf_obs_geocv, n
```

Partial areas, e.g. \(\mathrm{AUC}_{0\text{-}6}\) of every sample:

```python
from pkpdutils.nca import partial_auc

area = partial_auc(
    individuals, 0.5, 6.0
)  # DataArray over the sample dims, attrs["units"]
```

The example is `examples/group_uncertainty.py`; the reference of the module is in [API: nca.uncertainty](api/nca.uncertainty.md).

## References

[^efron]: Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman & Hall/CRC; 1993, ch. 5 (delta method) and 6 (bootstrap). See [References](references.md#statistics).

# Statistics

Statistics on pharmacokinetic parameters: comparisons of two groups, geometric mean ratios, average bioequivalence, the classification of drug-drug interactions and the meta-analysis of published studies. Every function of `pkpdutils.stats` works on a `ParameterSample`, the values of one parameter over the individuals of a group or the summary statistics of the group, taken from an `NCAResult` or a `FitResult` with `sample` or typed in from a publication.

## Concepts

**Log-normal parameters.** Exposure, clearance, volume and half-life are positive and skewed: their logarithms are close to normal. The statistics therefore run on the log scale by default (`Scale.LOG`): differences of the logarithms are ratios of geometric means, intervals are symmetric on the log scale and asymmetric around the ratio, and the geometric mean and the geometric coefficient of variation \(\mathrm{CV}_g = \sqrt{e^{\sigma^2} - 1}\) describe a group. `Scale.LINEAR` compares arithmetic means, for parameters like \(t_\mathrm{max}\) or an effect which may be negative; `summarize` on `Scale.LINEAR` tolerates a non-positive value and reports `geomean`/`geocv` as `NaN` instead, while `Scale.LOG` raises on one.

**Individual and summary data.** With the individual values of a group every test of scipy is available; a publication often gives only the mean, the standard deviation and the number of subjects. A summary sample is analysed with the Welch t test from its moments; on the log scale the moments of the logarithm follow from the log-normal relations \(\sigma^2 = \ln(1 + \mathrm{sd}^2/\mathrm{mean}^2)\) and \(\mu = \ln\mathrm{mean} - \sigma^2/2\), or directly from the geometric mean and CV when they are reported.

**Designs.** In a parallel design two groups of different subjects are compared with the Welch interval. In a paired or crossover design every subject receives both treatments, and the within-subject differences remove the between-subject variability: a 2x2 crossover (two sequences RT and TR, two periods) is analysed with sequence, period and subject effects, which also tests the period and the carryover effect and gives the within-subject CV.

**Bioequivalence.** Two formulations are bioequivalent when the 90 % confidence interval of the geometric mean ratio of \(\mathrm{AUC}\) and \(C_\mathrm{max}\) lies within 80-125 %[^fda_be]. That is the two one-sided tests procedure of Schuirmann at \(\alpha = 0.05\)[^schuirmann].

**Drug-drug interactions.** A perpetrator is classified by how much it changes the \(\mathrm{AUC}\) of a sensitive substrate: a strong, moderate or weak inhibitor raises it at least 5-fold, 2- to 5-fold or 1.25- to 2-fold, a strong, moderate or weak inducer lowers it by at least 80 %, 50-80 % or 20-50 %[^fda_ddi]; the EMA guideline uses the same thresholds[^ema_ddi]. With an interval of the ratio the classification is conservative and marked as uncertain when the interval spans a boundary.

**Meta-analysis.** Effects of several studies (Hedges' g, a mean difference or the log ratio of geometric means, the effect native to pharmacokinetics) are pooled with inverse variance weights. The fixed effect model assumes one true effect; the random effects model of DerSimonian and Laird adds the between-study variance \(\tau^2\) to every weight and widens the interval when the studies disagree[^dl]. \(Q\), \(I^2\) and \(H^2\) measure that disagreement[^higgins].

## Math

**Two-sample t tests.** With the means \(\bar a\), \(\bar b\), the variances \(s_a^2\), \(s_b^2\) and the sizes \(n_a\), \(n_b\) (on the analysis scale), Welch's statistic is \(t = (\bar a - \bar b) / \sqrt{s_a^2/n_a + s_b^2/n_b}\) with the Welch-Satterthwaite degrees of freedom; Student's uses the pooled variance \(s_p^2 = ((n_a-1)s_a^2 + (n_b-1)s_b^2)/(n_a+n_b-2)\) with \(n_a + n_b - 2\); the paired test is the one-sample test of the differences. The interval of the effect is \(\hat\theta \pm t_{1-\alpha/2,\nu}\,\mathrm{se}\), exponentiated on the log scale. The standardized effect sizes are Cohen's \(d = (\bar a - \bar b)/s_p\) and Hedges' \(g = J d\) with \(J = 1 - 3/(4N - 9)\), \(N = n_a + n_b\)[^hedges]. The rank tests (Mann-Whitney, Wilcoxon) and the permutation test report the same interval-free `TestResult`, with `effect` the difference or the ratio of the medians of the raw values for the rank tests.

**Geometric mean ratio.** Paired: \(d_i = \ln t_i - \ln r_i\), \(\ln\mathrm{GMR} = \bar d\), \(\mathrm{se} = s_d/\sqrt{n}\), \(n - 1\) degrees of freedom. Parallel: \(\ln\mathrm{GMR} = \bar{\ln t} - \bar{\ln r}\) with the Welch standard error. The interval of the ratio is \(\exp(\ln\mathrm{GMR} \pm t\,\mathrm{se})\). A paired `ratio` with no pair of finite values raises `ValueError`.

**Pairing and degenerate samples.** Paired analyses (`compare(paired=True)`, `ratio`, `tost`) match the two samples with `paired_values`: by label when both samples carry labels, so the order of the individuals does not matter and an individual only one sample holds is dropped, and by position otherwise, which then needs equal sizes. A pair is dropped when either of its values is missing, which is logged at debug level; the analysis runs on the remaining pairs, so a missing parameter of one subject costs that subject and does not shift the pairing of the others. A sample of a single value and two samples without variance leave the statistic undefined: `statistic`, `p_value`, `df` and the interval come back as `NaN` instead of raising, while the effect itself (the difference or the ratio of the means) stays finite. A sample without a finite value at all (a parameter no subject of the group has) gives `NaN` throughout an unpaired `compare` or `ratio` and raises on a paired one, where no pair remains; `tost` reports `NaN` p values and no bioequivalence when the design leaves no standard error; `effect_size` gives a `NaN` effect and variance for such a group instead of raising, and the pooling drops that study with a warning naming it, keeps it with a `NaN` weight in `MetaResult.to_dataframe` and raises only when no study is left; a variance which is zero or negative is an error rather than a missing value, it carries an infinite weight and is named in a `ValueError`. `ParameterSample` rejects a negative `sd` or `geocv`, a non-positive `geomean` and an `n` which is not a whole number.

**Strings instead of enumeration members.** Every option of `pkpdutils.stats` is taken either as its enumeration member or as the string of the member, so `compare(a, b, scale="log", test="paired_t")`, `multiple_comparison(p, method="holm")`, `effect_size(control, treatment, "log_ratio")` and `tost(test, reference, design="parallel")` run the analysis their members name. An unknown string raises a `ValueError` listing the members rather than falling back to a default.

**2x2 crossover.** With the log values \(y_{i1}\), \(y_{i2}\) of subject \(i\) in the two periods, the period differences \(d_i = (y_{i2} - y_{i1})/2\) and the totals \(u_i = y_{i1} + y_{i2}\), and the sequences A (test in period 2) and B (test in period 1)[^chow]:

\[\hat F = \bar d_A - \bar d_B, \qquad \hat P = \bar d_A + \bar d_B, \qquad \hat C = \bar u_A - \bar u_B,\]

with \(\mathrm{var}(\hat F) = \mathrm{var}(\hat P) = \sigma_d^2 (1/n_A + 1/n_B)\) from the pooled within-sequence variance \(\sigma_d^2\) with \(n_A + n_B - 2\) degrees of freedom, and \(\mathrm{var}(\hat C)\) from the pooled variance of the totals. \(\hat F\) is the treatment effect of the analysis of variance with sequence, period and subject-within-sequence effects, its residual variance is \(\sigma_e^2 = 2\sigma_d^2\), and the within-subject CV is \(\sqrt{e^{\sigma_e^2} - 1}\). The analysis also rejects a reference sample whose `sequence` coordinate disagrees with the test's, and a design where both sequences have the test in the same period.

**Two one-sided tests.** For the limits \(\theta_L < 1 < \theta_U\), \(t_L = (\ln\mathrm{GMR} - \ln\theta_L)/\mathrm{se}\) and \(t_U = (\ln\theta_U - \ln\mathrm{GMR})/\mathrm{se}\) are tested one-sided at \(\alpha\); rejecting both is the same as the \(1 - 2\alpha\) interval lying within the limits[^schuirmann].

**Multiple comparisons.** Bonferroni \(\tilde p_i = \min(1, m p_i)\); Holm sorts the p values and takes \(\tilde p_{(i)} = \max_{j \le i}\min(1, (m-j+1)p_{(j)})\)[^holm]; Benjamini-Hochberg takes \(\tilde p_{(i)} = \min_{j \ge i}\min(1, m p_{(j)}/j)\)[^bh].

**Effect sizes of a study.** Hedges' g as above with \(\mathrm{var}(d) = N/(n_C n_T) + d^2/(2N)\) and \(\mathrm{var}(g) = J^2\mathrm{var}(d)\); the mean difference \(\bar x_T - \bar x_C\) with \(s_T^2/n_T + s_C^2/n_C\); the log ratio \(\mu_T - \mu_C\) with \(\sigma_T^2/n_T + \sigma_C^2/n_C\).

**Pooling.** With \(w_i = 1/v_i\): \(\hat\theta_F = \sum w_i\theta_i/\sum w_i\), \(\mathrm{se} = 1/\sqrt{\sum w_i}\). Heterogeneity \(Q = \sum w_i(\theta_i - \hat\theta_F)^2\), \(C = \sum w_i - \sum w_i^2/\sum w_i\), \(\tau^2 = \max(0, (Q - (k-1))/C)\), \(I^2 = 100\max(0, (Q-(k-1))/Q)\) in percent, \(H^2 = Q/(k-1)\). Random effects: \(w_i^* = 1/(v_i + \tau^2)\) and the same pooling[^dl][^higgins]. The intervals of the effects and the pooled effects are normal, \(\hat\theta \pm z_{1-\alpha/2}\,\mathrm{se}\).

## Results

| function | result | fields |
| --- | --- | --- |
| `summarize` | `Summary` | `n`, `mean`, `sd`, `se`, `cv`, `geomean`, `geocv`, `median`, `q25`, `q75`, `min`, `max`, `ci_low`, `ci_high` |
| `compare` | `TestResult` | `test`, `statistic`, `p_value`, `effect`, `ci_low`, `ci_high`, `df`, `cohen_d`, `hedges_g`, `n_a`, `n_b` |
| `ratio` | `RatioResult` | `gmr`, `ci_low`, `ci_high`, `log_ratio`, `se_log`, `df`, `paired`, `n_test`, `n_reference` |
| `tost`, `bioequivalence` | `BEParameter`, `BEResult` | `gmr`, `ci_low`, `ci_high`, `bioequivalent`, `p_lower`, `p_upper`, `p_value`, `design`, `cv_intra`, `p_period`, `p_sequence` |
| `ddi_classification` | `DDIResult` | `kind`, `strength`, `auc_ratio`, `cmax_ratio`, `ci_low`, `ci_high`, `uncertain`, `kind_low`, `strength_low`, `kind_high`, `strength_high` |
| `effect_size` | `EffectSize` | `estimate`, `variance`, `se`, `ci_low`, `ci_high`, `kind`, `n_control`, `n_treatment`, `label` |
| `fixed_effect`, `random_effects` | `PooledEffect` | `estimate`, `se`, `ci_low`, `ci_high`, `z`, `p_value`, `weights`, `model`, `tau2` |
| `heterogeneity` | `Heterogeneity` | `q`, `df`, `p_value`, `i2`, `h2`, `tau2` |
| `meta_analysis` | `MetaResult` | `effects`, `fixed`, `random`, `heterogeneity`, `to_dataframe()` |

The `effect` of `compare` is `a - b` on the linear scale and the ratio of the geometric means `a / b` on the log scale; `ratio`, `tost` and `ddi_classification` report test over reference and with over without the perpetrator. `p_value` of a `BEParameter` is the larger of the two one-sided p values, `bioequivalent` is `p_value < (1 - ci_level) / 2`, which is the interval within the limits.

**Confidence levels and argument order.** `ci_level` is 0.95 everywhere except `ratio`, `tost` and `bioequivalence`, which default to 0.90, the regulatory interval of the two one-sided tests: the 90 % interval of the ratio is the interval the bioequivalence decision reads[^schuirmann]. Every function takes `ci_level` as a keyword, so a comparison at another level is one argument away. The samples of a comparison are given test (or treatment) first: `compare(a, b)`, `ratio(test, reference)`, `tost(test, reference)`, `bioequivalence(test, reference)`; the meta-analysis reverses it, `effect_size(control, treatment)` and `Study(label, control, treatment)`, the convention of its own literature[^hedges].

## API

A sample from a result or from numbers:

```python
from pkpdutils import ParameterSample
from pkpdutils.stats import Scale, summarize

auc = result.sample(
    "auc_inf_obs", "individual"
)  # individual values with labels and coordinates
published = ParameterSample(
    mean=45.2, sd=12.1, n=12, name="cl", unit="l/hr"
)  # summary data
geometric = ParameterSample(geomean=42.0, geocv=0.28, n=12)
summarize(auc)  # geometric mean with its interval, quantiles, CV
summarize(auc, scale=Scale.LINEAR).ci_low
```

Compare two groups:

```python
from pkpdutils import compare
from pkpdutils.stats import Alternative, TestMethod, multiple_comparison

smokers = result.sample(
    "cl", "individual", group="smokers"
)  # "group" is another sample dimension of result
non_smokers = result.sample("cl", "individual", group="non-smokers")
test = compare(
    smokers, non_smokers
)  # Welch t on the log scale, effect = ratio of geometric means
test.p_value, test.effect, (test.ci_low, test.ci_high), test.hedges_g
compare(smokers, non_smokers, test=TestMethod.MANN_WHITNEY)
compare(before, after, paired=True, alternative=Alternative.LESS)
compare(published_a, published_b)  # Welch t from mean, sd, n
multiple_comparison([t.p_value for t in tests])  # Holm
```

Ratios and bioequivalence:

```python
from pkpdutils import bioequivalence, ratio

r = ratio(
    test_auc, reference_auc
)  # paired by label when both carry the same subjects, 90 % interval
be = bioequivalence(test_result, reference_result, parameters=["auc_inf_obs", "cmax"])
be.bioequivalent, be["cmax"].gmr, be.to_dataframe()
```

`ratio_table` formats the ratios of a study the way a paper prints them: one row per parameter with the point estimate and its interval in percent of the reference, the numbers rounded to `digits` significant digits as strings. It takes a mapping of `RatioResult` objects or the result of `bioequivalence`, which adds the within-subject coefficient of variation, the acceptance limits and the verdict.

```python
from pkpdutils.stats import ratio_table

# parameter, unit, n_test, n_reference, gmr, ci_low, ci_high, ci_level and,
# for a bioequivalence result, cv_intra, limits and bioequivalent
ratio_table(be)
ratio_table({"auc_inf_obs": r}, percent=False, digits=4)  # plain ratios
```

| parameter | unit | n_test | n_reference | gmr | ci_low | ci_high | ci_level | cv_intra | limits | bioequivalent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auc_inf_obs | hour * milligram / liter | 12 | 12 | 95.6 % | 88.6 % | 103 % | 90 % | 10.3 % | 80.0 - 125.0 % | True |

A 2x2 crossover is recognized from the coordinates `period` (1 or 2) and `sequence` along the individual dimension of both batches; they are given to `Timecourses.from_arrays` as `coords={"individual": ids, "period": ("individual", periods), "sequence": ("individual", sequences)}` and travel through the NCA to the result. Without them two results with the same individuals are paired, otherwise the groups are parallel; `design=Design.PARALLEL` overrides the detection.

Drug-drug interactions:

```python
from pkpdutils import ddi_classification
from pkpdutils.stats import DDIThresholds, substrate_sensitivity

ddi = ddi_classification(
    ratio(inhibited_auc, control_auc), cmax_ratio=ratio(inhibited_cmax, control_cmax)
)
ddi.kind, ddi.strength, ddi.uncertain
ddi_classification(3.2, ci=(2.4, 4.3), thresholds=DDIThresholds.ema())
substrate_sensitivity(6.1)
```

`ddi_table` does the same over several parameters of two results: it takes every parameter from both, forms the ratio with and without the perpetrator and classifies it, so that the exposure and the maximum are read next to each other. The classes are defined for the \(\mathrm{AUC}\) and are applied to every parameter of the table.

```python
from pkpdutils.stats import ddi_table

ddi_table(with_inhibitor, without_inhibitor, ["auc_inf_obs", "cmax"], dim="individual")
```

| parameter | unit | n_test | n_reference | ratio | ci_low | ci_high | kind | strength | uncertain | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auc_inf_obs | hour * milligram / liter | 12 | 12 | 2.62 | 2.33 | 2.94 | inhibitor | moderate | False | FDA 2020 |
| cmax | milligram / liter | 12 | 12 | 1.40 | 1.28 | 1.53 | inhibitor | weak | False | FDA 2020 |

Meta-analysis:

```python
from pkpdutils import meta_analysis
from pkpdutils.stats import EffectKind, Study, effects_from_arrays, random_effects

studies = [
    Study(
        "Smith 1990",
        control=ParameterSample(mean=1.2, sd=0.4, n=10),
        treatment=ParameterSample(mean=2.0, sd=0.6, n=10),
    )
]
meta = meta_analysis(studies, EffectKind.LOG_RATIO)
meta.random.estimate, meta.heterogeneity.i2, meta.to_dataframe()
random_effects(
    effects_from_arrays(log_ratios, variances, labels=labels, kind=EffectKind.LOG_RATIO)
)  # effects computed elsewhere
```

`effects_from_arrays` defaults to `EffectKind.HEDGES_G` like `effect_size` and `meta_analysis`, so the kind of an effect computed elsewhere is given explicitly. Every scalar result carries `to_dict` (`Summary`, `TestResult`, `RatioResult`, `BEParameter`, `BEResult`, `EffectSize`, `PooledEffect`, `Heterogeneity`, `Study`, `MetaResult`, `DDIResult`) and every collection a `to_dataframe` (`BEResult`, `MetaResult`).

Figures: `plot_parameters`, `plot_ratio` and `plot_forest`, see [Plotting](plotting.md). Examples: `examples/bioequivalence.py`, `examples/ddi.py` and `examples/meta_analysis.py`. The reference of the modules is in [API: stats](api/stats.md), [API: stats.bioequivalence](api/stats.bioequivalence.md), [API: stats.ddi](api/stats.ddi.md) and [API: stats.meta](api/stats.meta.md).

## References

[^fda_be]: U.S. Food and Drug Administration. *Statistical Approaches to Establishing Bioequivalence.* 2001. See [References](references.md#regulatory-guidance).
[^schuirmann]: Schuirmann DJ. *J Pharmacokinet Biopharm.* 1987;15:657-680. See [References](references.md#statistics).
[^fda_ddi]: U.S. Food and Drug Administration. *Clinical Drug Interaction Studies.* 2020. See [References](references.md#regulatory-guidance).
[^ema_ddi]: European Medicines Agency. *Guideline on the investigation of drug interactions.* 2012. See [References](references.md#regulatory-guidance).
[^chow]: Chow SC, Liu JP. *Design and Analysis of Bioavailability and Bioequivalence Studies.* 3rd ed. 2009, ch. 3. See [References](references.md#statistics).
[^hedges]: Hedges LV. *J Educ Stat.* 1981;6:107-128. See [References](references.md#statistics).
[^dl]: DerSimonian R, Laird N. *Control Clin Trials.* 1986;7:177-188. See [References](references.md#statistics).
[^higgins]: Higgins JPT, Thompson SG. *Stat Med.* 2002;21:1539-1558. See [References](references.md#statistics).
[^holm]: Holm S. *Scand J Stat.* 1979;6:65-70. See [References](references.md#statistics).
[^bh]: Benjamini Y, Hochberg Y. *J R Stat Soc B.* 1995;57:289-300. See [References](references.md#statistics).

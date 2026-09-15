# References

`pkpdutils` implements the standard methods of pharmacokinetic data analysis. These are the textbooks, guidances and publications behind them; cite them when you report an analysis, and cite `pkpdutils` itself as described in [Home](index.md#how-to-cite). Every user guide page cites the entries it builds on.

## Textbooks

**Gabrielsson & Weiner.** The reference for the non-compartmental parameters and their interpretation.

> Gabrielsson J, Weiner D.
> **Pharmacokinetic and Pharmacodynamic Data Analysis: Concepts and Applications.**
> 5th edition. Swedish Pharmaceutical Press; 2016.

**Rowland & Tozer.** Clinical pharmacokinetics, the physiological meaning of clearance, volume and half-life.

> Rowland M, Tozer TN.
> **Clinical Pharmacokinetics and Pharmacodynamics: Concepts and Applications.**
> 4th edition. Lippincott Williams & Wilkins; 2011.

**Gibaldi & Perrier.** The derivations of the area and moment methods.

> Gibaldi M, Perrier D.
> **Pharmacokinetics.**
> 2nd edition. Marcel Dekker; 1982.

## Non-compartmental analysis

**Phoenix WinNonlin.** The rules for the selection of the terminal phase (best fit by adjusted R²) and the linear-up/log-down trapezoidal rule follow the Phoenix NCA implementation.

> Certara.
> **Phoenix WinNonlin User's Guide: Noncompartmental Analysis.**
> Certara USA, Inc.

## Regulatory guidance

**FDA drug interaction guidance.** The thresholds of the classification of inhibitors, inducers and sensitive substrates.

> U.S. Food and Drug Administration.
> **Clinical Drug Interaction Studies - Cytochrome P450 Enzyme- and Transporter-Mediated Drug Interactions. Guidance for Industry.**
> 2020.

**EMA drug interaction guideline.**

> European Medicines Agency.
> **Guideline on the investigation of drug interactions.** CPMP/EWP/560/95/Rev. 1.
> 2012.

**FDA bioequivalence guidance.** The 80–125 % acceptance range of the 90 % confidence interval of the geometric mean ratio.

> U.S. Food and Drug Administration.
> **Statistical Approaches to Establishing Bioequivalence. Guidance for Industry.**
> 2001.

## Statistics

**Two one-sided tests.**

> Schuirmann DJ.
> **A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability.**
> *Journal of Pharmacokinetics and Biopharmaceutics.* 1987;15(6):657-680.
> [doi:10.1007/BF01068419](https://doi.org/10.1007/BF01068419)

**Dose proportionality.**

> Smith BP, Vandenhende FR, DeSante KA, Farid NA, Welch PA, Callaghan JT, Forgue ST.
> **Confidence interval criteria for assessment of dose proportionality.**
> *Pharmaceutical Research.* 2000;17(10):1278-1283.
> [doi:10.1023/A:1026451721686](https://doi.org/10.1023/A:1026451721686)

**Effect sizes.** The small sample correction \(J\) of Hedges' g and its variance.

> Hedges LV.
> **Distribution theory for Glass's estimator of effect size and related estimators.**
> *Journal of Educational Statistics.* 1981;6(2):107-128.
> [doi:10.3102/10769986006002107](https://doi.org/10.3102/10769986006002107)

**Random effects meta-analysis.** The between-study variance \(\tau^2\) of the random effects model.

> DerSimonian R, Laird N.
> **Meta-analysis in clinical trials.**
> *Controlled Clinical Trials.* 1986;7(3):177-188.
> [doi:10.1016/0197-2456(86)90046-2](https://doi.org/10.1016/0197-2456(86)90046-2)

**Crossover designs.** The period-difference analysis of the 2x2 crossover, the tests of the period and the carryover effect and the within-subject CV.

> Chow SC, Liu JP.
> **Design and Analysis of Bioavailability and Bioequivalence Studies.**
> 3rd edition. Chapman & Hall/CRC; 2009.

**Heterogeneity.** \(I^2\) and \(H^2\) of a meta-analysis.

> Higgins JPT, Thompson SG.
> **Quantifying heterogeneity in a meta-analysis.**
> *Statistics in Medicine.* 2002;21(11):1539-1558.
> [doi:10.1002/sim.1186](https://doi.org/10.1002/sim.1186)

**Multiple comparisons.**

> Holm S.
> **A simple sequentially rejective multiple test procedure.**
> *Scandinavian Journal of Statistics.* 1979;6(2):65-70.

> Benjamini Y, Hochberg Y.
> **Controlling the false discovery rate: a practical and powerful approach to multiple testing.**
> *Journal of the Royal Statistical Society B.* 1995;57(1):289-300.
> [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

**Agreement of two measurements.** The Bland-Altman plot of `plot_bland_altman`.

> Bland JM, Altman DG.
> **Statistical methods for assessing agreement between two methods of clinical measurement.**
> *The Lancet.* 1986;327(8476):307-310.
> [doi:10.1016/S0140-6736(86)90837-8](https://doi.org/10.1016/S0140-6736(86)90837-8)

**Meta-analysis reference data.** The BCG vaccine trials used as the regression reference of the meta-analysis (`tests/data/reference/meta_bcg.json`), analysed with the R package `metafor`.

> Colditz GA, Brewer TF, Berkey CS, et al.
> **Efficacy of BCG vaccine in the prevention of tuberculosis: meta-analysis of the published literature.**
> *JAMA.* 1994;271(9):698-702.

> Viechtbauer W.
> **Conducting meta-analyses in R with the metafor package.**
> *Journal of Statistical Software.* 2010;36(3):1-48.
> [doi:10.18637/jss.v036.i03](https://doi.org/10.18637/jss.v036.i03)

**Bootstrap.** The parametric bootstrap and the delta method of the uncertainty of the NCA parameters (ch. 5 and 6) and the residual bootstrap of a fit (ch. 9).

> Efron B, Tibshirani RJ.
> **An Introduction to the Bootstrap.**
> Chapman & Hall/CRC; 1993.

**Nonlinear regression.** The covariance of the parameters from the Jacobian, the t based confidence intervals and the delta method for derived parameters.

> Seber GAF, Wild CJ.
> **Nonlinear Regression.**
> Wiley; 1989.

**Model selection by AICc and Akaike weights.** The ranking of the candidate models of a fit and the correction for the small sample size.

> Burnham KP, Anderson DR.
> **Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach.**
> 2nd edition. Springer; 2002.

## Data formats

The exchange formats of `pkpdutils.io`: the event records of NONMEM and Monolix, the two tables of PKNCA and the CDISC ADaM dataset of a non-compartmental analysis.

**NONMEM event records.** The one row per event data format, `EVID`, `MDV`, `AMT`, `RATE`, and the repeated doses of `ADDL`, `II` and `SS`.

> Bauer RJ.
> **NONMEM Tutorial Part I: Description of Commands and Options, with Simple Examples of Population Analysis.**
> *CPT: Pharmacometrics & Systems Pharmacology.* 2019;8(8):525-537.
> [doi:10.1002/psp4.12404](https://doi.org/10.1002/psp4.12404)

**Monolix data format.** The column names of the same format in the MonolixSuite (`AMOUNT`, `OBSERVATION`, `INFUSION DURATION`, `ADDITIONAL DOSES`, `INTERDOSE INTERVAL`, `STEADY STATE`).

> Lixoft.
> **MonolixSuite documentation: data format.**
> [monolix.lixoft.com/data-format](https://monolix.lixoft.com/data-format/)

**PKNCA.** The two table layout of the concentrations and the doses of the R package for automatic non-compartmental analysis.

> Denney W, Duvvuri S, Buckeridge C.
> **Simple, automatic noncompartmental analysis: the PKNCA R package.**
> *Journal of Pharmacokinetics and Pharmacodynamics.* 2015;42:S65.
> [cran.r-project.org/package=PKNCA](https://cran.r-project.org/package=PKNCA)

**CDISC ADaM ADNCA.** The analysis dataset of the input data of a non-compartmental analysis (`USUBJID`, `PARAMCD`, `AVAL`, `AFRLT`, `ARRLT`, `DOSEA`, `DTYPE`).

> CDISC.
> **ADaM Implementation Guide for Non-compartmental Analysis Input Data (ADNCA).**
> 2024.

> CDISC.
> **Analysis Data Model (ADaM) Implementation Guide.**
> [cdisc.org/standards/foundational/adam](https://www.cdisc.org/standards/foundational/adam)

## Software

**pint, xarray, scipy.** The libraries the package is built on.

> Hoyer S, Hamman J.
> **xarray: N-D labeled arrays and datasets in Python.**
> *Journal of Open Research Software.* 2017;5(1):10.
> [doi:10.5334/jors.148](https://doi.org/10.5334/jors.148)

> Virtanen P, Gommers R, Oliphant TE, et al.
> **SciPy 1.0: fundamental algorithms for scientific computing in Python.**
> *Nature Methods.* 2020;17:261-272.
> [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

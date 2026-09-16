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

**Shargel & Yu.** A general biopharmaceutics and pharmacokinetics textbook, parallel to Rowland & Tozer and Gibaldi & Perrier.

> Ducharme MP, Shargel L, Yu ABC.
> **Shargel and Yu's Applied Biopharmaceutics & Pharmacokinetics.**
> 8th edition. McGraw Hill; 2022. ISBN 978-1-260-14299-0.

**Bonate.** General pharmacokinetic-pharmacodynamic modeling and simulation, complementary to Gabrielsson & Weiner.

> Bonate PL.
> **Pharmacokinetic-Pharmacodynamic Modeling and Simulation.**
> 2nd edition. Springer; 2011.
> [doi:10.1007/978-1-4419-9485-1](https://doi.org/10.1007/978-1-4419-9485-1)

## Non-compartmental analysis

**Phoenix WinNonlin.** The rules for the selection of the terminal phase (best fit by adjusted R²) and the linear-up/log-down trapezoidal rule follow the Phoenix NCA implementation.

> Certara.
> **Phoenix WinNonlin User's Guide: Noncompartmental Analysis.**
> Certara USA, Inc.

**Non-compartmental analysis review.** The review-article companion to Gabrielsson & Weiner, an NCA methodology overview.

> Gabrielsson J, Weiner D.
> **Non-compartmental analysis.**
> *Methods in Molecular Biology.* 2012;929:377-389.
> [doi:10.1007/978-1-62703-050-2_16](https://doi.org/10.1007/978-1-62703-050-2_16)

**AUC integration methods.** The classic comparisons of numerical integration algorithms behind the trapezoidal rules of `AUCMethod`, and the direct justification of the linear-up/log-down rule.

> Yeh KC, Kwan KC.
> **A comparison of numerical integrating algorithms by trapezoidal, Lagrange, and spline approximation.**
> *Journal of Pharmacokinetics and Biopharmaceutics.* 1978;6(1):79-98.
> [doi:10.1007/BF01066064](https://doi.org/10.1007/BF01066064)

> Chiou WL.
> **Critical evaluation of the potential error in pharmacokinetic studies of using the linear trapezoidal rule method for the calculation of the area under the plasma level-time curve.**
> *Journal of Pharmacokinetics and Biopharmaceutics.* 1978;6(6):539-546.
> [doi:10.1007/BF01062108](https://doi.org/10.1007/BF01062108)

> Purves RD.
> **Optimum numerical integration methods for estimation of area-under-the-curve (AUC) and area-under-the-moment-curve (AUMC).**
> *Journal of Pharmacokinetics and Biopharmaceutics.* 1992;20(3):211-226.
> [doi:10.1007/BF01062525](https://doi.org/10.1007/BF01062525)

**Sparse and destructive sampling.** The variance of an AUC estimated from group data with one time point per subject, and its extension to sparse serial sampling.

> Bailer AJ.
> **Testing for the equality of area under the curves when using destructive measurement techniques.**
> *Journal of Pharmacokinetics and Biopharmaceutics.* 1988;16(3):303-309.
> [doi:10.1007/BF01062139](https://doi.org/10.1007/BF01062139)

> Nedelman JR, Gibiansky E, Lau DTW.
> **Applying Bailer's method for AUC confidence intervals to sparse sampling.**
> *Pharmaceutical Research.* 1995;12(1):124-128.
> [doi:10.1023/A:1016255124336](https://doi.org/10.1023/A:1016255124336)

**Delta method vs. bootstrap for AUC ratios.** A pharmacokinetics-specific comparison of confidence interval methods for an exposure metric.

> Jaki T, Wolfsegger MJ, Ploner M.
> **Confidence intervals for ratios of AUCs in the case of serial sampling: a comparison of seven methods.**
> *Pharmaceutical Statistics.* 2009;8(1):12-24.
> [doi:10.1002/pst.321](https://doi.org/10.1002/pst.321)

**NonCompart.** A comparable open-source, CDISC SDTM-oriented non-compartmental analysis implementation.

> Bae KS.
> **NonCompart: Noncompartmental Analysis for Pharmacokinetic Data.**
> CRAN package.
> [cran.r-project.org/package=NonCompart](https://cran.r-project.org/package=NonCompart)

**NonCompart validation report.** The published per-subject Phoenix WinNonlin results for the theophylline and the indomethacin dataset, the reference values of [Validation](validation.md).

> Han S.
> **Validation of Noncompartmental Analysis Performed by NonCompart R package.**
> 2018.
> [asancpt.github.io/NonCompart-tests](https://asancpt.github.io/NonCompart-tests/)

**Theophylline dataset.** The twelve subject oral single dose study distributed as `datasets::Theoph` of R, one of the two validation datasets.

> Boeckmann AJ, Sheiner LB, Beal SL.
> **NONMEM Users Guide: Part V.**
> NONMEM Project Group, University of California, San Francisco; 1994.

**Indomethacin dataset.** The six subject intravenous bolus study distributed as `datasets::Indometh` of R, the second validation dataset.

> Kwan KC, Breault GO, Umbenhauer ER, McMahon FG, Duggan DE.
> **Kinetics of indomethacin absorption, elimination, and enterohepatic circulation in man.**
> *Journal of Pharmacokinetics and Biopharmaceutics.* 1976;4(3):255-280.
> [doi:10.1007/BF01063617](https://doi.org/10.1007/BF01063617)

## Regulatory guidance

**FDA drug interaction guidance.** The thresholds of the classification of inhibitors, inducers and sensitive substrates.

> U.S. Food and Drug Administration.
> **Clinical Drug Interaction Studies - Cytochrome P450 Enzyme- and Transporter-Mediated Drug Interactions. Guidance for Industry.**
> 2020.

**EMA drug interaction guideline.**

> European Medicines Agency.
> **Guideline on the investigation of drug interactions.** CPMP/EWP/560/95/Rev. 1.
> 2012.

**FDA bioequivalence guidance.** The 80-125 % acceptance range of the 90 % confidence interval of the geometric mean ratio.

> U.S. Food and Drug Administration.
> **Statistical Approaches to Establishing Bioequivalence. Guidance for Industry.**
> 2026.

**FDA bioavailability guidance.** General bioavailability studies and AUC-based exposure endpoints.

> U.S. Food and Drug Administration.
> **Bioavailability Studies Submitted in NDAs or INDs - General Considerations. Guidance for Industry.**
> 2022.
> [fda.gov](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/bioavailability-studies-submitted-ndas-or-inds-general-considerations)

**FDA bioequivalence guidance for ANDAs.** The Cmax/AUC bioequivalence acceptance criteria for generic drugs.

> U.S. Food and Drug Administration.
> **Bioequivalence Studies With Pharmacokinetic Endpoints for Drugs Submitted Under an ANDA. Guidance for Industry.**
> 2026.
> [fda.gov](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/bioequivalence-studies-pharmacokinetic-endpoints-drugs-submitted-under-abbreviated-new-drug)

**EMA bioequivalence guideline.** European bioequivalence acceptance criteria and design requirements.

> European Medicines Agency.
> **Guideline on the Investigation of Bioequivalence.** CPMP/EWP/QWP/1401/98 Rev. 1.
> 2010.
> [ema.europa.eu](https://www.ema.europa.eu/en/investigation-bioequivalence-scientific-guideline)

**ICH M13A.** The current harmonized (FDA/EMA/PMDA) bioequivalence design and analysis standard for immediate-release solid oral dosage forms.

> International Council for Harmonisation.
> **ICH Harmonised Guideline: Bioequivalence for Immediate-Release Solid Oral Dosage Forms M13A.**
> 2024.
> [database.ich.org](https://database.ich.org/sites/default/files/ICH_M13A_Step4_Final_Guideline_2024_0723.pdf)

**FDA population pharmacokinetics guidance.** Context for group and batch analyses and the handoff from a non-compartmental to a population pharmacokinetic analysis.

> U.S. Food and Drug Administration.
> **Population Pharmacokinetics. Guidance for Industry.**
> 2022.
> [fda.gov](https://www.fda.gov/media/128793/download)

**FDA exposure-response guidance.** Regulatory context for exposure-response and pharmacodynamic modeling.

> U.S. Food and Drug Administration.
> **Exposure-Response Relationships - Study Design, Data Analysis, and Regulatory Applications. Guidance for Industry.**
> 2003.
> [fda.gov](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/exposure-response-relationships-study-design-data-analysis-and-regulatory-applications)

**ICH E9(R1).** The conceptual framing of what a comparison or an effect estimate represents.

> International Council for Harmonisation.
> **ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials to the Guideline on Statistical Principles for Clinical Trials E9(R1).**
> 2019.
> [database.ich.org](https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf)

**FDA drug interaction table.** The FDA's living reference tables of clinical index substrates, inhibitors and inducers.

> U.S. Food and Drug Administration.
> **Drug Development and Drug Interactions: Table of Substrates, Inhibitors and Inducers.**
> [fda.gov](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers)

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

**Meta-analysis.** The standard meta-analysis textbook, complementary to DerSimonian & Laird and Higgins & Thompson above.

> Borenstein M, Hedges LV, Higgins JPT, Rothstein HR.
> **Introduction to Meta-Analysis.**
> 2nd edition. Wiley; 2021.
> [doi:10.1002/9781119558378](https://doi.org/10.1002/9781119558378)

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

**Drug-drug interaction study design.** The industry perspective on study design and classification that FDA's drug interaction guidance later formalized.

> Bjornsson TD, Callaghan JT, Einolf HJ, et al.
> **The conduct of in vitro and in vivo drug-drug interaction studies: a Pharmaceutical Research and Manufacturers of America (PhRMA) perspective.**
> *Journal of Clinical Pharmacology.* 2003;43(5):443-469.
> [doi:10.1177/0091270003252519](https://doi.org/10.1177/0091270003252519)

**Draper and Smith.** The confidence band of a simple linear regression, drawn around the terminal regression of the NCA figure.

> Draper NR, Smith H.
> **Applied Regression Analysis.**
> 3rd edition. Wiley; 1998.
> [doi:10.1002/9781118625590](https://doi.org/10.1002/9781118625590)

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
> 2021.
> [cdisc.org/standards/foundational/adam/adamig-non-compartmental-analysis-input-data-v1-0](https://www.cdisc.org/standards/foundational/adam/adamig-non-compartmental-analysis-input-data-v1-0)

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

## Further reading

- [PKanalix documentation](https://monolixsuite.slp-software.com/pkanalix/2024R1/) - Lixoft/Simulations Plus's documentation for PKanalix, a GUI application for non-compartmental and compartmental PK analysis covering NCA rules, custom parameters, bioequivalence and regulatory reporting, for analysts who want a graphical cross-check to code-based NCA.
- [PKNCA package documentation site](https://humanpred.github.io/pknca/) - the official documentation and vignette site for the PKNCA R package, with worked examples, AUC/half-life method articles, sparse sampling and bioequivalence vignettes, for analysts comparing `pkpdutils`'s NCA conventions against PKNCA's.
- [PKNCA - theophylline vignette](https://cran.r-project.org/web/packages/PKNCA/vignettes/v02-example-theophylline.html) - the worked `pk.nca` example on `datasets::Theoph` whose printed results are the PKNCA reference values of [Validation](validation.md).
- [PKNCA - FDA-oriented introduction vignette](https://humanpred.github.io/pknca/articles/v31-FDA-introduction.html) - frames PKNCA's design goals (regulatory readiness, reproducibility, CDISC-aligned data structures) for a regulatory audience, for analysts preparing regulatory NCA submissions.
- [PKNCA training session vignette](https://humanpred.github.io/pknca/articles/v30-training-session.html) - a step-by-step walkthrough of an NCA workflow in R, for analysts new to R-based NCA.
- [Pumas - handling missing and BLQ data](https://docs.pumas.ai/stable/nca/blq_handling/) - the BLQ conventions of the Pumas NCA (`:first`, `:middle`, `:last` with `:keep`, `:drop` and numeric imputation), the positional axis `BLQRules` implements, for analysts porting an analysis between the two.
- [NonCompart on CRAN](https://cran.r-project.org/package=NonCompart) - an alternative open-source, CDISC SDTM-oriented NCA implementation in R with automatic/manual slope selection and multiple trapezoidal methods, for readers comparing implementations of the same NCA rules `pkpdutils` implements.
- [CDISC "Introduction to PK Analysis" course](https://www.cdisc.org/education/course/introduction-pk-analysis) - CDISC's on-demand course introducing PK analysis concepts alongside CDISC data standards, for clinical data and statistical programmers who need the CDISC ADaM/ADNCA context that `io.py`'s `read_adnca` targets.
- [CDISC ADaMIG for Non-compartmental Analysis Input Data v1.0](https://www.cdisc.org/standards/foundational/adam/adamig-non-compartmental-analysis-input-data-v1-0) - the implementation guide page itself, useful for programmers building ADNCA-compliant datasets, in addition to being cited above as the formal CDISC ADaM ADNCA reference.
- [FDA "Drug Development and Drug Interactions" table](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers) - the FDA's living reference tables of clinical index substrates, inhibitors and inducers with strong/moderate/weak classification, the practical companion to `stats/ddi.py`'s `DDIThresholds`.
- [Holford NHG - Advanced Pharmacometrics teaching page](https://holford.fmhs.auckland.ac.nz/teaching/pharmacometrics/advanced) - Nick Holford's pharmacometrics course materials at the University of Auckland, covering PK/PD modeling concepts beyond NCA, for readers wanting the population-modeling perspective on the same parameters `pkpdutils` computes non-compartmentally.
- [Mould & Upton, "Basic concepts in population modeling, simulation, and model-based drug development."](https://doi.org/10.1038/psp.2012.4) *CPT:PSP.* 2012;1(9):e6 - part 1 of a three-part introductory tutorial series for pharmacometrics newcomers.
- [Mould & Upton, "...part 2: introduction to pharmacokinetic modeling methods."](https://doi.org/10.1038/psp.2013.14) *CPT:PSP.* 2013;2(4):e38 - part 2, focused on PK modeling methods, a natural next step after this library's NCA and fitting docs.
- [Upton & Mould, "...part 3: introduction to pharmacodynamic modeling methods."](https://doi.org/10.1038/psp.2013.71) *CPT:PSP.* 2014;3:e88 - part 3, focused on PD modeling methods, a companion to [Pharmacodynamics](pd.md).
- [PKNCA GitHub repository](https://github.com/humanpred/pknca) - the source repository, for readers who want to compare `pkpdutils`'s NCA implementation choices against PKNCA's source directly.

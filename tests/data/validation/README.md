# Validation data

The datasets and the published reference values `tests/nca/test_validation.py` and `scripts/validation.py` compare the non-compartmental analysis against. The page [`docs/validation.md`](../../../docs/validation.md) explains the comparison, the option mapping between the tools and the known differences; this file names where every file came from and under which license.

## The datasets

| file | dataset | subjects | columns |
| --- | --- | --- | --- |
| `theoph.csv` | `datasets::Theoph` of R, the theophylline study | 12 | `Subject`, `Wt` (kg), `Dose` (mg/kg), `Time` (h), `conc` (mg/L) |
| `indometh.csv` | `datasets::Indometh` of R, the indomethacin study | 6 | `Subject`, `time` (h), `conc` (µg/mL) |

Both files are copies of the CSV exports of the [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) mirror, with the row-number column of the export removed and nothing else changed:

- <https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv>
- <https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Indometh.csv>

Retrieved on 2026-09-16.

**License.** The `datasets` package is part of the R distribution and is licensed GPL-2 / GPL-3, as R itself. The Rdatasets mirror redistributes it unchanged under the license of the originating package. The two files here are redistributed under the same terms and are used as test fixtures only.

**Origin of the data.**

- `Theoph`: the theophylline pharmacokinetics data of the NONMEM user guide, twelve subjects who received a single oral dose and gave eleven samples over about 25 hours. Boeckmann AJ, Sheiner LB, Beal SL. *NONMEM Users Guide: Part V.* NONMEM Project Group, University of California, San Francisco; 1994. Also in Davidian M, Giltinan DM. *Nonlinear Models for Repeated Measurement Data.* Chapman & Hall; 1995, and Pinheiro JC, Bates DM. *Mixed-effects Models in S and S-PLUS.* Springer; 2000.
- `Indometh`: six subjects who received a single intravenous bolus of 25 mg indomethacin and gave eleven samples over eight hours. Kwan KC, Breault GO, Umbenhauer ER, McMahon FG, Duggan DE. **Kinetics of indomethacin absorption, elimination, and enterohepatic circulation in man.** *Journal of Pharmacokinetics and Biopharmaceutics.* 1976;4(3):255-280. [doi:10.1007/BF01063617](https://doi.org/10.1007/BF01063617)

The `Indometh` dataset itself carries no dose column, and the R documentation only says that "each of the six subjects were given an intravenous injection of indometacin". The 25 mg bolus is the dose of the study of Kwan et al. and is what the published WinNonlin comparison used (`dose=25, adm="Bolus"`); it is taken from there, not derived from the data.

## The reference values

`reference.json` holds every number the analysis is compared against, one entry per dataset, subject and parameter with its `value`, its `unit`, the identifier of its `source` and the relative `tolerance` of the comparison. Its `sources` section carries the tool, the author, the URL, the retrieval date and the settings of every source. Nothing in the file is computed by `pkpdutils`, and no number without a published source is in it.

### Phoenix WinNonlin 6.3 and 7.0

Han S. **Validation of Noncompartmental Analysis Performed by NonCompart R package.** 2018-08-07. <https://asancpt.github.io/NonCompart-tests/>

The report validates the NonCompart R package against Phoenix WinNonlin on these two datasets and publishes the raw WinNonlin per-subject output as CSV files. Four of them are the source of the four WinNonlin cases:

| case of `reference.json` | file |
| --- | --- |
| `theoph-winnonlin-linear` | [`Final_Parameters_Pivoted_Theoph_Linear.csv`](https://raw.githubusercontent.com/asancpt/NonCompart-tests/master/Final_Parameters_Pivoted_Theoph_Linear.csv) |
| `theoph-winnonlin-linear-log` | [`Final_Parameters_Pivoted_Theoph_Log.csv`](https://raw.githubusercontent.com/asancpt/NonCompart-tests/master/Final_Parameters_Pivoted_Theoph_Log.csv) |
| `indometh-winnonlin-linear` | [`Final_Parameters_Pivoted_Indometh_Linear.csv`](https://raw.githubusercontent.com/asancpt/NonCompart-tests/master/Final_Parameters_Pivoted_Indometh_Linear.csv) |
| `indometh-winnonlin-linear-log` | [`Final_Parameters_Pivoted_Indometh_Log.csv`](https://raw.githubusercontent.com/asancpt/NonCompart-tests/master/Final_Parameters_Pivoted_Indometh_Log.csv) |

The report states the settings of every run: `tblNCA(Theoph, "Subject", "Time", "conc", dose=320, concUnit="mg/L")` with `down="Log"` for the mixed rule, and `tblNCA(Indometh, "Subject", "time", "conc", dose=25, adm="Bolus", concUnit="mg/L", R2ADJ=0.8)`, again with `down="Log"` for the mixed rule. The WinNonlin columns are mapped onto the variables of `pkpdutils` one to one; the percentages (`AUC_%Extrap_obs`, `AUC_%Back_Ext_obs`) are stored as the fractions `pkpdutils` reports. The report's files are retrieved on 2026-09-16 and carry 8 to 15 significant digits per number.

The report also publishes runs of the indomethacin dataset as an infusion and as an extravascular dose. They are not used, see the "What is not covered" section of `docs/validation.md`.

### PKNCA

Denney B. **Computing NCA Parameters for Theophylline.** PKNCA vignette. <https://cran.r-project.org/web/packages/PKNCA/vignettes/v02-example-theophylline.html>

The vignette runs `pk.nca` on `datasets::Theoph` with the PKNCA defaults (`auc.method = "lin up/log down"`, `min.hl.points = 3`, `allow.tmax.in.half.life = FALSE`) and prints

- the per-subject results of subject 1 (`cmax`, `tmax`, `tlast`, `clast.obs`, `lambda.z`) and of subject 6 (the same plus `auclast`), which are the case `theoph-pknca`;
- the summary over all twelve subjects, which is the case `theoph-pknca-summary`. PKNCA prints a coefficient of variation in percent; it is stored as the fraction `pkpdutils` reports.

The vignette prints no per-subject clearance, volume or mean residence time, so the reference file carries none for PKNCA. It also prints `auclast` over the interval 0 to 24 h, which is not comparable to `partial_auc` and is not recorded, see `docs/validation.md`.

No R installation was used: every number of this directory is transcribed from the sources above.

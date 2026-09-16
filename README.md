# pkpdutils: pharmacokinetic and pharmacodynamic analysis
[![GitHub Actions CI/CD Status](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml)
[![Documentation](https://img.shields.io/badge/docs-pkpdutils-3f51b5.svg)](https://matthiaskoenig.github.io/pkpdutils)
[![Version](https://img.shields.io/pypi/v/pkpdutils.svg)](https://pypi.org/project/pkpdutils/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pkpdutils.svg)](https://pypi.org/project/pkpdutils/)
[![MIT License](https://img.shields.io/pypi/l/pkpdutils.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

`pkpdutils` is a python library for the pharmacokinetic (PK) and pharmacodynamic (PD) analysis of timecourses and parameters. It was formerly published as `pkdb-analysis`, the analysis toolbox of [PK-DB](https://pk-db.com); version 1.0.0 is a rewrite without any PK-DB dependency.

Features include

- **non-compartmental analysis** - exposure, peak, terminal phase, clearance and volume parameters of concentration and effect timecourses, single dose and multiple dosing (every dosing interval, steady state, accumulation), with units
- **data formats** - read event records (NONMEM, Monolix), the two PKNCA tables and the CDISC ADaM ADNCA dataset, write event records back
- **uncertainty** - bootstrap and delta method propagation for group timecourses (mean ± SD), summary statistics over individuals
- **curve fitting** - exponential, Bateman, Emax, dose proportionality and covariate models with standard errors, confidence intervals and model comparison
- **statistics on parameters** - significance tests, geometric mean ratios, bioequivalence, classification of drug–drug interactions, meta-analysis
- **figures** - timecourses, NCA diagnostics, fits, parameter distributions, forest and ratio plots

All data structures are [xarray](https://xarray.dev) datasets with [pint](https://pint.readthedocs.io) units, so many timecourses are analysed in one vectorized call.

The documentation is available at [https://matthiaskoenig.github.io/pkpdutils](https://matthiaskoenig.github.io/pkpdutils).

If you have any questions or issues please [open an issue](https://github.com/matthiaskoenig/pkpdutils/issues).

## Quickstart

A study of twelve subjects in three dose groups, from the event table it arrives in to the parameter table and the figure of the report. The table is [study.csv](docs/data/study.csv), which the first walk-through of the [Workflows](https://matthiaskoenig.github.io/pkpdutils/workflows/) builds:

```python
import pandas as pd

from pkpdutils import Route, Timecourses, nca, summary_table
from pkpdutils.plot import plot_mean_timecourse

# [study.csv](https://raw.githubusercontent.com/matthiaskoenig/pkpdutils/develop/docs/data/study.csv):
# ID, TIME, DV, AMT, EVID and the dose group
events = pd.read_csv("study.csv")
batch = Timecourses.from_events(
    events,
    time_unit="hr",
    unit="mg/l",
    dose_unit="mg",
    route=Route.ORAL,
    covariates=["dose"],
)
result = nca(batch)
print(
    summary_table(
        result,
        "individual",
        by="dose",
        parameters=["auc_inf_obs", "cmax", "thalf", "cl_f"],
    ).to_string(index=False)
)
plot_mean_timecourse(batch, by="dose").savefig("study_curves.png", dpi=120)
```

![The mean curve of every dose group with its standard deviation, linear and semi-logarithmic](https://raw.githubusercontent.com/matthiaskoenig/pkpdutils/develop/docs/images/nca_batch_curves.png)

The same steps with the table built in place, the parameters printed and four more walk-throughs (bioequivalence, drug-drug interaction, steady state, dose proportionality) are in the [Workflows](https://matthiaskoenig.github.io/pkpdutils/workflows/) of the documentation; the [Gallery](https://matthiaskoenig.github.io/pkpdutils/gallery/) shows a figure and a snippet for every example of the repository.

## How to cite
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

If you use `pkpdutils` please cite the archived software on [Zenodo](https://doi.org/10.5281/zenodo.3997539):

> König, M. & Grzegorzewski, J. (2026). *pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters* (Version 1.0.0) \[Computer software\]. Zenodo. https://doi.org/10.5281/zenodo.22772056

## Installation

`pkpdutils` requires python >= 3.13 and is available from [pypi](https://pypi.python.org/pypi/pkpdutils):

```bash
uv add pkpdutils
```

or with pip

```bash
pip install pkpdutils
```

See [Installation](https://matthiaskoenig.github.io/pkpdutils/installation/) for details and [Development](https://matthiaskoenig.github.io/pkpdutils/development/) for working on the repository.

## License

- Source Code: [MIT](https://opensource.org/license/MIT)
- Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

## Funding

Matthias König is supported by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 "QuaLiPerF (Quantifying Liver Perfusion-Function Relationship in Complex Resection - A Systems Medicine Approach)" by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA).

Matthias König was supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054).

© 2018-2026 Matthias König & Jan Grzegorzewski.

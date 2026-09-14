# pkpdutils: pharmacokinetic and pharmacodynamic analysis
[![GitHub Actions CI/CD Status](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml) [![Documentation](https://img.shields.io/badge/docs-pkpdutils-3f51b5.svg)](https://matthiaskoenig.github.io/pkpdutils) [![Version](https://img.shields.io/pypi/v/pkpdutils.svg)](https://pypi.org/project/pkpdutils/) [![Python Versions](https://img.shields.io/pypi/pyversions/pkpdutils.svg)](https://pypi.org/project/pkpdutils/) [![MIT License](https://img.shields.io/pypi/l/pkpdutils.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

`pkpdutils` is a python library for the pharmacokinetic (PK) and pharmacodynamic (PD) analysis of timecourses and parameters. The source code is available from [https://github.com/matthiaskoenig/pkpdutils](https://github.com/matthiaskoenig/pkpdutils). The package was formerly published as `pkdb-analysis`, the analysis toolbox of [PK-DB](https://pk-db.com); version 1.0.0 is a rewrite without any PK-DB dependency.

## Background

A pharmacokinetic study measures the concentration of a substance over time; the parameters which describe such a curve, the exposure `AUC`, the peak `Cmax`, the half-life, the clearance and the volume of distribution, are what studies report, compare and pool. `pkpdutils` computes these parameters from timecourses without a model of the body (non-compartmental analysis), fits the curves and the parameters which need a model of the curve (exponentials, Emax, dose proportionality, covariates), propagates the uncertainty of group data, and provides the statistics used on the parameters: significance tests, bioequivalence, the classification of drug–drug interactions and meta-analysis.

All data structures are [xarray](https://xarray.dev) datasets with [pint](https://pint.readthedocs.io) units, so many timecourses, e.g. all individuals of a study or all runs of a simulation scan, are analysed in one vectorized call.

## Features

- **[Timecourses](timecourses.md)** - `Timecourse` for one curve, `Timecourses` for many, with doses, routes, uncertainties and metadata.
- **[Units](units.md)** - every timecourse and result carries its units, parameters are derived in the units of the input.
- **[Non-compartmental analysis](nca.md)** - exposure, peak, terminal phase, clearance and volume parameters of concentration curves, single dose and steady state, vectorized over a batch, with flags and units.
- **[Plotting](plotting.md)** - timecourses and NCA diagnostics as matplotlib figures.

The methods behind the package are cited in [References](references.md).

## Quickstart

```python
from pkpdutils import Dose, Route, Timecourse, nca_single

tc = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12, 24],
    value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2, 0.03],
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
result = nca_single(tc)
print(result.to_dataframe().T)
```

Continue with [Installation](installation.md), [Timecourses](timecourses.md) and [Non-compartmental analysis](nca.md).

## How to cite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

If you use `pkpdutils` please cite the archived software on [Zenodo](https://doi.org/10.5281/zenodo.3997539):

> König, M. & Grzegorzewski, J. (2026). *pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3997539

## License

- Source Code: [MIT](https://opensource.org/license/MIT)
- Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

## Funding

Matthias König is supported by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 "QuaLiPerF (Quantifying Liver Perfusion-Function Relationship in Complex Resection - A Systems Medicine Approach)" by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA).

Matthias König was supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054).

# Timecourses

A pharmacokinetic timecourse is the concentration of a substance in a tissue over time after a dose; a pharmacodynamic timecourse is an effect over time. `pkpdutils` represents one curve as a `Timecourse` and many curves as a `Timecourses` batch, which is the input of every analysis of the package. The first analysis is the [non-compartmental analysis](nca.md).

## Concepts

**Single curve.** A `Timecourse` holds the sampling times and the values with their units, an optional dose, and metadata (substance, label, tissue). It is a frozen [pydantic](https://docs.pydantic.dev) model: the arrays are converted to `float64`, sorted by time, and duplicate times or an unknown unit raise a `ValueError` when the object is created (a dimensionless value is spelled `unit="dimensionless"`, the empty string is not a unit). Missing values are `NaN` in `value`; every analysis drops them.

**Group data.** Publications report the mean curve of a group with the standard deviation or the standard error and the number of subjects. A `Timecourse` carries these as `sd`, `se` and `n`; the missing one of `sd` and `se` is derived from the other with \(\mathrm{se} = \mathrm{sd}/\sqrt{n}\). The uncertainty analyses of the package propagate them to the parameters, see [Uncertainty](uncertainty.md).

**Doses and routes.** A `Dose` has an `amount` with a dose unit (an amount or an amount per body weight, see [Units](units.md)), a `Route`, the `time` of the administration and, for an infusion, its `duration`. The route decides which parameters an analysis can report: after an intravenous bolus the clearance and the volume are absolute (`cl`, `vz`), after an extravascular dose they are relative to the unknown fraction absorbed (`cl_f`, `vz_f`), and an infusion shifts the mean residence time by half its duration. `Route.ORAL` stands for every extravascular route, and a route is also accepted as a string (`"oral"`, `"iv_bolus"`, ignoring the case). A batch has one route and either all or none of its curves carry a dose; curves with different routes go into separate batches, which keeps the route a property of the whole batch and every analysis unambiguous.

**Batches.** `Timecourses` wraps an [xarray](https://xarray.dev) dataset with a `time` dimension and any number of *sample dimensions*: the individuals of a study, the groups of a publication, the doses of a dose escalation, the dimensions of a simulation scan. Every analysis of the package is vectorized over the sample dimensions and returns a dataset over the same dimensions, so the parameters of a thousand curves are one call. Curves with different sampling times are stored per sample and padded with `NaN`, the `times` and `values` properties return the padded `(samples..., time)` arrays.

**Repeated dosing.** A timecourse is accompanied by its dosing protocol, not a single dose: the vector of the doses given and the times they were given, see [Dosing protocols](#dosing-protocols).

## Dosing protocols

A `Dosing` is a frozen model of the doses given and the times they were given: `amounts`, `times` and `durations` (`None` unless the route is `IV_INFUSION`), one `unit` and one `route` for the whole protocol. The doses are sorted by time on construction and duplicate times raise. `Dosing.single(dose)` wraps a single `Dose` into a protocol of one, `Dosing.from_doses(doses)` builds one from a list of `Dose` objects sharing a unit and a route, and `Dosing.regimen(dose, interval, n_doses)` builds a regular protocol at `dose.time + k * interval`; `DosingRegimen(dose=..., interval=..., n_doses=...).dosing()` delegates to the same constructor and stays the convenient way to describe a regimen.

`n_doses`, `doses` (the protocol as a list of `Dose`), `first` and `last` (the first and the last `Dose`), `intervals` (`np.diff(times)`), `tau` (the common interval when every interval is equal within a relative tolerance, `None` for an irregular protocol or a single dose), `is_regular`, `total_amount` and `shifted(offset)` (a copy with every time shifted by `-offset`) read and transform a protocol.

`Timecourse.dosing` carries the protocol of one curve, `None` without dose information. The constructor also accepts a single `dose: Dose` keyword for backwards compatibility, converted into a protocol of one dose (giving both `dose` and `dosing` raises); `Timecourse.dose` is a read-only property returning the first dose of the protocol (or `None`), so `tc.dose.amount`, `tc.dose.route` and `tc.dose.time` keep working for a single dose curve. `relative_to_dose(which="first" | "last")` shifts the curve and its protocol so that the chosen dose is at time 0, which the non-compartmental analysis of a multiple dose curve uses to report the point parameters from the last dose on, see [Non-compartmental analysis](nca.md).

```python
from pkpdutils import Dose, Dosing, DosingRegimen, Route, Timecourse

dose = Dose(amount=100, unit="mg", time=0, route=Route.ORAL)
protocol = Dosing.regimen(dose, interval=12, n_doses=4)  # 4 doses every 12 hr
same = DosingRegimen(dose=dose, interval=12, n_doses=4).dosing()
print(protocol.tau, protocol.total_amount)  # 12.0, 400.0

tc = Timecourse(
    time=[0.5, 1, 2, 11.5, 12.5, 13, 14, 23.5, 47.5],
    value=[0.9, 1.7, 2.6, 0.4, 1.0, 1.8, 2.5, 0.5, 0.2],
    time_unit="hr",
    unit="mg/l",
    dosing=protocol,
    substance="drug",
)
print(tc.dose.amount)  # the first dose, 100 mg
shifted = tc.relative_to_dose(which="last")
print(shifted.dosing.last.time)  # 0.0
print(shifted.dosing.first.time)  # -36.0
```

## Data layout of a batch

| variable | dimensions | content |
| --- | --- | --- |
| `value` | `(*sample, time)` | the values, `NaN` for missing points |
| `sd`, `se` | `(*sample, time)` | standard deviation and error of group data (optional) |
| `n` | `(*sample)` | number of subjects of group data (optional), one number per sample; an `n` which varies over the time points of a curve is reduced to its maximum with a warning |
| `dose_amount`, `dose_time`, `dose_duration` | `(*sample, dose_index)` | the dosing protocol of every sample (optional), the doses at the front of the row and the remaining columns `NaN`; `dose_duration` is `NaN` without infusion |
| `time` (coordinate) | `(time)` | the shared sampling grid, or an integer index for ragged data |
| `times` | `(*sample, time)` | the sampling times per sample, only for ragged data |

Every variable carries `attrs["units"]`; the dataset carries `substance`, `time_unit` and `unit` in its `attrs`, `tissue` when the curves name one and `route` only when doses are present. Any further metadata (sex, body weight, study) is a coordinate on a sample dimension and travels with the results. Several sample dimensions span their cartesian product: a combination without data is a sample of `NaN` values, which iteration and `sel`/`isel` return as a `Timecourse` with `NaN` values and without a dose. The dose dimension is called `dose_index` so that `dose` stays free as a sample dimension (the dose groups of a dose proportionality study, the dose axis of a simulation scan); a single dose batch has one dose column, and `n_doses`, `first_dose_amount`, `last_dose_time` and `dosing_of` read the protocol of a sample back.

## API

A single curve:

```python
from pkpdutils import Dose, Route, Timecourse

tc = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12, 24],
    value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2, 0.03],
    sd=[0.3, 0.5, 0.4, 0.3, 0.1, 0.05, 0.01],
    n=12,
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
print(tc.se)  # derived from sd and n
print(tc.value_q)  # a pint quantity
print(tc.to_dataframe())
```

A batch from arrays, with the individuals as coordinate labels:

```python
import numpy as np
from pkpdutils import Timecourses

time = np.array([0.5, 1, 2, 4, 8, 12, 24])
values = np.random.default_rng(0).uniform(0, 3, size=(3, time.size))
tcs = Timecourses.from_arrays(
    time,
    values,
    time_unit="hr",
    unit="mg/l",
    dims=("individual",),
    coords={"individual": ["s1", "s2", "s3"]},
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
print(tcs.ds)
print(tcs.sel(individual="s2"))  # a Timecourse
for tc in tcs:  # iteration over the samples
    print(tc.label)
```

The curve with the uncertainty of its group and the batch of three individuals, as `examples/timecourses.py` builds them:

![One group curve with error bars next to a batch of three individual curves](images/timecourses.png)

From a long table (one row per sample and time point) with a dose column, and from a list of `Timecourse` objects:

```python
tcs = Timecourses.from_dataframe(
    df,
    sample=["study", "group"],
    time_unit="hr",
    unit="mg/l",
    sd="sd",
    n="n",
    dose_amount="dose",
    dose_unit="mg",
    route=Route.ORAL,
)
tcs = Timecourses.from_timecourses([tc_a, tc_b], dim="group")
tcs = tc.to_batch(dim="individual", label="s1")  # one curve as a batch of one
```

The labels of the samples keep the dtype of what they came from: a subject column of integers gives an integer coordinate in `from_dataframe`, as the `labels` of `from_timecourses` do.

`Timecourses.relative_to_dose(which="first" | "last")` is the batch counterpart of `Timecourse.relative_to_dose`: every sample is shifted by the time of its own first (or last) dose, and its protocol with it. Equal shifts keep the layout of the batch; shifts which differ from sample to sample move the samples against each other, so the values are placed on the union of the shifted grids with `NaN` where a sample has no point at the time of another.

```python
aligned = tcs.relative_to_dose()  # every first dose at time 0
last = tcs.relative_to_dose(which="last")
```

### Selecting, grouping and averaging a batch

A study arrives as one batch whose groups are coordinates on the individual dimension (the treatment, the dose group, the sex), so the four methods below cut the batch into the pieces an analysis or a figure needs. `select` keeps a batch (`sel` returns a single `Timecourse` and needs a label for every sample dimension), `groupby` walks the groups of a coordinate in the order of their first appearance, `mean` reduces a sample dimension to the group curve with its spread, and `dose_normalized` divides the values by the dose so that the curves of a dose escalation can be overlaid.

```python
arm = tcs.select(treatment="test")  # a label, a list of labels or a slice
heavy = tcs.select(weight=slice(80.0, 100.0))  # a coordinate along a sample dimension

for dose, group in tcs.groupby("dose_group"):
    print(dose, group.n_samples)

group_curve = tcs.select(treatment="test").mean("individual")
print(group_curve.sd, group_curve.n)  # mean +- SD of the group, n subjects
normalized = tcs.dose_normalized()  # values per dose, unit "unit / dose_unit"
```

`mean(dim, spread="sd" | "se", min_n=1)` averages the samples which have a finite value at a time point, carries their standard deviation and standard error (the statistic `spread` names is the one computed from the curves, the other follows from \(\mathrm{se} = \mathrm{sd}/\sqrt{n}\)) and the number of subjects `n`, and sets a point covered by fewer than `min_n` samples to `NaN`. The samples need a shared sampling grid; a ragged batch is placed on the union of its grids first, and `relative_to_dose` aligns samples which were dosed at different times. The group curve carries the dosing protocol of its samples when they share one and the protocol of the first sample with a warning when they do not; it is a `Timecourses` again, so `nca` propagates its spread to the parameters, see [Uncertainty](uncertainty.md).

From a simulation: a dataset with a `_time` dimension and scan dimensions, e.g. the `XResult` of [sbmlsim](https://matthiaskoenig.github.io/sbmlsim):

```python
tcs = Timecourses.from_xresult(
    xres, "[Cve_mid]", dose=Dose(amount=7.5, unit="mg", route=Route.IV_BOLUS)
)
tcs = Timecourses.from_dataset(ds, "[Cve]", unit="mmol/l", time_unit="min")
```

The complete example is `examples/timecourses.py`; the reference of the module is in [API: timecourse](api/timecourse.md).

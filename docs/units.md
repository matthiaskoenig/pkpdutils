# Units

Every timecourse and every result of `pkpdutils` carries its units. The package uses [pint](https://pint.readthedocs.io) with one registry per process, `pkpdutils.units.ureg`; quantities of two registries cannot be combined, which is why nothing in the package creates a registry of its own and why an application which mixes its own quantities with those of the package should use `ureg` as well.

## Concepts

Units enter as strings on the data model (`time_unit="hr"`, `unit="ng/ml"`, `Dose(amount=100, unit="mg")`) and are validated when the object is created; an unknown unit raises a `ValueError`. The numerics of the package run on plain arrays in the units of the input, so nothing is converted behind your back: an `AUC` of a curve in `ng/ml` over `hr` is in `ng/ml·hr`. Results carry the derived unit in `attrs["units"]` of every variable, and the single sample accessors return pint quantities which convert with `.to("mg/l*hr")`.

Two families of parameters have conventional units the package converts to: volumes are reported in `liter` (or `liter/kg` for doses per body weight) and clearances in `liter/hour` (or `liter/hour/kg`), see `normalize_volume` and `normalize_clearance`.

## Dose units

A dose is an amount, in mass (`mg`, `g`), in substance (`mmol`, `µmol`) or in activity (`IU`, for insulin, heparin, vaccines and enzyme replacement), or such an amount per body weight (`mg/kg`, `µmol/kg`, `IU/kg`). `check_dose_unit` accepts exactly these six dimensionalities. Concentrations in mass per volume with a dose in substance (or the other way round) give parameters in mixed units such as `mmol/(ng/ml)`; convert one of them with the molar mass of the substance before the analysis when clearances in `liter/hour` are wanted.

## Custom units

The registry defines `none` (dimensionless count, for data without a unit) and `IU` (international units, a dimension of its own) in addition to the pint defaults, which already know `percent`. A dimensionless quantity is spelled `"dimensionless"`: the empty string is not a unit and `parse_unit("")` says so, because an empty unit composes into the derived units of a result as `"()"`.

## Converting a result

The analysis reports its parameters in the units it derived from the data, which are rarely the units a report asks for: an exposure in `hour * nanogram / milliliter` is written `h*ng/mL` in a submission and a clearance in `liter / hour` is often wanted in `mL/min`. `ParameterResult.to_units` converts the named variables of a finished result, and `NCAOptions.units` does the same as part of the analysis; the numbers of the analysis itself never change, only how the result reports them.

A converted parameter takes its uncertainty, summary and dose normalized variables with it: `auc_inf_obs_se`, `auc_inf_obs_ci_low`, `auc_inf_obs_median` are converted to the same unit and `auc_inf_dn`, the exposure per dose, keeps its dose and follows the numerator (`h*ng/mL` gives `h*ng/mL/mg`). The dimensionless companions (`x_cv`, `x_geocv`, `x_n`) are left alone, and a unit of another dimensionality raises.

```python
import numpy as np

from pkpdutils import Dose, NCAOptions, Route, Timecourse, Timecourses, nca

time = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
curve = Timecourse(
    time=time,
    value=10.0 * (np.exp(-0.2 * time) - np.exp(-1.5 * time)),
    time_unit="hr",
    unit="ng/ml",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    label="s1",
)
batch = Timecourses.from_timecourses([curve])
result = nca(batch)
converted = result.to_units({"cl_f": "mL/min"})
print(result.units("cl_f"), float(result.ds["cl_f"][0]))
print(converted.units("cl_f"), float(converted.ds["cl_f"][0]))

# the same conversion as part of the analysis
options = NCAOptions(units={"cl_f": "mL/min"})
print(nca(batch, options=options).units("cl_f"))
```

```text
liter / hour 2325.782438663154
milliliter / minute 38763.0406443859
milliliter / minute
```

## API

```python
from pkpdutils.units import Q_, check_dose_unit, normalize_clearance, normalize_volume

dose = Q_(100, "mg")
cl = Q_(120, "ml/min")
print(dose, normalize_clearance(cl), normalize_volume(Q_(4200, "ml")))
check_dose_unit("mg/kg")  # ok
try:
    check_dose_unit("mg/l")  # a concentration is not a dose
except ValueError as error:
    print(error)
```

```text
100 milligram 7.199999999999999 liter / hour 4.2 liter
A dose must be in ('[mass]', '[substance]', '[activity_amount]', '[mass] / [mass]', '[substance] / [mass]', '[activity_amount] / [mass]'), not '[mass] / [length] ** 3' ('mg/l')
```

The reference of the module is in [API: units](api/units.md).

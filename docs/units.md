# Units

Every timecourse and every result of `pkpdutils` carries its units. The package uses [pint](https://pint.readthedocs.io) with one registry per process, `pkpdutils.units.ureg`; quantities of two registries cannot be combined, which is why nothing in the package creates a registry of its own and why an application which mixes its own quantities with those of the package should use `ureg` as well.

## Concepts

Units enter as strings on the data model (`time_unit="hr"`, `unit="ng/ml"`, `Dose(amount=100, unit="mg")`) and are validated when the object is created; an unknown unit raises a `ValueError`. The numerics of the package run on plain arrays in the units of the input, so nothing is converted behind your back: an `AUC` of a curve in `ng/ml` over `hr` is in `ng/ml·hr`. Results carry the derived unit in `attrs["units"]` of every variable, and the single sample accessors return pint quantities which convert with `.to("mg/l*hr")`.

Two families of parameters have conventional units the package converts to: volumes are reported in `liter` (or `liter/kg` for doses per body weight) and clearances in `liter/hour` (or `liter/hour/kg`), see `normalize_volume` and `normalize_clearance`.

## Dose units

A dose is an amount, in mass (`mg`, `g`) or in substance (`mmol`, `µmol`), or such an amount per body weight (`mg/kg`, `µmol/kg`). `check_dose_unit` accepts exactly these four dimensionalities. Concentrations in mass per volume with a dose in substance (or the other way round) give parameters in mixed units such as `mmol/(ng/ml)`; convert one of them with the molar mass of the substance before the analysis when clearances in `liter/hour` are wanted.

## Custom units

The registry defines `none` (dimensionless count, for data without a unit) and `IU` (international units, a dimension of its own) in addition to the pint defaults, which already know `percent`.

## API

```python
from pkpdutils.units import Q_, check_dose_unit, normalize_clearance, ureg

dose = Q_(100, "mg")
cl = Q_(120, "ml/min")
print(normalize_clearance(cl))  # 7.2 liter / hour
check_dose_unit("mg/kg")  # ok
check_dose_unit("mg/l")  # ValueError
```

The reference of the module is in [API: units](api/units.md).

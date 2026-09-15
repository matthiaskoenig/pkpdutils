"""Dose proportionality of the exposure from an NCA over a dose escalation.

Run from the root of the repository with `python -m examples.dose_proportionality`.
Writes `dose_proportionality.png` into the working directory.
"""

import numpy as np

from pkpdutils import Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.fit import fit_table, proportionality_test
from pkpdutils.fit.models import Power
from pkpdutils.plot import plot_dose_proportionality

# a dose escalation whose exposure grows slightly faster than the dose (b = 1.15)
time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
doses = np.array([25.0, 50.0, 100.0, 200.0, 400.0])
rng = np.random.default_rng(4)
values = np.stack(
    [
        d**1.15 / 10 * np.exp(-0.25 * time) * rng.lognormal(0, 0.04, time.size)
        for d in doses
    ]
)
batch = Timecourses.from_arrays(
    time,
    values,
    time_unit="hr",
    unit="mg/l",
    dims=("dose",),
    coords={"dose": doses},
    dose={"amount": doses, "unit": "mg"},
    route=Route.IV_BOLUS,
    substance="drug",
)

if __name__ == "__main__":
    result = nca(batch)
    # the dose coordinate of the NCA result carries no unit, the fit needs one
    ds = result.ds.assign_coords(dose=("dose", doses, {"units": "mg"}))
    power = fit_table(Power(), ds, "dose", "auc_inf_obs", dim="dose")
    test = proportionality_test(power, dose_range=(25.0, 400.0))

    console.rule("Power model AUC = a dose^b")
    console.print(
        power.to_dataframe().T.loc[["a", "b", "b_se", "b_ci_low", "b_ci_high", "r2"]]
    )
    console.rule("Confidence interval criterion of Smith et al. over 25-400 mg")
    verdict = (
        "proportional"
        if bool(test.proportional)
        else "inconclusive"
        if bool(test.inconclusive)
        else "not proportional"
    )
    console.print(
        f"b = {float(test.slope):.3f} "
        f"[{float(test.ci_low):.3f}, {float(test.ci_high):.3f}], "
        f"acceptance bounds [{test.bounds[0]:.3f}, {test.bounds[1]:.3f}]"
    )
    console.print("verdict:", verdict)
    plot_dose_proportionality(power, test=test).savefig(
        "dose_proportionality.png", dpi=120
    )
    console.print("written: dose_proportionality.png")

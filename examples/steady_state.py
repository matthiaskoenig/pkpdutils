"""Steady state parameters and superposition.

Run from the root of the repository with `python -m examples.steady_state`.
Writes `steady_state.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, DosingRegimen, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod, superposition
from pkpdutils.plot import plot_timecourse

k, c0 = 0.15, 8.0
dose = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
t = np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])
single = Timecourse(
    time=t,
    value=c0 * np.exp(-k * t),
    time_unit="hr",
    unit="mg/l",
    dose=dose,
    substance="drug",
    label="single dose",
)

if __name__ == "__main__":
    regimen = DosingRegimen(dose=dose, interval=12, n_doses=10)
    predicted = superposition(single, regimen, NCAOptions(auc_method=AUCMethod.LOG))
    console.rule("Predicted multiple dose curve")
    console.print(predicted.to_dataframe().tail())

    console.rule("Steady state parameters of the last interval")
    last = predicted.time >= regimen.dose_times()[-1]
    interval = Timecourse(
        time=predicted.time[last],
        value=predicted.value[last],
        time_unit="hr",
        unit="mg/l",
        dose=dose.model_copy(update={"time": float(regimen.dose_times()[-1])}),
        substance="drug",
        label="steady state",
    )
    result = nca_single(
        interval,
        NCAOptions(
            regimen=DosingRegimen(dose=dose, interval=12), auc_method=AUCMethod.LOG
        ),
    )
    for name in (
        "auc_tau",
        "cmax",
        "cmin_ss",
        "ctrough",
        "cavg",
        "fluctuation",
        "swing",
        "accumulation_ratio",
        "cl_ss",
    ):
        console.print(f"{name:<20} {result.to_quantities()[name]:~P}")
    console.print(
        "predicted accumulation 1/(1-exp(-k tau)) =", 1 / (1 - np.exp(-k * 12))
    )

    fig = plot_timecourse(predicted)
    fig.savefig("steady_state.png", dpi=120)
    console.print("written: steady_state.png")

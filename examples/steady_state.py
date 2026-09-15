"""Multiple dosing: the parameters of every dosing interval and the steady state.

Run from the root of the repository with `python -m examples.steady_state`.
Writes `steady_state.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, Dosing, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod, superposition
from pkpdutils.plot import plot_timecourse

k, c0, tau, n_doses = 0.15, 8.0, 12.0, 10
dose = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
# an intravenous bolus: the curve starts at the dose with C0
t = np.array([0, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])
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
    options = NCAOptions(auc_method=AUCMethod.LOG)
    protocol = Dosing.regimen(dose, interval=tau, n_doses=n_doses)
    predicted = superposition(single, protocol, options)
    console.rule(f"Predicted curve of {n_doses} doses every {tau:g} hr")
    console.print(predicted.to_dataframe().tail())

    # the protocol drives the analysis: every dosing interval, the steady state
    # parameters of the last one and the point parameters from the last dose on
    result = nca_single(predicted, options)

    console.rule("Parameters of the dosing intervals")
    console.print(
        result.intervals()[
            ["interval", "interval_start", "interval_auc", "interval_cmax"]
        ]
    )

    console.rule("Steady state parameters of the last interval")
    quantities = result.to_quantities()
    for name in (
        "n_doses",
        "tau",
        "auc_tau",
        "cmax_ss",
        "cmin_ss",
        "ctrough",
        "cavg",
        "fluctuation",
        "swing",
        "accumulation_ratio",
        "accumulation_ratio_obs",
        "cl_ss",
    ):
        console.print(f"{name:<25} {quantities[name]:~P}")
    console.print(
        "predicted accumulation 1/(1-exp(-k tau)) =", 1 / (1 - np.exp(-k * tau))
    )

    fig = plot_timecourse(predicted)
    fig.savefig("steady_state.png", dpi=120)
    console.print("written: steady_state.png")

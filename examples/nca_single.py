"""Non-compartmental analysis of one timecourse.

Run from the root of the repository with `python -m examples.nca_single`.
Writes `nca_single.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod, TerminalMethod, TerminalPhase
from pkpdutils.plot import plot_nca

# caffeine after an oral dose, mean concentrations of a group
tc = Timecourse(
    time=[0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24],
    value=[0.9, 1.7, 2.6, 2.9, 2.8, 2.5, 2.2, 1.6, 1.2, 0.6, 0.1],
    sd=[0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1, 0.03],
    n=12,
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
    label="healthy volunteers",
)

if __name__ == "__main__":
    console.rule("Default options: linear-up/log-down, best fit terminal phase")
    result = nca_single(tc)
    for name, quantity in result.to_quantities().items():
        console.print(f"{name:<22} {quantity:~P}")
    console.print("flags:", result.flags())

    console.rule(
        "The rule of pkdb_analysis 0.3.1: linear trapezoid, all points after tmax"
    )
    old = nca_single(
        tc,
        NCAOptions(
            auc_method=AUCMethod.LINEAR,
            terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX),
        ),
    )
    console.print(old.to_dataframe().T)

    fig = plot_nca(tc, result)
    fig.savefig("nca_single.png", dpi=120)
    console.print("written: nca_single.png")
    console.print(np.round(result["auc_inf_obs"].values, 3))

"""Uncertainty of the NCA parameters of group timecourses.

A publication reports the mean concentration of a group with its standard
deviation and the number of subjects. The bootstrap and the delta method
propagate this uncertainty to the parameters; individual curves are summarized
over the individuals instead.

Run from the root of the repository with `python -m examples.group_uncertainty`.
Writes `group_uncertainty.png` into the working directory.
"""

import numpy as np

from pkpdutils import Dose, NCAOptions, Route, Timecourse, Timecourses, nca, nca_single
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod, UncertaintyMethod, partial_auc
from pkpdutils.nca.options import BootstrapSpread
from pkpdutils.plot import plot_timecourse

t = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
mean = np.array([1.9, 2.6, 2.4, 1.8, 1.3, 0.95, 0.5, 0.12])
sd = np.array([0.4, 0.5, 0.4, 0.3, 0.25, 0.2, 0.12, 0.04])
group = Timecourse(
    time=t,
    value=mean,
    sd=sd,
    n=10,
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
    label="group mean",
)

if __name__ == "__main__":
    console.rule("Bootstrap (default for group data): uncertainty of the mean curve")
    boot = nca_single(group, NCAOptions(seed=1, n_boot=2000))
    df = boot.to_dataframe().T
    console.print(
        df.loc[
            [
                n
                for n in df.index
                if n.startswith(("auc_inf_obs", "cmax", "thalf", "cl_f"))
            ]
        ]
    )

    console.rule(
        "Bootstrap with the spread of individuals (sd) instead of the mean (se)"
    )
    spread = nca_single(
        group, NCAOptions(seed=1, n_boot=2000, bootstrap_spread=BootstrapSpread.SD)
    )
    console.print(
        spread.to_quantities()["auc_inf_obs_sd"],
        spread.to_quantities()["auc_inf_obs_geocv"],
    )

    console.rule("Delta method")
    delta = nca_single(group, NCAOptions(uncertainty=UncertaintyMethod.DELTA))
    for name in ("auc_inf_obs", "cmax", "thalf"):
        console.print(
            f"{name:<12} {delta.to_quantities()[name]:~P}  se {delta.to_quantities()[name + '_se']:~P}"
        )

    console.rule("Individuals: summarize over the sample dimension")
    rng = np.random.default_rng(2)
    curves = [
        Timecourse(
            time=t,
            value=mean * rng.lognormal(0, 0.15, size=t.size),
            time_unit="hr",
            unit="mg/l",
            dose=Dose(amount=100, unit="mg", route=Route.ORAL),
            substance="caffeine",
            label=f"s{i}",
        )
        for i in range(8)
    ]
    individuals = Timecourses.from_timecourses(curves)
    summary = nca(individuals, NCAOptions(auc_method=AUCMethod.LINEAR_LOG)).summarize(
        "individual"
    )
    console.print(
        summary.to_dataframe().T.loc[
            [
                "auc_inf_obs",
                "auc_inf_obs_sd",
                "auc_inf_obs_ci_low",
                "auc_inf_obs_ci_high",
                "auc_inf_obs_geomean",
                "auc_inf_obs_geocv",
                "n",
            ]
        ]
    )

    console.rule("Partial AUC 0-6 h of the individuals")
    console.print(partial_auc(individuals, 0.5, 6.0).values.round(3))

    fig = plot_timecourse(group)
    fig.savefig("group_uncertainty.png", dpi=120)
    console.print("written: group_uncertainty.png")

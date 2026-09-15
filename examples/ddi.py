"""Classification of a drug-drug interaction from the exposure with and without a perpetrator.

Run from the root of the repository with `python -m examples.ddi`.
Writes `ddi.png` into the working directory.
"""

import numpy as np

from pkpdutils import Route, Timecourses, compare, ddi_classification, nca, ratio
from pkpdutils.console import console
from pkpdutils.plot import plot_ratio
from pkpdutils.stats import DDIThresholds, substrate_sensitivity

# a substrate given alone (control) and with a moderate CYP inhibitor to two
# parallel groups of 10 subjects: the inhibitor lowers the clearance to 35 %
TIME = np.array([0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48])
N = 10
rng = np.random.default_rng(8)


def batch(clearance_factor: float, label: str) -> Timecourses:
    ke = 0.12 * clearance_factor
    values = np.stack(
        [
            rng.lognormal(np.log(8), 0.2)
            * np.exp(-ke * TIME)
            * rng.lognormal(0, 0.05, TIME.size)
            for _ in range(N)
        ]
    )
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"{label}{i}" for i in range(N)],
            "treatment": ("individual", [label] * N),
        },
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="substrate",
    )


if __name__ == "__main__":
    control = nca(batch(1.0, "control"))
    inhibited = nca(batch(0.35, "inhibitor"))

    console.rule("Exposure ratios with / without the inhibitor")
    auc_ratio = ratio(
        inhibited.sample("auc_inf_obs", "individual"),
        control.sample("auc_inf_obs", "individual"),
    )
    cmax_ratio = ratio(
        inhibited.sample("cmax", "individual"), control.sample("cmax", "individual")
    )
    for r in (auc_ratio, cmax_ratio):
        console.print(
            f"{r.name:<12} GMR {r.gmr:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] (90 %)"
        )

    console.rule("Welch t test of the log AUC")
    test = compare(
        inhibited.sample("auc_inf_obs", "individual"),
        control.sample("auc_inf_obs", "individual"),
    )
    console.print(
        f"{test.test}: t = {test.statistic:.2f}, p = {test.p_value:.2g}, Hedges' g = {test.hedges_g:.2f}"
    )

    console.rule("Classification (FDA 2020)")
    ddi = ddi_classification(auc_ratio, cmax_ratio=cmax_ratio)
    console.print(f"{ddi.strength} {ddi.kind}, uncertain: {ddi.uncertain}")
    console.print(
        "EMA:", ddi_classification(auc_ratio, thresholds=DDIThresholds.ema()).to_dict()
    )
    console.print("substrate sensitivity:", substrate_sensitivity(auc_ratio))
    plot_ratio(
        {"auc_inf_obs": auc_ratio, "cmax": cmax_ratio},
        limits=None,
        thresholds=DDIThresholds.fda(),
    ).savefig("ddi.png", dpi=120)
    console.print("written: ddi.png")

"""Average bioequivalence of a test against a reference formulation in a 2x2 crossover.

Run from the root of the repository with `python -m examples.bioequivalence`.
Writes `bioequivalence.png` and `bioequivalence_parameters.png` into the working directory.
"""

import numpy as np

from pkpdutils import Route, Timecourses, bioequivalence, nca
from pkpdutils.console import console
from pkpdutils.plot import plot_parameters, plot_ratio

# 12 subjects, sequence RT receives the reference in period 1 and the test in
# period 2, sequence TR the other way round; the test formulation has a
# slightly lower bioavailability (ratio 0.93) and a slower absorption
N = 12
TIME = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24])
SUBJECTS = [f"s{i:02d}" for i in range(N)]
SEQUENCE = ["RT"] * 6 + ["TR"] * 6
PERIOD_TEST = np.where(np.array(SEQUENCE) == "RT", 2, 1)
PERIOD_REF = 3 - PERIOD_TEST
rng = np.random.default_rng(12)
subject_scale = rng.lognormal(0, 0.25, N)  # between-subject variability of the exposure


def curves(bioavailability: float, ka: float, period: np.ndarray) -> np.ndarray:
    ke = 0.15
    period_effect = np.where(period == 2, 1.05, 1.0)  # period 2 runs 5 % higher
    values = []
    for scale, _p in zip(subject_scale * period_effect, period, strict=True):
        c = (
            bioavailability
            * scale
            * 100
            * ka
            / (ka - ke)
            * (np.exp(-ke * TIME) - np.exp(-ka * TIME))
            / 30
        )
        values.append(c * rng.lognormal(0, 0.06, TIME.size))
    return np.stack(values)


def _format_float(value: float) -> str:
    return f"{value:.4g}"


def batch(values: np.ndarray, period: np.ndarray) -> Timecourses:
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": SUBJECTS,
            "period": ("individual", period),
            "sequence": ("individual", SEQUENCE),
        },
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )


if __name__ == "__main__":
    reference = nca(batch(curves(1.0, 1.5, PERIOD_REF), PERIOD_REF))
    test = nca(batch(curves(0.93, 0.9, PERIOD_TEST), PERIOD_TEST))

    console.rule("Two one-sided tests, 2x2 crossover, 90 % intervals, 80-125 %")
    result = bioequivalence(
        test, reference, parameters=["auc_inf_obs", "auc_last", "cmax"]
    )
    columns = [
        "parameter",
        "gmr",
        "ci_low",
        "ci_high",
        "bioequivalent",
        "p_value",
        "cv_intra",
        "p_period",
        "p_sequence",
        "design",
    ]
    console.print(
        result.to_dataframe()[columns].to_string(
            index=False, float_format=_format_float
        ),
        soft_wrap=True,
    )
    console.print("bioequivalent:", result.bioequivalent)
    plot_ratio(result).savefig("bioequivalence.png", dpi=120)
    plot_parameters(test, "cmax", "individual", by="sequence", log_y=True).savefig(
        "bioequivalence_parameters.png", dpi=120
    )
    console.print("written: bioequivalence.png, bioequivalence_parameters.png")

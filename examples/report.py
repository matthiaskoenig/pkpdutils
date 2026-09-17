"""The study report of a single dose study: the tables and the figures in one HTML file.

Run from the root of the repository with `python -m examples.report`.
Writes `report.html`, `markdown/report.md` with its figures and `report.png`
into the working directory.
"""

import numpy as np

from pkpdutils import Route, Timecourses, nca, study_report
from pkpdutils.console import console
from pkpdutils.plot import plot_mean_timecourse

# twelve subjects of two arms, an oral single dose of 100 mg, a one compartment
# profile with first order absorption and log-normal between-subject variability
N = 12
TIME = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0])
SUBJECTS = [f"s{i + 1:02d}" for i in range(N)]
ARM = ["fasted"] * (N // 2) + ["fed"] * (N // 2)


def curves() -> np.ndarray:
    rng = np.random.default_rng(7)
    ke = 0.15
    values = []
    for i in range(N):
        # the fed arm absorbs more slowly and reaches a lower peak
        ka = 1.4 if ARM[i] == "fasted" else 0.7
        scale = rng.lognormal(0.0, 0.22)
        c = (
            scale
            * 100
            * ka
            / (ka - ke)
            * (np.exp(-ke * TIME) - np.exp(-ka * TIME))
            / 30
        )
        values.append(c * rng.lognormal(0.0, 0.05, TIME.size))
    return np.stack(values)


if __name__ == "__main__":
    batch = Timecourses.from_arrays(
        TIME,
        curves(),
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": SUBJECTS, "arm": ("individual", ARM)},
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )
    result = nca(batch)

    report = study_report(
        batch,
        result,
        dim="individual",
        by="arm",
        subtitle="Single dose, 100 mg oral, twelve subjects in two arms",
    )
    console.rule("Study report")
    for section in report.sections:
        console.print(f"{section.kind:<7} {section.heading or section.caption}")
    report.write_html("report.html")
    # the markdown goes into a directory of its own, it writes its figures next to it
    report.write_markdown("markdown/report.md")

    # the figure of the gallery is the mean curve of the report
    plot_mean_timecourse(batch, by="arm").savefig("report.png", dpi=120)
    console.print("written: report.html, markdown/report.md, report.png")

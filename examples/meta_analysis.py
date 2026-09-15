"""Meta-analysis of the effect of smoking on the clearance of caffeine over published studies.

Run from the root of the repository with `python -m examples.meta_analysis`.
Writes `meta_analysis.png` into the working directory.
"""

from pkpdutils import ParameterSample, meta_analysis
from pkpdutils.console import console
from pkpdutils.plot import plot_forest
from pkpdutils.stats import EffectKind, Study

# clearance of caffeine (ml/min/kg) in non-smokers (control) and smokers as
# mean, sd and n; illustrative values in the range of the literature
STUDIES = [
    Study(
        "study A",
        ParameterSample(mean=1.2, sd=0.4, n=10),
        ParameterSample(mean=2.0, sd=0.6, n=10),
        "smokers",
    ),
    Study(
        "study B",
        ParameterSample(mean=1.4, sd=0.5, n=8),
        ParameterSample(mean=2.3, sd=0.8, n=9),
        "smokers",
    ),
    Study(
        "study C",
        ParameterSample(mean=1.1, sd=0.3, n=12),
        ParameterSample(mean=1.6, sd=0.5, n=12),
        "smokers",
    ),
    Study(
        "study D",
        ParameterSample(mean=1.3, sd=0.4, n=15),
        ParameterSample(mean=2.4, sd=0.9, n=14),
        "smokers",
    ),
    Study(
        "study E",
        ParameterSample(mean=1.2, sd=0.4, n=6),
        ParameterSample(mean=1.5, sd=0.4, n=6),
        "smokers",
    ),
]

if __name__ == "__main__":
    for kind in (EffectKind.LOG_RATIO, EffectKind.HEDGES_G):
        console.rule(f"{kind}")
        result = meta_analysis(STUDIES, kind)
        console.print(result.to_dataframe())
        console.print("fixed effect:  ", result.fixed.to_dict())
        console.print("random effects:", result.random.to_dict())
        het = result.heterogeneity
        console.print(
            f"Q = {het.q:.2f}, df = {het.df}, p = {het.p_value:.2g}, I2 = {het.i2:.1f} %, tau2 = {het.tau2:.3g}"
        )
    plot_forest(meta_analysis(STUDIES, EffectKind.LOG_RATIO)).savefig(
        "meta_analysis.png", dpi=120
    )
    console.print("written: meta_analysis.png")

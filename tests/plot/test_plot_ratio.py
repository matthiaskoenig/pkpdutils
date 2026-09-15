import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.figure import Figure

from pkpdutils.plot import plot_ratio
from pkpdutils.stats import DDIThresholds, ParameterSample, ratio
from pkpdutils.stats.bioequivalence import BEResult, tost

matplotlib.use("Agg")
RNG = np.random.default_rng(1)
REF = np.exp(RNG.normal(np.log(100), 0.2, 10))
TEST = REF * np.exp(RNG.normal(np.log(0.97), 0.1, 10))


def test_plot_ratio_of_ratio_results() -> None:
    ratios = {
        "auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF)),
        "cmax": ratio(ParameterSample(values=TEST * 1.1), ParameterSample(values=REF)),
    }
    fig = plot_ratio(ratios)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["auc", "cmax"]
    assert "90" in ax.get_title()
    xs = sorted(
        float(np.asarray(line.get_xdata())[0])
        for line in ax.get_lines()
        if line.get_linestyle() == "--"
    )
    assert xs == [0.8, 1.25]
    matplotlib.pyplot.close("all")


def test_plot_ratio_of_bioequivalence_and_thresholds() -> None:
    be = BEResult(
        parameters={
            "auc": tost(ParameterSample(values=TEST), ParameterSample(values=REF))
        },
        bioequivalent=True,
        limits=(0.8, 1.25),
        ci_level=0.90,
    )
    fig = plot_ratio(be, limits=None, thresholds=DDIThresholds.fda())
    ax = fig.axes[0]
    dotted = [
        float(np.asarray(line.get_xdata())[0])
        for line in ax.get_lines()
        if line.get_linestyle() == ":"
    ]
    assert sorted(dotted) == [0.2, 0.5, 0.8, 1.25, 2.0, 5.0]
    assert any("strong" in t.get_text() for t in ax.texts)
    matplotlib.pyplot.close("all")


def test_plot_ratio_raises_for_an_empty_mapping() -> None:
    with pytest.raises(ValueError):
        plot_ratio({})

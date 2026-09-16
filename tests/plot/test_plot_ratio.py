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


def test_plot_ratio_annotates_the_estimates_and_renames_the_rows() -> None:
    ratios = {
        "auc_inf_obs": ratio(ParameterSample(values=TEST), ParameterSample(values=REF)),
        "cmax": ratio(ParameterSample(values=TEST * 1.1), ParameterSample(values=REF)),
    }
    fig = plot_ratio(ratios, labels={"auc_inf_obs": "AUC(0-inf)"})
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["AUC(0-inf)", "cmax"]
    texts = [a.get_text() for a in ax.texts]
    assert len(texts) == 2
    for name, text in zip(ratios, texts, strict=True):
        r = ratios[name]
        assert text.startswith(f"{r.gmr:.3g} [")
        assert f"{r.ci_low:.3g}" in text and f"{r.ci_high:.3g}" in text
    # the annotations are written into room made to the right of the data
    assert ax.get_xlim()[1] > max(r.ci_high for r in ratios.values())
    matplotlib.pyplot.close(fig)


def test_plot_ratio_without_annotation_writes_no_text() -> None:
    ratios = {"auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF))}
    fig = plot_ratio(ratios, annotate=False)
    assert not fig.axes[0].texts
    matplotlib.pyplot.close(fig)


def test_plot_ratio_drops_the_unity_tick_when_thresholds_crowd_it() -> None:
    ratios = {"auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF))}
    with_thresholds = plot_ratio(ratios, thresholds=DDIThresholds.fda())
    ticks = [t.get_text() for t in with_thresholds.axes[0].get_xticklabels()]
    assert "1" not in ticks
    assert "0.8" in ticks and "1.25" in ticks
    plain = plot_ratio(ratios)
    assert "1" in [t.get_text() for t in plain.axes[0].get_xticklabels()]
    matplotlib.pyplot.close("all")

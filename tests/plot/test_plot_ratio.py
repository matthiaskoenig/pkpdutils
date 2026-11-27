from typing import Any

import numpy as np
import pytest
from sbmlsim_core.chart import Chart

from pkpdutils.plot import new_chart, plot_ratio
from pkpdutils.stats import DDIThresholds, ParameterSample, ratio
from pkpdutils.stats.bioequivalence import BEResult, tost

RNG = np.random.default_rng(1)
REF = np.exp(RNG.normal(np.log(100), 0.2, 10))
TEST = REF * np.exp(RNG.normal(np.log(0.97), 0.1, 10))


def kinds(plot: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in plot["elements"] if e["kind"] == kind]


def texts(plot: dict[str, Any]) -> list[str]:
    return [e["text"] for e in kinds(plot, "text")]


def test_plot_ratio_of_ratio_results() -> None:
    ratios = {
        "auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF)),
        "cmax": ratio(ParameterSample(values=TEST * 1.1), ParameterSample(values=REF)),
    }
    chart = plot_ratio(ratios)
    assert isinstance(chart, Chart)
    plot = chart.spec()["plots"][0]
    assert plot["x"]["scale"] == "log"
    assert plot["y"]["tick_labels"] == ["auc", "cmax"]
    assert "90" in plot["title"]
    limits = sorted(
        e["value"]
        for e in kinds(plot, "rule")
        if e["style"]["line"]["type"] == "dashed"
    )
    assert limits == [0.8, 1.25]


def test_plot_ratio_draws_into_a_cell_of_a_given_chart() -> None:
    chart = new_chart(12, 4, columns=2)
    ratios = {"auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF))}
    assert plot_ratio(ratios, chart=chart, column=1) is chart
    (plot,) = chart.spec()["plots"]
    assert plot["column"] == 1
    assert plot["y"]["tick_labels"] == ["auc"]


def test_plot_ratio_of_bioequivalence_and_thresholds() -> None:
    be = BEResult(
        parameters={
            "auc": tost(ParameterSample(values=TEST), ParameterSample(values=REF))
        },
        bioequivalent=True,
        limits=(0.8, 1.25),
        ci_level=0.90,
    )
    plot = plot_ratio(be, limits=None, thresholds=DDIThresholds.fda()).spec()["plots"][
        0
    ]
    # the thresholds are the ticks, the classes the shaded bands between them
    assert {0.2, 0.5, 0.8, 1.25, 2.0, 5.0} <= set(plot["x"]["ticks"])
    assert any("strong" in t for t in texts(plot))


def test_plot_ratio_raises_without_a_ratio() -> None:
    # without a row the figure was an empty log axis with the title
    # "geometric mean ratios with  % intervals"; the empty input raises
    empty = BEResult(
        parameters={}, bioequivalent=False, limits=(0.8, 1.25), ci_level=0.90
    )
    for ratios in ({}, empty):
        with pytest.raises(ValueError, match="at least one ratio"):
            plot_ratio(ratios)


def test_plot_ratio_annotates_the_estimates_and_renames_the_rows() -> None:
    ratios = {
        "auc_inf_obs": ratio(ParameterSample(values=TEST), ParameterSample(values=REF)),
        "cmax": ratio(ParameterSample(values=TEST * 1.1), ParameterSample(values=REF)),
    }
    plot = plot_ratio(ratios, labels={"auc_inf_obs": "AUC(0-inf)"}).spec()["plots"][0]
    assert plot["y"]["tick_labels"] == ["AUC(0-inf)", "cmax"]
    written = texts(plot)
    assert len(written) == 2
    for name, text in zip(ratios, written, strict=True):
        r = ratios[name]
        assert text.startswith(f"{r.gmr:.3g} [")
        assert f"{r.ci_low:.3g}" in text and f"{r.ci_high:.3g}" in text
    # the annotations are written into room made to the right of the data
    highest = max(r.ci_high for r in ratios.values())
    assert plot["x"]["max"] > highest
    assert all(e["x"] > highest for e in kinds(plot, "text"))


def test_plot_ratio_without_annotation_writes_no_text() -> None:
    ratios = {"auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF))}
    assert not texts(plot_ratio(ratios, annotate=False).spec()["plots"][0])


def test_plot_ratio_drops_the_unity_tick_when_thresholds_crowd_it() -> None:
    ratios = {"auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF))}
    crowded = plot_ratio(ratios, thresholds=DDIThresholds.fda()).spec()["plots"][0]
    assert "1" not in crowded["x"]["tick_labels"]
    assert "0.8" in crowded["x"]["tick_labels"]
    assert "1.25" in crowded["x"]["tick_labels"]
    plain = plot_ratio(ratios).spec()["plots"][0]
    assert "1" in plain["x"]["tick_labels"]


def test_plot_ratio_shades_the_interaction_classes() -> None:
    ratios = {"auc": ratio(ParameterSample(values=TEST), ParameterSample(values=REF))}
    plot = plot_ratio(ratios, thresholds=DDIThresholds.fda()).spec()["plots"][0]
    assert len(kinds(plot, "area")) == 7
    written = texts(plot)
    for name in (
        "strong inducer",
        "moderate inducer",
        "weak inducer",
        "no interaction",
        "weak inhibitor",
        "moderate inhibitor",
        "strong inhibitor",
    ):
        assert name in written
    assert plot["x"]["min"] < 0.2 / 2 and plot["x"]["max"] > 5 * 2
    # the threshold values are the ticks
    assert {0.2, 0.5, 0.8, 1.25, 2.0, 5.0} <= set(plot["x"]["ticks"])

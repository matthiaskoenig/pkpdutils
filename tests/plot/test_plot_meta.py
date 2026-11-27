from typing import Any

import numpy as np
import pytest
from sbmlsim_core.chart import Chart

from pkpdutils.plot import new_chart, plot_forest
from pkpdutils.stats import EffectKind, ParameterSample, Study, meta_analysis


def studies() -> list[Study]:
    rng = np.random.default_rng(9)
    return [
        Study(
            label=f"study {i}",
            control=ParameterSample(values=np.exp(rng.normal(np.log(100), 0.3, 10))),
            treatment=ParameterSample(values=np.exp(rng.normal(np.log(140), 0.3, 10))),
        )
        for i in range(4)
    ]


def texts(plot: dict[str, Any]) -> list[str]:
    return [e["text"] for e in plot["elements"] if e["kind"] == "text"]


def test_plot_forest_log_ratio_is_a_ratio_axis() -> None:
    result = meta_analysis(studies(), EffectKind.LOG_RATIO)
    chart = plot_forest(result)
    assert isinstance(chart, Chart)
    plot = chart.spec()["plots"][0]
    assert plot["x"]["scale"] == "log"
    labels = plot["y"]["tick_labels"]
    assert labels[:4] == ["study 0", "study 1", "study 2", "study 3"]
    assert "fixed" in labels[4].lower() and "random" in labels[5].lower()
    # the first study is the top row
    assert plot["y"]["reverse"] is True
    assert plot["x"]["label"].startswith("ratio")
    # the ticks of the log axis are plain numbers, not powers of ten
    assert plot["x"]["tick_labels"]
    assert all("^" not in t for t in plot["x"]["tick_labels"])


def test_plot_forest_hedges_g_linear_axis() -> None:
    result = meta_analysis(studies(), EffectKind.HEDGES_G)
    plot = plot_forest(result).spec()["plots"][0]
    assert plot["x"]["scale"] == "linear"
    assert "Hedges" in plot["x"]["label"]
    assert "I2" in plot["title"] or "I²" in plot["title"]


def test_plot_forest_draws_into_a_cell_of_a_given_chart() -> None:
    result = meta_analysis(studies(), EffectKind.LOG_RATIO)
    chart = new_chart(12, 4, columns=2, title="mine")
    assert plot_forest(result, chart=chart, row=0, column=1) is chart
    spec = chart.spec()
    # the chart of the caller keeps its title and its grid
    assert spec["title"] == "mine"
    assert [(p["row"], p["column"]) for p in spec["plots"]] == [(0, 1)]
    with pytest.raises(ValueError):
        plot_forest(result, chart=chart, row=0, column=2)


def test_plot_forest_annotates_the_effects_and_the_weights() -> None:
    result = meta_analysis(studies(), EffectKind.LOG_RATIO)
    plot = plot_forest(result).spec()["plots"][0]
    written = texts(plot)
    # one per study and one per pooled effect
    assert len(written) == result.n_studies + 2
    for text, weight in zip(written, result.random.weights, strict=False):
        assert "[" in text and "]" in text
        assert f"{100 * weight:.1f} %" in text
    assert "%" not in written[-1] and "%" not in written[-2]
    # the column is written into room made right of the data
    data_end = max(
        float(np.nanmax(np.asarray(e["x"], dtype=float)))
        for e in plot["elements"]
        if e["kind"] == "curve"
    )
    assert plot["x"]["max"] > data_end
    assert all(e["x"] > data_end for e in plot["elements"] if e["kind"] == "text")


def test_plot_forest_without_annotation_writes_no_text() -> None:
    result = meta_analysis(studies(), EffectKind.HEDGES_G)
    assert not texts(plot_forest(result, annotate=False).spec()["plots"][0])


def test_plot_forest_draws_the_pooled_effects_as_diamonds() -> None:
    result = meta_analysis(studies(), EffectKind.LOG_RATIO)
    plot = plot_forest(result).spec()["plots"][0]
    assert len([e for e in plot["elements"] if e["kind"] == "polygon"]) == 2
    # the marker area of a study grows with its weight
    sizes = [
        e["style"]["marker"]["size"]
        for e in plot["elements"]
        if e["kind"] == "curve" and e["style"]["marker"]["type"] == "square"
    ]
    assert np.argmax(sizes) == np.argmax(result.random.weights)

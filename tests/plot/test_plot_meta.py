import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib.figure import Figure

from pkpdutils.plot import plot_forest
from pkpdutils.stats import EffectKind, ParameterSample, Study, meta_analysis

matplotlib.use("Agg")


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


def test_plot_forest_log_ratio_is_a_ratio_axis() -> None:
    result = meta_analysis(studies(), EffectKind.LOG_RATIO)
    fig = plot_forest(result)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[:4] == ["study 0", "study 1", "study 2", "study 3"]
    assert "fixed" in labels[4].lower() and "random" in labels[5].lower()
    assert ax.get_xlabel().startswith("ratio")
    matplotlib.pyplot.close("all")


def test_plot_forest_hedges_g_linear_axis() -> None:
    result = meta_analysis(studies(), EffectKind.HEDGES_G)
    _fig, ax = matplotlib.pyplot.subplots()
    plot_forest(result, ax=ax)
    assert ax.get_xscale() == "linear"
    assert "Hedges" in ax.get_xlabel()
    assert "I2" in ax.get_title() or "I²" in ax.get_title()
    matplotlib.pyplot.close("all")

import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib.figure import Figure

from pkpdutils import Route, Timecourses, nca
from pkpdutils.plot import plot_parameters
from pkpdutils.stats import Scale

matplotlib.use("Agg")


def nca_result():
    time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
    rng = np.random.default_rng(2)
    n = 8
    curves = np.stack(
        [rng.lognormal(np.log(10), 0.2) * np.exp(-0.2 * time) for _ in range(n)]
    )
    batch = Timecourses.from_arrays(
        time,
        curves,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"s{i}" for i in range(n)],
            "sex": ("individual", ["F", "M"] * 4),
        },
        dose={"amount": np.full(n, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    return nca(batch)


def test_plot_parameters_groups() -> None:
    result = nca_result()
    fig = plot_parameters(result, "auc_inf_obs", "individual", by="sex", log=True)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["F", "M"]
    assert ax.get_ylabel().startswith("auc_inf_obs [")
    matplotlib.pyplot.close("all")


def test_plot_parameters_single_group_linear() -> None:
    result = nca_result()
    fig, ax = matplotlib.pyplot.subplots()
    out = plot_parameters(result, "cmax", "individual", scale=Scale.LINEAR, ax=ax)
    assert out is fig
    assert [t.get_text() for t in ax.get_xticklabels()] == ["cmax"]
    assert len(ax.containers) >= 1  # the error bar of the mean
    matplotlib.pyplot.close("all")

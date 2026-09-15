import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.figure import Figure

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca, nca_single
from pkpdutils.plot import (
    PlotStyle,
    plot_intervals,
    plot_nca,
    plot_nca_grid,
    plot_timecourse,
)

matplotlib.use("Agg")


def multiple_dose_tc(
    n_doses: int = 3, tau: float = 12.0, c0: float = 10.0, k: float = 0.2
) -> Timecourse:
    """Superposition of a mono-exponential bolus curve given every `tau` hours."""
    time = np.sort(
        np.concatenate([np.arange(0, n_doses * tau + 0.01, 0.5), [n_doses * tau + 24]])
    )
    value = np.zeros_like(time)
    for k_dose in range(n_doses):
        shifted = time - k_dose * tau
        value += np.where(shifted >= 0, c0 * np.exp(-k * shifted), 0.0)
    dosing = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=tau, n_doses=n_doses
    )
    return Timecourse(
        time=time, value=value, time_unit="hr", unit="mg/l", dosing=dosing
    )


def oral(ka: float = 2.0, label: str = "a") -> Timecourse:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])
    c = 10 * ka / (ka - 0.3) * (np.exp(-0.3 * t) - np.exp(-ka * t))
    return Timecourse(
        time=t,
        value=c,
        sd=0.1 * c,
        n=8,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
        label=label,
    )


def test_plot_timecourse_single_and_batch() -> None:
    fig = plot_timecourse(oral())
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time [hr]" and ax.get_ylabel() == "caffeine [mg/l]"
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    fig2 = plot_timecourse(batch, log=True, errorbars=False)
    assert fig2.axes[0].get_yscale() == "log"
    labels = [line.get_label() for line in fig2.axes[0].get_lines()]
    assert "slow" in labels and "fast" in labels
    matplotlib.pyplot.close("all")


def test_plot_nca_single() -> None:
    tc = oral()
    result = nca_single(tc)
    fig = plot_nca(tc, result, title="test")
    assert len(fig.axes) == 2
    assert fig.axes[1].get_yscale() == "log"
    assert "test" in fig.axes[0].get_title()
    matplotlib.pyplot.close(fig)


def test_plot_nca_from_batch_result_with_indexers_and_flags_in_title() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    result = nca(batch)
    fig = plot_nca(batch.sel(individual="fast"), result, individual="fast")
    assert "fast" in fig.axes[0].get_title()
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l")
    fig2 = plot_nca(short, nca_single(short))
    assert "TOO_FEW_POINTS" in fig2.axes[0].get_title()
    matplotlib.pyplot.close("all")


def test_plot_nca_iv_bolus_marks_c0() -> None:
    t = np.array([0.5, 1, 2, 4, 8])
    tc = Timecourse(
        time=t,
        value=10 * np.exp(-0.5 * t),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    fig = plot_nca(tc, nca_single(tc), style=PlotStyle(fit_color="red"))
    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert "C0" in labels
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid() -> None:
    batch = Timecourses.from_timecourses(
        [oral(k, str(k)) for k in (1.0, 2.0, 3.0, 4.0)]
    )
    fig = plot_nca_grid(batch, nca(batch, NCAOptions()), ncols=2)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 4
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_by_broadcasts_coordinate_to_sample_dims() -> None:
    n_dose, n_individual, n_time = 2, 3, 4
    time = np.array([0.5, 1.0, 2.0, 4.0])
    values = np.arange(n_dose * n_individual * n_time, dtype=float).reshape(
        n_dose, n_individual, n_time
    )
    tcs = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "individual"),
        coords={"dose": ["low", "high"], "individual": [0, 1, 2]},
    )
    # "group" is assigned with dims in the opposite order of the sample
    # dimensions ("dose", "individual") of the batch, to catch a naive
    # flatten that ignores the coordinate's own dimension order.
    group = np.array([[f"g{i}{d}" for d in range(n_dose)] for i in range(n_individual)])
    tcs.ds = tcs.ds.assign_coords(group=(("individual", "dose"), group))
    expected = [str(group[i, d]) for d in range(n_dose) for i in range(n_individual)]
    fig = plot_timecourse(tcs, by="group")
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert labels == expected
    matplotlib.pyplot.close(fig)


def test_plot_nca_breaks_line_at_nan_without_bridging() -> None:
    tc = oral()
    value = tc.value.copy()
    value[4] = np.nan
    tc_nan = tc.model_copy(update={"value": value})
    fig = plot_nca(tc_nan, nca_single(tc_nan))
    data_line = next(
        line for line in fig.axes[0].get_lines() if line.get_label() == "data"
    )
    # the raw (unmasked) values are plotted, matplotlib breaks the line at the
    # NaN itself instead of the previous, gap-bridging masked plot
    assert np.isnan(data_line.get_ydata()).any()
    matplotlib.pyplot.close(fig)


def test_plot_nca_multiple_dose_shades_the_steady_state_interval() -> None:
    tc = multiple_dose_tc()
    result = nca_single(tc, NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_nca(tc, result)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert any("tau" in label for label in legend_labels)
    matplotlib.pyplot.close(fig)


def test_plot_intervals_one_line_per_sample() -> None:
    a = multiple_dose_tc(n_doses=3, c0=10.0)
    b = multiple_dose_tc(n_doses=3, c0=20.0)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    result = nca(batch, NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_intervals(result)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_ylabel().startswith("interval_auc [")
    assert ax.get_xlabel() == "interval"
    fig.canvas.draw()
    tick_labels = [
        label.get_text() for label in ax.xaxis.get_ticklabels() if label.get_text()
    ]
    assert tick_labels and all("." not in label for label in tick_labels), tick_labels
    matplotlib.pyplot.close(fig)


def test_plot_intervals_one_sample_with_indexers() -> None:
    a = multiple_dose_tc(n_doses=3, c0=10.0)
    b = multiple_dose_tc(n_doses=3, c0=20.0)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    result = nca(batch, NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_intervals(result, individual="two")
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    matplotlib.pyplot.close(fig)


def test_plot_intervals_raises_without_intervals() -> None:
    tc = oral()
    result = nca_single(tc)
    with pytest.raises(ValueError):
        plot_intervals(result)


def test_plot_intervals_raises_for_a_non_interval_name() -> None:
    tc = multiple_dose_tc(n_doses=3)
    result = nca_single(tc, NCAOptions(auc_method=AUCMethod.LOG))
    with pytest.raises(ValueError):
        plot_intervals(result, name="cmax")

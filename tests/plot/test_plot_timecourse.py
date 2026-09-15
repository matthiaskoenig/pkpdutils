import warnings

import matplotlib
import matplotlib.pyplot
import numpy as np

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.plot import plot_timecourse

matplotlib.use("Agg")


def multiple_dose_tc(n_doses: int = 4, tau: float = 12.0) -> Timecourse:
    """A bolus curve given every `tau` hours, `n_doses` times."""
    time = np.arange(0, n_doses * tau + 0.01, 1.0)
    value = np.zeros_like(time)
    for k in range(n_doses):
        shifted = time - k * tau
        value += np.where(shifted >= 0, 10.0 * np.exp(-0.15 * shifted), 0.0)
    dosing = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=tau, n_doses=n_doses
    )
    return Timecourse(
        time=time, value=value, time_unit="hr", unit="mg/l", dosing=dosing
    )


def test_plot_timecourse_draws_a_dotted_line_per_dose() -> None:
    tc = multiple_dose_tc(n_doses=4)
    fig = plot_timecourse(tc)
    ax = fig.axes[0]
    dose_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert len(dose_lines) == 4
    for line in dose_lines:
        assert line.get_color() == "gray"
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_single_dose_draws_no_dose_line() -> None:
    tc = Timecourse(
        time=np.array([0.5, 1, 2, 4, 8]),
        value=10.0 * np.exp(-0.15 * np.array([0.5, 1, 2, 4, 8])),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    fig = plot_timecourse(tc)
    ax = fig.axes[0]
    dose_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert len(dose_lines) == 0
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_log_without_positive_values_stays_linear() -> None:
    # B28: a curve with no positive value used to raise "UserWarning: Data
    # has no positive values, and therefore cannot be log-scaled" at draw time.
    tc = Timecourse(
        time=[0, 1, 2, 4],
        value=[0.0, 0.0, 0.0, 0.0],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_timecourse(tc, log_y=True)
        fig.canvas.draw()
    assert fig.axes[0].get_yscale() == "linear"
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_batch_draws_no_dose_lines() -> None:
    a = multiple_dose_tc(n_doses=3)
    b = multiple_dose_tc(n_doses=3)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    fig = plot_timecourse(batch)
    ax = fig.axes[0]
    dose_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert len(dose_lines) == 0
    matplotlib.pyplot.close(fig)

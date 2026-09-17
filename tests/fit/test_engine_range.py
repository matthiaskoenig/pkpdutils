"""Tests of data beyond the range of double precision (#73), with warnings as errors.

A floating-point warning of one row would abort a whole batch under a strict
warning filter; such a row is flagged instead and the other rows are fitted as
if it were not there.
"""

import math

import numpy as np
import pytest
import xarray as xr

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.fit import (
    FitFlag,
    FitOptions,
    Weighting,
    fit,
    fit_table,
    fit_timecourses,
)
from pkpdutils.fit.engine import covariance
from pkpdutils.fit.model import Model
from pkpdutils.fit.models import (
    Allometric,
    Bateman,
    BiExp,
    Emax,
    Imax,
    Linear,
    LogLinear,
    MonoExp,
    Power,
    SigmoidEmax,
    SigmoidImax,
    TriExp,
)

pytestmark = pytest.mark.filterwarnings("error")

T = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
EXTREME = np.logspace(300, -300, T.size)

#: data from 1e-300 to 1e300 in the values (flagged) or in `x` only, and a
#: subnormal `x` on which `numpy.polyfit` in the initial guess raises
SPANS = {
    "decreasing": (T, EXTREME, True),
    "alternating": (T, np.tile([1e300, 1e-300], T.size // 2), True),
    "x_and_y": (np.logspace(-300, 300, T.size), np.logspace(-300, 300, T.size), True),
    "x": (np.logspace(-300, 300, T.size), np.linspace(1.0, 10.0, T.size), False),
    "subnormal_x": (
        np.linspace(5e-324, 4e-320, T.size),
        np.linspace(1, 10, T.size),
        False,
    ),
}

MODELS = [
    MonoExp(),
    BiExp(),
    TriExp(),
    Bateman(),
    Bateman(lag=True),
    Emax(),
    SigmoidEmax(),
    Imax(),
    SigmoidImax(),
    Linear(),
    LogLinear(),
    Power(),
    Allometric(),
    Allometric(exponent=0.75),
]


@pytest.mark.parametrize("span", sorted(SPANS))
@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.name)
def test_the_model_library_on_data_beyond_double_precision(
    model: Model, span: str
) -> None:
    """Nothing escapes, an extreme row is flagged, no parameter without a finite cost."""
    x, y, flagged = SPANS[span]
    for options in (
        FitOptions(seed=0, n_starts=3),
        FitOptions(seed=0, weighting=Weighting.INV_Y2, bootstrap=3),
    ):
        result = fit(model, x, y, options=options)
        flags = int(result["flags"].values)
        if flagged:
            assert flags & (FitFlag.NOT_CONVERGED | FitFlag.OVERFLOW)
        if not math.isfinite(float(result["cost"].values)):
            for name in model.parameter_names:
                assert np.isnan(float(result[name].values))


def test_an_unweighted_fit_whose_sum_of_squares_overflows_is_flagged() -> None:
    """The search used to run on an infinite cost and report `a = 2.4e172`, `k = 55`."""
    result = fit(MonoExp(), T, EXTREME)
    assert result.flags() == ["NOT_CONVERGED", "OVERFLOW"]
    assert np.isnan(float(result["a"].values)) and np.isnan(float(result["k"].values))
    assert int(result["n_points"].values) == T.size


def test_statistics_beyond_double_precision_are_nan_and_flagged() -> None:
    """A weighted fit of values near 1e250 keeps its parameters, `rmse` and `r2` overflow."""
    y = 1e250 * 10.0 * np.exp(-0.3 * T)
    result = fit(MonoExp(), T, y, options=FitOptions(weighting=Weighting.INV_Y2))
    assert result.flags() == ["OVERFLOW"]
    assert float(result["k"].values) == pytest.approx(0.3)
    assert np.isnan(float(result["rmse"].values)) and np.isnan(
        float(result["r2"].values)
    )
    assert math.isfinite(float(result["aicc"].values))


def test_an_extreme_row_leaves_the_other_rows_of_a_batch_unchanged() -> None:
    """The extreme row is flagged, every other row is the fit of the row alone."""
    good = BiExp().predict(T, np.array([8.0, 2.0, 2.0, 0.2]))
    rows = np.stack([good, EXTREME, 2.0 * good])
    batch = fit(BiExp(), T, rows, dims=("individual",))
    assert batch.decode_flags(int(batch["flags"].values[1])) == [
        "NOT_CONVERGED",
        "OVERFLOW",
    ]
    for i in (0, 2):
        alone = fit(BiExp(), T, rows[i])
        for name in ("flags", "a1", "k1", "a2", "k2", "auc", "a1_se", "r2", "aicc"):
            assert float(batch[name].values[i]) == float(alone[name].values)

    curves = Timecourses.from_timecourses(
        [
            Timecourse(
                time=T,
                value=values,
                time_unit="hr",
                unit="mg/l",
                dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
                label=label,
            )
            for label, values in zip(("a", "b", "c"), rows, strict=True)
        ]
    )
    result = fit_timecourses(BiExp(), curves)
    assert "OVERFLOW" in result.decode_flags(int(result["flags"].values[1]))
    assert float(result["k2"].values[0]) == float(batch["k2"].values[0])

    doses = np.array([1.0, 10.0, 100.0, 1000.0])
    auc = np.stack([3.0 * doses**0.9, np.logspace(300, -300, 4)])
    table = xr.Dataset({"auc": (("subject", "dose"), auc)}, coords={"dose": doses})
    power = fit_table(Power(), table, "dose", "auc", dim="dose")
    assert power.decode_flags(int(power["flags"].values[0])) == []
    assert float(power["b"].values[0]) == pytest.approx(0.9)
    assert power.decode_flags(int(power["flags"].values[1]))


def test_a_covariance_which_is_not_finite_is_an_overflow() -> None:
    jac = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    for arguments in ((jac * 1e200, 1.0), (jac, math.inf), (jac * 1e-100, 1e300)):
        cov, condition = covariance(*arguments, 4, 2)
        assert condition == FitFlag.OVERFLOW and np.isnan(cov).all()
    assert covariance(np.ones((4, 2)), 1.0, 4, 2)[1] == FitFlag.SINGULAR
    assert covariance(jac, 1.0, 4, 2)[1] == FitFlag.NONE

import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca, partial_auc

K, C0 = 0.5, 10.0


def batch() -> Timecourses:
    t = np.array([0.5, 1, 2, 4, 6, 8, 12])
    curves = [
        Timecourse(
            time=t,
            value=C0 * np.exp(-K * t),
            time_unit="hr",
            unit="mg/l",
            dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
            substance="x",
            label="a",
        ),
        Timecourse(
            time=t,
            value=2 * C0 * np.exp(-K * t),
            time_unit="hr",
            unit="mg/l",
            dose=Dose(amount=200, unit="mg", route=Route.IV_BOLUS),
            substance="x",
            label="b",
        ),
    ]
    return Timecourses.from_timecourses(curves)


def test_partial_auc_analytic_log_rule() -> None:
    area = partial_auc(batch(), 1.0, 6.0, NCAOptions(auc_method=AUCMethod.LOG))
    expected = C0 / K * (np.exp(-K * 1.0) - np.exp(-K * 6.0))
    np.testing.assert_allclose(area.values, [expected, 2 * expected], rtol=1e-9)
    assert area.dims == ("individual",)
    assert area.attrs["units"] == "hour * milligram / liter"
    assert area.name == "auc_partial"


def test_partial_auc_bounds_between_points_and_outside() -> None:
    area = partial_auc(batch(), 1.5, 5.0, NCAOptions(auc_method=AUCMethod.LOG))
    expected = C0 / K * (np.exp(-K * 1.5) - np.exp(-K * 5.0))
    assert float(area.values[0]) == pytest.approx(expected, rel=1e-9)
    outside = partial_auc(batch(), 1.0, 20.0)
    assert np.isnan(outside.values).all()
    before = partial_auc(batch(), 0.0, 4.0)
    assert np.isnan(before.values).all()


def test_partial_auc_whole_range_equals_auc_last() -> None:
    tcs = batch()
    options = NCAOptions(auc_method=AUCMethod.LINEAR)
    whole = partial_auc(tcs, 0.5, 12.0, options)
    # auc_last of a bolus includes the inserted (0, C0) segment, so compare against the area from the first sample
    result = nca(
        tcs.__class__.from_timecourses(
            [tc.model_copy(update={"dose": None}) for tc in tcs]
        ),
        options,
    )
    np.testing.assert_allclose(whole.values, result["auc_last"].values, rtol=1e-12)


def test_partial_auc_relative_to_dose_time() -> None:
    t = np.array([10.5, 11, 12, 14, 16, 18, 22])
    tc = Timecourse(
        time=t,
        value=C0 * np.exp(-K * (t - 10)),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS, time=10.0),
        substance="x",
    )
    area = partial_auc(
        Timecourses.from_timecourses([tc]),
        1.0,
        6.0,
        NCAOptions(auc_method=AUCMethod.LOG),
    )
    expected = C0 / K * (np.exp(-K * 1.0) - np.exp(-K * 6.0))
    assert float(area.values[0]) == pytest.approx(expected, rel=1e-9)


def test_partial_auc_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="t_end"):
        partial_auc(batch(), 4.0, 1.0)

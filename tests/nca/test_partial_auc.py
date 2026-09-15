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
    area = partial_auc(batch(), 1.0, 6.0, options=NCAOptions(auc_method=AUCMethod.LOG))
    expected = C0 / K * (np.exp(-K * 1.0) - np.exp(-K * 6.0))
    np.testing.assert_allclose(area.values, [expected, 2 * expected], rtol=1e-9)
    assert area.dims == ("individual",)
    assert area.attrs["units"] == "hour * milligram / liter"
    assert area.name == "auc_partial"


def test_partial_auc_bounds_between_points_and_outside() -> None:
    area = partial_auc(batch(), 1.5, 5.0, options=NCAOptions(auc_method=AUCMethod.LOG))
    expected = C0 / K * (np.exp(-K * 1.5) - np.exp(-K * 5.0))
    assert float(area.values[0]) == pytest.approx(expected, rel=1e-9)
    outside = partial_auc(batch(), 1.0, 20.0)
    assert np.isnan(outside.values).all()
    # from time 0 the bolus is back extrapolated to C0, so the area exists and
    # the falling segments make the linear-up/log-down rule exact
    before = partial_auc(batch(), 0.0, 4.0)
    exact = C0 / K * (1 - np.exp(-K * 4.0))
    np.testing.assert_allclose(before.values, [exact, 2 * exact], rtol=1e-9)


def test_partial_auc_whole_range_equals_auc_last() -> None:
    tcs = batch()
    options = NCAOptions(auc_method=AUCMethod.LINEAR)
    whole = partial_auc(tcs, 0.5, 12.0, options=options)
    # auc_last of a bolus includes the inserted (0, C0) segment, so compare against the area from the first sample
    result = nca(
        tcs.__class__.from_timecourses(
            [tc.model_copy(update={"dosing": None}) for tc in tcs]
        ),
        options=options,
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
        options=NCAOptions(auc_method=AUCMethod.LOG),
    )
    expected = C0 / K * (np.exp(-K * 1.0) - np.exp(-K * 6.0))
    assert float(area.values[0]) == pytest.approx(expected, rel=1e-9)


def test_partial_auc_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="t_end"):
        partial_auc(batch(), 4.0, 1.0)


T_SPARSE = np.array([0.5, 1, 2, 4, 6, 8, 12])


def routed_batch(route: Route) -> Timecourses:
    """One monoexponential curve with a dose of the given route."""
    duration = 0.5 if route is Route.IV_INFUSION else None
    curve = Timecourse(
        time=T_SPARSE,
        value=C0 * np.exp(-K * T_SPARSE),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=route, duration=duration),
        substance="x",
    )
    return Timecourses.from_timecourses([curve])


def test_partial_auc_from_zero_back_extrapolates_a_bolus() -> None:
    # B29: AUC(0-t) was NaN for every curve whose first sample is after 0
    options = NCAOptions(auc_method=AUCMethod.LOG)
    area = float(
        partial_auc(routed_batch(Route.IV_BOLUS), 0.0, 12.0, options=options).values[0]
    )
    # the log rule with the back extrapolated C0 is exact for a bolus
    assert area == pytest.approx(C0 / K * (1 - np.exp(-K * 12.0)), rel=1e-9)


def test_partial_auc_from_zero_starts_at_zero_for_an_extravascular_dose() -> None:
    options = NCAOptions(auc_method=AUCMethod.LOG)
    batch = routed_batch(Route.ORAL)
    area = float(partial_auc(batch, 0.0, 12.0, options=options).values[0])
    observed = float(partial_auc(batch, 0.5, 12.0, options=options).values[0])
    # the concentration is 0 at the dose: the first segment is the triangle
    # from (0, 0) to the first sample
    rise = 0.5 * 0.5 * C0 * np.exp(-K * 0.5)
    assert area == pytest.approx(observed + rise)


def test_partial_auc_from_zero_is_nan_for_an_infusion() -> None:
    options = NCAOptions(auc_method=AUCMethod.LOG)
    batch = routed_batch(Route.IV_INFUSION)
    assert np.isnan(partial_auc(batch, 0.0, 12.0, options=options).values).all()
    assert np.isfinite(partial_auc(batch, 0.5, 12.0, options=options).values).all()


def test_partial_auc_before_the_dose_is_nan() -> None:
    batch = routed_batch(Route.ORAL)
    assert np.isnan(partial_auc(batch, -1.0, 12.0).values).all()

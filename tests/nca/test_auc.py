import numpy as np
import pytest

from pkpdutils.nca.auc import (
    auc_aumc,
    insert_point,
    interpolate_at,
    pack_valid,
    segment_areas,
)
from pkpdutils.nca.options import AUCMethod


def monoexp(t: np.ndarray, c0: float = 10.0, k: float = 0.5) -> np.ndarray:
    return c0 * np.exp(-k * t)


def test_pack_valid_moves_valid_points_to_front() -> None:
    t = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, np.nan, np.nan]])
    c = np.array([[1.0, np.nan, 3.0, 4.0], [1.0, 2.0, np.nan, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    np.testing.assert_array_equal(n_valid, [3, 2])
    np.testing.assert_allclose(tp[0, :3], [0.0, 2.0, 3.0])
    np.testing.assert_allclose(cp[0, :3], [1.0, 3.0, 4.0])
    assert np.isnan(tp[0, 3]) and np.isnan(cp[0, 3])
    np.testing.assert_allclose(cp[1, :2], [1.0, 2.0])


def test_linear_trapezoid_on_line() -> None:
    t = np.array([[0.0, 1.0, 2.0, 4.0]])
    c = np.array(
        [[0.0, 2.0, 4.0, 8.0]]
    )  # c = 2t: auc = t^2 (linear trapezoid is exact for AUC)
    tp, cp, n_valid = pack_valid(t, c)
    auc, aumc = auc_aumc(tp, cp, n_valid, AUCMethod.LINEAR)
    assert auc[0] == pytest.approx(16.0)
    # the standard linear-trapezoidal AUMC formula dt*(t1*c1+t2*c2)/2 is NOT exact
    # for AUMC even on a straight concentration line, because the integrand t*c(t)
    # is quadratic; per-segment: 1*(0*0+1*2)/2 + 1*(1*2+2*4)/2 + 2*(2*4+4*8)/2
    # = 1.0 + 5.0 + 40.0 = 46.0 (the exact quadratic integral would be 2/3*64=42.667,
    # but that is a different, non-standard formula not used here)
    assert aumc[0] == pytest.approx(46.0)


def test_log_trapezoid_exact_on_monoexponential() -> None:
    t = np.array([[0.0, 1.0, 3.0, 8.0, 20.0]])
    c = monoexp(t)
    tp, cp, n_valid = pack_valid(t, c)
    auc, aumc = auc_aumc(tp, cp, n_valid, AUCMethod.LOG)
    # analytic: auc(0-20) = c0/k (1 - e^{-20k}); aumc = c0/k^2 (1 - e^{-20k}(1 + 20k))
    k, c0 = 0.5, 10.0
    assert auc[0] == pytest.approx(c0 / k * (1 - np.exp(-20 * k)), rel=1e-10)
    assert aumc[0] == pytest.approx(
        c0 / k**2 * (1 - np.exp(-20 * k) * (1 + 20 * k)), rel=1e-10
    )


def test_linear_log_uses_log_only_on_falling_segments() -> None:
    t = np.array([[0.0, 1.0, 2.0, 4.0]])
    c = np.array([[0.0, 4.0, 2.0, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    areas, _ = segment_areas(tp, cp, n_valid, AUCMethod.LINEAR_LOG)
    assert areas[0, 0] == pytest.approx(2.0)  # rising: linear 0.5*(0+4)*1
    assert areas[0, 1] == pytest.approx((4 - 2) / np.log(2) * 1)  # falling: log
    assert areas[0, 2] == pytest.approx((2 - 1) / np.log(2) * 2)


def test_segments_beyond_valid_points_are_zero() -> None:
    t = np.array([[0.0, 1.0, 2.0, np.nan]])
    c = np.array([[1.0, 1.0, 1.0, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    areas, moments = segment_areas(tp, cp, n_valid, AUCMethod.LINEAR)
    np.testing.assert_allclose(areas[0], [1.0, 1.0, 0.0])
    assert not np.isnan(moments).any()


def test_auc_until_t_end() -> None:
    t = np.array([[0.0, 1.0, 2.0, 3.0]])
    c = np.array([[1.0, 1.0, 1.0, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    auc, _ = auc_aumc(tp, cp, n_valid, AUCMethod.LINEAR, t_end=np.array([2.0]))
    assert auc[0] == pytest.approx(2.0)


def test_auc_between_t_start_and_t_end() -> None:
    t = np.array([[-1.0, 0.0, 1.0, 2.0, 3.0]])
    c = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    start, end = np.array([0.0]), np.array([2.0])
    auc, aumc = auc_aumc(tp, cp, n_valid, AUCMethod.LINEAR, t_start=start, t_end=end)
    assert auc[0] == pytest.approx(2.0)  # without t_start the segment [-1, 0] adds 1
    assert aumc[0] == pytest.approx(0.5 + 1.5)  # dt (t1 c1 + t2 c2) / 2 per segment
    # a segment starting before t_start is dropped as a whole
    partial, _ = auc_aumc(
        tp, cp, n_valid, AUCMethod.LINEAR, t_start=np.array([-0.5]), t_end=end
    )
    assert partial[0] == pytest.approx(2.0)


def test_interpolate_at() -> None:
    t = np.array([[0.0, 1.0, 2.0, 4.0], [0.0, 2.0, np.nan, np.nan]])
    c = np.array([[0.0, 4.0, 2.0, 1.0], [1.0, 3.0, np.nan, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    lin = interpolate_at(tp, cp, n_valid, np.array([3.0, 1.0]), AUCMethod.LINEAR)
    np.testing.assert_allclose(lin, [1.5, 2.0])
    log = interpolate_at(tp, cp, n_valid, np.array([3.0, 1.0]), AUCMethod.LINEAR_LOG)
    assert log[0] == pytest.approx(
        2.0 * np.exp(np.log(1.0 / 2.0) * 0.5)
    )  # log-down between (2,2) and (4,1)
    assert log[1] == pytest.approx(2.0)  # rising segment stays linear
    at_point = interpolate_at(tp, cp, n_valid, np.array([2.0, 2.0]), AUCMethod.LINEAR)
    np.testing.assert_allclose(at_point, [2.0, 3.0])
    outside = interpolate_at(tp, cp, n_valid, np.array([5.0, -1.0]), AUCMethod.LINEAR)
    assert np.isnan(outside).all()


def test_insert_point_keeps_time_order() -> None:
    t = np.array([[0.0, 1.0, 4.0, np.nan]])
    c = np.array([[0.0, 4.0, 1.0, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    tp2, cp2, n2 = insert_point(tp, cp, n_valid, np.array([2.0]), np.array([2.5]))
    assert n2[0] == 4
    np.testing.assert_allclose(tp2[0, :4], [0.0, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(cp2[0, :4], [0.0, 4.0, 2.5, 1.0])
    tp3, _, n3 = insert_point(tp, cp, n_valid, np.array([np.nan]), np.array([np.nan]))
    assert n3[0] == 3 and tp3.shape[1] == tp.shape[1] + 1

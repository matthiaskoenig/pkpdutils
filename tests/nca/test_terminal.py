import numpy as np
import pytest

from pkpdutils.nca.auc import pack_valid
from pkpdutils.nca.options import NCAFlag, TerminalMethod, TerminalPhase
from pkpdutils.nca.terminal import terminal_fit, window_statistics


def oral_curve(
    t: np.ndarray, ka: float = 2.0, ke: float = 0.3, f: float = 10.0
) -> np.ndarray:
    return f * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))


def test_window_statistics_matches_numpy_polyfit() -> None:
    x = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    y = np.array([[5.0, 3.9, 3.1, 1.9, 1.2]])
    valid = np.ones_like(x, dtype=bool)
    stats = window_statistics(x, y, valid)
    for s in range(3):
        slope, intercept = np.polyfit(x[0, s:], y[0, s:], 1)
        assert stats["slope"][0, s] == pytest.approx(slope)
        assert stats["intercept"][0, s] == pytest.approx(intercept)
        assert stats["n"][0, s] == 5 - s


def test_monoexponential_recovers_slope_exactly() -> None:
    t = np.linspace(0, 10, 11)[None, :]
    c = 8.0 * np.exp(-0.4 * t)
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase())
    assert fit.slope[0] == pytest.approx(-0.4)
    assert fit.intercept[0] == pytest.approx(np.log(8.0))
    assert fit.r2[0] == pytest.approx(1.0)
    assert fit.flags[0] == 0
    # exclude_cmax: the maximum at index 0 is excluded, the best window has 10 points
    assert fit.n_points[0] == 10
    assert fit.t_first[0] == pytest.approx(1.0)


def test_best_fit_prefers_more_points_within_tolerance() -> None:
    t = np.linspace(0, 10, 11)[None, :]
    c = 8.0 * np.exp(-0.4 * t)
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(
        tp, cp, n_valid, np.array([0]), TerminalPhase(exclude_cmax=False)
    )
    assert fit.n_points[0] == 11
    assert fit.start[0] == 0


def test_best_fit_skips_absorption_phase() -> None:
    t = np.array([[0.25, 0.5, 1, 2, 4, 6, 8, 12, 24]], dtype=float)
    c = oral_curve(t)
    tp, cp, n_valid = pack_valid(t, c)
    tmax_idx = np.array([int(np.argmax(c[0]))])
    fit = terminal_fit(tp, cp, n_valid, tmax_idx, TerminalPhase())
    assert fit.slope[0] == pytest.approx(-0.3, rel=0.02)
    assert fit.n_points[0] >= 3
    assert fit.t_first[0] > t[0, tmax_idx[0]]


def test_best_fit_start_snaps_to_a_regressable_point() -> None:
    # the zero at t = 2 cannot be regressed: the window of the suffix sums that
    # starts there is the window starting at t = 3, so the reported first point
    # must be t = 3 and not the excluded zero
    t = np.array([[0, 1, 2, 3, 4, 5]], dtype=float)
    c = np.array([[10.0, 8.0, 0.0, 4.0, 2.0, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase())
    assert fit.t_first[0] == pytest.approx(3.0)
    assert fit.n_points[0] == 3
    slope = np.polyfit([3.0, 4.0, 5.0], np.log([4.0, 2.0, 1.0]), 1)[0]
    assert fit.slope[0] == pytest.approx(slope)
    assert tp[0, fit.start[0]] == pytest.approx(3.0)


def test_last_n_and_all_after_tmax() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=float)
    c = np.array([[1, 5, 4, 3, 2.2, 1.5, 1.1]])
    tp, cp, n_valid = pack_valid(t, c)
    tmax_idx = np.array([1])
    last3 = terminal_fit(
        tp,
        cp,
        n_valid,
        tmax_idx,
        TerminalPhase(method=TerminalMethod.LAST_N, n_points=3),
    )
    assert last3.n_points[0] == 3
    assert last3.t_first[0] == pytest.approx(4.0)
    after = terminal_fit(
        tp, cp, n_valid, tmax_idx, TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX)
    )
    assert after.n_points[0] == 5
    assert after.t_first[0] == pytest.approx(2.0)
    slope, intercept = np.polyfit(t[0, 2:], np.log(c[0, 2:]), 1)
    assert after.slope[0] == pytest.approx(slope)
    assert after.intercept[0] == pytest.approx(intercept)


def test_manual_mask() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5]], dtype=float)
    c = np.array([[1, 5, 4, 3, 2, 1.4]])
    tp, cp, n_valid = pack_valid(t, c)
    mask = np.array([[False, False, True, False, True, True]])
    fit = terminal_fit(
        tp,
        cp,
        n_valid,
        np.array([1]),
        TerminalPhase(method=TerminalMethod.MANUAL, points=(2, 4, 5)),
        manual_mask=mask,
    )
    slope, _ = np.polyfit([2, 4, 5], np.log([4, 2, 1.4]), 1)
    assert fit.slope[0] == pytest.approx(slope)
    assert fit.n_points[0] == 3


def test_too_few_points_flag() -> None:
    t = np.array([[0, 1, 2, 3]], dtype=float)
    c = np.array([[1, 5, 4, 3]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([1]), TerminalPhase())
    assert np.isnan(fit.slope[0])
    assert fit.flags[0] & NCAFlag.TOO_FEW_POINTS
    assert fit.start[0] == -1


def test_positive_slope_flag() -> None:
    t = np.array([[0, 1, 2, 3, 4]], dtype=float)
    c = np.array([[1, 1.1, 1.3, 1.6, 2.0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(
        tp, cp, n_valid, np.array([4]), TerminalPhase(exclude_cmax=False)
    )
    assert np.isnan(fit.slope[0])
    assert fit.flags[0] & NCAFlag.POSITIVE_SLOPE


def test_zero_values_are_excluded_from_regression() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5]], dtype=float)
    c = np.array([[5, 4, 3, 2, 0, 0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(
        tp,
        cp,
        n_valid,
        np.array([0]),
        TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX),
    )
    slope, _ = np.polyfit([1, 2, 3], np.log([4, 3, 2]), 1)
    assert fit.slope[0] == pytest.approx(slope)
    assert fit.n_points[0] == 3


def test_min_adj_r2_rejects_poor_fit() -> None:
    t = np.array([[0, 1, 2, 3, 4, 5]], dtype=float)
    c = np.array([[9, 5, 6, 2, 4, 1.0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase(min_adj_r2=0.999))
    assert np.isnan(fit.slope[0])
    assert fit.flags[0] & NCAFlag.TOO_FEW_POINTS


def test_batch_rows_are_independent() -> None:
    t = np.tile(np.linspace(0, 10, 11), (3, 1))
    ks = np.array([0.2, 0.5, 1.0])
    c = 5.0 * np.exp(-ks[:, None] * t)
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.zeros(3, dtype=int), TerminalPhase())
    np.testing.assert_allclose(fit.slope, -ks)


def test_last_n_honours_exclude_cmax() -> None:
    # B18: `LAST_N` regressed the whole absorption phase when `n_points` reached
    # beyond the maximum
    t = np.array([[0.5, 1, 2, 4, 8, 12, 24.0]])
    c = np.array([[1.2, 2.5, 2.1, 1.3, 0.5, 0.2, 0.05]])
    tp, cp, n_valid = pack_valid(t, c)
    tmax_idx = np.array([1])
    excluded = terminal_fit(
        tp,
        cp,
        n_valid,
        tmax_idx,
        TerminalPhase(method=TerminalMethod.LAST_N, n_points=7, exclude_cmax=True),
    )
    assert excluded.n_points[0] == 5
    assert excluded.t_first[0] == pytest.approx(2.0)
    included = terminal_fit(
        tp,
        cp,
        n_valid,
        tmax_idx,
        TerminalPhase(method=TerminalMethod.LAST_N, n_points=7, exclude_cmax=False),
    )
    assert included.n_points[0] == 7
    assert included.t_first[0] == pytest.approx(0.5)


def test_t_last_is_the_last_regressed_point() -> None:
    # the window ends at the last regressable point, not at the padded end
    t = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    c = np.array([[8.0, 4.0, 2.0, 1.0, 0.5, np.nan]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([0]), TerminalPhase())
    assert fit.t_first[0] == pytest.approx(1.0)
    assert fit.t_last[0] == pytest.approx(4.0)


def test_t_last_of_a_manual_window() -> None:
    t = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    c = np.array([[8.0, 4.0, 2.0, 1.0, 0.5, 0.25]])
    tp, cp, n_valid = pack_valid(t, c)
    mask = np.zeros_like(c, dtype=bool)
    mask[0, 1:4] = True
    fit = terminal_fit(
        tp,
        cp,
        n_valid,
        np.array([0]),
        TerminalPhase(method=TerminalMethod.MANUAL, points=(1, 2, 3)),
        manual_mask=mask,
    )
    assert fit.n_points[0] == 3
    assert fit.t_first[0] == pytest.approx(1.0)
    assert fit.t_last[0] == pytest.approx(3.0)


def test_t_last_is_nan_without_a_fit() -> None:
    t = np.array([[0.0, 1.0]])
    c = np.array([[1.0, 2.0]])
    tp, cp, n_valid = pack_valid(t, c)
    fit = terminal_fit(tp, cp, n_valid, np.array([1]), TerminalPhase())
    assert np.isnan(fit.t_last[0])

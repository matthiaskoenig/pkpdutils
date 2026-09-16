import numpy as np
import pytest

from pkpdutils import Dose, NCAOptions, Route, Timecourse, Timecourses, nca, nca_single
from pkpdutils.nca.options import TerminalMethod, TerminalPhase
from pkpdutils.nca.terminal import CANDIDATE_DIM

CANDIDATE_VARIABLES = ("candidate_t_first", "candidate_n_points", "candidate_r2_adj")


def curve(label: str = "a", k: float = 0.2) -> Timecourse:
    """A mono-exponential oral curve with eight points after the maximum."""
    t = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0])
    ka = 2.0
    c = 10.0 * ka / (ka - k) * (np.exp(-k * t) - np.exp(-ka * t))
    return Timecourse(
        time=t,
        value=c,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        label=label,
    )


def test_the_default_result_carries_no_candidate_windows() -> None:
    result = nca_single(curve())
    assert not [name for name in CANDIDATE_VARIABLES if name in result.ds.data_vars]
    assert CANDIDATE_DIM not in result.ds.dims


def test_keep_candidates_leaves_every_parameter_unchanged() -> None:
    tc = curve()
    plain = nca_single(tc)
    kept = nca_single(
        tc, options=NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    )
    for name in plain.parameters:
        np.testing.assert_allclose(
            float(plain[name].values), float(kept[name].values), rtol=1e-12
        )
    assert int(kept["flags"].values) == int(plain["flags"].values)


def test_the_candidate_windows_are_the_windows_after_the_maximum() -> None:
    tc = curve()
    result = nca_single(
        tc, options=NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    )
    starts = result.ds["candidate_t_first"].to_numpy()
    counts = result.ds["candidate_n_points"].to_numpy()
    # the maximum sits at 1.5 h, so the windows start at 2 h and hold 8, 7, ...
    # down to the three points of the last window
    np.testing.assert_allclose(starts, [2.0, 3.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(counts, [7.0, 6.0, 5.0, 4.0, 3.0])
    assert result.ds["candidate_t_first"].dims == (CANDIDATE_DIM,)
    assert result.units("candidate_t_first") == "hour"
    assert result.units("candidate_r2_adj") == "dimensionless"
    # they are point variables, so they stay out of the parameter table
    assert set(CANDIDATE_VARIABLES) <= set(result.point_variables)
    assert not set(CANDIDATE_VARIABLES) & set(result.to_dataframe().columns)


def test_the_chosen_window_is_the_candidate_with_the_largest_adjusted_r2() -> None:
    tc = curve()
    result = nca_single(
        tc, options=NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    )
    starts = result.ds["candidate_t_first"].to_numpy()
    r2_adj = result.ds["candidate_r2_adj"].to_numpy()
    chosen = float(result["lambda_z_t_first"].values)
    assert chosen in set(starts.tolist())
    picked = starts == chosen
    np.testing.assert_allclose(
        float(result["lambda_z_r2_adj"].values), float(r2_adj[picked][0])
    )
    assert float(r2_adj[picked][0]) >= r2_adj.max() - 1e-4


def test_the_candidates_of_another_rule_are_the_same_windows() -> None:
    # the table describes the windows the selection may choose from, so it does
    # not depend on the rule which picks one of them
    tc = curve()
    best = nca_single(
        tc, options=NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    )
    last_three = nca_single(
        tc,
        options=NCAOptions(
            terminal=TerminalPhase(
                method=TerminalMethod.LAST_N, n_points=3, keep_candidates=True
            )
        ),
    )
    np.testing.assert_allclose(
        best.ds["candidate_t_first"].to_numpy(),
        last_three.ds["candidate_t_first"].to_numpy(),
    )
    assert float(last_three["lambda_z_t_first"].values) == 8.0


def test_without_exclude_cmax_the_windows_start_before_the_maximum() -> None:
    tc = curve()
    result = nca_single(
        tc,
        options=NCAOptions(
            terminal=TerminalPhase(exclude_cmax=False, keep_candidates=True)
        ),
    )
    starts = result.ds["candidate_t_first"].to_numpy()
    assert starts.min() == 0.25 and starts.size == 9


def test_a_batch_of_several_samples_keeps_no_candidate_table() -> None:
    batch = Timecourses.from_timecourses([curve("a", 0.2), curve("b", 0.3)])
    options = NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    result = nca(batch, options=options)
    plain = nca(batch)
    assert CANDIDATE_DIM not in result.ds.dims
    for name in plain.parameters:
        np.testing.assert_allclose(
            plain[name].to_numpy(), result[name].to_numpy(), rtol=1e-12
        )


def test_a_batch_of_one_sample_carries_the_candidate_table() -> None:
    batch = Timecourses.from_timecourses([curve("a")])
    result = nca(
        batch, options=NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    )
    assert result.ds["candidate_t_first"].dims == ("individual", CANDIDATE_DIM)


def test_a_curve_without_a_terminal_phase_has_no_candidate_window() -> None:
    # three points, two of them after the maximum: no window of three points
    tc = Timecourse(
        time=[0.5, 1.0, 2.0],
        value=[1.0, 3.0, 2.0],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=10, unit="mg", route=Route.ORAL),
    )
    result = nca_single(
        tc, options=NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    )
    assert CANDIDATE_DIM not in result.ds.dims
    assert not np.isfinite(float(result["lambda_z"].values))


@pytest.mark.parametrize("keep", [False, True])
def test_the_candidate_table_of_the_fit_follows_the_option(keep: bool) -> None:
    from pkpdutils.nca.auc import pack_valid
    from pkpdutils.nca.terminal import terminal_fit

    tc = curve()
    t = tc.time[None, :]
    c = tc.value[None, :]
    tp, cp, n_valid = pack_valid(t, c)
    imax = np.argmax(cp, axis=1)
    fit = terminal_fit(tp, cp, n_valid, imax, TerminalPhase(keep_candidates=keep))
    if not keep:
        assert fit.candidates is None
        return
    assert fit.candidates is not None
    assert list(fit.candidates.columns) == ["row", "start_time", "n", "r2_adj", "slope"]
    assert (fit.candidates["row"] == 0).all()
    assert (fit.candidates["slope"] < 0).all()

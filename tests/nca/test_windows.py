"""Per-sample terminal windows (`TerminalPhase.windows`) and `NCAResult.terminal_windows`."""

import numpy as np
import pytest

from pkpdutils import (
    NCAOptions,
    Route,
    TerminalMethod,
    TerminalPhase,
    Timecourses,
    nca,
)

TIME = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
LABELS = ["s1", "s2", "s3"]


def batch() -> Timecourses:
    k = np.array([0.15, 0.2, 0.25])
    return Timecourses.from_arrays(
        TIME,
        10.0 * np.exp(-k[:, None] * TIME[None, :]),
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": LABELS},
        dose={"amount": np.full(3, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="drug",
    )


def test_a_window_of_one_sample_regresses_the_points_inside_it() -> None:
    windows = {"s2": (4.0, 24.0)}
    result = nca(batch(), options=NCAOptions(terminal=TerminalPhase(windows=windows)))
    plain = nca(batch())
    assert float(result["lambda_z_t_first"].sel(individual="s2")) == 4.0
    assert float(result["lambda_z_t_last"].sel(individual="s2")) == 24.0
    assert float(result["lambda_z_n_points"].sel(individual="s2")) == 5.0
    # a mono-exponential gives the same rate constant from every window
    assert float(result["lambda_z"].sel(individual="s2")) == pytest.approx(0.2)
    # the other samples follow the batch rule and are unchanged
    for label in ("s1", "s3"):
        assert float(result["lambda_z_t_first"].sel(individual=label)) == float(
            plain["lambda_z_t_first"].sel(individual=label)
        )
        assert float(result["auc_inf_obs"].sel(individual=label)) == pytest.approx(
            float(plain["auc_inf_obs"].sel(individual=label))
        )


def test_the_star_key_names_every_other_sample() -> None:
    windows = {"s1": (8.0, 24.0), "*": (2.0, 12.0)}
    result = nca(batch(), options=NCAOptions(terminal=TerminalPhase(windows=windows)))
    assert result["lambda_z_t_first"].to_numpy().tolist() == [8.0, 2.0, 2.0]
    assert result["lambda_z_t_last"].to_numpy().tolist() == [24.0, 12.0, 12.0]


def test_terminal_windows_re_runs_the_analysis_identically() -> None:
    data = batch()
    options = NCAOptions(terminal=TerminalPhase(method=TerminalMethod.BEST_FIT))
    result = nca(data, options=options)
    windows = result.terminal_windows()
    assert set(windows) == set(LABELS)
    assert windows["s1"] == (
        float(result["lambda_z_t_first"].sel(individual="s1")),
        float(result["lambda_z_t_last"].sel(individual="s1")),
    )
    reviewed = nca(
        data,
        options=options.model_copy(update={"terminal": TerminalPhase(windows=windows)}),
    )
    for name in result.ds.data_vars:
        if result.ds[str(name)].dtype.kind not in "fi":
            continue
        np.testing.assert_allclose(
            reviewed[str(name)].to_numpy(),
            result[str(name)].to_numpy(),
            equal_nan=True,
            err_msg=str(name),
        )


def test_a_sample_without_a_terminal_phase_carries_no_window() -> None:
    flat = Timecourses.from_arrays(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.ones((1, 4)),
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["flat"]},
        dose={"amount": np.full(1, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="drug",
    )
    assert nca(flat).terminal_windows() == {}


def test_an_unknown_label_and_a_reversed_window_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be"):
        TerminalPhase(windows={"s1": (12.0, 4.0)})
    options = NCAOptions(terminal=TerminalPhase(windows={"s9": (4.0, 24.0)}))
    with pytest.raises(ValueError, match="name no sample of the batch"):
        nca(batch(), options=options)

"""Regressions for shared NCA preprocessing and bounded uncertainty work."""

import importlib
import json
import warnings

import numpy as np
import pytest

from pkpdutils import Timecourse, Timecourses
from pkpdutils.nca import NCAOptions, engine, nca, nca_single


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e-6, 1e-200, 1e200])
def test_delta_retains_spread_below_floating_point_increment(scale: float) -> None:
    se = scale * 1e-15
    tc = Timecourse(
        time=[0, 1, 2, 3],
        value=[scale] * 4,
        se=[se] * 4,
        n=10,
        time_unit="h",
        unit="mg/l",
    )
    result = nca_single(
        tc,
        options=NCAOptions(
            auc_method="linear", uncertainty="delta", delta_step=1e-4, n_workers=1
        ),
    )
    # Trapezoidal area has weights (1/2, 1, 1, 1/2), independent of scale.
    assert float(result.ds.auc_last_se) == pytest.approx(
        np.sqrt(2.5) * se, rel=1e-6, abs=0
    )


def test_lognormal_bootstrap_keeps_finite_draws_at_large_spread() -> None:
    tc = Timecourse(
        time=[0, 1, 2, 3],
        value=[1, 0.7, 0.5, 0.3],
        se=[1e155] * 4,
        n=10,
        time_unit="h",
        unit="mg/l",
    )
    result = nca_single(
        tc,
        options=NCAOptions(
            uncertainty="bootstrap",
            bootstrap_distribution="lognormal",
            n_boot=20,
            seed=1,
            n_workers=1,
        ),
    )
    assert np.isfinite(float(result.ds.cmax_geomean))
    assert 0 < float(result.ds.cmax_se) < np.inf


@pytest.mark.parametrize(
    "values,times,kind,parameter",
    [
        ([0, 0, 0], [0, 1, 2], "concentration", "cmax"),
        ([-3, -1, -2], [0, 1, 2], "effect", "emax_obs"),
        ([0, np.finfo(float).max, 0], [0, 1e-308, 2e-308], "concentration", "cmax"),
    ],
)
def test_delta_steps_at_zero_negative_and_upper_finite_limit(
    values: list[float], times: list[float], kind: str, parameter: str
) -> None:
    tc = Timecourse(
        time=times,
        value=values,
        se=[0, 1, 0],
        n=10,
        time_unit="h",
        unit="mg/l",
    )
    result = nca_single(
        tc,
        options=NCAOptions(
            auc_method="linear", uncertainty="delta", kind=kind, n_workers=1
        ),
    )
    assert float(result.ds[f"{parameter}_se"]) == pytest.approx(1.0)


def test_delta_retains_smallest_positive_spread_at_zero() -> None:
    smallest = np.nextafter(0.0, 1.0)
    tc = Timecourse(
        time=[0, 1, 2],
        value=[0, 0, 0],
        se=[0, smallest, 0],
        time_unit="h",
        unit="mg/l",
    )
    result = nca_single(tc, options=NCAOptions(uncertainty="delta", n_workers=1))
    assert float(result.ds.cmax_se) == smallest


def test_linear_auc_keeps_finite_area_when_concentration_sum_overflows() -> None:
    tc = Timecourse(
        time=[0, 1e-308, 2e-308],
        value=[1e308] * 3,
        time_unit="h",
        unit="mg/l",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = nca_single(
            tc, options=NCAOptions(uncertainty="none", auc_method="linear")
        )
    assert float(result.ds.auc_last) == pytest.approx(2.0)


def test_zero_counts_on_empty_points_do_not_warn_in_uncertainty() -> None:
    batch = Timecourses.from_arrays(
        [0, 1, 2],
        [[np.nan] * 3],
        sd=[[0.0] * 3],
        n=0,
        time_unit="h",
        unit="mg/l",
        dims=("individual",),
    )
    # A dataset from another producer may carry SD without a derived SE.
    batch = Timecourses(batch.ds.drop_vars("se"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = nca(batch, options=NCAOptions(n_boot=10, seed=1))
    assert np.isnan(result.ds.auc_last_se.to_numpy()).all()


def curve(*, offset: float = 0.0, rate: float = 0.3) -> Timecourse:
    t = np.arange(8.0)
    c = 10.0 * np.exp(-rate * t)
    return Timecourse(
        time=t + offset,
        value=c,
        se=c * 0.05,
        n=12,
        time_unit="h",
        unit="mg/l",
        substance="x",
    )


@pytest.mark.parametrize("method", ["none", "bootstrap", "delta"])
def test_named_partial_auc_uses_blq_and_propagates_uncertainty(method: str) -> None:
    tc = Timecourse(
        time=[1, 2, 3, 4, 5],
        value=[10, 0.1, 5, 3, 1],
        se=[1, 0.01, 0.5, 0.3, 0.1],
        n=12,
        time_unit="h",
        unit="mg/l",
        substance="x",
    )
    result = nca_single(
        tc,
        options=NCAOptions(
            auc_method="linear",
            lloq=0.5,
            partial_aucs={"my_auc": (1, 5)},
            uncertainty=method,
            n_boot=40,
            seed=1,
            n_workers=1,
            chunk_rows=11,
        ),
    )
    assert float(result.ds.my_auc) == pytest.approx(21.0)
    assert float(result.ds.my_auc) == pytest.approx(float(result.ds.auc_last))
    if method != "none":
        assert float(result.ds.my_auc_se) > 0
        assert float(result.ds.my_auc_se) == pytest.approx(float(result.ds.auc_last_se))


@pytest.mark.parametrize("offset,rate", [(1e8, 0.3), (0, 1e-8)])
def test_terminal_regression_handles_offsets_and_small_slopes(
    offset: float, rate: float
) -> None:
    result = nca_single(
        curve(offset=offset, rate=rate), options=NCAOptions(uncertainty="none")
    )
    assert float(result.ds.lambda_z) == pytest.approx(rate, rel=1e-6)
    assert float(result.ds.lambda_z_r2) == pytest.approx(1.0)
    assert np.isfinite(float(result.ds.lambda_z_stderr))


def test_delta_bounds_generated_perturbations(monkeypatch: pytest.MonkeyPatch) -> None:
    # the blocking of the perturbations is the python core's, and the kernel
    # answers the delta method of a group batch itself, so this one is forced
    monkeypatch.setattr(engine, "_force_python", True)
    core = importlib.import_module("pkpdutils.nca.nca")
    original = core.run_analysis_rows
    seen = []

    def observe(t, c, **kwargs):
        seen.append(c.shape)
        return original(t, c, **kwargs)

    monkeypatch.setattr(core, "run_analysis_rows", observe)
    options = NCAOptions(uncertainty="delta", chunk_rows=3, n_workers=1)
    result = nca(Timecourses.from_timecourses([curve(), curve()]), options=options)
    assert seen[0] == (2, 8)
    assert len(seen) > 2
    assert all(rows <= 3 for rows, _ in seen[1:])
    assert np.isfinite(result.ds.auc_last_se).all()


def test_bootstrap_reduces_each_block_before_next_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the order of the analysis and the reduction of a block is the python
    # core's; the kernel blocks its own replicates and is held by its own tests
    monkeypatch.setattr(engine, "_force_python", True)
    core = importlib.import_module("pkpdutils.nca.nca")
    uncertainty = importlib.import_module("pkpdutils.nca.uncertainty")
    original_run = core.run_analysis_rows
    original_reduce = uncertainty.reduce_replicates
    events = []

    def observe_run(*args, **kwargs):
        events.append("analyse")
        return original_run(*args, **kwargs)

    def observe_reduce(*args, **kwargs):
        events.append("reduce")
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(core, "run_analysis_rows", observe_run)
    monkeypatch.setattr(uncertainty, "reduce_replicates", observe_reduce)
    nca(
        Timecourses.from_timecourses([curve(), curve(), curve()]),
        options=NCAOptions(
            uncertainty="bootstrap",
            n_boot=10,
            chunk_rows=10,
            n_workers=1,
            seed=1,
        ),
    )
    assert events == [
        "analyse",
        "analyse",
        "reduce",
        "analyse",
        "reduce",
        "analyse",
        "reduce",
    ]


def test_result_records_resolved_options() -> None:
    result = nca_single(curve(), options=NCAOptions(seed=7, n_boot=10, n_workers=1))
    saved = json.loads(result.ds.attrs["nca_options"])
    assert saved["uncertainty"] == "bootstrap"
    assert saved["seed"] == 7
    from pkpdutils.nca.options import deserialize_nca_options

    assert deserialize_nca_options(result.ds.attrs["nca_options"]).n_boot == 10


def test_options_provenance_preserves_typed_window_labels() -> None:
    from pkpdutils.nca.options import (
        TerminalPhase,
        deserialize_nca_options,
        serialize_nca_options,
    )

    windows = {
        ("arm", 1): (1.0, 7.0),
        2: (2.0, 7.0),
        "2": (3.0, 7.0),
        np.int64(3): (4.0, 7.0),
    }
    options = NCAOptions(terminal=TerminalPhase(windows=windows), uncertainty="none")
    restored = deserialize_nca_options(serialize_nca_options(options))
    assert restored == options
    assert restored.terminal.windows == windows
    assert ("arm", 1) in restored.terminal.windows
    assert 2 in restored.terminal.windows
    assert "2" in restored.terminal.windows

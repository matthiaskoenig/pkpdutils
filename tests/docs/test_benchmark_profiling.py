"""Check profiling accounting and engine selection without timing thresholds."""

import importlib
import json
import pstats
import subprocess
import sys
from pathlib import Path

import pytest

from pkpdutils.nca import engine
from scripts import benchmark


def test_nested_phase_times_are_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((1.0, 2.0, 5.0, 8.0))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    timer = benchmark.PhaseTimer()
    child = timer.wrap("input", lambda: None)
    parent = timer.wrap("dispatch", child)
    parent()
    assert timer.seconds["input"] == 3.0
    assert timer.seconds["dispatch"] == 4.0
    assert timer.calls == {"dispatch": 1, "input": 1}
    assert timer.stack == []


@pytest.mark.parametrize("forced", [False, True])
def test_profile_measures_real_engine_and_restores_patches(
    forced: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(engine.COMPARE_VARIABLE, raising=False)
    case = benchmark.engine_case(2, 12)
    module = importlib.import_module("pkpdutils.nca.nca")
    original = (
        engine.batch_dict,
        engine.run_kernel,
        module._to_result,
        engine._force_python,
    )
    sample = benchmark.profile_once(case, forced)
    assert sample["total"] == pytest.approx(
        sum(sample[phase] for phase in benchmark.PHASES)
    )
    assert all(value >= 0.0 for value in sample.values())
    assert sample["result"] > 0.0
    assert sample["python"] > 0.0 if forced else sample["binding"] > 0.0
    assert sample["binding"] == 0.0 if forced else sample["python"] == 0.0
    assert (
        engine.batch_dict,
        engine.run_kernel,
        module._to_result,
        engine._force_python,
    ) == original


def test_failed_profile_restores_engine_and_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = benchmark.engine_case(1, 12)
    original = engine.batch_dict
    previous = engine._force_python

    def fail(self: benchmark.EngineCase) -> None:
        raise ValueError("profile failed")

    monkeypatch.setattr(benchmark.EngineCase, "run", fail)
    with pytest.raises(ValueError, match="profile failed"):
        benchmark.profile_once(case, not previous)
    assert engine.batch_dict is original
    assert engine._force_python is previous


@pytest.mark.skipif(sys.platform == "win32", reason="benchmark RSS uses Unix resource")
def test_profile_cli_saves_samples_and_readable_function_profiles(
    tmp_path: Path,
) -> None:
    output = tmp_path / "timings.json"
    profiles = tmp_path / "profiles"
    completed = subprocess.run(
        [
            sys.executable,
            str(benchmark.PAGE_PATH.parent.parent / "scripts" / "benchmark.py"),
            "engine-100x12",
            "--repeat",
            "2",
            "--n-workers",
            "1",
            "--profile-dir",
            str(profiles),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        # The benchmark must never time the suite's differential comparison.
        env={
            key: value
            for key, value in benchmark.os.environ.items()
            if key not in (engine.COMPARE_VARIABLE, benchmark.CHILD_ENV)
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "| engine-100x12 | kernel |" in completed.stdout
    assert "| engine-100x12 | python |" in completed.stdout
    report = json.loads(output.read_text())
    assert report["repeat"] == 2
    assert report["worker_policy"] == 1
    assert report["metadata"]["versions"]["pkpdutils-core"]
    row = report["rows"][0]
    assert row["validated"]
    assert row["options"]["n_workers"] == 1
    assert row["speedup"] == pytest.approx(row["python_s"] / row["kernel_s"])
    for half in ("kernel", "python"):
        assert len(row["samples_s"][half]) == 2
        assert len(row["profile"][half]["samples_s"]) == 2
        assert row["profile"][half]["cprofile"]["top_self"]
        assert row["profile"][half]["cprofile"]["n_workers"] == 1
        path = profiles / f"engine-100x12-{half}.prof"
        assert pstats.Stats(str(path)).get_stats_profile().func_profiles
        assert "function calls" in path.with_suffix(".txt").read_text()


@pytest.mark.parametrize("flag", ["--repeat", "--n-workers"])
def test_cli_rejects_nonpositive_counts(flag: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(benchmark.PAGE_PATH.parent.parent / "scripts" / "benchmark.py"),
            "engine-1x12",
            flag,
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "must be at least 1" in completed.stderr


def test_phase_schema_changes_make_documentation_stale() -> None:
    page = benchmark.PAGE_PATH.read_text(encoding="utf-8")
    assert benchmark.page_complaints(page.replace("| binding ms |", "| stale ms |"))


def test_benchmark_rejects_differential_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    with pytest.raises(RuntimeError, match="differential gate is armed"):
        benchmark.check_dispatch("engine-1x12", benchmark.engine_case(1, 12))


def test_published_tables_match_saved_samples() -> None:
    report = json.loads(benchmark.DATA_PATH.read_text(encoding="utf-8"))
    page = benchmark.PAGE_PATH.read_text(encoding="utf-8")
    rows = report["rows"]
    timings = benchmark.region(page, benchmark.TIMINGS_START, benchmark.TIMINGS_END)
    profiles = benchmark.region(page, benchmark.PROFILES_START, benchmark.PROFILES_END)
    assert benchmark.engine_table(rows) in timings
    assert benchmark.summary_sentences(rows) in timings
    assert benchmark.profile_table(rows, representative=True) in profiles
    assert benchmark.profile_summary_sentences(rows) in profiles
    serial = benchmark.region(page, benchmark.SERIAL_START, benchmark.SERIAL_END)
    assert benchmark.engine_table(report["serial_rows"]) in serial

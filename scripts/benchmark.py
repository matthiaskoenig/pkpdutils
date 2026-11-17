"""Benchmark of the hot paths of `pkpdutils` and of its two engines.

Every case builds its data first and then runs one operation `--repeat` times;
the table reports the median wall time and the peak resident set size of the
process the case ran in. Each case runs in a fresh interpreter, so its memory
is its own and its caches (the unit caches of `pkpdutils.units`) start empty,
as they do in a script a user runs.

```bash
uv run python scripts/benchmark.py all
uv run python scripts/benchmark.py nca-large bootstrap --repeat 5
uv run python scripts/benchmark.py engines
```

The `engine-*` cases are the second table: one single dose analysis measured
through both engines of `pkpdutils.nca`, the rust kernel and the python core,
alternating in one process, and the ratio of the two. `--write` runs every one
of them and pastes the table into `docs/benchmark.md`, `--check` exits 1 when
that page no longer matches the cases of this script (the timings themselves
are not compared, they are machine specific), and
`tests/docs/test_benchmark.py` runs the check:

```bash
uv run python scripts/benchmark.py --write
uv run python scripts/benchmark.py --check
```

The absolute numbers are machine specific and only comparable against another
run of the same script on the same machine; the point of the table is the
before and after of a change.
"""

import argparse
import contextlib
import functools
import gc
import itertools
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pkpdutils_core

import pkpdutils
from pkpdutils import (
    Acceptance,
    Dose,
    Kind,
    NCAOptions,
    NCAResult,
    Route,
    Timecourse,
    Timecourses,
    fit_timecourses,
    nca,
)
from pkpdutils.fit.models import MonoExp
from pkpdutils.fit.options import FitOptions
from pkpdutils.nca import engine
from pkpdutils.nca.options import UncertaintyMethod
from pkpdutils.parallel import NCA_WORKER_THRESHOLD, worker_pool

#: set in the environment of the child process which runs one case
CHILD_ENV = "PKPDUTILS_BENCHMARK_CHILD"

#: the page the comparison of the two engines is pasted into
PAGE_PATH: Path = Path(__file__).resolve().parent.parent / "docs" / "benchmark.md"


@dataclass(frozen=True)
class Case:
    """One benchmark: a label of the problem size and the operation to time."""

    size: str
    run: Callable[[], Any]


def mono_batch(
    n_rows: int,
    n_time: int = 12,
    *,
    spread: bool = False,
    seed: int = 0,
) -> Timecourses:
    """A batch of `n_rows` noisy mono-exponential iv bolus curves."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.25, 24.0, n_time)
    dose = 100.0
    v = rng.uniform(20.0, 60.0, size=n_rows)[:, None]
    k = rng.uniform(0.05, 0.3, size=n_rows)[:, None]
    values = (
        dose / v * np.exp(-k * t[None, :]) * rng.normal(1.0, 0.05, (n_rows, n_time))
    )
    extra: dict[str, Any] = {"sd": 0.2 * values, "n": 8} if spread else {}
    return Timecourses.from_arrays(
        t,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        dose=Dose(amount=dose, unit="mg", route=Route.IV_BOLUS),
        substance="drug",
        **extra,
    )


def multi_dose_batch(
    n_rows: int, n_time: int = 60, n_doses: int = 5, tau: float = 12.0, seed: int = 0
) -> Timecourses:
    """A batch of multiple dose curves, the superposition of `n_doses` iv boluses."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.25, tau * n_doses, n_time)
    v = rng.uniform(20.0, 60.0, size=n_rows)[:, None]
    k = rng.uniform(0.05, 0.3, size=n_rows)[:, None]
    dose, times = 100.0, tau * np.arange(n_doses, dtype=float)
    values = np.zeros((n_rows, n_time))
    for start in times:
        dt = t[None, :] - start
        values += np.where(dt >= 0, dose / v * np.exp(-k * np.maximum(dt, 0.0)), 0.0)
    values = values * rng.normal(1.0, 0.05, (n_rows, n_time))
    return Timecourses.from_arrays(
        t,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        dose={
            "amount": np.broadcast_to(np.full(n_doses, dose), (n_rows, n_doses)).copy(),
            "time": np.broadcast_to(times, (n_rows, n_doses)).copy(),
            "unit": "mg",
        },
        route=Route.IV_BOLUS,
        substance="drug",
    )


def case_nca_small() -> Case:
    """The analysis of a small batch, repeated: the fixed cost of a call."""
    batch = mono_batch(100)
    options = NCAOptions(uncertainty=UncertaintyMethod.NONE)
    return Case(
        "20 x (100 curves x 12 points)",
        lambda: [nca(batch, options=options) for _ in range(20)],
    )


def case_nca_large() -> Case:
    """The analysis of a large batch: the vectorized core."""
    batch = mono_batch(100_000)
    options = NCAOptions(uncertainty=UncertaintyMethod.NONE)
    return Case("100 000 curves x 12 points", lambda: nca(batch, options=options))


def case_nca_multiple() -> Case:
    """The analysis of a multiple dose batch: the dosing intervals."""
    batch = multi_dose_batch(5_000)
    options = NCAOptions(uncertainty=UncertaintyMethod.NONE)
    return Case(
        "5 000 curves x 60 points x 5 doses", lambda: nca(batch, options=options)
    )


def case_bootstrap() -> Case:
    """The parametric bootstrap of group curves."""
    batch = mono_batch(100, spread=True)
    options = NCAOptions(
        uncertainty=UncertaintyMethod.BOOTSTRAP, n_boot=1_000, seed=1234
    )
    return Case(
        "100 group curves x 1 000 replicates", lambda: nca(batch, options=options)
    )


def case_delta() -> Case:
    """The delta method on group curves."""
    batch = mono_batch(1_000, spread=True)
    options = NCAOptions(uncertainty=UncertaintyMethod.DELTA)
    return Case("1 000 group curves x 12 points", lambda: nca(batch, options=options))


def case_fit() -> Case:
    """A batch fit of a mono-exponential model."""
    batch = mono_batch(500)
    options = FitOptions(n_starts=1)
    return Case(
        "500 curves x 12 points, MonoExp",
        lambda: fit_timecourses(MonoExp(), batch, options=options),
    )


def case_constructors() -> Case:
    """The construction of single timecourses, i.e. pydantic and the unit checks."""
    t = np.linspace(0.25, 24.0, 12)
    values = 100.0 / 40.0 * np.exp(-0.1 * t)
    dose = Dose(amount=100.0, unit="mg", route=Route.IV_BOLUS)
    return Case(
        "5 000 x Timecourse(12 points)",
        lambda: [
            Timecourse(
                time=t,
                value=values,
                time_unit="hr",
                unit="mg/l",
                dose=dose,
                substance="drug",
            )
            for _ in range(5_000)
        ],
    )


def case_iterate() -> Case:
    """The iteration over the curves of a batch."""
    batch = mono_batch(5_000)
    return Case("5 000 curves of a batch", lambda: [tc.size for tc in batch])


def case_mean_ragged() -> Case:
    """Aggregation when every sample has distinct actual sampling times."""
    n_rows, n_time = 3_000, 20
    times = np.arange(n_time)[None, :] + np.arange(n_rows)[:, None] / n_rows
    batch = Timecourses.from_arrays(
        times,
        np.exp(-0.1 * times),
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
    )
    return Case("3 000 curves x 20 distinct times", lambda: batch.mean("individual"))


def case_nested_samples() -> Case:
    """Construction of observed pairs instead of a Cartesian grid."""
    frame = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(1_000), 2),
            "group": np.repeat(np.arange(1_000), 2),
            "time": np.tile([0.0, 1.0], 1_000),
            "value": np.tile([2.0, 1.0], 1_000),
        }
    )
    return Case(
        "1 000 observed subject/group pairs",
        lambda: Timecourses.from_dataframe(
            frame, sample=["subject", "group"], time_unit="hr", unit="mg/l"
        ),
    )


def case_delta_long() -> Case:
    """Delta uncertainty on long curves, including perturbation preparation."""
    batch = mono_batch(40, n_time=256, spread=True)
    options = NCAOptions(uncertainty=UncertaintyMethod.DELTA, chunk_rows=128)
    return Case("40 group curves x 256 points", lambda: nca(batch, options=options))


def case_worker_churn() -> Case:
    """Repeated worker-size changes without retaining every idle pool."""

    def run() -> None:
        for size in range(1, 9):
            with worker_pool("thread", size) as pool:
                list(pool.map(abs, [-1] * size))

    return Case("8 successive worker counts", run)


CASES: dict[str, Callable[[], Case]] = {
    "nca-small": case_nca_small,
    "nca-large": case_nca_large,
    "nca-multiple": case_nca_multiple,
    "bootstrap": case_bootstrap,
    "delta": case_delta,
    "fit": case_fit,
    "constructors": case_constructors,
    "iterate": case_iterate,
    "mean-ragged": case_mean_ragged,
    "nested-samples": case_nested_samples,
    "delta-long": case_delta_long,
    "worker-churn": case_worker_churn,
}


# --- the two engines of the analysis ------------------------------------------


@dataclass(frozen=True)
class EngineCase:
    """One analysis measured through both engines: the batch and its options.

    The batch and the options are fields and not a closure, because
    `measure_engines` asks the dispatch of `pkpdutils.nca` about them before it
    times anything: a case whose kernel half quietly ran the python core would
    otherwise be published as a ratio of 1.0.
    """

    size: str
    timecourses: Timecourses
    options: NCAOptions

    def run(self) -> NCAResult:
        """Analyse the batch through whichever engine the dispatch picks."""
        return nca(self.timecourses, options=self.options)


def engine_size(n_rows: int, n_time: int) -> str:
    """The size label of an engine case, in the thousands style of the table."""
    curves = "curve" if n_rows == 1 else "curves"
    return f"{n_rows:,} {curves} x {n_time} points".replace(",", " ")


def engine_case(n_rows: int, n_time: int) -> EngineCase:
    """The single dose analysis of `n_rows` curves of `n_time` points.

    The analysis both engines carry, with the defaults a caller gets: no
    uncertainty to propagate, no partial area, no acceptance criterion, one
    dose per row. What the kernel does not carry is `refusal_rows`.

    Args:
        n_rows: number of curves of the batch
        n_time: number of time points per curve

    Returns:
        The case.
    """
    return EngineCase(
        engine_size(n_rows, n_time),
        mono_batch(n_rows, n_time),
        NCAOptions(uncertainty=UncertaintyMethod.NONE),
    )


#: the shapes of the engine comparison, `(n_rows, n_time)` per case, in the
#: order of the table: the four of the kernel's own benchmark (one curve, 1 000
#: by 200, 10 000 by 200, 100 000 by 12), so the two tables can be read against
#: each other, and the ones which locate the crossover, sparse and dense, since
#: the work of an analysis is its points and not its rows
ENGINE_SHAPES: dict[str, tuple[int, int]] = {
    "engine-1x12": (1, 12),
    "engine-100x12": (100, 12),
    "engine-10x200": (10, 200),
    "engine-250x12": (250, 12),
    "engine-1000x12": (1_000, 12),
    "engine-100x200": (100, 200),
    "engine-10000x12": (10_000, 12),
    "engine-1000x200": (1_000, 200),
    "engine-100000x12": (100_000, 12),
    "engine-10000x200": (10_000, 200),
}

ENGINE_CASES: dict[str, Callable[[], EngineCase]] = {
    name: functools.partial(engine_case, n_rows, n_time)
    for name, (n_rows, n_time) in ENGINE_SHAPES.items()
}


@contextlib.contextmanager
def python_path(forced: bool) -> Generator[None]:
    """Run the body with the python path of `nca` forced, or with the dispatch.

    `pkpdutils.nca.engine._force_python` is the switch the package documents
    for its tests and for this benchmark: while it is set, the dispatch refuses
    every batch and the python core runs it, which is how the two engines are
    measured in one process.

    Args:
        forced: whether the python core is to run the batch.

    Yields:
        Nothing; the body runs inside the `yield`.
    """
    previous = engine._force_python
    engine._force_python = forced
    try:
        yield
    finally:
        engine._force_python = previous


def check_dispatch(name: str, case: EngineCase) -> None:
    """Refuse to measure a case whose two halves are not the two engines.

    Four questions, every one of them a way for this table to be a plausible
    table of nonsense rather than a failure, and none of them answerable from
    the numbers afterwards:

    - **does the dispatch send this batch to the kernel at all**
      (`pkpdutils.nca.engine.unsupported_reason`), or is the kernel column the
      python core under another name
    - **is the forced python path really forced**, or is the python column the
      kernel twice
    - **is the differential gate armed** (`engine.comparing()`,
      `PKPDUTILS_COMPARE_ENGINES`)? While it is, every batch which reaches the
      kernel is analysed by the python core as well and compared, so the kernel
      column would carry both engines and the ratio would be about half of the
      truth, with nothing in the table to show it. Continuous integration has a
      job which sets that variable, so an inherited environment is a real way
      to get here and not a thought experiment.
    - **does the kernel accept the batch when it is really called**? A
      refusal of `run_kernel` is answered by `nca` with a warning and the
      python path (`pkpdutils.nca.nca.FALLBACK_WARNING`), so the kernel column
      would time a refused call plus the python core. The predicate cannot
      answer this one, which is the whole reason the kernel is the authority
      and the predicate a mirror of it, so the kernel is called here once.

    Args:
        name: name of the case
        case: the case.

    Raises:
        RuntimeError: if the kernel does not carry the analysis, if the forced
            python path is not forced, if the differential gate is armed or if
            the kernel refuses the batch the predicate let through.
    """
    if engine.comparing():
        raise RuntimeError(
            f"the differential gate is armed ({engine.COMPARE_VARIABLE} is "
            "set), so every batch of the kernel half would be analysed by the "
            "python core as well and the ratio of the case "
            f"'{name}' would be about half of the truth; unset the variable "
            "and measure again"
        )
    reason = engine.unsupported_reason(case.timecourses, case.options)
    if reason is not None:
        raise RuntimeError(f"the kernel does not carry the case '{name}': {reason}")
    with python_path(True):
        forced = engine.unsupported_reason(case.timecourses, case.options)
    if forced != engine.FORCED_REASON:
        raise RuntimeError(
            f"the python path of the case '{name}' is not forced: the dispatch "
            f"answers {forced!r} instead of the forced reason"
        )
    try:
        engine.run_kernel(case.timecourses, case.options)
    except engine.UnsupportedError as refusal:
        raise RuntimeError(
            f"the kernel refuses the case '{name}' although the predicate let "
            "it through, so its kernel half would time a refused call and the "
            f"python core behind it: {refusal}"
        ) from refusal


def refusal_rows() -> list[tuple[str, str]]:
    """What the dispatch sends down the python path, with the reason it gives.

    One row per analysis the kernel of this installation does not carry, in the
    order `pkpdutils.nca.engine.unsupported_reason` tests them. The reason is
    read from the dispatch and never written here, so the table of the page is
    the kernel's own sentence and cannot drift from it.

    Returns:
        The analysis and the reason, one pair per row.

    Raises:
        RuntimeError: if the dispatch carries an analysis this table claims it
            refuses, which would make the row of the page a wrong claim.
    """
    batch = mono_batch(2)
    group = mono_batch(2, spread=True)
    cases: list[tuple[str, Timecourses, NCAOptions]] = [
        ("an effect curve", batch, NCAOptions(kind=Kind.EFFECT)),
        (
            "the acceptance criteria of the terminal phase",
            batch,
            NCAOptions(acceptance=Acceptance(span_min=2.0)),
        ),
        (
            "a named partial area",
            batch,
            NCAOptions(partial_aucs={"auc_0_4": (0.0, 4.0)}),
        ),
        ("the uncertainty of a group curve", group, NCAOptions()),
        (
            "the reporting units of a result",
            batch,
            NCAOptions(units={"auc_last": "mg * min / l"}),
        ),
    ]
    rows: list[tuple[str, str]] = []
    for what, timecourses, options in cases:
        reason = engine.unsupported_reason(timecourses, options)
        if reason is None:
            raise RuntimeError(
                f"the kernel carries '{what}', which this table reports as an "
                "analysis of the python core"
            )
        rows.append((what, reason))
    return rows


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB."""
    # `resource` is unix only and a measurement needs it; the staleness check
    # of `docs/benchmark.md` does not, and it runs on every platform of the
    # test matrix (`tests/docs/test_benchmark.py`), so this module stays
    # importable where the module is not
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure(name: str, repeat: int) -> dict[str, Any]:
    """Build the data of a case and time its operation `repeat` times.

    Args:
        name: name of the case.
        repeat: number of timed runs after one warm-up run.

    Returns:
        The name, the size label, the median time in seconds and the peak RSS.
    """
    case = CASES[name]()
    first_start = time.perf_counter()
    case.run()
    first_call_s = time.perf_counter() - first_start
    times: list[float] = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        case.run()
        times.append(time.perf_counter() - start)
    return {
        "case": name,
        "size": case.size,
        "median_s": statistics.median(times),
        "first_call_s": first_call_s,
        "peak_rss_mb": peak_rss_mb(),
    }


def measure_engines(name: str, repeat: int) -> dict[str, Any]:
    """Time one analysis through both engines, alternating in one process.

    The batch is built once and analysed `repeat` times by each engine, kernel
    and python run after run, so that both halves are measured in the same
    window: a ratio of two medians taken one after the other is an artefact of
    whatever else the machine did in between.

    Args:
        name: name of the engine case.
        repeat: number of timed runs per engine, after one warm-up run each.

    Returns:
        The name, the size label, the median time of each engine in seconds,
        their ratio and the peak RSS of the process, which ran both.
    """
    case = ENGINE_CASES[name]()
    check_dispatch(name, case)
    times: dict[str, list[float]] = {"kernel": [], "python": []}
    halves = (("kernel", False), ("python", True))
    for forced in (False, True):
        # one warm-up run per engine, so that no timed run pays the first touch
        # of a cache either of them keeps
        with python_path(forced):
            case.run()
    for _ in range(repeat):
        for half, forced in halves:
            with python_path(forced):
                gc.collect()
                start = time.perf_counter()
                case.run()
                times[half].append(time.perf_counter() - start)
    kernel_s = statistics.median(times["kernel"])
    python_s = statistics.median(times["python"])
    return {
        "case": name,
        "size": case.size,
        "kernel_s": kernel_s,
        "python_s": python_s,
        "speedup": python_s / kernel_s,
        "peak_rss_mb": peak_rss_mb(),
    }


def run_isolated(name: str, repeat: int) -> dict[str, Any]:
    """Run one case in a fresh interpreter and read its result.

    Args:
        name: name of the case.
        repeat: number of timed runs.

    Returns:
        The row of the case.

    Raises:
        RuntimeError: if the child process fails or writes no result.
    """
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name, "--repeat", str(repeat)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, CHILD_ENV: "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(f"case '{name}' failed:\n{completed.stderr}")
    for line in completed.stdout.splitlines():
        if line.startswith("{"):
            row: dict[str, Any] = json.loads(line)
            return row
    raise RuntimeError(f"case '{name}' wrote no result:\n{completed.stdout}")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """The rows as a markdown table.

    Args:
        rows: the results of the cases.

    Returns:
        The table.
    """
    header = "| case | size | first call s | median s | peak RSS MB |"
    lines = [header, "|---|---|---|---|---|"]
    lines.extend(
        f"| {row['case']} | {row['size']} | {row['first_call_s']:.3f} | "
        f"{row['median_s']:.3f} | "
        f"{row['peak_rss_mb']:.0f} |"
        for row in rows
    )
    return "\n".join(lines)


def engine_table(rows: list[dict[str, Any]]) -> str:
    """The comparison of the two engines as a markdown table.

    The number of points of a row is read from `ENGINE_SHAPES` rather than
    measured, so that the table of a run and the table the staleness check
    builds carry it the same way.

    Args:
        rows: the results of the engine cases.

    Returns:
        The table.
    """
    header = "| case | size | points | kernel ms | python ms | speedup | peak RSS MB |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for row in rows:
        n_rows, n_time = ENGINE_SHAPES[row["case"]]
        points = f"{n_rows * n_time:,}".replace(",", " ")
        lines.append(
            f"| {row['case']} | {row['size']} | {points} | "
            f"{1e3 * row['kernel_s']:.2f} | {1e3 * row['python_s']:.2f} | "
            f"{row['speedup']:.2f}x | {row['peak_rss_mb']:.0f} |"
        )
    return "\n".join(lines)


def engine_points(row: dict[str, Any]) -> int:
    """The work of an engine case: its curves times its points per curve.

    Args:
        row: the result of an engine case.

    Returns:
        The number of points of its batch.
    """
    n_rows, n_time = ENGINE_SHAPES[row["case"]]
    return n_rows * n_time


def summary_sentences(rows: list[dict[str, Any]]) -> str:
    """The load-bearing figures of the table, as sentences of the page.

    The prose of `docs/benchmark.md` is written around these rather than
    repeating them, because a figure a reader copies into a paragraph is the
    one staleness the structural check of `page_complaints` cannot see: it
    maps every measured cell to `#` on purpose, so a `--write` on another
    machine moves the table and leaves a hand-copied number behind. These
    sentences are computed from the very rows the table is formatted from, so
    the two cannot disagree.

    The figures: the largest and the smallest ratio of the run, the two shapes
    the crossover lies between, the largest amount the kernel path costs below
    it, the range the ratios below it span, the two shapes of nearly equal work
    whose ratios differ most (`shape_pair`) and the shape the clear wins start
    at (`CLEAR_WIN`, far enough from 1 to survive the few percent a rerun moves
    a ratio by).

    The last two are here because the prose of the page claimed them by hand
    twice and was falsified by a `--write` twice: "within a few percent of each
    other up to the crossover" and "the same few percent apart" on two shapes
    of nearly the same work are both figures of a run, and a figure of a run
    belongs in this region or nowhere.

    The two shapes and the clear win are read by walking the sweep in the order
    of the work (`crossover`, `clear_win`) and not off the two sorted halves of
    it, because this sweep is deliberately not monotone in its ratio: the row
    above `pkpdutils.parallel.NCA_WORKER_THRESHOLD` holds two parallel analyses
    against each other where the rows below it hold two serial ones, and a
    machine of more cores can put it below 1.00x between two rows which are
    well above it. `below[-1]` and `above[0]` would then be the largest and the
    smallest shape of the whole run, and the sentence would name a crossover of
    1 200 000 and 3 000 points with a cost of some 50 ms which is no cost of a
    crossover at all. A sweep which crosses 1.00x more than once names no
    single crossover, and the sentences say that rather than inventing one.

    Args:
        rows: the results of the engine cases, in the order of the table.

    Returns:
        The sentences, as one line of markdown.
    """
    ordered = sorted(rows, key=engine_points)
    best = max(ordered, key=lambda row: row["speedup"])
    worst = min(ordered, key=lambda row: row["speedup"])
    sentences = [
        f"The largest ratio of this run is {best['speedup']:.2f}x, at "
        f"{best['size']}, and the smallest {worst['speedup']:.2f}x, at "
        f"{worst['size']}."
    ]
    below = [row for row in ordered if row["speedup"] < 1.0]
    above = [row for row in ordered if row["speedup"] >= 1.0]
    crossed = crossover(ordered)
    if not below:
        sentences.append(
            "Every shape of this run is at or above 1.00x, so the crossover "
            "is below the smallest one measured."
        )
    elif not above:
        sentences.append(
            "Every shape of this run is below 1.00x, so the crossover is "
            "above the largest one measured."
        )
    elif crossed is None:
        changes = sign_changes(ordered)
        if len(changes) == 1:
            left, right = changes[0]
            crossing = (
                f"The ratio of this run crosses 1.00x downwards, from "
                f"{left['size']} at {left['speedup']:.2f}x to {right['size']} "
                f"at {right['speedup']:.2f}x, so the sweep names no crossover"
            )
        else:
            crossing = (
                f"The ratio of this run crosses 1.00x at {len(changes)} shapes "
                "of the sweep rather than at one, so it names no single "
                "crossover"
            )
        sentences.append(
            crossing
            + "; the shapes below 1.00x are "
            + ", ".join(f"{row['size']} at {row['speedup']:.2f}x" for row in below)
            + ", and on those the kernel path costs at most "
            f"{1e3 * max(row['kernel_s'] - row['python_s'] for row in below):.2f} "
            "ms more per call."
        )
    else:
        last, first = crossed
        sentences.append(
            f"The two engines cost the same between {engine_points(last):,} "
            f"and {engine_points(first):,} points: {last['size']} is "
            f"{last['speedup']:.2f}x and {first['size']} is "
            f"{first['speedup']:.2f}x.".replace(",", " ")
        )
        up_to = [row for row in ordered if engine_points(row) <= engine_points(last)]
        cost = max(row["kernel_s"] - row["python_s"] for row in up_to)
        sentences.append(
            f"Below the crossover the kernel path costs at most "
            f"{1e3 * cost:.2f} ms more per call, and the ratios there run from "
            f"{min(row['speedup'] for row in up_to):.2f}x to "
            f"{max(row['speedup'] for row in up_to):.2f}x."
        )
    pair = shape_pair(ordered)
    if pair is not None:
        apart, together = pair
        sentences.append(
            f"Of two shapes of nearly equal work the ratios differ most "
            f"between {apart['size']} at {apart['speedup']:.2f}x and "
            f"{together['size']} at {together['speedup']:.2f}x."
        )
    win = clear_win(ordered)
    if win is not None:
        sentences.append(
            f"From {win['size']} on every shape of this run is a clear win, "
            f"the first of them at {win['speedup']:.2f}x."
        )
    return " ".join(sentences)


def sign_changes(
    ordered: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every adjacent pair of the sweep whose ratios lie on two sides of 1.00x.

    Args:
        ordered: the results of the engine cases, sorted by `engine_points`.

    Returns:
        The pairs, in the order of the work.
    """
    return [
        (left, right)
        for left, right in itertools.pairwise(ordered)
        if (left["speedup"] < 1.0) != (right["speedup"] < 1.0)
    ]


def crossover(
    ordered: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The two neighbouring shapes the crossover of the sweep lies between.

    The crossover of a sweep is a pair of neighbours in the order of the work
    whose ratios lie on two sides of 1.00x, and it is one only when the sweep
    has exactly one such pair and crosses upwards there: a sweep which crosses
    back has as many crossovers as it has sign changes and none of them is
    "the" crossover of the page.

    Args:
        ordered: the results of the engine cases, sorted by `engine_points`.

    Returns:
        The last shape below 1.00x and the first at or above it, or `None` for
        a sweep which names no single crossover.
    """
    changes = sign_changes(ordered)
    if len(changes) != 1:
        return None
    last, first = changes[0]
    return None if last["speedup"] >= 1.0 else (last, first)


#: how far the work of two shapes may differ for them to be the sparse and the
#: dense case of one size of the sweep (`shape_pair`)
NEARLY_EQUAL: float = 2.0


def shape_pair(
    ordered: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The two shapes of nearly equal work whose ratios differ most.

    The figure behind "what crosses over is the work and not the shape": the
    sweep carries a sparse and a dense case at every size, within
    `NEARLY_EQUAL` of each other in points, and this is the pair of them the
    claim holds least well for, so the page states how far it holds instead of
    asserting that it does.

    A pair which straddles `pkpdutils.parallel.NCA_WORKER_THRESHOLD` is left
    out, and the threshold is read from the library rather than written here:
    such a pair holds a batch both engines run in parallel against one both run
    serially, which is a figure about the worker pools and no figure about the
    shape of a batch.

    Args:
        ordered: the results of the engine cases, sorted by `engine_points`.

    Returns:
        The pair in the order of the work, or `None` when no two shapes of the
        sweep are near enough in work and on one side of the threshold.
    """
    pairs = [
        (left, right)
        for left, right in itertools.pairwise(ordered)
        if engine_points(right) <= NEARLY_EQUAL * engine_points(left)
        and (
            (ENGINE_SHAPES[left["case"]][0] < NCA_WORKER_THRESHOLD)
            == (ENGINE_SHAPES[right["case"]][0] < NCA_WORKER_THRESHOLD)
        )
    ]
    if not pairs:
        return None
    return max(pairs, key=lambda pair: abs(pair[0]["speedup"] - pair[1]["speedup"]))


def clear_win(ordered: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The smallest shape from which every larger one of the run is a clear win.

    Read from the large end of the sweep and not as the first row at or above
    `CLEAR_WIN`: the smallest shape above the ratio is the first clear win only
    while the sweep stays above it, and a run in which a larger shape falls
    back would publish the first of the two and say nothing of the second.

    Args:
        ordered: the results of the engine cases, sorted by `engine_points`.

    Returns:
        The smallest shape of the suffix every ratio of which is at or above
        `CLEAR_WIN`, or `None` when the largest shape of the run is below it.
    """
    win: dict[str, Any] | None = None
    for row in reversed(ordered):
        if row["speedup"] < CLEAR_WIN:
            break
        win = row
    return win


#: the ratio from which a win is clear of the few percent a rerun moves one by
CLEAR_WIN: float = 1.25


def refusal_table() -> str:
    """The analyses the kernel does not carry as a markdown table.

    Returns:
        The table, with the dispatch's own reason per row.
    """
    lines = ["| analysis | the reason the dispatch gives |", "|---|---|"]
    lines.extend(f"| {what} | {reason} |" for what, reason in refusal_rows())
    return "\n".join(lines)


def skeleton_rows() -> list[dict[str, Any]]:
    """One row per engine case, with its labels and without a measurement.

    The structure of the table is a function of the cases alone, so the check
    of the page builds it without analysing anything: a case added, removed,
    renamed or given another size label makes the page stale and a timing never
    does.

    Returns:
        The rows `engine_table` would format, every number 0.
    """
    return [
        {
            "case": name,
            "size": engine_size(n_rows, n_time),
            "kernel_s": 0.0,
            "python_s": 0.0,
            "speedup": 0.0,
            "peak_rss_mb": 0.0,
        }
        for name, (n_rows, n_time) in ENGINE_SHAPES.items()
    ]


def machine_line(repeat: int) -> str:
    """The sentence which names the machine and the day of a run.

    Args:
        repeat: number of timed runs per engine.

    Returns:
        The sentence, as one line of markdown.
    """
    # the wheel of `pkpdutils-core` carries no stub beside its compiled module,
    # so the one lookup of its version is suppressed by name
    kernel = pkpdutils_core.__version__  # ty: ignore[missing-attribute]
    return (
        f"Measured on {date.today().isoformat()}, python "
        f"{platform.python_version()} on {platform.system()} "
        f"{platform.machine()} with {os.cpu_count()} cores, `pkpdutils` "
        f"{pkpdutils.__version__} with `pkpdutils-core` {kernel}, {repeat} "
        "timed runs per engine and case after one warm-up run each."
    )


#: a measured cell of a table, which the staleness check does not compare
NUMBER = re.compile(r"^\d+(\.\d+)?(e[-+]?\d+)?x?$")

#: the region of `docs/benchmark.md` the comparison of the two engines is in
TIMINGS_START, TIMINGS_END = "<!-- timings:start -->", "<!-- timings:end -->"

#: the region of `docs/benchmark.md` the refusals of the kernel are in
REFUSALS_START, REFUSALS_END = "<!-- refusals:start -->", "<!-- refusals:end -->"


def table_structure(text: str) -> list[list[str]]:
    """The cells of every markdown table row of a text, measurements aside.

    A cell which is a number (or a ratio such as `4.87x`) is read as `#`, every
    other cell as it stands: the names of the cases, the size labels, the
    header and the number of columns are the structure of the table, the
    timings are not, since they differ between machines and between runs.

    Args:
        text: the markdown of a region of the page.

    Returns:
        One list of cells per table row.
    """
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(["#" if NUMBER.match(cell) else cell for cell in cells])
    return rows


def region(page: str, start: str, end: str) -> str:
    """The text of the page between two markers.

    Args:
        page: the markdown of the page
        start: the opening marker
        end: the closing marker.

    Returns:
        The text between them.

    Raises:
        ValueError: if the page holds no such region.
    """
    first, last = page.find(start), page.find(end)
    if first < 0 or last < first:
        raise ValueError(f"the page holds no region '{start}' ... '{end}'")
    return page[first + len(start) : last]


def replace_region(page: str, start: str, end: str, body: str) -> str:
    """The page with the text between two markers replaced.

    Args:
        page: the markdown of the page
        start: the opening marker
        end: the closing marker
        body: what goes between them.

    Returns:
        The page.

    Raises:
        ValueError: if the page holds no such region.
    """
    first, last = page.find(start), page.find(end)
    if first < 0 or last < first:
        raise ValueError(f"the page holds no region '{start}' ... '{end}'")
    inner = body.strip("\n")
    return f"{page[: first + len(start)]}\n{inner}\n{page[last:]}"


def render_page(page: str, rows: list[dict[str, Any]], repeat: int) -> str:
    """The page with the two generated tables of this run pasted into it.

    The timing region carries the machine, the table and the sentences of
    `summary_sentences`, which are the figures the prose of the page leans on,
    so that a rerun moves them with the table.

    Args:
        page: the markdown of the page
        rows: the results of the engine cases
        repeat: number of timed runs per engine, named on the page.

    Returns:
        The markdown of the page.
    """
    timings = (
        f"{machine_line(repeat)}\n\n{engine_table(rows)}\n\n{summary_sentences(rows)}"
    )
    page = replace_region(page, TIMINGS_START, TIMINGS_END, timings)
    return replace_region(page, REFUSALS_START, REFUSALS_END, refusal_table())


def page_complaints(page: str) -> list[str]:
    """Every way the page no longer matches what this script writes.

    The timings are not compared: they are the run the page names and they
    differ between machines, which is why `--write` is how they are refreshed.
    Compared are the structure of the timing table (`table_structure`: the
    header, the cases, their size labels, the number of columns and that every
    measured cell is a number) and the table of the refusals word for word,
    since that one is the dispatch's own text and no measurement.

    Args:
        page: the markdown of the page.

    Returns:
        One sentence per complaint, empty for a current page.
    """
    complaints: list[str] = []
    found = table_structure(region(page, TIMINGS_START, TIMINGS_END))
    expected = table_structure(engine_table(skeleton_rows()))
    if found != expected:
        complaints.append(
            "the timing table of the page is not the table of the cases of "
            f"the script: {first_difference(expected, found)}"
        )
    refusals = region(page, REFUSALS_START, REFUSALS_END).strip()
    if refusals != refusal_table():
        complaints.append(
            "the refusals of the page are not the ones the dispatch gives: "
            + first_difference(
                table_structure(refusal_table()), table_structure(refusals)
            )
        )
    return complaints


def first_difference(expected: list[list[str]], found: list[list[str]]) -> str:
    """The first row in which two tables differ, both sides named.

    Args:
        expected: the rows the script writes
        found: the rows the page holds.

    Returns:
        The sentence.
    """
    for number, (left, right) in enumerate(zip(expected, found, strict=False)):
        if left != right:
            return f"row {number} is {right} and should be {left}"
    if len(expected) != len(found):
        return (
            f"the page holds {len(found)} row(s) and the script writes {len(expected)}"
        )
    return "nothing, the two agree"


def main() -> int:
    """Parse the arguments and run the requested cases.

    Returns:
        The exit status, 1 for a stale page under `--check`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cases",
        nargs="*",
        default=["all"],
        choices=["all", "engines", *CASES, *ENGINE_CASES],
        help=(
            "cases to run, 'all' for every one of them (the default), "
            "'engines' for the comparison of the two engines"
        ),
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="timed runs per case (default 3)"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"run the engine cases and rewrite {PAGE_PATH.name}",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=f"exit 1 if {PAGE_PATH.name} no longer matches the cases",
    )
    args = parser.parse_args()

    if args.check:
        complaints = page_complaints(PAGE_PATH.read_text(encoding="utf-8"))
        for complaint in complaints:
            print(complaint)
        if complaints:
            print(f"{PAGE_PATH} is not current, run {Path(__file__).name} --write")
            return 1
        return 0

    names = case_names(args)
    if len(names) == 1 and os.environ.get(CHILD_ENV) == "1":
        # one case in a fresh interpreter: report the row as json
        name = names[0]
        row = (
            measure_engines(name, args.repeat)
            if name in ENGINE_CASES
            else measure(name, args.repeat)
        )
        print(json.dumps(row), flush=True)
        return 0

    started = time.perf_counter()
    rows = [run_isolated(name, args.repeat) for name in names]
    hot = [row for row in rows if row["case"] in CASES]
    engines = [row for row in rows if row["case"] in ENGINE_CASES]
    if hot:
        print(markdown_table(hot))
        print()
    if engines:
        print(engine_table(engines))
        print()
    if args.write:
        page = render_page(PAGE_PATH.read_text(encoding="utf-8"), engines, args.repeat)
        PAGE_PATH.write_text(page, encoding="utf-8")
        print(f"wrote {PAGE_PATH}")
    print(
        f"{len(names)} cases, {args.repeat} repeats, "
        f"{time.perf_counter() - started:.1f} s total"
    )
    return 0


def case_names(args: argparse.Namespace) -> list[str]:
    """The cases to run, in the order of the tables.

    `--write` runs the engine cases whatever was asked for, since the page it
    writes is their table.

    Args:
        args: the parsed arguments.

    Returns:
        The names of the cases.
    """
    if args.write:
        return list(ENGINE_CASES)
    if "all" in args.cases:
        return [*CASES, *ENGINE_CASES]
    names: list[str] = []
    for case in args.cases:
        names.extend(ENGINE_CASES if case == "engines" else [case])
    return list(dict.fromkeys(names))


if __name__ == "__main__":
    raise SystemExit(main())

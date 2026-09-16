"""Benchmark of the hot paths of `pkpdutils`.

Every case builds its data first and then runs one operation `--repeat` times;
the table reports the median wall time and the peak resident set size of the
process the case ran in. Each case runs in a fresh interpreter, so its memory
is its own and its caches (the unit caches of `pkpdutils.units`) start empty,
as they do in a script a user runs.

```bash
uv run python scripts/benchmark.py all
uv run python scripts/benchmark.py nca-large bootstrap --repeat 5
```

The absolute numbers are machine specific and only comparable against another
run of the same script on the same machine; the point of the table is the
before and after of a change.
"""

import argparse
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pkpdutils import (
    Dose,
    NCAOptions,
    Route,
    Timecourse,
    Timecourses,
    fit_timecourses,
    nca,
)
from pkpdutils.fit.models import MonoExp
from pkpdutils.fit.options import FitOptions
from pkpdutils.nca.options import UncertaintyMethod

#: set in the environment of the child process which runs one case
CHILD_ENV = "PKPDUTILS_BENCHMARK_CHILD"


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


CASES: dict[str, Callable[[], Case]] = {
    "nca-small": case_nca_small,
    "nca-large": case_nca_large,
    "nca-multiple": case_nca_multiple,
    "bootstrap": case_bootstrap,
    "delta": case_delta,
    "fit": case_fit,
    "constructors": case_constructors,
    "iterate": case_iterate,
}


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB."""
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
    case.run()
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
    header = "| case | size | median s | peak RSS MB |"
    lines = [header, "|---|---|---|---|"]
    lines.extend(
        f"| {row['case']} | {row['size']} | {row['median_s']:.3f} | "
        f"{row['peak_rss_mb']:.0f} |"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    """Parse the arguments and run the requested cases."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cases",
        nargs="*",
        default=["all"],
        choices=["all", *CASES],
        help="cases to run, 'all' for every one of them (the default)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="timed runs per case (default 3)"
    )
    args = parser.parse_args()
    names = list(CASES) if "all" in args.cases else list(dict.fromkeys(args.cases))

    if len(names) == 1 and os.environ.get(CHILD_ENV) == "1":
        # one case in a fresh interpreter: report the row as json
        print(json.dumps(measure(names[0], args.repeat)), flush=True)
        return
    started = time.perf_counter()
    rows = [run_isolated(name, args.repeat) for name in names]
    print(markdown_table(rows))
    print(
        f"\n{len(names)} cases, {args.repeat} repeats, "
        f"{time.perf_counter() - started:.1f} s total"
    )


if __name__ == "__main__":
    main()

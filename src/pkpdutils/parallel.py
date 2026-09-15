"""The shared worker pools of the analyses.

The non-compartmental analysis and the fit both spread their rows over
workers, and both used to create a fresh `concurrent.futures.Executor` per
call. With python's `forkserver` and `spawn` start methods (the default on
macOS and Windows, and on Linux from python 3.14) every worker of a new
process pool imports `pkpdutils`, `numpy`, `scipy`, `xarray` and `pint` from
scratch, about 0.7 s per pool, so a one-shot analysis was slower with workers
than without them. This module therefore keeps one executor per kind and
size, created on first use and closed at interpreter exit, so that the
start-up is paid once per process instead of once per call.

The two analyses use different kinds of workers:

- the NCA core is vectorized numpy over a chunk of rows and releases the GIL
  for most of its time, so its chunks run in **threads**: no pickling, no
  copy of the batch, and a pool that starts in half a millisecond;
- a fit row is a python-heavy `scipy.optimize.least_squares` search, so the
  rows run in **processes**, which is where the GIL is actually escaped.

`resolve_workers` turns `n_workers` into a worker count: `None` is automatic
and stays serial below a row threshold, `1` is serial and any other number is
taken as given. `split_rows` cuts the rows into about one contiguous slice
per worker, bounded from below so that a worker gets enough work to pay for
itself and from above so that the memory of the vectorized core stays bounded
(`NCAOptions.chunk_rows`).
"""

import atexit
import logging
import os
import threading
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal

logger = logging.getLogger(__name__)

#: the kinds of executor this module hands out
ExecutorKind = Literal["thread", "process"]

#: the live executors, keyed by kind and number of workers
_EXECUTORS: dict[tuple[str, int], Executor] = {}

#: guards `_EXECUTORS`; the NCA runs its chunks in threads, so two threads can
#: ask for an executor at the same time
_LOCK = threading.Lock()


def executor(kind: ExecutorKind, n_workers: int) -> Executor:
    """The shared executor of a kind and size, created on first use.

    The executor is cached and reused for the life of the process and closed
    by an `atexit` handler (`shutdown_executors`), so the start-up of a
    process pool is paid once and not once per call. A process executor uses
    the default start method of the platform; with `forkserver` or `spawn`
    the caller must run under an `if __name__ == "__main__":` guard.

    Args:
        kind: `"thread"` for a `ThreadPoolExecutor`, `"process"` for a
            `ProcessPoolExecutor`
        n_workers: number of workers, at least 1

    Returns:
        The executor; two calls with the same kind and size return the same
        object.
    """
    size = max(1, int(n_workers))
    key = (str(kind), size)
    with _LOCK:
        pool = _EXECUTORS.get(key)
        if pool is None:
            pool = (
                ThreadPoolExecutor(max_workers=size, thread_name_prefix="pkpdutils")
                if kind == "thread"
                else ProcessPoolExecutor(max_workers=size)
            )
            logger.debug("created the shared %s executor of %d workers", kind, size)
            _EXECUTORS[key] = pool
        return pool


def shutdown_executors() -> None:
    """Close every shared executor and forget it.

    Registered with `atexit`, so a script does not have to close the pools it
    used; a later call to `executor` creates a new one.
    """
    with _LOCK:
        pools = list(_EXECUTORS.values())
        _EXECUTORS.clear()
    for pool in pools:
        pool.shutdown(wait=True)


def _forget_executors() -> None:
    """Drop the executors inherited by a forked child process.

    The worker threads and the worker processes of an executor do not survive
    `fork`: the child inherits the objects but none of the workers behind
    them, so using one would block forever. The child therefore starts with an
    empty cache and creates its own executor when it needs one. The lock is
    replaced as well, since it may have been held by another thread of the
    parent at the moment of the fork.
    """
    global _LOCK
    _LOCK = threading.Lock()
    _EXECUTORS.clear()


atexit.register(shutdown_executors)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_executors)


def resolve_workers(
    n_workers: int | None,
    n_rows: int,
    *,
    threshold: int = 20_000,
    max_workers: int = 8,
) -> int:
    """The number of workers of a run over `n_rows` rows.

    `None` is the automatic default: a run below `threshold` rows is serial,
    since the pool costs more than it saves, and a larger one uses one worker
    per core up to `max_workers` (the scaling of the shared-memory core
    flattens there). An explicit `n_workers` is taken as given, `1` being the
    serial run.

    Args:
        n_workers: the option, `None` for automatic
        n_rows: number of rows of the run

    Keyword Args:
        threshold: rows from which the automatic default uses workers
        max_workers: upper bound of the automatic worker count

    Returns:
        The number of workers, 1 for a serial run.
    """
    if n_workers is not None:
        return max(1, int(n_workers))
    if n_rows < threshold:
        return 1
    return max(1, min(os.cpu_count() or 1, max_workers))


def split_rows(
    n_rows: int,
    n_workers: int,
    *,
    min_rows: int = 1_000,
    max_rows: int | None = None,
) -> list[slice]:
    """Cut `n_rows` rows into contiguous slices, about one per worker.

    The slices are contiguous and cover every row in order, so a chunk of an
    array is a view and not a copy. There are about `n_workers` of them: never
    more than one per `min_rows` rows, so that a worker gets enough work to
    pay for its share of the overhead (a batch below `min_rows` rows stays one
    slice), and never a slice longer than `max_rows`, the bound on the memory
    of the vectorized core, which can force more slices than there are
    workers.

    Args:
        n_rows: number of rows, 0 or more
        n_workers: number of workers, 1 or more

    Keyword Args:
        min_rows: fewest rows a slice carries while there is more than one
        max_rows: most rows a slice carries, `None` for no bound

    Returns:
        The slices in row order; empty for `n_rows = 0`.
    """
    if n_rows <= 0:
        return []
    count = min(max(1, int(n_workers)), max(1, n_rows // max(1, int(min_rows))))
    if max_rows is not None:
        count = max(count, -(-n_rows // max(1, int(max_rows))))
    count = min(count, n_rows)
    base, extra = divmod(n_rows, count)
    slices: list[slice] = []
    start = 0
    for index in range(count):
        stop = start + base + (1 if index < extra else 0)
        slices.append(slice(start, stop))
        start = stop
    return slices

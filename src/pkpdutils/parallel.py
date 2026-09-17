"""The shared worker pools of the analyses.

The non-compartmental analysis and the fit both spread their rows over
workers, and both used to create a fresh `concurrent.futures.Executor` per
call. A process pool starts its workers with `forkserver` or `spawn`
(`PROCESS_START_METHOD`), so every worker of a new pool imports `pkpdutils`,
`numpy`, `scipy`, `xarray` and `pint` from scratch, about 0.7 s per pool, and
a one-shot analysis was slower with workers than without them. This module
therefore keeps one executor per kind and size, created on first use and
closed at interpreter exit, so that the start-up is paid once per process
instead of once per call.

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

The two pools live side by side in one process, so the process pool never
forks. With the `fork` start method, the default of python 3.13 on Linux, a
worker would be forked from a parent whose NCA threads may hold a lock at
that moment, and the child would inherit the locked lock and could block
forever; python 3.13 warns about it (`DeprecationWarning: This process is
multi-threaded, use of fork() may lead to deadlocks in the child`), and python
3.14 no longer forks by default. The process pool therefore starts its
workers with `PROCESS_START_METHOD` on every python version, the default of
python 3.14: `forkserver` on Linux and the other POSIX platforms which offer
it, `spawn` on macOS and Windows, whatever `multiprocessing.set_start_method`
chose for the rest of the program. Both start a fresh interpreter which
imports the main module without running it, so a pooled call needs an
`if __name__ == "__main__":` guard, and what it sends to the workers (a model
of the fit) must be importable, not defined in an interactive session.
"""

import atexit
import logging
import multiprocessing
import os
import sys
import threading
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal

logger = logging.getLogger(__name__)

#: the kinds of executor this module hands out
ExecutorKind = Literal["thread", "process"]

#: the start methods the process pool uses, the ones which do not fork the caller
StartMethod = Literal["forkserver", "spawn"]


def _process_start_method() -> StartMethod:
    """The start method of the workers of the process pool.

    The default start method of python 3.14 on the platform, which is never
    `fork`: `spawn` on macOS, whose system libraries may start threads of
    their own, and wherever `forkserver` is not offered (Windows), and
    `forkserver` on the other platforms.

    Returns:
        `"forkserver"` or `"spawn"`.
    """
    if sys.platform == "darwin":
        return "spawn"
    if "forkserver" in multiprocessing.get_all_start_methods():
        return "forkserver"
    return "spawn"


#: start method of the workers of the process pool on every python version,
#: the default of python 3.14 (`forkserver` on Linux, `spawn` on macOS and
#: Windows) instead of the `fork` of python 3.13 on Linux, which may deadlock
#: next to the threads of the NCA
PROCESS_START_METHOD: StartMethod = _process_start_method()

#: rows from which the automatic `n_workers` of the NCA uses the thread pool,
#: the measured break-even of the vectorized core against the pool
NCA_WORKER_THRESHOLD = 20_000

#: the live executors, keyed by kind and number of workers
_EXECUTORS: dict[tuple[str, int], Executor] = {}

#: guards `_EXECUTORS`; the NCA runs its chunks in threads, so two threads can
#: ask for an executor at the same time
_LOCK = threading.Lock()


def _is_dead(pool: Executor) -> bool:
    """Whether an executor can no longer run work.

    A `ProcessPoolExecutor` whose worker died (`BrokenProcessPool`) and a
    `ThreadPoolExecutor` whose initializer raised are marked broken and reject
    every later submission; a pool that was shut down does the same. A cached
    pool in that state is replaced rather than handed out again.

    Args:
        pool: the executor.

    Returns:
        Whether it is broken or shut down.
    """
    return bool(
        getattr(pool, "_broken", False)
        or getattr(pool, "_shutdown", False)
        or getattr(pool, "_shutdown_thread", False)
    )


def executor(kind: ExecutorKind, n_workers: int) -> Executor:
    """The shared executor of a kind and size, created on first use.

    The executor is cached and reused for the life of the process and closed
    by an `atexit` handler (`shutdown_executors`), so the start-up of a
    process pool is paid once and not once per call. A cached pool that is
    broken or shut down is dropped and replaced, so that one dead worker does
    not fail every later call of the process. A process executor starts its
    workers with `PROCESS_START_METHOD` (`forkserver` or `spawn`, never
    `fork`, whatever the default of the platform or
    `multiprocessing.set_start_method` says), so the caller must run under an
    `if __name__ == "__main__":` guard.

    The pools are not re-entrant: work running in a worker of a pool must not
    submit to that same pool and wait for the result, which deadlocks once
    every worker waits (calling `pkpdutils.nca.nca` from a chunk of an NCA
    that is already running in the shared thread pool, for instance). The
    analyses of the package never do.

    Args:
        kind: `"thread"` for a `ThreadPoolExecutor`, `"process"` for a
            `ProcessPoolExecutor`
        n_workers: number of workers, at least 1

    Returns:
        The executor; two calls with the same kind and size return the same
        object while it is usable.
    """
    size = max(1, int(n_workers))
    key = (str(kind), size)
    with _LOCK:
        pool = _EXECUTORS.pop(key, None)
        if pool is not None and _is_dead(pool):
            logger.debug("the shared %s executor of %d workers died", kind, size)
            pool.shutdown(wait=False)
            pool = None
        if pool is None:
            pool = (
                ThreadPoolExecutor(max_workers=size, thread_name_prefix="pkpdutils")
                if kind == "thread"
                else ProcessPoolExecutor(
                    max_workers=size,
                    mp_context=multiprocessing.get_context(PROCESS_START_METHOD),
                )
            )
            logger.debug("created the shared %s executor of %d workers", kind, size)
        _EXECUTORS[key] = pool
        return pool


def evict(kind: ExecutorKind, n_workers: int) -> None:
    """Drop the shared executor of a kind and size and shut it down.

    The caller of a pool that failed (a worker process that died takes the
    whole `ProcessPoolExecutor` with it) evicts it before it retries: the next
    `executor` call then builds a fresh pool. Evicting an executor that is not
    cached does nothing.

    Args:
        kind: the kind of the executor.
        n_workers: the number of workers it was created with.
    """
    size = max(1, int(n_workers))
    with _LOCK:
        pool = _EXECUTORS.pop((str(kind), size), None)
    if pool is not None:
        pool.shutdown(wait=False)


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
    threshold: int = NCA_WORKER_THRESHOLD,
    max_workers: int = 8,
) -> int:
    """The number of workers of a run over `n_rows` rows.

    `None` is the automatic default: a run below `threshold` rows is serial,
    since the pool costs more than it saves, and a larger one uses one worker
    per usable core up to `max_workers` (the scaling of the shared-memory core
    flattens there). The cores are counted with `os.process_cpu_count`, which
    honours the CPU affinity of the process, a cgroup quota and
    `PYTHON_CPU_COUNT`, so a process pinned to two cores of a cluster node
    uses two workers. An explicit `n_workers` is taken as given, `1` being the
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
    return max(1, min(os.process_cpu_count() or 1, max_workers))


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

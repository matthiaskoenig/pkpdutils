"""Tests of the shared worker pools (`pkpdutils.parallel`)."""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest

from pkpdutils.parallel import evict, executor, resolve_workers, split_rows


def die_on_zero(value: int) -> int:
    """A worker task which kills its process for one of the values."""
    if value == 0:
        os._exit(1)
    return value


def test_executor_is_shared_per_kind_and_size() -> None:
    pool = executor("thread", 2)
    assert executor("thread", 2) is pool
    assert isinstance(pool, ThreadPoolExecutor)
    assert executor("thread", 3) is not pool
    assert executor("thread", 3) is executor("thread", 3)


def test_executor_runs_work() -> None:
    pool = executor("thread", 2)
    assert list(pool.map(abs, [-1, 2, -3])) == [1, 2, 3]


def test_process_executor_is_shared() -> None:
    pool = executor("process", 2)
    assert executor("process", 2) is pool
    assert isinstance(pool, ProcessPoolExecutor)
    assert executor("process", 2) is not executor("thread", 2)


def test_evict_drops_the_cached_executor() -> None:
    pool = executor("thread", 4)
    evict("thread", 4)
    fresh = executor("thread", 4)
    assert fresh is not pool
    assert list(fresh.map(abs, [-1, 2])) == [1, 2]
    # evicting an executor which is not cached does nothing
    evict("thread", 4)
    evict("process", 7)


def test_a_broken_process_pool_is_replaced() -> None:
    # a worker which dies breaks the whole pool; the shared one is dropped, so
    # that one dead worker does not fail every later call of the process
    pool = executor("process", 2)
    with pytest.raises(BrokenProcessPool):
        list(pool.map(die_on_zero, [0, 1, 2]))
    fresh = executor("process", 2)
    assert fresh is not pool
    assert list(fresh.map(abs, [-1, 2, -3])) == [1, 2, 3]


def test_resolve_workers_counts_the_usable_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the affinity of the process, a cgroup quota and `PYTHON_CPU_COUNT` are
    # honoured, `os.cpu_count` (every core of the machine) is not
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3)
    assert resolve_workers(None, 10**6) == 3
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64)
    assert resolve_workers(None, 10**6) == 8
    monkeypatch.setattr(os, "process_cpu_count", lambda: None)
    assert resolve_workers(None, 10**6) == 1


def test_resolve_workers_automatic() -> None:
    expected = max(1, min(os.process_cpu_count() or 1, 8))
    assert resolve_workers(None, 0) == 1
    assert resolve_workers(None, 19_999) == 1
    assert resolve_workers(None, 20_000) == expected
    assert resolve_workers(None, 1_000_000) == expected


def test_resolve_workers_explicit_and_thresholds() -> None:
    assert resolve_workers(1, 10**6) == 1
    assert resolve_workers(3, 0) == 3
    # an explicit count is kept, even above the cores of the machine
    assert resolve_workers(64, 10**6) == 64
    assert resolve_workers(None, 300, threshold=200) == max(
        1, min(os.process_cpu_count() or 1, 8)
    )
    assert resolve_workers(None, 300, threshold=200, max_workers=2) == 2


@pytest.mark.parametrize("n_rows", [0, 1, 7, 999, 1_000, 30_000])
def test_split_rows_covers_the_rows(n_rows: int) -> None:
    chunks = split_rows(n_rows, 8)
    assert sum(chunk.stop - chunk.start for chunk in chunks) == n_rows
    start = 0
    for chunk in chunks:
        assert chunk.start == start
        assert chunk.stop >= chunk.start
        start = chunk.stop
    assert start == n_rows


def test_split_rows_edge_cases() -> None:
    assert split_rows(0, 8) == []
    assert split_rows(1, 8) == [slice(0, 1)]
    # below `min_rows` there is nothing to share out
    assert split_rows(999, 8) == [slice(0, 999)]
    # 100 rows over 8 workers: four chunks of 13 and four of 12
    sizes = [chunk.stop - chunk.start for chunk in split_rows(100, 8, min_rows=10)]
    assert sizes == [13, 13, 13, 13, 12, 12, 12, 12]


def test_split_rows_one_chunk_per_worker() -> None:
    chunks = split_rows(32_000, 8)
    assert len(chunks) == 8
    assert {chunk.stop - chunk.start for chunk in chunks} == {4_000}
    assert split_rows(100, 1, min_rows=1) == [slice(0, 100)]


def test_split_rows_min_rows_bounds_the_count() -> None:
    # 8 workers, but only 3 chunks of at least 1 000 rows fit
    chunks = split_rows(3_500, 8)
    assert len(chunks) == 3
    assert [chunk.stop - chunk.start for chunk in chunks] == [1_167, 1_167, 1_166]


def test_split_rows_max_rows_forces_more_chunks() -> None:
    chunks = split_rows(30_000, 2, max_rows=5_000)
    assert len(chunks) == 6
    assert max(chunk.stop - chunk.start for chunk in chunks) == 5_000
    # `max_rows` also cuts a batch that is smaller than `min_rows`
    chunks = split_rows(5, 1, max_rows=2)
    assert chunks == [slice(0, 2), slice(2, 4), slice(4, 5)]


def test_split_rows_never_more_chunks_than_rows() -> None:
    assert split_rows(3, 8, min_rows=1, max_rows=1) == [
        slice(0, 1),
        slice(1, 2),
        slice(2, 3),
    ]

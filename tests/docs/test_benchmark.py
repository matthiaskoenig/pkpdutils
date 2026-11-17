"""The performance page shows the cases the benchmark really runs.

The timings of `docs/benchmark.md` are machine specific and are not compared
here: what is compared is everything the script controls, i.e. the cases, their
size labels, the columns of the table, that every measured cell is a number,
and the refusals of the kernel word for word, since those are the dispatch's
own sentences and no measurement (`scripts.benchmark.page_complaints`).

The sentences under the table are the one part of the page a reader takes a
figure from, and no check can see whether they were measured, so what is
asserted here is that they cannot claim a crossover a run does not have: the
sweep is not monotone in its ratio by construction - one of its shapes is run
in parallel by both engines - and a sweep which crosses 1.00x more than once
names no crossover at all (`scripts.benchmark.summary_sentences`).
"""

from pathlib import Path
from typing import Any

from scripts.benchmark import (
    ENGINE_SHAPES,
    PAGE_PATH,
    clear_win,
    crossover,
    engine_points,
    engine_size,
    page_complaints,
    sign_changes,
    summary_sentences,
)


def test_page_is_current() -> None:
    page = Path(PAGE_PATH).read_text(encoding="utf-8")
    assert page_complaints(page) == [], (
        "docs/benchmark.md is stale: run `uv run python scripts/benchmark.py --write`"
    )


def sweep(ratios: dict[str, float]) -> list[dict[str, Any]]:
    """The rows of a run whose cases carry the given ratios.

    The two halves of a case are built from its ratio and its work: the kernel
    half is the fixed cost of a call plus a time per point and the python half
    is that times the ratio, so the cost sentence of a sweep is a function of
    the sweep alone and the largest shapes are the expensive ones, as they are
    in a run.
    """
    rows = []
    for name, speedup in ratios.items():
        n_rows, n_time = ENGINE_SHAPES[name]
        kernel_s = 3e-3 + 4e-8 * n_rows * n_time
        rows.append(
            {
                "case": name,
                "size": engine_size(n_rows, n_time),
                "kernel_s": kernel_s,
                "python_s": kernel_s * speedup,
                "speedup": speedup,
                "peak_rss_mb": 200.0,
            }
        )
    return rows


#: the sweep as the page measured it, which crosses 1.00x once and then holds
#: one row below the trend of the rows around it without falling back under it
MONOTONE: dict[str, float] = {
    "engine-1x12": 0.87,
    "engine-100x12": 0.94,
    "engine-10x200": 0.97,
    "engine-250x12": 1.04,
    "engine-1000x12": 1.44,
    "engine-100x200": 1.94,
    "engine-10000x12": 4.43,
    "engine-1000x200": 9.79,
    "engine-100000x12": 1.98,
    "engine-10000x200": 24.94,
}

#: the same sweep on a machine of more cores, where the one shape both engines
#: run in parallel falls below 1.00x: a 2 % move of a row two orders of
#: magnitude of points away from the shapes the crossover really lies between
NON_MONOTONE: dict[str, float] = {**MONOTONE, "engine-100000x12": 0.98}


def test_the_crossover_is_the_one_sign_change_of_the_sweep() -> None:
    rows = sweep(MONOTONE)
    ordered = sorted(rows, key=engine_points)
    crossed = crossover(ordered)
    assert crossed is not None
    assert [row["size"] for row in crossed] == [
        "10 curves x 200 points",
        "250 curves x 12 points",
    ]
    sentences = summary_sentences(rows)
    assert "The two engines cost the same between 2 000 and 3 000 points" in sentences
    assert "10 curves x 200 points is 0.97x" in sentences
    assert "250 curves x 12 points is 1.04x" in sentences
    # the bound is read from the shapes below the crossover, which is what the
    # sentence says, and not from every shape of the run below 1.00x
    assert "Below the crossover the kernel path costs at most 0.39 ms" in sentences


def test_a_sweep_which_crosses_twice_names_no_crossover() -> None:
    # the defect this guards: the two sorted halves of such a sweep put the
    # largest shape of the run at `below[-1]` and the smallest at `above[0]`,
    # so the sentence reads as a crossover between 1 200 000 and 3 000 points,
    # downwards, with a cost of 1.02 ms taken from a shape 400 times the size
    # of the one the crossover really lies at
    rows = sweep(NON_MONOTONE)
    ordered = sorted(rows, key=engine_points)
    below = [row for row in ordered if row["speedup"] < 1.0]
    above = [row for row in ordered if row["speedup"] >= 1.0]
    assert (below[-1]["size"], above[0]["size"]) == (
        "100 000 curves x 12 points",
        "250 curves x 12 points",
    )
    assert engine_points(below[-1]) > engine_points(above[0])
    assert len(sign_changes(ordered)) == 3
    assert crossover(ordered) is None
    sentences = summary_sentences(rows)
    assert "cost the same between" not in sentences
    assert "Below the crossover" not in sentences
    assert "crosses 1.00x at 3 shapes of the sweep rather than at one" in sentences
    assert "names no single crossover" in sentences
    assert "100 000 curves x 12 points at 0.98x" in sentences


def test_the_clear_win_is_the_shape_the_wins_start_at() -> None:
    rows = sweep(MONOTONE)
    win = clear_win(sorted(rows, key=engine_points))
    assert win is not None and win["size"] == "1 000 curves x 12 points"
    assert (
        "From 1 000 curves x 12 points on every shape of this run is a clear win"
        in summary_sentences(rows)
    )


def test_a_sweep_which_falls_back_names_a_later_clear_win() -> None:
    # 1.44x with a shape of 0.98x above it is no shape the kernel stays ahead
    # from, so the sentence names the shape the suffix of clear wins starts at
    # and not the first row which reaches 1.25x
    rows = sweep(NON_MONOTONE)
    win = clear_win(sorted(rows, key=engine_points))
    assert win is not None and win["size"] == "10 000 curves x 200 points"
    assert "From 10 000 curves x 200 points on every shape of this run" in (
        summary_sentences(rows)
    )


def test_a_sweep_below_one_everywhere_names_no_crossover_and_no_win() -> None:
    rows = sweep(dict.fromkeys(ENGINE_SHAPES, 0.9))
    sentences = summary_sentences(rows)
    assert "Every shape of this run is below 1.00x" in sentences
    assert "clear win" not in sentences

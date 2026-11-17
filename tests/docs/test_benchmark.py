"""The performance page shows the cases the benchmark really runs.

The timings of `docs/benchmark.md` are machine specific and are not compared
here: what is compared is everything the script controls, i.e. the cases, their
size labels, the columns of the table, that every measured cell is a number,
and the refusals of the kernel word for word, since those are the dispatch's
own sentences and no measurement (`scripts.benchmark.page_complaints`).
"""

from pathlib import Path

from scripts.benchmark import PAGE_PATH, page_complaints


def test_page_is_current() -> None:
    page = Path(PAGE_PATH).read_text(encoding="utf-8")
    assert page_complaints(page) == [], (
        "docs/benchmark.md is stale: run `uv run python scripts/benchmark.py --write`"
    )

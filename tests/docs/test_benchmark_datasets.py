"""The page of the benchmark datasets shows what its code prints."""

from pathlib import Path

from scripts.benchmark_datasets import PAGE_PATH, render


def test_page_is_current() -> None:
    page = Path(PAGE_PATH).read_text(encoding="utf-8")
    assert render(page) == page, (
        "docs/benchmark_datasets.md is stale: run "
        "`uv run python scripts/benchmark_datasets.py`"
    )

"""Run the code of `docs/benchmark_datasets.md` and paste what it prints.

The page compares `pkpdutils` against Phoenix WinNonlin, PKNCA and NonCompart on
the benchmark datasets of `docs/data/benchmarks/`, and every table on it is the
printed output of the python snippet above it. This script runs the snippets
of the page in order and in one namespace, in a temporary working directory
with the contents of `docs/data/` next to it (as `tests/docs/test_snippets.py`
runs them) and with warnings as errors, and writes what every snippet printed
into the region between the markers `<!-- output:start -->` and
`<!-- output:end -->` which follows it:

```bash
uv run python scripts/benchmark_datasets.py          # rewrite the page
uv run python scripts/benchmark_datasets.py --check  # exit 1 if it is stale
```

`tests/docs/test_benchmark_datasets.py` runs the check, so the page can not
drift from the code it shows. The PKNCA and NonCompart results the page reads
are written by `scripts/benchmark_datasets.R`.
"""

import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

REPO_DIR: Path = Path(__file__).parent.parent
PAGE_PATH: Path = REPO_DIR / "docs" / "benchmark_datasets.md"
DATA_DIR: Path = REPO_DIR / "docs" / "data"

OUTPUT_START = "<!-- output:start -->"
OUTPUT_END = "<!-- output:end -->"

#: a python snippet or an output region of the page, in the order of the page
PART = re.compile(
    r"^```python\n(?P<code>.*?)^```\n"
    rf"|^{re.escape(OUTPUT_START)}\n.*?^{re.escape(OUTPUT_END)}\n",
    re.S | re.M,
)


def run_snippets(snippets: list[str]) -> list[str]:
    """Run the snippets in one namespace and return what every one printed.

    Args:
        snippets: the code of every snippet, in the order of the page.

    Returns:
        The printed output of every snippet.
    """
    outputs: list[str] = []
    namespace: dict[str, object] = {"__name__": "__main__"}
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        for data in DATA_DIR.iterdir():
            if data.is_dir():
                shutil.copytree(data, Path(tmp) / data.name)
            else:
                shutil.copy(data, Path(tmp) / data.name)
        os.chdir(tmp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for number, code in enumerate(snippets, start=1):
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        exec(compile(code, f"<snippet {number}>", "exec"), namespace)
                    outputs.append(buffer.getvalue())
        finally:
            os.chdir(cwd)
    return outputs


def output_region(text: str) -> str:
    """The markdown of an output region: a text fence between the markers.

    Args:
        text: what a snippet printed.

    Returns:
        The region, ending with a newline.
    """
    return f"{OUTPUT_START}\n```text\n{text.rstrip()}\n```\n{OUTPUT_END}\n"


def render(page: str) -> str:
    """The page with the output of every snippet pasted below it.

    Args:
        page: the markdown of the page.

    Returns:
        The markdown with every output region rewritten.

    Raises:
        ValueError: if an output region has no snippet before it, or its
            snippet printed nothing.
    """
    parts = list(PART.finditer(page))
    snippets = [m["code"] for m in parts if m["code"] is not None]
    outputs = iter(run_snippets(snippets))
    rendered: list[str] = []
    last = 0
    printed: str | None = None
    for match in parts:
        rendered.append(page[last : match.start()])
        last = match.end()
        if match["code"] is not None:
            printed = next(outputs)
            rendered.append(match.group(0))
            continue
        if not printed:
            raise ValueError(
                f"the output region at offset {match.start()} follows no snippet "
                "which prints"
            )
        rendered.append(output_region(printed))
        printed = None
    rendered.append(page[last:])
    return "".join(rendered)


def main(argv: list[str]) -> int:
    page = PAGE_PATH.read_text(encoding="utf-8")
    new = render(page)
    if "--check" in argv:
        if new != page:
            print(f"{PAGE_PATH} is not current, run {Path(__file__).name}")
            return 1
        return 0
    PAGE_PATH.write_text(new, encoding="utf-8")
    print(f"wrote {PAGE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

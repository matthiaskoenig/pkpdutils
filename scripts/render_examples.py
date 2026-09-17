"""Render the figures of the examples into the documentation.

Every example of `tests/examples/test_examples.py` is run as a module in a
temporary working directory, with the `Agg` backend and warnings as errors, and
the PNG files it writes are copied into `docs/images/` under the name the
example gave them (`<example>.png`, `<example>_<suffix>.png`). The images are
committed, so the documentation build stays a plain `zensical build`; run the
script after an example or a plot function changed:

```bash
uv run python scripts/render_examples.py
uv run python scripts/render_examples.py nca_single formats
```

The script prints the files it wrote and exits non-zero when an example fails
or writes no figure at all.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DIR: Path = Path(__file__).parent.parent
IMAGES_DIR: Path = REPO_DIR / "docs" / "images"
EXAMPLE_TEST: Path = REPO_DIR / "tests" / "examples" / "test_examples.py"


def example_modules() -> list[str]:
    """The example modules, read from the list the test suite runs."""
    spec = importlib.util.spec_from_file_location("_example_scripts", EXAMPLE_TEST)
    if spec is None or spec.loader is None:  # pragma: no cover - a broken checkout
        raise RuntimeError(f"cannot read the example list from {EXAMPLE_TEST}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.SCRIPTS)


def render(module: str, work_dir: Path) -> list[Path]:
    """Run one example in `work_dir` and return the PNG files it wrote."""
    env = dict(os.environ, PYTHONPATH=str(REPO_DIR), MPLBACKEND="Agg", PYTHONUTF8="1")
    result = subprocess.run(
        [sys.executable, "-W", "error", "-m", module],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"{module} failed with exit code {result.returncode}")
    return sorted(work_dir.glob("*.png"))


def main() -> int:
    """Render the requested examples and copy their figures to `docs/images/`."""
    modules = example_modules()
    names = [module.rpartition(".")[2] for module in modules]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "examples",
        nargs="*",
        default=names,
        choices=names,
        help="examples to render, every one of them by default",
    )
    args = parser.parse_args()
    selected = [f"examples.{name}" for name in dict.fromkeys(args.examples)]

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    failed: list[str] = []
    for module in selected:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                figures = render(module, Path(tmp))
                if not figures:
                    failed.append(f"{module}: no figure written")
                    continue
                for figure in figures:
                    if figure.name in written:
                        failed.append(
                            f"{module}: {figure.name} was already written by "
                            f"{written[figure.name]}"
                        )
                        continue
                    shutil.copyfile(figure, IMAGES_DIR / figure.name)
                    written[figure.name] = module
                    size = (IMAGES_DIR / figure.name).stat().st_size
                    print(f"docs/images/{figure.name} ({size} bytes, {module})")
        except RuntimeError as error:
            failed.append(str(error))

    print(f"\n{len(written)} figures of {len(selected)} examples")
    stale = sorted(
        path.name for path in IMAGES_DIR.glob("*.png") if path.name not in written
    )
    if stale and len(selected) == len(modules):
        print(f"not written by any example: {', '.join(stale)}")
    for message in failed:
        print(message, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

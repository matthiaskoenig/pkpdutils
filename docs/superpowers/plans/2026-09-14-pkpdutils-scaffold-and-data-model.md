# pkpdutils Scaffold and Data Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `pkdb_analysis` package by the empty but fully tooled `pkpdutils` package with its units and timecourse data model, so that the NCA, fitting and statistics plans have a repository with green checks to build on.

**Architecture:** Phases 1 and 2 of the spec. The reference values of the old NCA are frozen first, then everything PK-DB specific is deleted, the tooling of `sbmlsim` is copied and adapted (uv, hatchling, ruff, ty, tox, pre-commit, workflows, rulesets, Zensical docs), and the package gets `units.py`, `console.py`, `log.py` and `timecourse.py` with `Dose`, `Route`, `DosingRegimen`, `Timecourse` (pydantic, one curve) and `Timecourses` (xarray batch). Every task leaves `pytest`, `ruff check`, `ruff format --check`, `ty check` and `zensical build` green.

**Tech Stack:** python 3.13/3.14, uv, hatchling, numpy, xarray, pandas, pint, pydantic v2, rich, pytest + pytest-xdist, ruff, ty, tox, Zensical + mkdocstrings-python.

**Spec:** `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`

**Later plans** (one each, written after this one is executed): NCA (phase 3), uncertainty (phase 4), fitting and PD (phase 5), statistics (phase 6), release (phase 7).

## Global Constraints

- package name `pkpdutils`, import name `pkpdutils`, repository stays `/home/mkoenig/git/pkdb_analysis` locally (the GitHub rename is a manual step of the maintainer)
- `requires-python = ">=3.13"`, `.python-version` is `3.14`, tox envs `ty, py3.{13,14}`
- runtime dependencies exactly: `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib`, `rich`
- `[tool.ty.terminal] error-on-warning = true`: the tree has zero ty diagnostics; a suppression is rule specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`
- every module, class and function of `src/pkpdutils` has full type annotations and a google style docstring (ruff `D`); `examples/` and `tests/` are exempt from `D`
- library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting (ruff `G`), never prints, never calls `plt.show()`
- markdown has no hard line wraps: a paragraph, list item or table row is one line
- commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`
- work happens on the branch `redesign`; the reference source for tooling is the sibling checkout `/home/mkoenig/git/sbmlsim` (read only)
- commands below run from the repository root; prefix with `uv run` when the environment is not activated

---

### Task 1: Freeze the reference values of the old NCA

The old `TimecoursePK` is deleted in Task 2; its numeric results on the test fixtures are the regression reference of the NCA plan. They are computed with the old code, in a throwaway environment with only its direct dependencies, and stored as JSON.

**Files:**
- Create: `tests/data/reference/nca_reference.json`
- Create (scratch, not committed): `/tmp/claude-1000/-home-mkoenig-git-pkdb-analysis/d6cf6b42-d640-496e-9375-e7189581a805/scratchpad/freeze_nca.py`

**Interfaces:**
- Produces: `tests/data/reference/nca_reference.json`, a list of cases `{"name", "file", "time_unit", "unit", "dose", "dose_unit", "substance", "time", "concentration", "parameters": {"auc": {"magnitude", "unit"}, ...}, "regression": {"slope", "intercept", "r_value", "p_value", "std_err", "max_idx"}}`. The NCA plan reproduces `auc`, `aucinf`, `tmax`, `cmax`, `kel`, `thalf`, `vd`, `cl` with `auc_method=LINEAR` and `TerminalPhase(method=ALL_AFTER_TMAX)`.

- [ ] **Step 1: Write the freeze script**

```python
"""Freeze the results of the old TimecoursePK as regression reference.

Run from the repository root before the old package is deleted:

    PYTHONPATH=src uv run --no-project --python 3.13 \
        --with "numpy<2.3,scipy,pint,matplotlib,rich,pandas" \
        python <this file>
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pint

from pkdb_analysis.pk.pharmacokinetics import TimecoursePK

REPO = Path.cwd()
DATA = REPO / "tests" / "data" / "pk"
OUT = REPO / "tests" / "data" / "reference" / "nca_reference.json"

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def case(name, file, time, conc, time_unit, unit, dose, dose_unit, substance):
    tcpk = TimecoursePK(
        time=Q_(np.asarray(time, dtype=float), time_unit),
        concentration=Q_(np.asarray(conc, dtype=float), unit),
        dose=Q_(dose, dose_unit),
        ureg=ureg,
        substance=substance,
    )
    pk = tcpk.pk

    def num(x):
        x = float(x)
        return None if math.isnan(x) else x

    parameters = {}
    for key in pk.parameters:
        q = getattr(pk, key)
        parameters[key] = {"magnitude": num(q.magnitude), "unit": str(q.units)}
    regression = {
        "slope": num(pk.slope.magnitude),
        "slope_unit": str(pk.slope.units),
        "intercept": num(pk.intercept.magnitude),
        "intercept_unit": str(pk.intercept.units),
        "r_value": num(pk.r_value),
        "p_value": num(pk.p_value),
        "std_err": num(pk.std_err),
        "max_idx": int(pk.max_idx),
    }
    return {
        "name": name,
        "file": file,
        "time_unit": time_unit,
        "unit": unit,
        "dose": dose,
        "dose_unit": dose_unit,
        "substance": substance,
        "time": [float(v) for v in time],
        "concentration": [None if math.isnan(float(v)) else float(v) for v in conc],
        "parameters": parameters,
        "regression": regression,
    }


cases = []

# synthetic mono-exponential curve (example0 of the old package)
t = np.linspace(0, 3, num=50)
c = 10.0 * np.exp(-1.0 * t)
cases.append(case("synthetic_monoexp", None, t, c, "hr", "nmol/l", 0.01, "mol", "substance"))

# NaN handling (test_pharmacokinetics_nan of the old package)
cases.append(
    case(
        "nan_values",
        None,
        [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 24.0],
        [0.00829241, 0.00610768, 0.00460769, 0.0050102, 0.00220838, 0.00190252, 0.00128454, 0.00094463, 0.00077496, np.nan],
        "hr",
        "g/l",
        0.35,
        "g",
        "substance",
    )
)

df = pd.read_csv(DATA / "Divoll1982_Fig1.tsv", sep="\t", na_values="NA")
cases.append(case("Divoll1982_Fig1", "Divoll1982_Fig1.tsv", df.time.values, df.apap.values, "hr", "µg/ml", 100, "mg", "acetaminophen"))

df = pd.read_csv(DATA / "Kim2011_Fig2.tsv", sep="\t", na_values="NA")
df = df[df.interventions == "paracetamol1000mg"]
cases.append(case("Kim2011_Fig2", "Kim2011_Fig2.tsv", df.time.values, df["mean"].values, "hr", "µg/ml", 100, "mg", "acetaminophen"))

df = pd.read_csv(DATA / "midazolam.tsv", sep="\t", na_values="NA")
cases.append(case("midazolam", "midazolam.tsv", df.time.values, df["Cve_mid"].values, "min", "mmol/l", 7.5, "mg", "midazolam"))

df = pd.read_csv(DATA / "data_example2.csv", sep="\t", na_values="NA")
for substance, dose_per_kg in [("caffeine", 2), ("caffeine", 4), ("paraxanthine", 2), ("paraxanthine", 4)]:
    data = df[(df.substance == substance) & (df.dose == dose_per_kg)]
    conc = data.caf.values if substance == "caffeine" else data.px.values
    cases.append(
        case(
            f"Benowitz1995_{substance}_{dose_per_kg}mgkg",
            "data_example2.csv",
            data.time.values,
            conc,
            "hr",
            "mg/l",
            dose_per_kg * 70,
            "mg",
            substance,
        )
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(cases, indent=2))
print(f"{len(cases)} cases written to {OUT}")
```

- [ ] **Step 2: Run the freeze script**

Run:
```bash
PYTHONPATH=src uv run --no-project --python 3.13 --with "numpy<2.3,scipy,pint,matplotlib,rich,pandas" python /tmp/claude-1000/-home-mkoenig-git-pkdb-analysis/d6cf6b42-d640-496e-9375-e7189581a805/scratchpad/freeze_nca.py
```
Expected: `10 cases written to .../tests/data/reference/nca_reference.json`. Warnings from the old code (positive slope, extrapolation) are expected on the paraxanthine cases. If the `pint` import of the old module fails on the `Quantity([])` guard, add `--with "pint<0.25"`.

- [ ] **Step 3: Check the JSON**

Run: `python -c "import json; d=json.load(open('tests/data/reference/nca_reference.json')); print(len(d), [c['name'] for c in d]); print(d[0]['parameters']['kel'])"`
Expected: 10 names, `kel` of `synthetic_monoexp` has magnitude close to `1.0` and unit `1 / hour`.

- [ ] **Step 4: Commit**

```bash
git add tests/data/reference/nca_reference.json
git commit -m "Freeze the NCA results of pkdb_analysis 0.3.1 as regression reference

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 2: Remove the old package, documentation, tests and tooling

**Files:**
- Delete: `src/pkdb_analysis/` (all), `docs_builder/` (all), `tests/__init__.py`, `tests/test_data.py`, `tests/test_inference_bw.py`, `tests/test_io.py`, `tests/test_pk.py`, `tests/test_query.py`, `tests/update_test_data.py`, `tests/data/testdata_concise_false.zip`, `tests/data/testdata_concise_true.zip`, `MANIFEST.in`, `RELEASE.md`, `.github/workflows/main.yml`, `.github/workflows/ruff.yml`, `.github/CONTRIBUTING.rst`, `uv.lock`
- Move: `tests/data/pk/*` to `tests/data/`

- [ ] **Step 1: Delete and move**

```bash
git rm -r -q src/pkdb_analysis docs_builder tests/__init__.py tests/test_data.py tests/test_inference_bw.py tests/test_io.py tests/test_pk.py tests/test_query.py tests/update_test_data.py tests/data/testdata_concise_false.zip tests/data/testdata_concise_true.zip MANIFEST.in RELEASE.md .github/workflows/main.yml .github/workflows/ruff.yml .github/CONTRIBUTING.rst uv.lock
git mv tests/data/pk/data_example1.csv tests/data/pk/data_example2.csv tests/data/pk/Divoll1982_Fig1.tsv tests/data/pk/Kim2011_Fig2.tsv tests/data/pk/Lane2014_Fig1.tsv tests/data/pk/midazolam.tsv tests/data/
```

- [ ] **Step 2: Verify the tree**

Run: `git status --short | head -5 && find src tests -type f | sort`
Expected: `src` does not exist or is empty; `tests/data/` holds the six data files and `reference/nca_reference.json`, nothing else.

- [ ] **Step 3: Commit**

```bash
git commit -q -m "Remove the PK-DB specific package, its documentation and tests

Everything of pkdb_analysis that is not the NCA algorithm or the meta-analysis math
is PK-DB specific and goes; both are ported in pkpdutils with tests against
tests/data/reference/nca_reference.json.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 3: Project scaffolding and the package skeleton

**Files:**
- Create: `pyproject.toml` (overwrite), `.python-version` (overwrite), `tox.ini` (overwrite), `.ruff.toml` (overwrite), `.pre-commit-config.yaml` (overwrite), `.bumpversion.toml` (overwrite), `.gitignore` (overwrite), `conftest.py`, `src/pkpdutils/__init__.py`, `src/pkpdutils/console.py`, `src/pkpdutils/log.py`, `src/pkpdutils/py.typed`
- Test: `tests/test_package.py`, `tests/test_log.py`

**Interfaces:**
- Produces: `pkpdutils.__version__ == "1.0.0.dev0"` (bumped to `1.0.0` in the release plan), `pkpdutils.console.console: rich.console.Console`, `pkpdutils.log.enable_rich_logging(level=logging.INFO, console=None) -> logging.Logger`, `pkpdutils.log.PACKAGE_LOGGER = "pkpdutils"`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pkpdutils"
dynamic = ["version"]
description = "pkpdutils are python utilities for the pharmacokinetic and pharmacodynamic analysis of timecourses and parameters."
readme = "README.md"
requires-python = ">=3.13"
license = "MIT"
license-files = ["LICENSE"]
authors = [
	{name="Matthias König", email="konigmatt@googlemail.com"},
]
maintainers = [
	{name="Matthias König", email="konigmatt@googlemail.com"}
]
classifiers = [
	"Development Status :: 5 - Production/Stable",
	"Intended Audience :: Science/Research",
	"Operating System :: OS Independent",
	"Programming Language :: Python :: 3.13",
	"Programming Language :: Python :: 3.14",
	"Programming Language :: Python :: Implementation :: CPython",
	"Topic :: Scientific/Engineering",
	"Topic :: Scientific/Engineering :: Bio-Informatics",
]
keywords = [
	"pharmacokinetics",
	"pharmacodynamics",
	"non-compartmental analysis",
	"bioequivalence",
	"drug-drug interaction",
	"meta-analysis",
]
dependencies = [
	# numerics, data structures and units
	"numpy>=2.2.2",
	"scipy>=1.15.0",
	"pandas>=2.2.0",
	"xarray>=2025.1.0",
	"pint>=0.25.3",
	"pydantic>=2.10.0",
	# figures and console output
	"matplotlib>=3.10.0",
	"rich>=14.0.0",
]

[project.optional-dependencies]
dev = [
	"bump-my-version>=1.5.1",
	"ruff>=0.16.6",
	"pre-commit>=4.6.2",
	"ty>=0.0.79",
	"tox>=4.61.2",
	"pytest>=9.1.1",
	"pytest-cov>=7.1.0",
	"pytest-xdist>=3.8",
	"zensical>=0.0.60",
	"mkdocstrings-python>=2.0.8",
]

[project.urls]
Homepage = "https://matthiaskoenig.github.io/pkpdutils"
Documentation = "https://matthiaskoenig.github.io/pkpdutils"
Repository = "https://github.com/matthiaskoenig/pkpdutils"
Issues = "https://github.com/matthiaskoenig/pkpdutils/issues"
Changelog = "https://github.com/matthiaskoenig/pkpdutils/tree/develop/release-notes"
Download = "https://pypi.org/project/pkpdutils"

[tool.hatch.version]
path = "./src/pkpdutils/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/pkpdutils"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# the tests run in parallel (pytest-xdist), `pytest -n 0` runs them in one process
addopts = "-n auto"

[tool.ty.environment]
# the package lives in the src layout, the examples are a package at the root of
# the repository; the checked python version is inferred from
# `project.requires-python` (i.e. the oldest supported version)
root = ["./src", "."]

[tool.ty.src]
include = ["src", "tests", "examples", "scripts"]

[tool.ty.terminal]
# warnings are failures: keep the codebase free of any diagnostic
error-on-warning = true
```

- [ ] **Step 2: Write `.python-version`, `tox.ini`, `.bumpversion.toml`, `.gitignore`, `conftest.py`**

`.python-version`:
```
3.14
```

`tox.ini`:
```ini
[tox]
envlist = ty, py3.{13,14}

[testenv]
package = wheel
wheel_build_env = .pkg
deps =
    pytest
    pytest-xdist
commands =
    pytest

[testenv:ty]
passenv =
    TY_OUTPUT_FORMAT
deps =
    ty
    pytest
commands =
    ty check
```

`.bumpversion.toml`:
```toml
[tool.bumpversion]
current_version = "1.0.0.dev0"
commit = true
parse = "(?P<major>\\d+)\\.(?P<minor>\\d+)\\.(?P<patch>\\d+)(\\.dev(?P<dev>\\d+))?"
serialize = ["{major}.{minor}.{patch}.dev{dev}", "{major}.{minor}.{patch}"]
search = "{current_version}"
replace = "{new_version}"
regex = false
ignore_missing_version = false
# no tag: the bump is merged into develop through a pull request, which
# rewrites the commit, so the tag is created on develop afterwards
tag = false
sign_tags = false
tag_name = "{new_version}"
tag_message = "Bump version: {current_version} → {new_version}"
allow_dirty = false
message = "Bump version: {current_version} → {new_version}"
commit_args = ""

[tool.bumpversion.parts.dev]
optional_value = "final"
values = ["0", "final"]

[[tool.bumpversion.files]]
filename = "./src/pkpdutils/__init__.py"

[[tool.bumpversion.files]]
filename = "./CITATION.cff"
```

`.gitignore`:
```
# python
__pycache__/
*.pyc

# packaging artifacts (hatchling builds into dist/)
dist/
*.egg-info/

# environments, tool caches
.venv/
.tox/
.pytest_cache/
.ruff_cache/
.coverage
coverage.xml
# build cache of zensical
.cache/

# rendered documentation, built by the `documentation` workflow
site/

# the lock file is not committed, pkpdutils is a library and pins nothing
uv.lock

# output of the examples, they write into the working directory
examples/**/results/
/results/

# editors, operating system
.idea/
.vscode/
*~
.DS_Store
.ipynb_checkpoints/
```

`conftest.py`:
```python
"""Test session configuration.

Two things are set up for the whole test session:

- **matplotlib never opens a window.** The `Agg` backend is selected before
  pyplot is imported anywhere, so a figure created by an example or by a test
  is rendered into a buffer instead of into a window.
- **the examples are importable.** They are not part of the package, they live
  in `examples/` at the root of the repository. pytest puts the directory of
  this file on `sys.path`, so that the tests can import the examples.
"""

import matplotlib

matplotlib.use("Agg", force=True)
```

- [ ] **Step 3: Copy `.ruff.toml` and `.pre-commit-config.yaml` from sbmlsim and adapt**

```bash
cp /home/mkoenig/git/sbmlsim/.ruff.toml .ruff.toml
cp /home/mkoenig/git/sbmlsim/.pre-commit-config.yaml .pre-commit-config.yaml
```

In `.ruff.toml` replace the `[lint.per-file-ignores]` block by:
```toml
[lint.per-file-ignores]
# the examples and tests are scripts and fixtures, a docstring on every helper adds nothing
"examples/**" = ["D"]
"tests/**" = ["D"]
"scripts/**" = ["D"]
```
and remove the sentence about `sbmlutils.factory` in the comment above it.

In `.pre-commit-config.yaml` replace every `exclude:` value by `'^(tests/data/|examples/.*\.(tsv|csv|txt|json|md|svg|html)$)'` and remove the sentence "The packaged models, the data of the examples ..." in the header comment (replace it with "The data of the examples and the fixtures of the tests are content, not source, so the whitespace hooks below exclude them.").

- [ ] **Step 4: Write the package skeleton**

`src/pkpdutils/__init__.py`:
```python
"""pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters."""

__version__ = "1.0.0.dev0"

__all__ = ["__version__"]
```

`src/pkpdutils/py.typed`: empty file.

`src/pkpdutils/console.py`:
```python
"""Shared rich console.

The console is used for the output of scripts and examples; library code logs
instead of printing, see `pkpdutils.log`.

```python
from pkpdutils.console import console

console.print(result)
console.rule("Section", style="white")
```

Importing this module has no side effects on the interpreter. To get rich
representations in an interactive session, install them explicitly with
`rich.pretty.install()`.
"""

from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "success": "green",
        "info": "blue",
        "warning": "orange3",
        "error": "red",
    }
)

console = Console(theme=custom_theme)
```

`src/pkpdutils/log.py`:
```python
"""Logging of the package.

`pkpdutils` follows the convention for libraries: it only gets loggers and logs
to them, it does not configure logging. Handlers, levels and formatting are left
to the application.

Modules get their logger from the standard library with

```python
import logging

logger = logging.getLogger(__name__)
```

All loggers are therefore below the `pkpdutils` logger, so an application
configures them in one place:

```python
import logging

logging.getLogger("pkpdutils").setLevel(logging.WARNING)
```

For scripts and interactive work the rich formatting can be enabled explicitly,
which is what the examples do:

```python
from pkpdutils import log

log.enable_rich_logging()
```
"""

import logging

from rich.console import Console
from rich.logging import RichHandler

from pkpdutils.console import console as default_console

#: name of the logger all loggers of the package are below
PACKAGE_LOGGER = "pkpdutils"


def enable_rich_logging(
    level: int = logging.INFO, console: Console | None = None
) -> logging.Logger:
    """Log the messages of the package on a rich console.

    This configures logging and is meant for scripts, examples and interactive
    work. Applications should configure logging themselves instead of calling
    this. Calling it repeatedly replaces the handler instead of adding a second
    one.

    Args:
        level: level from which messages are logged
        console: console to log on, the console of the package by default

    Returns:
        The `pkpdutils` logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)

    # remove a handler of an earlier call, logging twice is worse than not at all
    for handler in list(logger.handlers):
        if isinstance(handler, RichHandler):
            logger.removeHandler(handler)

    handler = RichHandler(
        markup=False,
        rich_tracebacks=True,
        show_time=False,
        console=console if console is not None else default_console,
    )
    handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))

    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
```

- [ ] **Step 5: Write the tests**

`tests/test_package.py`:
```python
import pkpdutils


def test_version() -> None:
    assert pkpdutils.__version__ == "1.0.0.dev0"
```

`tests/test_log.py`:
```python
import logging

from rich.console import Console
from rich.logging import RichHandler

from pkpdutils import log


def test_enable_rich_logging_adds_one_handler() -> None:
    console = Console(file=open("/dev/null", "w"), width=80)  # noqa: SIM115
    logger = log.enable_rich_logging(level=logging.DEBUG, console=console)
    log.enable_rich_logging(level=logging.DEBUG, console=console)
    handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
    assert logger.name == "pkpdutils"
    assert len(handlers) == 1
    assert logger.level == logging.DEBUG
    for handler in handlers:
        logger.removeHandler(handler)
```

- [ ] **Step 6: Create the environment and run everything**

Run:
```bash
uv sync --extra dev
uv run pytest
uv run ruff check && uv run ruff format --check
uv run ty check
```
Expected: `uv sync` succeeds with python 3.14; 2 tests pass; ruff and ty report nothing. If `uv sync` needs the interpreter: `uv python install 3.13 3.14`.

- [ ] **Step 7: Install the pre-commit hook and run it**

Run: `uv run pre-commit install && uv run pre-commit run --all-files`
Expected: all hooks pass (hooks may reformat data files once: `git add` the changes if any hook modified a file, then rerun).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -q -m "Scaffold pkpdutils: packaging, tooling and the package skeleton

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 4: Units

**Files:**
- Create: `src/pkpdutils/units.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces:
  - `ureg: pint.UnitRegistry` (one per process), `Q_ = ureg.Quantity`, `Quantity` (type alias of `pint.facets.plain.PlainQuantity`), `Unit` (type alias of `pint.facets.plain.PlainUnit`)
  - `parse_unit(unit: str) -> Unit` raises `ValueError` on an unknown unit
  - `check_dose_unit(unit: str) -> None` raises `ValueError` unless the unit reduces to `[mass]`, `[substance]`, `[mass]/[mass]` or `[substance]/[mass]`
  - `is_per_bodyweight(unit: str) -> bool`
  - `normalize_volume(q: Quantity) -> Quantity` converts a volume to `liter` and a volume per mass to `liter/kilogram`, leaves anything else unchanged
  - `normalize_clearance(q: Quantity) -> Quantity` converts to `liter/hour` or `liter/hour/kilogram`
  - `unit_str(unit: Unit | str) -> str` the canonical string of a unit (`str(parse_unit(unit))`)

- [ ] **Step 1: Write the failing tests**

`tests/test_units.py`:
```python
import pytest

from pkpdutils.units import (
    Q_,
    check_dose_unit,
    is_per_bodyweight,
    normalize_clearance,
    normalize_volume,
    parse_unit,
    unit_str,
    ureg,
)


def test_registry_custom_units() -> None:
    assert Q_(1, "percent").to("dimensionless").magnitude == pytest.approx(0.01)
    assert Q_(1, "none").dimensionless
    assert Q_(1, "IU").check("[activity_amount]")


def test_parse_unit() -> None:
    assert parse_unit("ng/ml") == ureg.Unit("nanogram / milliliter")
    with pytest.raises(ValueError, match="not_a_unit"):
        parse_unit("not_a_unit")


@pytest.mark.parametrize("unit", ["mg", "mmol", "mg/kg", "µmol/kg", "g"])
def test_check_dose_unit_valid(unit: str) -> None:
    check_dose_unit(unit)


@pytest.mark.parametrize("unit", ["mg/l", "hr", "l", "mg/kg/hr"])
def test_check_dose_unit_invalid(unit: str) -> None:
    with pytest.raises(ValueError, match="dose"):
        check_dose_unit(unit)


def test_is_per_bodyweight() -> None:
    assert is_per_bodyweight("mg/kg")
    assert not is_per_bodyweight("mg")


def test_normalize_volume() -> None:
    assert normalize_volume(Q_(2000, "ml")).magnitude == pytest.approx(2.0)
    assert str(normalize_volume(Q_(2000, "ml")).units) == "liter"
    q = normalize_volume(Q_(0.5, "m**3/kg"))
    assert q.magnitude == pytest.approx(500.0)
    assert str(q.units) == "liter / kilogram"
    unchanged = normalize_volume(Q_(1, "hr"))
    assert str(unchanged.units) == "hour"


def test_normalize_clearance() -> None:
    q = normalize_clearance(Q_(100, "ml/min"))
    assert q.magnitude == pytest.approx(6.0)
    assert str(q.units) == "liter / hour"
    q = normalize_clearance(Q_(1, "ml/min/kg"))
    assert q.magnitude == pytest.approx(0.06)
    assert str(q.units) == "liter / hour / kilogram"


def test_unit_str() -> None:
    assert unit_str("ng/ml") == "nanogram / milliliter"
    assert unit_str(parse_unit("hr")) == "hour"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_units.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.units'`.

- [ ] **Step 3: Write `src/pkpdutils/units.py`**

```python
"""Units of the package.

One [pint](https://pint.readthedocs.io) registry per process, `ureg`, is shared
by every timecourse, result and quantity of the package; quantities of different
registries cannot be combined, which is why nothing creates a registry of its
own. Numerics run on plain arrays in the units of the input, pint is used at the
boundaries: parsing unit strings, deriving the units of results and converting
volumes and clearances to their conventional units.

```python
from pkpdutils.units import Q_, ureg

dose = Q_(100, "mg")
time = Q_([0, 1, 2], "hr")
```
"""

import pint
from pint.facets.plain import PlainQuantity, PlainUnit

#: the unit registry of the package
ureg: pint.UnitRegistry = pint.UnitRegistry()
ureg.define("none = count")
ureg.define("IU = [activity_amount]")

#: shortcut for creating quantities
Q_ = ureg.Quantity

#: type of a quantity of the registry, for annotations
Quantity = PlainQuantity

#: type of a unit of the registry, for annotations
Unit = PlainUnit

#: dimensionalities a dose may have: an amount, or an amount per body weight
DOSE_DIMENSIONS: tuple[str, ...] = (
    "[mass]",
    "[substance]",
    "[mass] / [mass]",
    "[substance] / [mass]",
)


def parse_unit(unit: str) -> Unit:
    """Parse a unit string with the registry of the package.

    Args:
        unit: unit string, e.g. `"ng/ml"` or `"hr"`

    Returns:
        The unit.

    Raises:
        ValueError: if the string is not a unit of the registry.
    """
    try:
        return ureg.Unit(unit)
    except (pint.UndefinedUnitError, pint.DefinitionSyntaxError, AttributeError, TypeError) as err:
        raise ValueError(f"'{unit}' is not a unit: {err}") from err


def unit_str(unit: Unit | str) -> str:
    """Canonical string of a unit, e.g. `"nanogram / milliliter"` for `"ng/ml"`."""
    return str(parse_unit(unit) if isinstance(unit, str) else unit)


def check_dose_unit(unit: str) -> None:
    """Check that a unit is a dose unit.

    A dose is an amount of substance, as mass (`mg`) or as substance (`mmol`),
    or such an amount per body weight (`mg/kg`, `µmol/kg`).

    Args:
        unit: unit string of the dose

    Raises:
        ValueError: if the unit has another dimensionality.
    """
    u = parse_unit(unit)
    reduced = (1 * u).to_base_units().to_reduced_units()
    if not any(reduced.check(dimension) for dimension in DOSE_DIMENSIONS):
        raise ValueError(
            f"A dose must be in {DOSE_DIMENSIONS}, "
            f"not '{reduced.dimensionality}' ('{unit}')"
        )


def is_per_bodyweight(unit: str) -> bool:
    """Whether a dose unit is an amount per body weight, e.g. `"mg/kg"`."""
    reduced = (1 * parse_unit(unit)).to_base_units().to_reduced_units()
    return reduced.check("[mass] / [mass]") or reduced.check("[substance] / [mass]")


def normalize_volume(q: Quantity) -> Quantity:
    """Convert a volume to `liter` and a volume per body weight to `liter/kilogram`.

    Anything else is returned unchanged.
    """
    reduced = q.to_base_units().to_reduced_units()
    if reduced.check("[length] ** 3"):
        return q.to("liter")
    if reduced.check("[length] ** 3 / [mass]"):
        return q.to("liter / kilogram")
    return q


def normalize_clearance(q: Quantity) -> Quantity:
    """Convert a clearance to `liter/hour` and one per body weight to `liter/hour/kilogram`.

    Anything else is returned unchanged.
    """
    reduced = q.to_base_units().to_reduced_units()
    if reduced.check("[length] ** 3 / [time]"):
        return q.to("liter / hour")
    if reduced.check("[length] ** 3 / [time] / [mass]"):
        return q.to("liter / hour / kilogram")
    return q
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_units.py -n 0 -q`
Expected: all pass. If `Q_(1, "IU").check(...)` fails because pint spells the dimension differently, use `assert "[activity_amount]" in str(Q_(1, "IU").dimensionality)`.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: clean. If ty flags the `except` tuple, keep only `(pint.UndefinedUnitError, pint.DefinitionSyntaxError)` plus `AttributeError`.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/units.py tests/test_units.py
git commit -q -m "Add the unit registry and unit helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 5: GitHub workflows, rulesets and citation files

**Files:**
- Create: `.github/workflows/ci-cd.yml`, `.github/workflows/ruff.yml`, `.github/workflows/ty.yml`, `.github/workflows/docs.yml`, `.github/rulesets/apply.sh`, `.github/rulesets/develop.json`, `.github/rulesets/main.json`, `.github/rulesets/tags.json`, `.github/CODEOWNERS`, `.github/pull_request_template.md`, `.github/dependabot.yml`, `CITATION.cff`
- Modify: `.zenodo.json`

- [ ] **Step 1: Copy the GitHub files from sbmlsim**

```bash
mkdir -p .github/workflows .github/rulesets
cp /home/mkoenig/git/sbmlsim/.github/workflows/{ci-cd,ruff,ty,docs}.yml .github/workflows/
cp /home/mkoenig/git/sbmlsim/.github/rulesets/{apply.sh,develop.json,main.json,tags.json} .github/rulesets/
cp /home/mkoenig/git/sbmlsim/.github/{CODEOWNERS,pull_request_template.md,dependabot.yml} .github/
chmod +x .github/rulesets/apply.sh
sed -i 's/sbmlsim/pkpdutils/g' .github/workflows/*.yml .github/rulesets/apply.sh
```

- [ ] **Step 2: Remove the roadrunner specific steps**

pkpdutils has no compiled extension, so the steps "Install the python dev files" (the `deadsnakes` apt block) are deleted from `ci-cd.yml`, `ty.yml` and `docs.yml`. In `ci-cd.yml` delete the whole step from `- name: Install the python dev files (linux only)` to the line `shell: bash`; in `ty.yml` and `docs.yml` delete from `- name: Install the python dev files` through the `sudo apt-get install -y python3.14-dev` line. Verify with:

```bash
grep -n "deadsnakes\|python3.14-dev\|libroadrunner" .github/workflows/*.yml
```
Expected: no output.

- [ ] **Step 3: Write `CITATION.cff` and update `.zenodo.json`**

`CITATION.cff`:
```yaml
cff-version: 1.2.0
message: "If you use pkpdutils, please cite it as below."
title: "pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters"
abstract: "pkpdutils is a python library for the pharmacokinetic and pharmacodynamic analysis of timecourses and parameters: non-compartmental analysis, curve fitting, uncertainty propagation, significance tests, bioequivalence, drug-drug interaction classification and meta-analysis."
type: software
authors:
  - family-names: "König"
    given-names: "Matthias"
    orcid: "https://orcid.org/0000-0003-1725-179X"
    affiliation: "Humboldt-University Berlin, Faculty of Life Science, Institute for Biology, Berlin; University Lübeck; University Hospital Schleswig-Holstein, Campus Lübeck, First Department of Medicine, Germany"
  - family-names: "Grzegorzewski"
    given-names: "Jan"
version: "1.0.0.dev0"
date-released: "2020-08-24"
doi: "10.5281/zenodo.3997539"
license: MIT
repository-code: "https://github.com/matthiaskoenig/pkpdutils"
url: "https://matthiaskoenig.github.io/pkpdutils"
keywords:
  - pharmacokinetics
  - pharmacodynamics
  - non-compartmental analysis
  - bioequivalence
  - drug-drug interaction
  - meta-analysis
```

`.zenodo.json` (overwrite):
```json
{
  "upload_type": "software",
  "title": "pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters",
  "creators": [
    {
      "orcid": "0000-0003-1725-179X",
      "affiliation": "Humboldt-University Berlin, Institute for Theoretical Biology, Berlin",
      "name": "König, Matthias"
    },
    {
      "name": "Grzegorzewski, Jan"
    }
  ],
  "description": "<p><code>pkpdutils</code> is a python library for the pharmacokinetic and pharmacodynamic analysis of timecourses and parameters with source code available from <a href=\"https://github.com/matthiaskoenig/pkpdutils\">https://github.com/matthiaskoenig/pkpdutils</a>.</p>\n<p>Features include<ul><li>non-compartmental analysis of concentration and effect timecourses</li><li>uncertainty propagation for group data</li><li>curve fitting of exponential, Emax, dose proportionality and covariate models</li><li>significance tests, bioequivalence, drug-drug interaction classification and meta-analysis of pharmacokinetic parameters</li></ul></p>\n<p>The documentation is available on <a href=\"https://matthiaskoenig.github.io/pkpdutils\">https://matthiaskoenig.github.io/pkpdutils</a>. The package was formerly published as <code>pkdb-analysis</code>.</p>\n<p>If you have any questions or issues please <a href=\"https://github.com/matthiaskoenig/pkpdutils/issues\">open an issue</a></p>\n<h2>Funding</h2><p>Matthias König is supported by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 <strong><a href=\"https://qualiperf.de\">QuaLiPerF</a></strong> (Quantifying Liver Perfusion-Function Relationship in Complex Resection - A Systems Medicine Approach) by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA). Matthias König was supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (<strong>LiSyM</strong>, grant number 031L0054).</p>",
  "access_right": "open",
  "license": "MIT",
  "keywords": [
    "pharmacokinetics",
    "pharmacodynamics",
    "non-compartmental analysis",
    "bioequivalence",
    "drug-drug interaction",
    "meta-analysis"
  ]
}
```

- [ ] **Step 4: Validate the files**

Run: `python -c "import json,yaml;json.load(open('.zenodo.json'));[json.load(open(f'.github/rulesets/{n}.json')) for n in ('develop','main','tags')];print('ok')" ; uv run pre-commit run --all-files`
Expected: `ok` (if `yaml` is missing, drop the import; the pre-commit `check-yaml` hook covers the workflows) and all hooks pass.

- [ ] **Step 5: Commit**

```bash
git add .github CITATION.cff .zenodo.json
git commit -q -m "Add the CI, release and documentation workflows, the rulesets and the citation files

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 6: Documentation skeleton

**Files:**
- Create: `zensical.toml`, `docs/index.md`, `docs/installation.md`, `docs/development.md`, `docs/references.md`, `docs/robots.txt`, `docs/api/index.md`, `docs/api/units.md`, `docs/api/console.md`, `docs/api/log.md`, `docs/images/.gitkeep`, `scripts/llms_txt.py`

**Interfaces:**
- Produces: a building site; `docs/api/<module>.md` pages contain only `# <module>` and `::: pkpdutils.<module>`; later plans add pages and `nav` entries.

- [ ] **Step 1: Write `zensical.toml`**

```toml
[project]
site_name = "pkpdutils"
site_url = "https://matthiaskoenig.github.io/pkpdutils/"
site_description = "Pharmacokinetic and pharmacodynamic analysis of timecourses and parameters: non-compartmental analysis, curve fitting, uncertainty, significance tests, bioequivalence, drug-drug interactions and meta-analysis"
site_author = "Matthias König"

copyright = 'Copyright &copy; 2018-2026 <a href="https://livermetabolism.com">Matthias König</a>'

repo_url = "https://github.com/matthiaskoenig/pkpdutils"
repo_name = "matthiaskoenig/pkpdutils"
edit_uri = "edit/develop/docs/"

nav = [
  { "Home" = "index.md" },
  { "Installation" = "installation.md" },
  { "User guide" = [
    { "Units" = "units.md" },
    { "Timecourses" = "timecourses.md" },
  ] },
  { "References" = "references.md" },
  { "API reference" = [
    { "Overview" = "api/index.md" },
    { "pkpdutils" = [
      { "units" = "api/units.md" },
      { "timecourse" = "api/timecourse.md" },
      { "console" = "api/console.md" },
      { "log" = "api/log.md" },
    ] },
  ] },
  { "Contributing" = "development.md" },
]

[project.theme]
language = "en"
features = [
  "content.action.edit",
  "content.code.annotate",
  "content.code.copy",
  "content.tooltips",
  "navigation.footer",
  "navigation.indexes",
  "navigation.instant",
  "navigation.instant.prefetch",
  "navigation.sections",
  "navigation.top",
  "navigation.tracking",
  "search.highlight",
  "search.suggest",
  "toc.follow",
]

[[project.theme.palette]]
media = "(prefers-color-scheme)"
toggle.icon = "lucide/sun-moon"
toggle.name = "Switch to light mode"

[[project.theme.palette]]
media = "(prefers-color-scheme: light)"
scheme = "default"
primary = "teal"
accent = "teal"
toggle.icon = "lucide/sun"
toggle.name = "Switch to dark mode"

[[project.theme.palette]]
media = "(prefers-color-scheme: dark)"
scheme = "slate"
primary = "teal"
accent = "teal"
toggle.icon = "lucide/moon"
toggle.name = "Switch to system preference"

[[project.extra.social]]
icon = "fontawesome/brands/github"
link = "https://github.com/matthiaskoenig/pkpdutils"

[[project.extra.social]]
icon = "fontawesome/brands/python"
link = "https://pypi.org/project/pkpdutils/"

# API reference is rendered from the docstrings, see the `docs/api` pages
[project.plugins.mkdocstrings]
default_handler = "python"

[project.plugins.mkdocstrings.handlers.python]
paths = ["src"]

[project.plugins.mkdocstrings.handlers.python.options]
docstring_style = "google"
docstring_section_style = "table"
show_source = false
show_root_heading = false
show_root_toc_entry = false
show_root_full_path = false
show_symbol_type_heading = true
show_symbol_type_toc = true
members_order = "source"
separate_signature = true
signature_crossrefs = true
merge_init_into_class = true
filters = ["!^_"]

[project.markdown_extensions]
abbr = {}
admonition = {}
attr_list = {}
def_list = {}
footnotes = {}
md_in_html = {}
tables = {}
toc.permalink = true
pymdownx.arithmatex.generic = true
pymdownx.betterem = {}
pymdownx.caret = {}
pymdownx.details = {}
pymdownx.emoji.emoji_generator = "zensical.extensions.emoji.to_svg"
pymdownx.emoji.emoji_index = "zensical.extensions.emoji.twemoji"
pymdownx.highlight.anchor_linenums = true
pymdownx.highlight.line_spans = "__span"
pymdownx.highlight.pygments_lang_class = true
pymdownx.inlinehilite = {}
pymdownx.keys = {}
pymdownx.magiclink = {}
pymdownx.mark = {}
pymdownx.smartsymbols = {}
pymdownx.superfences = {}
pymdownx.tabbed.alternate_style = true
pymdownx.tasklist.custom_checkbox = true
pymdownx.tilde = {}

[project.extra_javascript]
# MathJax for the formulas of the user guide (pymdownx.arithmatex)
paths = [
  "javascripts/mathjax.js",
  "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js",
]
```

If `zensical build` rejects `[project.extra_javascript]` with `paths`, use the list form `extra_javascript = ["javascripts/mathjax.js", "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"]` under `[project]` instead (check `uv run zensical build --clean` output).

Create `docs/javascripts/mathjax.js`:
```javascript
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
```

The `nav` lists `units.md`, `timecourses.md` and `api/timecourse.md`, which Task 12 creates; until then Zensical warns about missing pages but builds. If the build fails on the missing pages instead, remove the three entries now and add them back in Task 12.

- [ ] **Step 2: Write the docs pages**

`docs/index.md`:
```markdown
# pkpdutils: pharmacokinetic and pharmacodynamic analysis
[![GitHub Actions CI/CD Status](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml) [![Documentation](https://img.shields.io/badge/docs-pkpdutils-3f51b5.svg)](https://matthiaskoenig.github.io/pkpdutils) [![Version](https://img.shields.io/pypi/v/pkpdutils.svg)](https://pypi.org/project/pkpdutils/) [![Python Versions](https://img.shields.io/pypi/pyversions/pkpdutils.svg)](https://pypi.org/project/pkpdutils/) [![MIT License](https://img.shields.io/pypi/l/pkpdutils.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

`pkpdutils` is a python library for the pharmacokinetic (PK) and pharmacodynamic (PD) analysis of timecourses and parameters. The source code is available from [https://github.com/matthiaskoenig/pkpdutils](https://github.com/matthiaskoenig/pkpdutils). The package was formerly published as `pkdb-analysis`, the analysis toolbox of [PK-DB](https://pk-db.com); version 1.0.0 is a rewrite without any PK-DB dependency.

## Background

A pharmacokinetic study measures the concentration of a substance over time; the parameters which describe such a curve, the exposure `AUC`, the peak `Cmax`, the half-life, the clearance and the volume of distribution, are what studies report, compare and pool. `pkpdutils` computes these parameters from timecourses without a model of the body (non-compartmental analysis), fits the curves and the parameters which need a model of the curve (exponentials, Emax, dose proportionality, covariates), propagates the uncertainty of group data, and provides the statistics used on the parameters: significance tests, bioequivalence, the classification of drug–drug interactions and meta-analysis.

All data structures are [xarray](https://xarray.dev) datasets with [pint](https://pint.readthedocs.io) units, so many timecourses, e.g. all individuals of a study or all runs of a simulation scan, are analysed in one vectorized call.

## Features

- **[Timecourses](timecourses.md)** — `Timecourse` for one curve, `Timecourses` for many, with doses, routes, uncertainties and metadata.
- **[Units](units.md)** — every timecourse and result carries its units, parameters are derived in the units of the input.

The methods behind the package are cited in [References](references.md).

## How to cite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

If you use `pkpdutils` please cite the archived software on [Zenodo](https://doi.org/10.5281/zenodo.3997539):

> König, M. & Grzegorzewski, J. (2026). *pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3997539

## License

- Source Code: [MIT](https://opensource.org/license/MIT)
- Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

## Funding

Matthias König is supported by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 "QuaLiPerF (Quantifying Liver Perfusion-Function Relationship in Complex Resection — A Systems Medicine Approach)" by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA).

Matthias König was supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054).
```

`docs/installation.md`:
````markdown
# Installation

`pkpdutils` requires python >= 3.13 and is available from [pypi](https://pypi.python.org/pypi/pkpdutils). It is tested on Linux, macOS and Windows and is pure python; every dependency ships binary wheels, so no compiler is needed.

## With uv

[uv](https://docs.astral.sh/uv/) is the recommended way to install the package. In a project it is added as a dependency:

```bash
uv add pkpdutils
```

Into an existing virtual environment it is installed through the pip interface of uv:

```bash
uv venv --python 3.14
uv pip install pkpdutils
```

## With pip

```bash
pip install pkpdutils
```

## Development version

The current state of the `develop` branch is installed directly from GitHub:

```bash
uv add "pkpdutils @ git+https://github.com/matthiaskoenig/pkpdutils.git@develop"
```

or, with pip,

```bash
pip install git+https://github.com/matthiaskoenig/pkpdutils.git@develop
```

To work on the repository itself, with the test and documentation tooling, see [Development](development.md).

## Dependencies

| package | used for |
| --- | --- |
| [numpy](https://numpy.org), [scipy](https://scipy.org) | numerics, integration, regression, optimization and statistics |
| [xarray](https://xarray.dev), [pandas](https://pandas.pydata.org) | timecourses and results as labeled arrays and tables |
| [pint](https://pint.readthedocs.io) | units and unit conversions |
| [pydantic](https://docs.pydantic.dev) | validated data structures and options |
| [matplotlib](https://matplotlib.org) | figures |
| [rich](https://rich.readthedocs.io) | console output of scripts and examples |

## Logging

`pkpdutils` does not configure logging. It logs to loggers below the `pkpdutils` logger and leaves handlers, levels and formatting to the application:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("pkpdutils").setLevel(logging.WARNING)
```

For scripts and interactive work the rich output of the package can be turned on explicitly:

```python
from pkpdutils import log

log.enable_rich_logging()
```
````

`docs/development.md`: copy `/home/mkoenig/git/sbmlsim/docs/development.md`, then

```bash
cp /home/mkoenig/git/sbmlsim/docs/development.md docs/development.md
sed -i 's/sbmlsim/pkpdutils/g' docs/development.md
```

and edit by hand: in the "Testing" section replace the example paths `tests/simulation/test_simulation.py` and `tests/simulation/test_simulation.py::test_timecourse` by `tests/test_units.py` and `tests/test_units.py::test_parse_unit`; delete the paragraph starting "Some tests are skipped on purpose"; in "Linting and formatting" delete the sentence about `sbmlutils.factory` and `F403`/`F405`; in "Type checking" delete the paragraph starting "libsbml, libsedml and roadrunner have no type stubs"; in "Examples" replace `python -m examples.timecourse` and `python -m examples.demo.demo` by `python -m examples.timecourses`; in "Documentation" replace `# simulation.timecourse` / `::: pkpdutils.simulation.timecourse` by `# timecourse` / `::: pkpdutils.timecourse` and `/creation.md` for `/creation/` by `/nca.md` for `/nca/`; in "Release" step 8 keep `uv venv --python 3.14` and `uv pip install pkpdutils`. The table of the required checks stays as it is (`tests`, `ruff`, `ty`, `docs`).

`docs/references.md`:
```markdown
# References

`pkpdutils` implements the standard methods of pharmacokinetic data analysis. These are the textbooks, guidances and publications behind them; cite them when you report an analysis, and cite `pkpdutils` itself as described in [Home](index.md#how-to-cite). Every user guide page cites the entries it builds on.

## Textbooks

**Gabrielsson & Weiner.** The reference for the non-compartmental parameters and their interpretation.

> Gabrielsson J, Weiner D.
> **Pharmacokinetic and Pharmacodynamic Data Analysis: Concepts and Applications.**
> 5th edition. Swedish Pharmaceutical Press; 2016.

**Rowland & Tozer.** Clinical pharmacokinetics, the physiological meaning of clearance, volume and half-life.

> Rowland M, Tozer TN.
> **Clinical Pharmacokinetics and Pharmacodynamics: Concepts and Applications.**
> 4th edition. Lippincott Williams & Wilkins; 2011.

**Gibaldi & Perrier.** The derivations of the area and moment methods.

> Gibaldi M, Perrier D.
> **Pharmacokinetics.**
> 2nd edition. Marcel Dekker; 1982.

## Non-compartmental analysis

**Phoenix WinNonlin.** The rules for the selection of the terminal phase (best fit by adjusted R²) and the linear-up/log-down trapezoidal rule follow the Phoenix NCA implementation.

> Certara.
> **Phoenix WinNonlin User's Guide: Noncompartmental Analysis.**
> Certara USA, Inc.

## Regulatory guidance

**FDA drug interaction guidance.** The thresholds of the classification of inhibitors, inducers and sensitive substrates.

> U.S. Food and Drug Administration.
> **Clinical Drug Interaction Studies — Cytochrome P450 Enzyme- and Transporter-Mediated Drug Interactions. Guidance for Industry.**
> 2020.

**EMA drug interaction guideline.**

> European Medicines Agency.
> **Guideline on the investigation of drug interactions.** CPMP/EWP/560/95/Rev. 1.
> 2012.

**FDA bioequivalence guidance.** The 80–125 % acceptance range of the 90 % confidence interval of the geometric mean ratio.

> U.S. Food and Drug Administration.
> **Statistical Approaches to Establishing Bioequivalence. Guidance for Industry.**
> 2001.

## Statistics

**Two one-sided tests.**

> Schuirmann DJ.
> **A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability.**
> *Journal of Pharmacokinetics and Biopharmaceutics.* 1987;15(6):657-680.
> [doi:10.1007/BF01068419](https://doi.org/10.1007/BF01068419)

**Dose proportionality.**

> Smith BP, Vandenhende FR, DeSante KA, Farid NA, Welch PA, Callaghan JT, Forgue ST.
> **Confidence interval criteria for assessment of dose proportionality.**
> *Pharmaceutical Research.* 2000;17(10):1278-1283.
> [doi:10.1023/A:1026451721686](https://doi.org/10.1023/A:1026451721686)

**Effect sizes.**

> Hedges LV.
> **Distribution theory for Glass's estimator of effect size and related estimators.**
> *Journal of Educational Statistics.* 1981;6(2):107-128.
> [doi:10.3102/10769986006002107](https://doi.org/10.3102/10769986006002107)

**Random effects meta-analysis.**

> DerSimonian R, Laird N.
> **Meta-analysis in clinical trials.**
> *Controlled Clinical Trials.* 1986;7(3):177-188.
> [doi:10.1016/0197-2456(86)90046-2](https://doi.org/10.1016/0197-2456(86)90046-2)

**Bootstrap.**

> Efron B, Tibshirani RJ.
> **An Introduction to the Bootstrap.**
> Chapman & Hall/CRC; 1993.

## Software

**pint, xarray, scipy.** The libraries the package is built on.

> Hoyer S, Hamman J.
> **xarray: N-D labeled arrays and datasets in Python.**
> *Journal of Open Research Software.* 2017;5(1):10.
> [doi:10.5334/jors.148](https://doi.org/10.5334/jors.148)

> Virtanen P, Gommers R, Oliphant TE, et al.
> **SciPy 1.0: fundamental algorithms for scientific computing in Python.**
> *Nature Methods.* 2020;17:261-272.
> [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
```

`docs/robots.txt`:
```
# The documentation is available as markdown for agents and language models:
#   https://matthiaskoenig.github.io/pkpdutils/llms.txt      index of all pages
#   https://matthiaskoenig.github.io/pkpdutils/llms-full.txt the complete documentation
# Every page is served as markdown next to its html, e.g. /nca.md for /nca/.

User-agent: *
Allow: /

Sitemap: https://matthiaskoenig.github.io/pkpdutils/sitemap.xml
```

`docs/api/index.md`:
```markdown
# API reference

The API reference is generated from the docstrings of the package.

## pkpdutils

| module | description |
| --- | --- |
| [units](units.md) | the unit registry of the package and unit helpers |
| [timecourse](timecourse.md) | `Timecourse`, `Timecourses`, `Dose`, `Route` and `DosingRegimen`, the data model |
| [console](console.md) | shared rich console |
| [log](log.md) | logging of the package |
```

`docs/api/units.md`:
```markdown
# units

::: pkpdutils.units
```

`docs/api/console.md`:
```markdown
# console

::: pkpdutils.console
```

`docs/api/log.md`:
```markdown
# log

::: pkpdutils.log
```

`docs/images/.gitkeep`: empty.

- [ ] **Step 3: Copy the llms script**

```bash
mkdir -p scripts
cp /home/mkoenig/git/sbmlsim/scripts/llms_txt.py scripts/llms_txt.py
sed -i 's/sbmlsim/pkpdutils/g' scripts/llms_txt.py
```

- [ ] **Step 4: Build the site**

Run: `uv run zensical build --clean && uv run python scripts/llms_txt.py && ls site/llms.txt site/llms-full.txt site/api/units/index.html`
Expected: build succeeds (warnings about the not yet existing `units.md`, `timecourses.md`, `api/timecourse.md` are acceptable, an error is not, see Step 1), the three files exist.

- [ ] **Step 5: Lint**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: clean (`scripts/` is type checked).

- [ ] **Step 6: Commit**

```bash
git add zensical.toml docs scripts
git commit -q -m "Add the documentation skeleton: Zensical site, API reference pages and agent files

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 7: README, CLAUDE.md, examples package and example runner

**Files:**
- Create: `README.md` (overwrite), `CLAUDE.md`, `examples/__init__.py`, `examples/README.md`, `tests/examples/__init__.py`, `tests/examples/test_examples.py`

**Interfaces:**
- Produces: `tests/examples/test_examples.py::SCRIPTS`, the list of example modules run by the test suite; Task 12 appends `"examples.timecourses"`.

- [ ] **Step 1: Write `README.md`**

````markdown
# pkpdutils: pharmacokinetic and pharmacodynamic analysis
[![GitHub Actions CI/CD Status](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/matthiaskoenig/pkpdutils/actions/workflows/ci-cd.yml)
[![Documentation](https://img.shields.io/badge/docs-pkpdutils-3f51b5.svg)](https://matthiaskoenig.github.io/pkpdutils)
[![Version](https://img.shields.io/pypi/v/pkpdutils.svg)](https://pypi.org/project/pkpdutils/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pkpdutils.svg)](https://pypi.org/project/pkpdutils/)
[![MIT License](https://img.shields.io/pypi/l/pkpdutils.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

`pkpdutils` is a python library for the pharmacokinetic (PK) and pharmacodynamic (PD) analysis of timecourses and parameters. It was formerly published as `pkdb-analysis`, the analysis toolbox of [PK-DB](https://pk-db.com); version 1.0.0 is a rewrite without any PK-DB dependency.

Features include

- **non-compartmental analysis** — exposure, peak, terminal phase, clearance and volume parameters of concentration and effect timecourses, single dose and steady state, with units
- **uncertainty** — bootstrap and delta method propagation for group timecourses (mean ± SD), summary statistics over individuals
- **curve fitting** — exponential, Bateman, Emax, dose proportionality and covariate models with standard errors, confidence intervals and model comparison
- **statistics on parameters** — significance tests, geometric mean ratios, bioequivalence, classification of drug–drug interactions, meta-analysis
- **figures** — timecourses, NCA diagnostics, fits, parameter distributions, forest and ratio plots

All data structures are [xarray](https://xarray.dev) datasets with [pint](https://pint.readthedocs.io) units, so many timecourses are analysed in one vectorized call.

The documentation is available at [https://matthiaskoenig.github.io/pkpdutils](https://matthiaskoenig.github.io/pkpdutils).

If you have any questions or issues please [open an issue](https://github.com/matthiaskoenig/pkpdutils/issues).

## How to cite
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3997539.svg)](https://doi.org/10.5281/zenodo.3997539)

If you use `pkpdutils` please cite the archived software on [Zenodo](https://doi.org/10.5281/zenodo.3997539):

> König, M. & Grzegorzewski, J. (2026). *pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3997539

## Installation

`pkpdutils` requires python >= 3.13 and is available from [pypi](https://pypi.python.org/pypi/pkpdutils):

```bash
uv add pkpdutils
```

or with pip

```bash
pip install pkpdutils
```

See [Installation](https://matthiaskoenig.github.io/pkpdutils/installation/) for details and [Development](https://matthiaskoenig.github.io/pkpdutils/development/) for working on the repository.

## License

- Source Code: [MIT](https://opensource.org/license/MIT)
- Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

## Funding

Matthias König is supported by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 "QuaLiPerF (Quantifying Liver Perfusion-Function Relationship in Complex Resection — A Systems Medicine Approach)" by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA).

Matthias König was supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054).

© 2018-2026 Matthias König & Jan Grzegorzewski.
````

- [ ] **Step 2: Write `CLAUDE.md`**

````markdown
# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project

`pkpdutils` is a python library for the pharmacokinetic and pharmacodynamic analysis of timecourses and parameters: non-compartmental analysis, uncertainty propagation, curve fitting, statistics on parameters (tests, bioequivalence, drug–drug interactions, meta-analysis) and figures. Pure library, no CLI entry points. Requires python >= 3.13, packaged with hatchling (version is read from `src/pkpdutils/__init__.py`). Runtime dependencies are `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib` and `rich`. It was `pkdb_analysis` until 0.3.1; the design of the rewrite is in `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`.

## Commands

```bash
# environment (uv based)
uv sync --extra dev
uv run pre-commit install

# tests
pytest                                   # all tests, in parallel with pytest-xdist
pytest -n 0                              # in one process, e.g. for --pdb
pytest tests/test_units.py               # single file
pytest tests/test_units.py::test_parse_unit  # single test
tox r -e py3.14                          # single tox env (py3.13, py3.14 available)
tox run-parallel                         # full matrix + ty

# lint / format / types
ruff check
ruff format
tox -e ty                       # ty type check (config in [tool.ty] in pyproject.toml)
uvx ty check                    # same check, straight from the working tree

# docs
uv run zensical build --clean
uv run python scripts/llms_txt.py
uv run zensical serve

# examples, they are modules of the `examples` package and not part of pkpdutils
python -m examples.timecourses
```

`develop` is the default branch and takes every change through a pull request; direct pushes are rejected by the rulesets in `.github/rulesets/` (applied with `.github/rulesets/apply.sh`), which require the `tests`, `ruff`, `ty` and `docs` checks. `main` only tracks the latest release and is fast-forwarded by the `sync-main` job of the release workflow, never by hand.

Release steps are in `docs/development.md`: the release is prepared on a branch, `uvx bump-my-version bump [major|minor|patch]` updates `src/pkpdutils/__init__.py` and `CITATION.cff` and commits without tagging, and the tag is created on `develop` after the pull request was merged, which triggers the PyPI release workflow.

Documentation is [Zensical](https://zensical.org/): markdown sources in `docs/`, configured in `zensical.toml`, built into the gitignored `site/`. The API reference is rendered from the docstrings by mkdocstrings; a page in `docs/api/` is just `::: pkpdutils.<module>`. Formulas use `pymdownx.arithmatex` with MathJax. `scripts/llms_txt.py` runs after the build and writes `llms.txt`, `llms-full.txt` and the markdown of every page into `site/`.

## Architecture

**`units.py`.** One pint `UnitRegistry` per process, `ureg`; `Q_` creates quantities, `Quantity` and `Unit` are the annotation types. `check_dose_unit` validates dose units, `normalize_volume` and `normalize_clearance` convert results to `liter`/`liter/kg` and `liter/hour`/`liter/hour/kg`. Numerics run on plain arrays in the units of the input, pint is used at the boundaries.

**`timecourse.py`.** `Timecourse` is one curve (pydantic, frozen): `time`, `value`, units, optional `sd`/`se`/`n`, a `Dose` (`amount`, `unit`, `Route`, `time`, `duration`) and metadata. `Timecourses` wraps an `xarray.Dataset` with a `time` dimension plus any sample dimensions, the variables `value` and optionally `sd`, `se`, `n`, the dose as `dose_amount`/`dose_time`/`dose_duration` and units in `attrs["units"]`; `times` and `values` return the `(samples..., time)` arrays every analysis works on. `DosingRegimen` describes repeated dosing for steady state analyses.

`console.py` (rich console, for scripts and examples) and `log.py` provide the shared output/logging. Modules get their logger from the standard library with `logging.getLogger(__name__)`. The package never configures logging: `log.enable_rich_logging()` is the opt-in for scripts. Library code logs, it does not print, and log calls use lazy `%s` formatting rather than f-strings (enforced by ruff `G`).

## Conventions

- Type checking is done with [ty](https://docs.astral.sh/ty/). `[tool.ty.terminal] error-on-warning = true` means warnings fail the check, so the tree must stay at zero diagnostics. Suppress a diagnostic with a rule-specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`. ty also runs as a pre-commit hook (`--extra dev`).
- Every module, class and function of the package carries full type annotations and a google style docstring (ruff `D`); `examples/`, `tests/` and `scripts/` are exempt from the docstring rules. Docstrings of parameters and methods carry the formula and a citation key of `docs/references.md`.
- Results are `xarray.Dataset` objects with one variable per parameter over the sample dimensions of the input and `attrs["units"]` on every variable.
- `examples/` at the top level holds the runnable examples, they are not part of the package and are run as modules (`python -m examples.timecourses`). An example writes into the current working directory and never opens a window; the `conftest.py` at the root selects the `Agg` backend for the test session. `tests/examples/test_examples.py` runs them.
- Library code does not call `plt.show()`; figures are returned or saved.
- Test fixtures live in `tests/data/`; `tests/data/reference/nca_reference.json` holds the results of `pkdb_analysis` 0.3.1 on them and is the regression reference of the NCA.
- Markdown carries no hard line wraps: a paragraph, a list item or a table row is a single line. Code fences, headings and the rows of badges keep their line structure.
- Release notes go in `release-notes/` as part of a release commit.
````

- [ ] **Step 3: Write the examples package and the runner**

`examples/__init__.py`:
```python
"""Runnable examples of pkpdutils, see README.md."""
```

`examples/README.md`:
````markdown
# Examples

Runnable examples for pkpdutils. They are **not** part of the package: they are not installed with `pip install pkpdutils`, they are read and run from a checkout of the repository.

Every example is a module of the `examples` package, so it is run from the root of the repository with

```bash
python -m examples.timecourses
```

An example writes what it creates into the current working directory. No example opens a window: matplotlib figures are saved to a file, never shown, so that the examples also run on a machine without a display.

## What is where

| path | content |
| --- | --- |
| `examples/timecourses.py` | creating `Timecourse` and `Timecourses` objects from arrays, data frames and a simulation-like dataset |
````

`tests/examples/__init__.py`: empty.

`tests/examples/test_examples.py`:
```python
"""Run the example scripts.

Every example is run as a module in a temporary working directory, so the
files it writes do not end up in the repository. An example which breaks
fails the test suite.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: the root of the repository, `python -m examples.<module>` is run from here
REPO_DIR = Path(__file__).parent.parent.parent

#: examples which run offline and without optional dependencies
SCRIPTS: list[str] = []


@pytest.mark.parametrize("module", SCRIPTS)
def test_example_script(module: str, tmp_path: Path) -> None:
    """Every example runs without an error and writes into the working directory."""
    env = dict(os.environ, PYTHONPATH=str(REPO_DIR), MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 4: Run the checks**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: tests pass (the parametrized example test is skipped as empty), ruff and ty clean.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md examples tests/examples
git commit -q -m "Add README, CLAUDE.md and the examples package with its test runner

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 8: `Route`, `Dose` and `DosingRegimen`

**Files:**
- Create: `src/pkpdutils/timecourse.py`
- Test: `tests/test_timecourse.py`

**Interfaces:**
- Produces:
  - `class Route(StrEnum)`: `IV_BOLUS = "iv_bolus"`, `IV_INFUSION = "iv_infusion"`, `ORAL = "oral"`; property `is_iv: bool`
  - `class Dose(BaseModel, frozen=True)`: `amount: float`, `unit: str`, `route: Route = Route.ORAL`, `time: float = 0.0`, `duration: float | None = None`; validation: `check_dose_unit(unit)`, `amount >= 0`, `duration` required and `> 0` for `IV_INFUSION`, forbidden otherwise; properties `quantity: Quantity`, `per_bodyweight: bool`
  - `class DosingRegimen(BaseModel, frozen=True)`: `dose: Dose`, `interval: float`, `n_doses: int | None = None`; `interval > 0`, `n_doses >= 1`; method `dose_times(time_unit_of_interval_is_the_timecourse_unit) -> np.ndarray` returns `dose.time + k*interval` for `k in range(n_doses)` (`n_doses` required for this call, `ValueError` otherwise)

- [ ] **Step 1: Write the failing tests**

`tests/test_timecourse.py` (start of the file; later tasks append):
```python
import numpy as np
import pytest

from pkpdutils.timecourse import Dose, DosingRegimen, Route


def test_route() -> None:
    assert Route.IV_BOLUS.is_iv
    assert Route.IV_INFUSION.is_iv
    assert not Route.ORAL.is_iv
    assert Route("oral") is Route.ORAL


def test_dose_defaults() -> None:
    dose = Dose(amount=100, unit="mg")
    assert dose.route is Route.ORAL
    assert dose.time == 0.0
    assert dose.duration is None
    assert dose.quantity.magnitude == 100
    assert str(dose.quantity.units) == "milligram"
    assert not dose.per_bodyweight


def test_dose_per_bodyweight() -> None:
    assert Dose(amount=2, unit="mg/kg").per_bodyweight


def test_dose_invalid_unit() -> None:
    with pytest.raises(ValueError, match="dose"):
        Dose(amount=1, unit="mg/l")


def test_dose_negative_amount() -> None:
    with pytest.raises(ValueError):
        Dose(amount=-1, unit="mg")


def test_dose_infusion_requires_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        Dose(amount=1, unit="mg", route=Route.IV_INFUSION)
    dose = Dose(amount=1, unit="mg", route=Route.IV_INFUSION, duration=0.5)
    assert dose.duration == 0.5


def test_dose_duration_only_for_infusion() -> None:
    with pytest.raises(ValueError, match="duration"):
        Dose(amount=1, unit="mg", route=Route.ORAL, duration=0.5)


def test_dose_is_frozen() -> None:
    dose = Dose(amount=1, unit="mg")
    with pytest.raises(ValueError):
        dose.amount = 2  # ty: ignore[invalid-assignment]


def test_dosing_regimen() -> None:
    regimen = DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12, n_doses=3)
    np.testing.assert_allclose(regimen.dose_times(), [0.0, 12.0, 24.0])


def test_dosing_regimen_validation() -> None:
    with pytest.raises(ValueError):
        DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=0)
    with pytest.raises(ValueError):
        DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12, n_doses=0)
    with pytest.raises(ValueError, match="n_doses"):
        DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12).dose_times()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_timecourse.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.timecourse'`.

- [ ] **Step 3: Write the first part of `src/pkpdutils/timecourse.py`**

```python
"""Timecourses, doses and dosing regimens.

The data model of the package:

- `Timecourse` is one curve, i.e. values over time with units, an optional
  uncertainty (`sd`/`se` and `n` for group data), a `Dose` and metadata.
- `Timecourses` is a batch of curves as an `xarray.Dataset` with a `time`
  dimension and any number of sample dimensions (individuals, groups, studies,
  the dimensions of a simulation scan). Every analysis of the package works on
  a `Timecourses` object and returns an `xarray.Dataset` over the same sample
  dimensions.
- `DosingRegimen` describes repeated dosing for steady state analyses.

```python
from pkpdutils.timecourse import Dose, Route, Timecourse

tc = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12],
    value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2],
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
```
"""

import logging
from enum import StrEnum
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.units import Q_, Quantity, check_dose_unit, is_per_bodyweight, unit_str

logger = logging.getLogger(__name__)


class Route(StrEnum):
    """Route of administration.

    `ORAL` stands for every extravascular route (oral, subcutaneous,
    intramuscular, ...): the substance has an absorption phase and the
    parameters which need the fraction absorbed are reported relative to it
    (`cl_f`, `vz_f`).
    """

    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    ORAL = "oral"

    @property
    def is_iv(self) -> bool:
        """Whether the route is intravenous (bolus or infusion)."""
        return self in (Route.IV_BOLUS, Route.IV_INFUSION)


class Dose(BaseModel):
    """A dose of the substance of a timecourse.

    Attributes:
        amount: amount of the dose (non-negative)
        unit: unit of the amount, an amount (`mg`, `mmol`) or an amount per body
            weight (`mg/kg`, `µmol/kg`), see `pkpdutils.units.check_dose_unit`
        route: route of administration
        time: time of the dose in the time unit of the timecourse
        duration: duration of the infusion in the time unit of the timecourse;
            required for `Route.IV_INFUSION`, not allowed otherwise
    """

    model_config = ConfigDict(frozen=True)

    amount: float = Field(ge=0)
    unit: str
    route: Route = Route.ORAL
    time: float = 0.0
    duration: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        check_dose_unit(self.unit)
        if self.route is Route.IV_INFUSION:
            if self.duration is None or self.duration <= 0:
                raise ValueError("An infusion needs a positive 'duration'")
        elif self.duration is not None:
            raise ValueError("'duration' is only allowed for Route.IV_INFUSION")
        return self

    @property
    def quantity(self) -> Quantity:
        """The dose as a quantity."""
        return Q_(self.amount, self.unit)

    @property
    def per_bodyweight(self) -> bool:
        """Whether the dose is an amount per body weight."""
        return is_per_bodyweight(self.unit)


class DosingRegimen(BaseModel):
    """Repeated administration of the same dose at a fixed interval.

    Attributes:
        dose: the dose given at every administration; its `time` is the time of
            the first dose
        interval: dosing interval `tau` in the time unit of the timecourse
        n_doses: number of doses, `None` for an unspecified number (steady
            state analyses only need `tau`)
    """

    model_config = ConfigDict(frozen=True)

    dose: Dose
    interval: float = Field(gt=0)
    n_doses: int | None = Field(default=None, ge=1)

    def dose_times(self) -> np.ndarray:
        """Times of the administrations, `dose.time + k * interval`.

        Raises:
            ValueError: if `n_doses` is `None`.
        """
        if self.n_doses is None:
            raise ValueError("'n_doses' is required for the dose times")
        return self.dose.time + self.interval * np.arange(self.n_doses, dtype=float)
```

(`Any`, `unit_str`, `logger` are used by Task 9; ruff flags them as unused until then, so add them in Task 9 instead if the lint check of this task must be clean: remove `Any` and `unit_str` from the imports and the `logger` line now, and add them back in Task 9.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_timecourse.py -n 0 -q`
Expected: all pass. If the frozen assignment raises `pydantic.ValidationError`, that is a `ValueError` subclass and the test passes.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/timecourse.py tests/test_timecourse.py
git commit -q -m "Add Route, Dose and DosingRegimen

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 9: `Timecourse`

**Files:**
- Modify: `src/pkpdutils/timecourse.py` (append)
- Test: `tests/test_timecourse.py` (append)

**Interfaces:**
- Produces: `class Timecourse(BaseModel, frozen=True)` with fields `time: np.ndarray`, `value: np.ndarray`, `time_unit: str`, `unit: str`, `sd: np.ndarray | None = None`, `se: np.ndarray | None = None`, `n: float | np.ndarray | None = None`, `dose: Dose | None = None`, `substance: str = "substance"`, `label: str | None = None`, `tissue: str | None = None`; properties `time_q`, `value_q`, `sd_q`, `se_q` (quantities, `None` when missing), `size: int`; methods `relative_to_dose() -> Timecourse`, `to_dataframe() -> pandas.DataFrame` (columns `time`, `value`, and `sd`, `se`, `n` when present), classmethod `from_dataframe(df, time="time", value="value", time_unit, unit, sd=None, se=None, n=None, **fields) -> Timecourse`.
- Validation: arrays converted to `float64`; `time` and `value` same length, at least 2 points; unsorted `time` is sorted with all arrays (logged warning), duplicate times raise `ValueError`; `NaN` in `time` raises; `sd`/`se` same length as `value`; `se` derived from `sd` and `n` (and `sd` from `se` and `n`) when missing; `n` scalar or array of the length of `value`; units parsed with `parse_unit` (a `ValueError` for unknown units).

- [ ] **Step 1: Append the failing tests**

Add `import pandas as pd` to the imports at the top of `tests/test_timecourse.py` and change the import line to `from pkpdutils.timecourse import Dose, DosingRegimen, Route, Timecourse` (ruff `E402` forbids imports in the middle of the file). Then append:
```python


def test_timecourse_basic() -> None:
    tc = Timecourse(
        time=[0, 1, 2, 4],
        value=[0.0, 2.0, 1.5, 0.5],
        time_unit="hr",
        unit="mg/l",
        substance="caffeine",
    )
    assert tc.size == 4
    assert tc.time.dtype == np.float64
    assert tc.value.dtype == np.float64
    assert tc.dose is None
    assert tc.sd is None and tc.se is None and tc.n is None
    assert str(tc.time_q.units) == "hour"
    assert str(tc.value_q.units) == "milligram / liter"
    assert tc.sd_q is None


def test_timecourse_sorts_unsorted_time() -> None:
    tc = Timecourse(time=[2, 0, 1], value=[1.5, 0.0, 2.0], time_unit="hr", unit="mg/l")
    np.testing.assert_allclose(tc.time, [0, 1, 2])
    np.testing.assert_allclose(tc.value, [0.0, 2.0, 1.5])


def test_timecourse_sorts_uncertainties_with_time() -> None:
    tc = Timecourse(
        time=[2, 0, 1], value=[1.5, 0.0, 2.0], sd=[0.3, 0.0, 0.2], n=5,
        time_unit="hr", unit="mg/l",
    )
    np.testing.assert_allclose(tc.sd, [0.0, 0.2, 0.3])


def test_timecourse_duplicate_time() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Timecourse(time=[0, 1, 1], value=[0, 1, 2], time_unit="hr", unit="mg/l")


def test_timecourse_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        Timecourse(time=[0, 1, 2], value=[0, 1], time_unit="hr", unit="mg/l")
    with pytest.raises(ValueError, match="length"):
        Timecourse(time=[0, 1, 2], value=[0, 1, 2], sd=[0, 1], time_unit="hr", unit="mg/l")


def test_timecourse_too_short() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        Timecourse(time=[0], value=[1], time_unit="hr", unit="mg/l")


def test_timecourse_nan_time() -> None:
    with pytest.raises(ValueError, match="NaN"):
        Timecourse(time=[0, np.nan, 2], value=[0, 1, 2], time_unit="hr", unit="mg/l")


def test_timecourse_nan_value_allowed() -> None:
    tc = Timecourse(time=[0, 1, 2], value=[0, np.nan, 2], time_unit="hr", unit="mg/l")
    assert np.isnan(tc.value[1])


def test_timecourse_invalid_unit() -> None:
    with pytest.raises(ValueError, match="not a unit"):
        Timecourse(time=[0, 1], value=[0, 1], time_unit="hr", unit="mg/foo")


def test_timecourse_derives_se_from_sd() -> None:
    tc = Timecourse(time=[0, 1], value=[1, 2], sd=[2.0, 4.0], n=4, time_unit="hr", unit="mg/l")
    np.testing.assert_allclose(tc.se, [1.0, 2.0])


def test_timecourse_derives_sd_from_se() -> None:
    tc = Timecourse(time=[0, 1], value=[1, 2], se=[1.0, 2.0], n=[4, 9], time_unit="hr", unit="mg/l")
    np.testing.assert_allclose(tc.sd, [2.0, 6.0])
    np.testing.assert_allclose(tc.n, [4, 9])


def test_timecourse_sd_without_n_keeps_se_none() -> None:
    tc = Timecourse(time=[0, 1], value=[1, 2], sd=[2.0, 4.0], time_unit="hr", unit="mg/l")
    assert tc.se is None
    assert tc.n is None


def test_timecourse_n_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        Timecourse(time=[0, 1], value=[1, 2], n=[4, 9, 1], time_unit="hr", unit="mg/l")


def test_timecourse_relative_to_dose() -> None:
    tc = Timecourse(
        time=[10, 11, 12], value=[0, 2, 1], time_unit="hr", unit="mg/l",
        dose=Dose(amount=1, unit="mg", time=10),
    )
    rel = tc.relative_to_dose()
    np.testing.assert_allclose(rel.time, [0, 1, 2])
    assert rel.dose is not None and rel.dose.time == 0.0
    np.testing.assert_allclose(tc.time, [10, 11, 12])


def test_timecourse_relative_to_dose_without_dose() -> None:
    tc = Timecourse(time=[10, 11], value=[0, 2], time_unit="hr", unit="mg/l")
    assert tc.relative_to_dose() is tc


def test_timecourse_dataframe_roundtrip() -> None:
    tc = Timecourse(
        time=[0, 1, 2], value=[0, 2, 1], sd=[0, 0.2, 0.1], n=5,
        time_unit="hr", unit="mg/l", substance="caffeine", label="a",
    )
    df = tc.to_dataframe()
    assert list(df.columns) == ["time", "value", "sd", "se", "n"]
    tc2 = Timecourse.from_dataframe(df, time_unit="hr", unit="mg/l", sd="sd", n="n", substance="caffeine")
    np.testing.assert_allclose(tc2.value, tc.value)
    np.testing.assert_allclose(tc2.sd, tc.sd)
    np.testing.assert_allclose(tc2.n, 5)


def test_timecourse_from_dataframe_columns() -> None:
    df = pd.DataFrame({"t": [0, 1, 2], "c": [0, 2, 1]})
    tc = Timecourse.from_dataframe(df, time="t", value="c", time_unit="min", unit="ng/ml")
    np.testing.assert_allclose(tc.time, [0, 1, 2])
    assert str(tc.time_q.units) == "minute"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_timecourse.py -n 0 -q`
Expected: FAIL with `ImportError: cannot import name 'Timecourse'`.

- [ ] **Step 3: Append `Timecourse` to `src/pkpdutils/timecourse.py`**

Add to the imports at the top of the module: `from typing import Any, Self`, `import pandas as pd`, and `from pkpdutils.units import Q_, Quantity, check_dose_unit, is_per_bodyweight, parse_unit`; add `logger = logging.getLogger(__name__)` below the imports if not present. Then append:

```python
def _as_float_array(name: str, values: Any) -> np.ndarray:
    """Convert to a 1-D float64 array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"'{name}' must be one dimensional, not {array.ndim}-D")
    return array


class Timecourse(BaseModel):
    """One curve of values over time with units, uncertainty, dose and metadata.

    Concentration timecourses of a substance and effect timecourses of a
    pharmacodynamic response use the same class; `value` is the generic name.
    A group timecourse (mean of several subjects) carries the standard
    deviation `sd` or the standard error `se` and the number of subjects `n`;
    an individual timecourse carries none of them.

    Validation converts the arrays to `float64`, sorts them by time, derives
    `se` from `sd` and `n` (or `sd` from `se` and `n`) and checks the units.

    Attributes:
        time: sampling times, strictly increasing after validation
        value: values at the sampling times, `NaN` for missing values
        time_unit: unit of `time`, e.g. `"hr"`
        unit: unit of `value`, e.g. `"ng/ml"`
        sd: standard deviation per time point (group data)
        se: standard error per time point (group data)
        n: number of subjects, one number or one per time point
        dose: the dose of the substance, `None` without dose information
        substance: name of the substance or of the effect
        label: label of the curve, e.g. the group or the individual
        tissue: tissue or matrix the values were measured in, e.g. `"plasma"`
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time: np.ndarray
    value: np.ndarray
    time_unit: str
    unit: str
    sd: np.ndarray | None = None
    se: np.ndarray | None = None
    n: float | np.ndarray | None = None
    dose: Dose | None = None
    substance: str = "substance"
    label: str | None = None
    tissue: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)

        time = _as_float_array("time", data.get("time"))
        value = _as_float_array("value", data.get("value"))
        if time.size < 2:
            raise ValueError("A timecourse needs at least 2 time points")
        if value.size != time.size:
            raise ValueError(
                f"'value' has length {value.size}, 'time' has length {time.size}"
            )
        if np.isnan(time).any():
            raise ValueError("'time' contains NaN")
        if np.unique(time).size != time.size:
            raise ValueError("'time' contains duplicate values")

        arrays: dict[str, np.ndarray | None] = {}
        for key in ("sd", "se"):
            raw = data.get(key)
            if raw is None:
                arrays[key] = None
                continue
            array = _as_float_array(key, raw)
            if array.size != time.size:
                raise ValueError(
                    f"'{key}' has length {array.size}, 'time' has length {time.size}"
                )
            arrays[key] = array

        n_raw = data.get("n")
        n: float | np.ndarray | None
        if n_raw is None:
            n = None
        elif np.ndim(n_raw) == 0:
            n = float(n_raw)
        else:
            n = _as_float_array("n", n_raw)
            if n.size != time.size:
                raise ValueError(
                    f"'n' has length {n.size}, 'time' has length {time.size}"
                )

        # derive the missing one of sd and se
        if n is not None:
            sqrt_n = np.sqrt(n)
            if arrays["se"] is None and arrays["sd"] is not None:
                arrays["se"] = arrays["sd"] / sqrt_n
            elif arrays["sd"] is None and arrays["se"] is not None:
                arrays["sd"] = arrays["se"] * sqrt_n

        # sort by time
        order = np.argsort(time)
        if not np.array_equal(order, np.arange(time.size)):
            logger.warning("Unsorted time points are sorted by time")
            time = time[order]
            value = value[order]
            for key, array in arrays.items():
                if array is not None:
                    arrays[key] = array[order]
            if isinstance(n, np.ndarray):
                n = n[order]

        data.update({"time": time, "value": value, "n": n, **arrays})
        return data

    @model_validator(mode="after")
    def _check_units(self) -> Self:
        parse_unit(self.time_unit)
        parse_unit(self.unit)
        return self

    @property
    def size(self) -> int:
        """Number of time points."""
        return int(self.time.size)

    @property
    def time_q(self) -> Quantity:
        """The times as a quantity."""
        return Q_(self.time, self.time_unit)

    @property
    def value_q(self) -> Quantity:
        """The values as a quantity."""
        return Q_(self.value, self.unit)

    @property
    def sd_q(self) -> Quantity | None:
        """The standard deviations as a quantity, `None` without `sd`."""
        return None if self.sd is None else Q_(self.sd, self.unit)

    @property
    def se_q(self) -> Quantity | None:
        """The standard errors as a quantity, `None` without `se`."""
        return None if self.se is None else Q_(self.se, self.unit)

    def relative_to_dose(self) -> "Timecourse":
        """Copy with the time shifted so that the dose is given at time 0.

        Returns the timecourse itself when it has no dose or the dose is at 0.
        """
        if self.dose is None or self.dose.time == 0.0:
            return self
        shift = self.dose.time
        return self.model_copy(
            update={
                "time": self.time - shift,
                "dose": self.dose.model_copy(update={"time": 0.0}),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """The curve as a data frame with the columns `time`, `value` and, when present, `sd`, `se`, `n`."""
        columns: dict[str, Any] = {"time": self.time, "value": self.value}
        if self.sd is not None:
            columns["sd"] = self.sd
        if self.se is not None:
            columns["se"] = self.se
        if self.n is not None:
            columns["n"] = np.broadcast_to(self.n, self.time.shape)
        return pd.DataFrame(columns)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        time_unit: str,
        unit: str,
        time: str = "time",
        value: str = "value",
        sd: str | None = None,
        se: str | None = None,
        n: str | None = None,
        **fields: Any,
    ) -> "Timecourse":
        """Create a timecourse from the columns of a data frame.

        Args:
            df: the data frame, one row per time point
            time_unit: unit of the time column
            unit: unit of the value column
            time: name of the time column
            value: name of the value column
            sd: name of the standard deviation column, `None` for none
            se: name of the standard error column, `None` for none
            n: name of the column with the number of subjects, `None` for none
            **fields: the remaining fields of `Timecourse` (`dose`, `substance`, `label`, `tissue`)

        Returns:
            The timecourse.
        """
        data: dict[str, Any] = {
            "time": df[time].to_numpy(),
            "value": df[value].to_numpy(),
            "time_unit": time_unit,
            "unit": unit,
            **fields,
        }
        if sd is not None:
            data["sd"] = df[sd].to_numpy()
        if se is not None:
            data["se"] = df[se].to_numpy()
        if n is not None:
            data["n"] = df[n].to_numpy()
        return cls(**data)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_timecourse.py -n 0 -q`
Expected: all pass. If pydantic complains about `np.ndarray` fields, `arbitrary_types_allowed=True` is missing in `model_config`. If `test_timecourse_sd_without_n_keeps_se_none` fails, the derivation branch runs without `n`; it must be inside `if n is not None`.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: clean. A ty complaint about `"Timecourse"` forward references in the return annotations is resolved by `from __future__ import annotations` at the top of the module (keep pydantic happy: it supports it).

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/timecourse.py tests/test_timecourse.py
git commit -q -m "Add Timecourse, the validated single curve

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 10: `Timecourses` batch container

**Files:**
- Modify: `src/pkpdutils/timecourse.py` (append)
- Test: `tests/test_timecourses.py`

**Interfaces:**
- Produces `class Timecourses`:
  - `TIME_DIM = "time"`; constructor `Timecourses(ds: xr.Dataset)` validates: `value` variable with a `time` dimension, `attrs["units"]` on `value` and on the `time` coordinate (or on `times`), `time_unit`/`unit` derivable
  - `from_arrays(time, values, *, time_unit, unit, dims=("individual",), coords=None, sd=None, se=None, n=None, dose=None, route=None, substance="substance") -> Timecourses`; `time` 1-D shared grid, or an array of shape `(*sample_shape, n_time)` for per sample grids; `values`, `sd`, `se` of shape `(*sample_shape, n_time)`; `n` scalar or of shape `sample_shape`; `dose: Dose | None` (one dose for all) or a dict `{"amount": array, "time": array, "duration": array | None}` of shape `sample_shape` plus `route`
  - `from_timecourses(timecourses: Sequence[Timecourse], dim="individual", labels=None) -> Timecourses`; shared grid when all `time` arrays are equal, else per sample grids padded with `NaN`; labels default to `Timecourse.label` or the index
  - properties: `ds: xr.Dataset`, `sample_dims: tuple[str, ...]`, `sample_shape: tuple[int, ...]`, `n_samples: int`, `n_time: int`, `time_unit: str`, `unit: str`, `substance: str`, `route: Route | None`, `times: np.ndarray` of shape `(*sample_shape, n_time)` (broadcast of the shared grid), `values: np.ndarray`, `sd: np.ndarray | None`, `se: np.ndarray | None`, `n: np.ndarray | None` (shape `sample_shape`), `dose_amount: np.ndarray | None`, `dose_time: np.ndarray | None`, `dose_duration: np.ndarray | None`, `dose_unit: str | None`, `has_uncertainty: bool`, `has_dose: bool`
  - `__len__` (= `n_samples`), `__iter__` yields `Timecourse` in C order of the sample dims, `sel(**indexers) -> Timecourse` (one sample by coordinate labels), `isel(**indexers) -> Timecourse`, `to_dataframe() -> pd.DataFrame` long format with the sample coordinates, `time`, `value` and the optional columns
  - dataset layout: dims `(*sample_dims, "time")`; shared grid stored as coordinate `time`; per sample grids stored as data variable `times` with `attrs["units"]` and an integer coordinate `time`; `value`, `sd`, `se` over `(*sample_dims, "time")`; `n` over `sample_dims`; `dose_amount`, `dose_time`, `dose_duration` over `sample_dims` with `attrs["units"]` (`dose_unit`, `time_unit`, `time_unit`); `ds.attrs`: `substance`, `route` (string or absent), `time_unit`, `unit`

- [ ] **Step 1: Write the failing tests**

`tests/test_timecourses.py`:
```python
import numpy as np
import pytest
import xarray as xr

from pkpdutils.timecourse import Dose, Route, Timecourse, Timecourses

T = np.array([0.0, 1.0, 2.0, 4.0])
V = np.array([[0.0, 2.0, 1.5, 0.5], [0.0, 3.0, 2.0, 1.0], [0.0, 1.0, 0.8, 0.3]])


def make_batch() -> Timecourses:
    return Timecourses.from_arrays(
        T, V, time_unit="hr", unit="mg/l", dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine",
    )


def test_from_arrays_layout() -> None:
    tcs = make_batch()
    assert tcs.sample_dims == ("individual",)
    assert tcs.sample_shape == (3,)
    assert tcs.n_samples == 3 and len(tcs) == 3
    assert tcs.n_time == 4
    assert tcs.time_unit == "hr" and tcs.unit == "mg/l"
    assert tcs.substance == "caffeine"
    assert tcs.route is Route.ORAL
    assert tcs.ds["value"].dims == ("individual", "time")
    assert tcs.ds["value"].attrs["units"] == "mg/l"
    assert tcs.ds["time"].attrs["units"] == "hr"
    np.testing.assert_allclose(tcs.times, np.broadcast_to(T, (3, 4)))
    np.testing.assert_allclose(tcs.values, V)
    assert tcs.has_dose and not tcs.has_uncertainty
    np.testing.assert_allclose(tcs.dose_amount, [100, 100, 100])
    np.testing.assert_allclose(tcs.dose_time, [0, 0, 0])
    assert tcs.dose_unit == "mg"
    assert tcs.sd is None and tcs.n is None


def test_from_arrays_two_sample_dims() -> None:
    values = np.stack([V, 2 * V])  # (dose, individual, time)
    tcs = Timecourses.from_arrays(
        T, values, time_unit="hr", unit="mg/l", dims=("dose", "individual"),
        coords={"dose": [50, 100], "individual": ["a", "b", "c"]},
        dose={"amount": np.array([[50, 50, 50], [100, 100, 100]]), "unit": "mg"},
        route=Route.ORAL,
    )
    assert tcs.sample_dims == ("dose", "individual")
    assert tcs.n_samples == 6
    np.testing.assert_allclose(tcs.dose_amount[1], [100, 100, 100])
    tc = tcs.sel(dose=100, individual="b")
    np.testing.assert_allclose(tc.value, 2 * V[1])
    assert tc.dose is not None and tc.dose.amount == 100


def test_from_arrays_uncertainty() -> None:
    sd = 0.1 * V
    tcs = Timecourses.from_arrays(T, V, time_unit="hr", unit="mg/l", sd=sd, n=np.array([5, 6, 7]))
    assert tcs.has_uncertainty
    np.testing.assert_allclose(tcs.sd, sd)
    np.testing.assert_allclose(tcs.se, sd / np.sqrt([[5], [6], [7]]))
    np.testing.assert_allclose(tcs.n, [5, 6, 7])
    tc = tcs.isel(individual=1)
    np.testing.assert_allclose(tc.n, 6)


def test_from_arrays_per_sample_times() -> None:
    times = np.array([[0, 1, 2, 4], [0, 2, 4, 8], [0, 0.5, 1, 2]])
    tcs = Timecourses.from_arrays(times, V, time_unit="hr", unit="mg/l")
    assert "times" in tcs.ds
    np.testing.assert_allclose(tcs.times, times)
    np.testing.assert_allclose(tcs.isel(individual=1).time, [0, 2, 4, 8])


def test_from_arrays_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        Timecourses.from_arrays(T, V[:, :3], time_unit="hr", unit="mg/l")


def test_iteration_yields_timecourses() -> None:
    tcs = make_batch()
    items = list(tcs)
    assert len(items) == 3
    assert all(isinstance(tc, Timecourse) for tc in items)
    assert items[2].label == "c"
    np.testing.assert_allclose(items[2].value, V[2])
    assert items[2].dose is not None and items[2].dose.route is Route.ORAL
    assert items[2].substance == "caffeine"


def test_from_timecourses_shared_grid() -> None:
    tcs = Timecourses.from_timecourses(
        [
            Timecourse(time=T, value=V[0], time_unit="hr", unit="mg/l", label="a"),
            Timecourse(time=T, value=V[1], time_unit="hr", unit="mg/l", label="b"),
        ]
    )
    assert "times" not in tcs.ds
    assert list(tcs.ds["individual"].values) == ["a", "b"]
    np.testing.assert_allclose(tcs.values, V[:2])


def test_from_timecourses_ragged() -> None:
    tcs = Timecourses.from_timecourses(
        [
            Timecourse(time=[0, 1, 2], value=[0, 2, 1], time_unit="hr", unit="mg/l"),
            Timecourse(time=[0, 1, 2, 4, 8], value=[0, 3, 2, 1, 0.5], time_unit="hr", unit="mg/l"),
        ],
        dim="subject",
    )
    assert tcs.sample_dims == ("subject",)
    assert tcs.n_time == 5
    assert np.isnan(tcs.values[0, 3:]).all()
    assert np.isnan(tcs.times[0, 3:]).all()
    tc = tcs.isel(subject=0)
    assert tc.size == 3
    np.testing.assert_allclose(tc.time, [0, 1, 2])


def test_from_timecourses_requires_same_units() -> None:
    with pytest.raises(ValueError, match="unit"):
        Timecourses.from_timecourses(
            [
                Timecourse(time=T, value=V[0], time_unit="hr", unit="mg/l"),
                Timecourse(time=T, value=V[1], time_unit="hr", unit="ng/ml"),
            ]
        )


def test_from_timecourses_uncertainty_and_dose() -> None:
    tcs = Timecourses.from_timecourses(
        [
            Timecourse(time=T, value=V[0], sd=0.1 * V[0], n=4, time_unit="hr", unit="mg/l", dose=Dose(amount=50, unit="mg")),
            Timecourse(time=T, value=V[1], sd=0.1 * V[1], n=6, time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg")),
        ]
    )
    assert tcs.has_uncertainty and tcs.has_dose
    np.testing.assert_allclose(tcs.n, [4, 6])
    np.testing.assert_allclose(tcs.dose_amount, [50, 100])


def test_to_dataframe_long() -> None:
    df = make_batch().to_dataframe()
    assert set(df.columns) == {"individual", "time", "value"}
    assert len(df) == 12
    assert df["value"].sum() == pytest.approx(V.sum())


def test_dataset_validation() -> None:
    ds = xr.Dataset({"foo": (("time",), np.zeros(3))}, coords={"time": [0, 1, 2]})
    with pytest.raises(ValueError, match="value"):
        Timecourses(ds)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_timecourses.py -n 0 -q`
Expected: FAIL with `ImportError: cannot import name 'Timecourses'`.

- [ ] **Step 3: Append `Timecourses` to `src/pkpdutils/timecourse.py`**

Add the imports `from collections.abc import Iterator, Mapping, Sequence` and `import xarray as xr` at the top. Append:

```python
#: name of the time dimension of a `Timecourses` dataset
TIME_DIM = "time"

#: name of the per sample time variable of a `Timecourses` dataset with ragged grids
TIMES_VAR = "times"


class Timecourses:
    """A batch of timecourses as an `xarray.Dataset`.

    The dataset has the dimension `time` and any number of sample dimensions,
    e.g. `individual`, `group`, `study`, or the dimensions of a simulation
    scan. Its variables are

    - `value` over `(*sample_dims, time)`, the values; `NaN` marks missing points,
    - `sd`, `se` over the same dimensions and `n` over the sample dimensions,
      for group data (optional),
    - `dose_amount`, `dose_time`, `dose_duration` over the sample dimensions
      (optional; `dose_duration` is `NaN` without infusion),
    - the coordinate `time` with the shared sampling grid, or, when the samples
      have different sampling times, the variable `times` over
      `(*sample_dims, time)` padded with `NaN` and an integer coordinate `time`.

    Every variable carries its unit in `attrs["units"]`; the dataset carries
    `substance`, `route`, `time_unit` and `unit` in its `attrs`. The properties
    `times` and `values` return the `(*sample_shape, n_time)` arrays every
    analysis of the package works on; iteration and `sel`/`isel` give single
    `Timecourse` objects.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a dataset, see the class documentation for its layout.

        Raises:
            ValueError: if the dataset does not have the layout.
        """
        if "value" not in ds:
            raise ValueError("The dataset needs a 'value' variable")
        if TIME_DIM not in ds["value"].dims:
            raise ValueError(f"'value' needs the dimension '{TIME_DIM}'")
        if ds["value"].dims[-1] != TIME_DIM:
            ds = ds.transpose(..., TIME_DIM)
        if "units" not in ds["value"].attrs:
            raise ValueError("'value' needs attrs['units']")
        time_var = ds[TIMES_VAR] if TIMES_VAR in ds else ds[TIME_DIM]
        if "units" not in time_var.attrs:
            raise ValueError("the time coordinate needs attrs['units']")
        self.ds: xr.Dataset = ds

    # --- layout -------------------------------------------------------------

    @property
    def sample_dims(self) -> tuple[str, ...]:
        """The dimensions other than `time`."""
        return tuple(str(d) for d in self.ds["value"].dims if d != TIME_DIM)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """The shape of the sample dimensions."""
        return tuple(int(self.ds.sizes[d]) for d in self.sample_dims)

    @property
    def n_samples(self) -> int:
        """Number of timecourses."""
        return int(np.prod(self.sample_shape, dtype=int)) if self.sample_shape else 1

    @property
    def n_time(self) -> int:
        """Number of time points (the length of the padded grid for ragged data)."""
        return int(self.ds.sizes[TIME_DIM])

    @property
    def time_unit(self) -> str:
        """Unit of the times."""
        time_var = self.ds[TIMES_VAR] if TIMES_VAR in self.ds else self.ds[TIME_DIM]
        return str(time_var.attrs["units"])

    @property
    def unit(self) -> str:
        """Unit of the values."""
        return str(self.ds["value"].attrs["units"])

    @property
    def substance(self) -> str:
        """Name of the substance or effect."""
        return str(self.ds.attrs.get("substance", "substance"))

    @property
    def route(self) -> Route | None:
        """Route of the doses, `None` without dose information."""
        route = self.ds.attrs.get("route")
        return None if route is None else Route(route)

    @property
    def has_uncertainty(self) -> bool:
        """Whether `sd` or `se` is present."""
        return "sd" in self.ds or "se" in self.ds

    @property
    def has_dose(self) -> bool:
        """Whether doses are present."""
        return "dose_amount" in self.ds

    # --- arrays -------------------------------------------------------------

    @property
    def times(self) -> np.ndarray:
        """Times as an array of shape `(*sample_shape, n_time)`."""
        if TIMES_VAR in self.ds:
            return self.ds[TIMES_VAR].transpose(*self.sample_dims, TIME_DIM).to_numpy()
        grid = self.ds[TIME_DIM].to_numpy().astype(np.float64)
        return np.broadcast_to(grid, (*self.sample_shape, grid.size)).copy()

    @property
    def values(self) -> np.ndarray:
        """Values as an array of shape `(*sample_shape, n_time)`."""
        return self.ds["value"].transpose(*self.sample_dims, TIME_DIM).to_numpy()

    def _optional(self, name: str) -> np.ndarray | None:
        if name not in self.ds:
            return None
        da = self.ds[name]
        if TIME_DIM in da.dims:
            return da.transpose(*self.sample_dims, TIME_DIM).to_numpy()
        return da.transpose(*self.sample_dims).to_numpy()

    @property
    def sd(self) -> np.ndarray | None:
        """Standard deviations, `None` without."""
        return self._optional("sd")

    @property
    def se(self) -> np.ndarray | None:
        """Standard errors, `None` without."""
        return self._optional("se")

    @property
    def n(self) -> np.ndarray | None:
        """Number of subjects per sample, `None` without."""
        return self._optional("n")

    @property
    def dose_amount(self) -> np.ndarray | None:
        """Dose amounts per sample, `None` without doses."""
        return self._optional("dose_amount")

    @property
    def dose_time(self) -> np.ndarray | None:
        """Dose times per sample, `None` without doses."""
        return self._optional("dose_time")

    @property
    def dose_duration(self) -> np.ndarray | None:
        """Infusion durations per sample (`NaN` without infusion), `None` without doses."""
        return self._optional("dose_duration")

    @property
    def dose_unit(self) -> str | None:
        """Unit of the doses, `None` without doses."""
        if not self.has_dose:
            return None
        return str(self.ds["dose_amount"].attrs["units"])

    # --- construction -------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        time: Any,
        values: Any,
        *,
        time_unit: str,
        unit: str,
        dims: Sequence[str] = ("individual",),
        coords: Mapping[str, Any] | None = None,
        sd: Any | None = None,
        se: Any | None = None,
        n: Any | None = None,
        dose: Dose | Mapping[str, Any] | None = None,
        route: Route | None = None,
        substance: str = "substance",
    ) -> "Timecourses":
        """Create a batch from arrays.

        Args:
            time: the sampling grid shared by all samples (1-D), or the times per
                sample with the shape of `values`
            values: values of shape `(*sample_shape, n_time)`
            time_unit: unit of the times
            unit: unit of the values
            dims: names of the sample dimensions, one per leading axis of `values`
            coords: coordinate values per sample dimension (labels of the samples)
            sd: standard deviations with the shape of `values`
            se: standard errors with the shape of `values`
            n: number of subjects, one number or an array of shape `sample_shape`
            dose: one `Dose` for all samples, or a mapping with `amount`
                (array of shape `sample_shape`), `unit`, and optionally `time`
                and `duration` arrays; the route is then given by `route`
            route: route of the doses when `dose` is a mapping
            substance: name of the substance or effect

        Returns:
            The batch.

        Raises:
            ValueError: if the shapes do not fit.
        """
        values_arr = np.asarray(values, dtype=np.float64)
        dims = tuple(dims)
        if values_arr.ndim != len(dims) + 1:
            raise ValueError(
                f"'values' has shape {values_arr.shape}, expected {len(dims) + 1} axes for dims {dims} + time"
            )
        sample_shape = values_arr.shape[:-1]
        n_time = values_arr.shape[-1]
        all_dims = (*dims, TIME_DIM)
        parse_unit(time_unit)
        parse_unit(unit)

        time_arr = np.asarray(time, dtype=np.float64)
        data_vars: dict[str, Any] = {"value": (all_dims, values_arr, {"units": unit})}
        coordinates: dict[str, Any] = dict(coords or {})
        if time_arr.ndim == 1:
            if time_arr.size != n_time:
                raise ValueError(
                    f"'time' has length {time_arr.size}, 'values' has shape {values_arr.shape}"
                )
            coordinates[TIME_DIM] = (TIME_DIM, time_arr, {"units": time_unit})
        else:
            if time_arr.shape != values_arr.shape:
                raise ValueError(
                    f"'time' has shape {time_arr.shape}, 'values' has shape {values_arr.shape}"
                )
            data_vars[TIMES_VAR] = (all_dims, time_arr, {"units": time_unit})
            coordinates[TIME_DIM] = (TIME_DIM, np.arange(n_time), {"units": time_unit})

        for name, raw in (("sd", sd), ("se", se)):
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float64)
            if arr.shape != values_arr.shape:
                raise ValueError(f"'{name}' has shape {arr.shape}, 'values' has shape {values_arr.shape}")
            data_vars[name] = (all_dims, arr, {"units": unit})
        if n is not None:
            n_arr = np.broadcast_to(np.asarray(n, dtype=np.float64), sample_shape).copy()
            data_vars["n"] = (dims, n_arr, {"units": "dimensionless"})
            if "sd" in data_vars and "se" not in data_vars:
                data_vars["se"] = (all_dims, data_vars["sd"][1] / np.sqrt(n_arr)[..., None], {"units": unit})
            elif "se" in data_vars and "sd" not in data_vars:
                data_vars["sd"] = (all_dims, data_vars["se"][1] * np.sqrt(n_arr)[..., None], {"units": unit})

        attrs: dict[str, Any] = {"substance": substance, "time_unit": time_unit, "unit": unit}
        if isinstance(dose, Dose):
            attrs["route"] = dose.route.value
            amount = np.full(sample_shape, dose.amount)
            dose_time = np.full(sample_shape, dose.time)
            duration = np.full(sample_shape, np.nan if dose.duration is None else dose.duration)
            data_vars["dose_amount"] = (dims, amount, {"units": dose.unit})
            data_vars["dose_time"] = (dims, dose_time, {"units": time_unit})
            data_vars["dose_duration"] = (dims, duration, {"units": time_unit})
        elif dose is not None:
            if route is None:
                raise ValueError("'route' is required when 'dose' is given as arrays")
            dose_unit = str(dose["unit"])
            check_dose_unit(dose_unit)
            attrs["route"] = route.value
            amount = np.broadcast_to(np.asarray(dose["amount"], dtype=np.float64), sample_shape).copy()
            dose_time = np.broadcast_to(np.asarray(dose.get("time", 0.0), dtype=np.float64), sample_shape).copy()
            duration = np.broadcast_to(np.asarray(dose.get("duration", np.nan), dtype=np.float64), sample_shape).copy()
            data_vars["dose_amount"] = (dims, amount, {"units": dose_unit})
            data_vars["dose_time"] = (dims, dose_time, {"units": time_unit})
            data_vars["dose_duration"] = (dims, duration, {"units": time_unit})

        ds = xr.Dataset(data_vars=data_vars, coords=coordinates, attrs=attrs)
        return cls(ds)

    @classmethod
    def from_timecourses(
        cls,
        timecourses: Sequence[Timecourse],
        dim: str = "individual",
        labels: Sequence[Any] | None = None,
    ) -> "Timecourses":
        """Create a batch from single timecourses along one sample dimension.

        The timecourses must share `time_unit`, `unit`, `substance` and the route
        of their doses. If all sampling grids are equal the grid becomes the
        `time` coordinate, otherwise the times are stored per sample and shorter
        curves are padded with `NaN`.

        Args:
            timecourses: the curves
            dim: name of the sample dimension
            labels: coordinate labels of the samples, the `label` of every
                timecourse (or its index when missing) by default

        Returns:
            The batch.

        Raises:
            ValueError: for an empty sequence or differing units.
        """
        if not timecourses:
            raise ValueError("At least one timecourse is required")
        first = timecourses[0]
        for tc in timecourses[1:]:
            if tc.time_unit != first.time_unit or tc.unit != first.unit:
                raise ValueError(
                    f"All timecourses need the same units: '{first.time_unit}'/'{first.unit}' "
                    f"and '{tc.time_unit}'/'{tc.unit}'"
                )
            if tc.substance != first.substance:
                raise ValueError("All timecourses need the same substance")

        if labels is None:
            labels = [tc.label if tc.label is not None else i for i, tc in enumerate(timecourses)]
        n_time = max(tc.size for tc in timecourses)
        shared = all(tc.size == first.size and np.array_equal(tc.time, first.time) for tc in timecourses)

        def padded(arrays: Sequence[np.ndarray | None]) -> np.ndarray | None:
            if any(a is None for a in arrays):
                return None
            out = np.full((len(arrays), n_time), np.nan)
            for i, a in enumerate(arrays):
                assert a is not None
                out[i, : a.size] = a
            return out

        values = padded([tc.value for tc in timecourses])
        assert values is not None
        time: np.ndarray = first.time if shared else padded([tc.time for tc in timecourses])  # ty: ignore[invalid-assignment]
        sd = padded([tc.sd for tc in timecourses])
        se = padded([tc.se for tc in timecourses])
        n_values = [tc.n for tc in timecourses]
        n: np.ndarray | None = None
        if all(v is not None for v in n_values):
            n = np.array([float(np.nanmax(v)) for v in n_values])  # ty: ignore[no-matching-overload]

        doses = [tc.dose for tc in timecourses]
        dose: Mapping[str, Any] | None = None
        route: Route | None = None
        if all(d is not None for d in doses):
            routes = {d.route for d in doses if d is not None}
            units = {d.unit for d in doses if d is not None}
            if len(routes) != 1 or len(units) != 1:
                raise ValueError("All doses need the same route and unit")
            route = routes.pop()
            dose = {
                "amount": np.array([d.amount for d in doses if d is not None]),
                "unit": units.pop(),
                "time": np.array([d.time for d in doses if d is not None]),
                "duration": np.array([np.nan if d.duration is None else d.duration for d in doses if d is not None]),
            }

        return cls.from_arrays(
            time,
            values,
            time_unit=first.time_unit,
            unit=first.unit,
            dims=(dim,),
            coords={dim: list(labels)},
            sd=sd,
            se=se,
            n=n,
            dose=dose,
            route=route,
            substance=first.substance,
        )

    # --- access -------------------------------------------------------------

    def __len__(self) -> int:
        """Number of timecourses."""
        return self.n_samples

    def _timecourse(self, sample: xr.Dataset, label: Any) -> Timecourse:
        """Build the `Timecourse` of a dataset without sample dimensions."""
        time = sample[TIMES_VAR].to_numpy() if TIMES_VAR in sample else sample[TIME_DIM].to_numpy().astype(np.float64)
        value = sample["value"].to_numpy()
        mask = ~np.isnan(time)
        data: dict[str, Any] = {
            "time": time[mask],
            "value": value[mask],
            "time_unit": self.time_unit,
            "unit": self.unit,
            "substance": self.substance,
            "label": None if label is None else str(label),
        }
        for name in ("sd", "se"):
            if name in sample:
                data[name] = sample[name].to_numpy()[mask]
        if "n" in sample:
            data["n"] = float(sample["n"].to_numpy())
        if self.has_dose:
            duration = float(sample["dose_duration"].to_numpy())
            route = self.route
            assert route is not None
            data["dose"] = Dose(
                amount=float(sample["dose_amount"].to_numpy()),
                unit=self.dose_unit or "mg",
                route=route,
                time=float(sample["dose_time"].to_numpy()),
                duration=None if np.isnan(duration) else duration,
            )
        return Timecourse(**data)

    def isel(self, **indexers: int) -> Timecourse:
        """One timecourse by integer position on every sample dimension."""
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(f"isel needs an index for every sample dimension, missing {sorted(missing)}")
        sample = self.ds.isel(**indexers)
        label = self._label(sample)
        return self._timecourse(sample, label)

    def sel(self, **indexers: Any) -> Timecourse:
        """One timecourse by coordinate label on every sample dimension."""
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(f"sel needs a label for every sample dimension, missing {sorted(missing)}")
        sample = self.ds.sel(**indexers)
        label = self._label(sample)
        return self._timecourse(sample, label)

    def _label(self, sample: xr.Dataset) -> Any:
        """Label of a selected sample: the coordinates of the sample dimensions joined by `|`."""
        parts = [str(sample[d].values) for d in self.sample_dims if d in sample.coords]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else "|".join(parts)

    def __iter__(self) -> Iterator[Timecourse]:
        """Iterate over the timecourses in C order of the sample dimensions."""
        for index in np.ndindex(*self.sample_shape):
            yield self.isel(**dict(zip(self.sample_dims, (int(i) for i in index), strict=True)))

    def to_dataframe(self) -> pd.DataFrame:
        """The batch as a long data frame: the sample coordinates, `time`, `value` and the optional columns."""
        names = ["value", *[v for v in ("sd", "se") if v in self.ds]]
        df = self.ds[names].to_dataframe().reset_index()
        if TIMES_VAR in self.ds:
            df[TIME_DIM] = self.ds[TIMES_VAR].to_dataframe().reset_index()[TIMES_VAR].to_numpy()
        if "n" in self.ds:
            df = df.merge(self.ds["n"].to_dataframe().reset_index(), on=list(self.sample_dims))
        columns = [*self.sample_dims, TIME_DIM, *names, *(["n"] if "n" in self.ds else [])]
        return df[columns]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_timecourses.py -n 0 -q`
Expected: all pass. Likely adjustments: `test_to_dataframe_long` expects exactly the columns `individual`, `time`, `value`, which the final `df[columns]` selection guarantees; if xarray puts `time` in a different dtype (`int64` for the padded grid), cast with `df[TIME_DIM] = df[TIME_DIM].astype(float)`.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: clean. Remove any `# ty: ignore[...]` comment that ty reports as unused (`error-on-warning` turns `unused-ignore-comment` into a failure).

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/timecourse.py tests/test_timecourses.py
git commit -q -m "Add Timecourses, the xarray batch of timecourses

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 11: `Timecourses.from_dataframe` and `from_dataset`/`from_xresult`

**Files:**
- Modify: `src/pkpdutils/timecourse.py` (append methods to `Timecourses`)
- Test: `tests/test_timecourses.py` (append)

**Interfaces:**
- Produces:
  - `Timecourses.from_dataframe(df, *, sample: Sequence[str], time_unit, unit, time="time", value="value", sd=None, se=None, n=None, dose_amount=None, dose_unit=None, dose_time=None, route=None, substance="substance") -> Timecourses`: long data frame, one row per (sample, time); the sample columns become the sample dimensions (a multi-column sample is the cartesian product, missing combinations are `NaN`); per sample grids when the times differ between samples
  - `Timecourses.from_dataset(ds: xr.Dataset, value: str, *, unit, time_unit, time_dim="_time", time: str | None = None, dose=None, substance=None) -> Timecourses`: a dataset with a time dimension `time_dim` (the time values are its coordinate, or the variable `time` when given), the variable `value` over `(time_dim, *scan dims)`; scan dims become sample dims
  - `Timecourses.from_xresult(xres: Any, selection: str, *, dose: Dose | None = None, substance: str | None = None) -> Timecourses`: for an sbmlsim `XResult` (duck typed: `xres.xds` is the dataset with `_time`, `xres.uinfo[selection]` and `xres.uinfo["time"]` are the unit strings); sbmlsim is not imported

- [ ] **Step 1: Append the failing tests**

Add `import pandas as pd` to the imports at the top of `tests/test_timecourses.py`, then append:

```python
def test_from_dataframe_shared_grid() -> None:
    rows = []
    for i, name in enumerate(["a", "b", "c"]):
        for t, v in zip(T, V[i], strict=True):
            rows.append({"individual": name, "time": t, "value": v, "dose": 100})
    df = pd.DataFrame(rows)
    tcs = Timecourses.from_dataframe(
        df, sample=["individual"], time_unit="hr", unit="mg/l",
        dose_amount="dose", dose_unit="mg", route=Route.ORAL,
    )
    assert tcs.sample_dims == ("individual",)
    assert list(tcs.ds["individual"].values) == ["a", "b", "c"]
    np.testing.assert_allclose(tcs.values, V)
    np.testing.assert_allclose(tcs.dose_amount, [100, 100, 100])
    assert "times" not in tcs.ds


def test_from_dataframe_ragged() -> None:
    df = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "b", "b"],
            "time": [0, 1, 2, 0, 4],
            "value": [0, 2, 1, 0, 3],
            "sd": [0, 0.2, 0.1, 0, 0.3],
            "n": [5, 5, 5, 8, 8],
        }
    )
    tcs = Timecourses.from_dataframe(df, sample=["subject"], time_unit="hr", unit="mg/l", sd="sd", n="n")
    assert "times" in tcs.ds
    assert tcs.n_time == 3
    np.testing.assert_allclose(tcs.n, [5, 8])
    tc = tcs.sel(subject="b")
    np.testing.assert_allclose(tc.time, [0, 4])
    np.testing.assert_allclose(tc.sd, [0, 0.3])


def test_from_dataframe_two_sample_columns() -> None:
    rows = []
    for dose in (50, 100):
        for name in ("a", "b"):
            for t in T:
                rows.append({"dose": dose, "individual": name, "time": t, "value": dose * t})
    tcs = Timecourses.from_dataframe(pd.DataFrame(rows), sample=["dose", "individual"], time_unit="hr", unit="mg/l")
    assert tcs.sample_dims == ("dose", "individual")
    assert tcs.sample_shape == (2, 2)
    np.testing.assert_allclose(tcs.sel(dose=100, individual="b").value, 100 * T)


def test_from_dataset_scan() -> None:
    time = np.linspace(0, 10, 11)
    scan = np.array([1.0, 2.0])
    values = np.exp(-scan[None, :] * time[:, None] / 5)  # (_time, dim_dose)
    ds = xr.Dataset(
        {"[Cve_mid]": (("_time", "dim_dose"), values)},
        coords={"_time": time, "dim_dose": scan},
    )
    tcs = Timecourses.from_dataset(ds, "[Cve_mid]", unit="mmol/l", time_unit="min", substance="midazolam")
    assert tcs.sample_dims == ("dim_dose",)
    assert tcs.n_time == 11
    np.testing.assert_allclose(tcs.values, values.T)
    np.testing.assert_allclose(tcs.ds["time"].values, time)
    assert tcs.unit == "mmol/l" and tcs.time_unit == "min"


def test_from_xresult_duck_typed() -> None:
    time = np.linspace(0, 10, 11)
    ds = xr.Dataset({"[Cve_mid]": (("_time",), np.exp(-time / 5))}, coords={"_time": time})

    class FakeXResult:
        xds = ds
        uinfo = {"[Cve_mid]": "mmol/l", "time": "min"}

    tcs = Timecourses.from_xresult(FakeXResult(), "[Cve_mid]", dose=Dose(amount=7.5, unit="mg", route=Route.IV_BOLUS))
    assert tcs.sample_dims == ()
    assert tcs.n_samples == 1
    assert tcs.unit == "mmol/l"
    tc = tcs.isel()
    assert tc.dose is not None and tc.dose.route is Route.IV_BOLUS
    np.testing.assert_allclose(tc.time, time)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_timecourses.py -n 0 -q -k "from_dataframe or from_dataset or from_xresult"`
Expected: FAIL with `AttributeError: type object 'Timecourses' has no attribute 'from_dataframe'`.

- [ ] **Step 3: Append the constructors to `Timecourses`**

Inside the class, after `from_timecourses`:

```python
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        sample: Sequence[str],
        time_unit: str,
        unit: str,
        time: str = "time",
        value: str = "value",
        sd: str | None = None,
        se: str | None = None,
        n: str | None = None,
        dose_amount: str | None = None,
        dose_unit: str | None = None,
        dose_time: str | None = None,
        route: Route | None = None,
        substance: str = "substance",
    ) -> "Timecourses":
        """Create a batch from a long data frame, one row per sample and time point.

        Args:
            df: the data frame
            sample: the columns which identify a sample; they become the sample
                dimensions, several columns give their cartesian product with
                `NaN` for missing combinations
            time_unit: unit of the time column
            unit: unit of the value column
            time: name of the time column
            value: name of the value column
            sd: name of the standard deviation column
            se: name of the standard error column
            n: name of the column with the number of subjects (constant per sample)
            dose_amount: name of the dose column (constant per sample)
            dose_unit: unit of the doses, required with `dose_amount`
            dose_time: name of the dose time column, 0 by default
            route: route of the doses, required with `dose_amount`
            substance: name of the substance or effect

        Returns:
            The batch.
        """
        sample = list(sample)
        if not sample:
            raise ValueError("'sample' needs at least one column")
        groups = df.groupby(sample, sort=True, dropna=False)
        keys = list(groups.groups)
        # from_timecourses decides between a shared grid and per sample grids
        timecourses = [
                Timecourse.from_dataframe(
                    g.sort_values(time),
                    time_unit=time_unit,
                    unit=unit,
                    time=time,
                    value=value,
                    sd=sd,
                    se=se,
                    n=n,
                    substance=substance,
                    label=str(key),
                    dose=_dose_of_group(g, dose_amount, dose_unit, dose_time, route),
                )
                for key, g in groups
            ]

        if len(sample) == 1:
            return cls.from_timecourses(timecourses, dim=sample[0], labels=list(keys))

        # several sample columns: build along one flat dimension, then unstack
        flat = cls.from_timecourses(timecourses, dim="_sample", labels=list(range(len(keys))))
        index = pd.MultiIndex.from_tuples([tuple(k) for k in keys], names=sample)
        ds = flat.ds.assign_coords(_sample=index).unstack("_sample")
        ds.attrs.update(flat.ds.attrs)
        for name in ds.data_vars:
            ds[name].attrs.update(flat.ds[name].attrs)
        if TIME_DIM in ds.coords:
            ds[TIME_DIM].attrs.update(flat.ds[TIME_DIM].attrs)
        return cls(ds.transpose(*sample, TIME_DIM))

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        value: str,
        *,
        unit: str,
        time_unit: str,
        time_dim: str = "_time",
        time: str | None = None,
        dose: Dose | None = None,
        substance: str | None = None,
    ) -> "Timecourses":
        """Create a batch from a dataset of a simulation, e.g. a parameter scan.

        Args:
            ds: dataset with the time dimension `time_dim` and the variable `value`
                over it and the scan dimensions
            value: name of the variable with the values
            unit: unit of the values
            time_unit: unit of the times
            time_dim: name of the time dimension
            time: name of the variable with the time values, the coordinate of
                `time_dim` by default
            dose: one dose for all samples
            substance: name of the substance, `value` by default

        Returns:
            The batch with the scan dimensions as sample dimensions.
        """
        da = ds[value]
        if time_dim not in da.dims:
            raise ValueError(f"'{value}' has no dimension '{time_dim}'")
        sample_dims = tuple(str(d) for d in da.dims if d != time_dim)
        values = da.transpose(*sample_dims, time_dim).to_numpy()
        grid = (ds[time] if time is not None else ds[time_dim]).to_numpy().astype(np.float64)
        if grid.ndim != 1:
            raise ValueError("The time values must be one dimensional")
        coords = {d: ds[d].to_numpy() for d in sample_dims if d in ds.coords}
        return cls.from_arrays(
            grid,
            values,
            time_unit=time_unit,
            unit=unit,
            dims=sample_dims,
            coords=coords,
            dose=dose,
            substance=value if substance is None else substance,
        )

    @classmethod
    def from_xresult(
        cls,
        xres: Any,
        selection: str,
        *,
        dose: Dose | None = None,
        substance: str | None = None,
    ) -> "Timecourses":
        """Create a batch from the result of an sbmlsim simulation.

        sbmlsim is not a dependency; an `XResult` is used by its attributes:
        `xres.xds` is the dataset with the `_time` dimension and `xres.uinfo`
        maps `selection` and `"time"` to unit strings.

        Args:
            xres: the `sbmlsim.result.XResult`
            selection: the variable of the result, e.g. `"[Cve_mid]"`
            dose: one dose for all samples
            substance: name of the substance, `selection` by default

        Returns:
            The batch with the scan dimensions as sample dimensions.
        """
        return cls.from_dataset(
            xres.xds,
            selection,
            unit=str(xres.uinfo[selection]),
            time_unit=str(xres.uinfo["time"]),
            dose=dose,
            substance=substance,
        )
```

and the module level helper (above the class):

```python
def _dose_of_group(
    g: pd.DataFrame,
    dose_amount: str | None,
    dose_unit: str | None,
    dose_time: str | None,
    route: Route | None,
) -> Dose | None:
    """The dose of the rows of one sample of a long data frame."""
    if dose_amount is None:
        return None
    if dose_unit is None or route is None:
        raise ValueError("'dose_unit' and 'route' are required with 'dose_amount'")
    amounts = g[dose_amount].dropna().unique()
    if amounts.size != 1:
        raise ValueError(f"The dose must be constant per sample, found {amounts}")
    time = 0.0
    if dose_time is not None:
        times = g[dose_time].dropna().unique()
        if times.size != 1:
            raise ValueError(f"The dose time must be constant per sample, found {times}")
        time = float(times[0])
    return Dose(amount=float(amounts[0]), unit=dose_unit, route=route, time=time)
```

The `isel()` call with no indexers in `test_from_xresult_duck_typed` needs `isel`/`sel` to accept an empty set of indexers when there are no sample dims: the `missing` check already passes, `self.ds.isel()` returns the dataset unchanged, and `_label` returns `None`.

Note: the `n` column of a group must be constant per sample; `Timecourse.from_dataframe` receives the column as an array and `from_timecourses` takes `nanmax`, which is the constant.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass. If `groupby(..., dropna=False)` with a single column returns scalar keys and with several columns tuples, `list(keys)` handles both; if pandas wraps single keys in one-element tuples (pandas >= 2.x with a list `by`), unwrap them: `keys = [k[0] if isinstance(k, tuple) and len(sample) == 1 else k for k in keys]`.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check --fix && uv run ruff format && uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pkpdutils/timecourse.py tests/test_timecourses.py
git commit -q -m "Add Timecourses constructors from data frames, datasets and sbmlsim results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 12: Top level exports, documentation of the data model and the example

**Files:**
- Modify: `src/pkpdutils/__init__.py`, `zensical.toml` (nav, if entries were removed in Task 6), `tests/examples/test_examples.py`, `examples/README.md`, `CLAUDE.md`
- Create: `docs/units.md`, `docs/timecourses.md`, `docs/api/timecourse.md`, `examples/timecourses.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Add the failing export test**

Append to `tests/test_package.py`:
```python
def test_exports() -> None:
    from pkpdutils import Dose, DosingRegimen, Q_, Route, Timecourse, Timecourses, ureg

    assert Route.ORAL.value == "oral"
    assert Dose and DosingRegimen and Timecourse and Timecourses and Q_ and ureg
```

Run: `uv run pytest tests/test_package.py -n 0 -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 2: Write `src/pkpdutils/__init__.py`**

```python
"""pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters.

The data model is `Timecourse` (one curve) and `Timecourses` (a batch as an
xarray dataset), see `pkpdutils.timecourse`; units are pint quantities of the
shared registry `ureg`, see `pkpdutils.units`.
"""

from pkpdutils.timecourse import Dose, DosingRegimen, Route, Timecourse, Timecourses
from pkpdutils.units import Q_, Quantity, ureg

__version__ = "1.0.0.dev0"

__all__ = [
    "Dose",
    "DosingRegimen",
    "Q_",
    "Quantity",
    "Route",
    "Timecourse",
    "Timecourses",
    "__version__",
    "ureg",
]
```

Run: `uv run pytest tests/test_package.py -n 0 -q`
Expected: PASS.

- [ ] **Step 3: Write the example**

`examples/timecourses.py`:
```python
"""Creating timecourses.

Run from the root of the repository with `python -m examples.timecourses`.
The example prints the objects and writes `timecourses.tsv` into the working
directory.
"""

import numpy as np
import pandas as pd
import xarray as xr

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.console import console


def single_timecourse() -> Timecourse:
    """One oral caffeine curve with a dose and group uncertainty."""
    return Timecourse(
        time=[0.5, 1, 2, 4, 8, 12, 24],
        value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2, 0.03],
        sd=[0.3, 0.5, 0.4, 0.3, 0.1, 0.05, 0.01],
        n=12,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
        label="healthy",
        tissue="plasma",
    )


def batch_from_arrays() -> Timecourses:
    """Three individuals on a shared sampling grid."""
    time = np.array([0.5, 1, 2, 4, 8, 12, 24])
    kel = np.array([0.1, 0.15, 0.2])
    values = 3.0 * np.exp(-kel[:, None] * time[None, :]) * (1 - np.exp(-2 * time[None, :]))
    return Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["s1", "s2", "s3"]},
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )


def batch_from_dataframe() -> Timecourses:
    """Two subjects with different sampling times from a long table."""
    df = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "a", "b", "b", "b"],
            "time": [0.5, 1, 2, 4, 1, 4, 8],
            "value": [1.0, 2.0, 1.5, 0.8, 1.8, 1.0, 0.4],
            "dose": [50, 50, 50, 50, 100, 100, 100],
        }
    )
    return Timecourses.from_dataframe(
        df,
        sample=["subject"],
        time_unit="hr",
        unit="mg/l",
        dose_amount="dose",
        dose_unit="mg",
        route=Route.ORAL,
        substance="caffeine",
    )


def batch_from_simulation() -> Timecourses:
    """A dataset shaped like an sbmlsim scan result: `_time` plus a scan dimension."""
    time = np.linspace(0, 24, 49)
    doses = np.array([25.0, 50.0, 100.0])
    values = doses[None, :] / 40 * np.exp(-0.15 * time[:, None])
    ds = xr.Dataset({"[Cve]": (("_time", "dim_dose"), values)}, coords={"_time": time, "dim_dose": doses})
    return Timecourses.from_dataset(ds, "[Cve]", unit="mg/l", time_unit="hr", substance="caffeine")


if __name__ == "__main__":
    tc = single_timecourse()
    console.rule("Timecourse")
    console.print(tc)
    console.print(tc.to_dataframe())

    console.rule("Timecourses from arrays")
    tcs = batch_from_arrays()
    console.print(tcs.ds)
    for item in tcs:
        console.print(item.label, item.value.round(3))

    console.rule("Timecourses from a data frame (ragged)")
    ragged = batch_from_dataframe()
    console.print(ragged.ds)
    console.print(ragged.sel(subject="b"))

    console.rule("Timecourses from a simulation dataset")
    scan = batch_from_simulation()
    console.print(scan.sample_dims, scan.sample_shape)

    tcs.to_dataframe().to_csv("timecourses.tsv", sep="\t", index=False)
    console.print("written: timecourses.tsv")
```

Register it: in `tests/examples/test_examples.py` set `SCRIPTS: list[str] = ["examples.timecourses"]`.

Run: `uv run python -m examples.timecourses && uv run pytest tests/examples -n 0 -q && rm -f timecourses.tsv`
Expected: the example prints the four sections and the test passes.

- [ ] **Step 4: Write the documentation pages**

`docs/api/timecourse.md`:
```markdown
# timecourse

::: pkpdutils.timecourse
```

`docs/units.md`:
````markdown
# Units

Every timecourse and every result of `pkpdutils` carries its units. The package uses [pint](https://pint.readthedocs.io) with one registry per process, `pkpdutils.units.ureg`; quantities of two registries cannot be combined, which is why nothing in the package creates a registry of its own and why an application which mixes its own quantities with those of the package should use `ureg` as well.

## Concepts

Units enter as strings on the data model (`time_unit="hr"`, `unit="ng/ml"`, `Dose(amount=100, unit="mg")`) and are validated when the object is created; an unknown unit raises a `ValueError`. The numerics of the package run on plain arrays in the units of the input, so nothing is converted behind your back: an `AUC` of a curve in `ng/ml` over `hr` is in `ng/ml·hr`. Results carry the derived unit in `attrs["units"]` of every variable, and the single sample accessors return pint quantities which convert with `.to("mg/l*hr")`.

Two families of parameters have conventional units the package converts to: volumes are reported in `liter` (or `liter/kg` for doses per body weight) and clearances in `liter/hour` (or `liter/hour/kg`), see `normalize_volume` and `normalize_clearance`.

## Dose units

A dose is an amount, in mass (`mg`, `g`) or in substance (`mmol`, `µmol`), or such an amount per body weight (`mg/kg`, `µmol/kg`). `check_dose_unit` accepts exactly these four dimensionalities. Concentrations in mass per volume with a dose in substance (or the other way round) give parameters in mixed units such as `mmol/(ng/ml)`; convert one of them with the molar mass of the substance before the analysis when clearances in `liter/hour` are wanted.

## Custom units

The registry defines `none` (dimensionless count, for data without a unit), `percent` and `IU` (international units, a dimension of its own) in addition to the pint defaults.

## API

```python
from pkpdutils.units import Q_, check_dose_unit, normalize_clearance, ureg

dose = Q_(100, "mg")
cl = Q_(120, "ml/min")
print(normalize_clearance(cl))  # 7.2 liter / hour
check_dose_unit("mg/kg")        # ok
check_dose_unit("mg/l")         # ValueError
```

The reference of the module is in [API: units](api/units.md).
````

`docs/timecourses.md`:
````markdown
# Timecourses

A pharmacokinetic timecourse is the concentration of a substance in a tissue over time after a dose; a pharmacodynamic timecourse is an effect over time. `pkpdutils` represents one curve as a `Timecourse` and many curves as a `Timecourses` batch, which is the input of every analysis of the package.

## Concepts

**Single curve.** A `Timecourse` holds the sampling times and the values with their units, an optional dose, and metadata (substance, label, tissue). It is a frozen [pydantic](https://docs.pydantic.dev) model: the arrays are converted to `float64`, sorted by time, and duplicate times or an unknown unit raise a `ValueError` when the object is created. Missing values are `NaN` in `value`; every analysis drops them.

**Group data.** Publications report the mean curve of a group with the standard deviation or the standard error and the number of subjects. A `Timecourse` carries these as `sd`, `se` and `n`; the missing one of `sd` and `se` is derived from the other with \(\mathrm{se} = \mathrm{sd}/\sqrt{n}\). The uncertainty analyses of the package propagate them to the parameters, see [Uncertainty](uncertainty.md).

**Doses and routes.** A `Dose` has an `amount` with a dose unit (an amount or an amount per body weight, see [Units](units.md)), a `Route`, the `time` of the administration and, for an infusion, its `duration`. The route decides which parameters an analysis can report: after an intravenous bolus the clearance and the volume are absolute (`cl`, `vz`), after an extravascular dose they are relative to the unknown fraction absorbed (`cl_f`, `vz_f`), and an infusion shifts the mean residence time by half its duration. `Route.ORAL` stands for every extravascular route.

**Batches.** `Timecourses` wraps an [xarray](https://xarray.dev) dataset with a `time` dimension and any number of *sample dimensions*: the individuals of a study, the groups of a publication, the doses of a dose escalation, the dimensions of a simulation scan. Every analysis of the package is vectorized over the sample dimensions and returns a dataset over the same dimensions, so the parameters of a thousand curves are one call. Curves with different sampling times are stored per sample and padded with `NaN`, the `times` and `values` properties return the padded `(samples..., time)` arrays.

**Repeated dosing.** A `DosingRegimen` is a dose given every `interval` for `n_doses` administrations; steady state analyses need the interval \(\tau\).

## Data layout of a batch

| variable | dimensions | content |
| --- | --- | --- |
| `value` | `(*sample, time)` | the values, `NaN` for missing points |
| `sd`, `se` | `(*sample, time)` | standard deviation and error of group data (optional) |
| `n` | `(*sample)` | number of subjects of group data (optional) |
| `dose_amount`, `dose_time`, `dose_duration` | `(*sample)` | the doses (optional), `dose_duration` is `NaN` without infusion |
| `time` (coordinate) | `(time)` | the shared sampling grid, or an integer index for ragged data |
| `times` | `(*sample, time)` | the sampling times per sample, only for ragged data |

Every variable carries `attrs["units"]`; the dataset carries `substance`, `route`, `time_unit` and `unit` in its `attrs`. Any further metadata (sex, body weight, study) is a coordinate on a sample dimension and travels with the results.

## API

A single curve:

```python
from pkpdutils import Dose, Route, Timecourse

tc = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12, 24],
    value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2, 0.03],
    sd=[0.3, 0.5, 0.4, 0.3, 0.1, 0.05, 0.01],
    n=12,
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
print(tc.se)               # derived from sd and n
print(tc.value_q)          # a pint quantity
print(tc.to_dataframe())
```

A batch from arrays, with the individuals as coordinate labels:

```python
import numpy as np
from pkpdutils import Timecourses

time = np.array([0.5, 1, 2, 4, 8, 12, 24])
values = np.random.default_rng(0).uniform(0, 3, size=(3, time.size))
tcs = Timecourses.from_arrays(
    time, values, time_unit="hr", unit="mg/l",
    dims=("individual",), coords={"individual": ["s1", "s2", "s3"]},
    dose=Dose(amount=100, unit="mg", route=Route.ORAL), substance="caffeine",
)
print(tcs.ds)
print(tcs.sel(individual="s2"))   # a Timecourse
for tc in tcs:                    # iteration over the samples
    print(tc.label)
```

From a long table (one row per sample and time point) with a dose column, and from a list of `Timecourse` objects:

```python
tcs = Timecourses.from_dataframe(
    df, sample=["study", "group"], time_unit="hr", unit="mg/l",
    sd="sd", n="n", dose_amount="dose", dose_unit="mg", route=Route.ORAL,
)
tcs = Timecourses.from_timecourses([tc_a, tc_b], dim="group")
```

From a simulation: a dataset with a `_time` dimension and scan dimensions, e.g. the `XResult` of [sbmlsim](https://matthiaskoenig.github.io/sbmlsim):

```python
tcs = Timecourses.from_xresult(xres, "[Cve_mid]", dose=Dose(amount=7.5, unit="mg", route=Route.IV_BOLUS))
tcs = Timecourses.from_dataset(ds, "[Cve]", unit="mmol/l", time_unit="min")
```

The complete example is `examples/timecourses.py`; the reference of the module is in [API: timecourse](api/timecourse.md).
````

If Task 6 removed the `units.md`, `timecourses.md` and `api/timecourse.md` entries from `nav` in `zensical.toml`, add them back now (see the `nav` in Task 6 Step 1). `docs/timecourses.md` links to `uncertainty.md`, which the uncertainty plan creates; Zensical warns about the missing target but builds. If the build treats it as an error, write the link as plain text `Uncertainty` for now and restore the link in that plan.

Update `examples/README.md` (the table already lists `examples/timecourses.py`, keep it) and `CLAUDE.md`: nothing to change, the architecture section already describes `timecourse.py`.

- [ ] **Step 5: Build the docs and run all checks**

Run:
```bash
uv run zensical build --clean && uv run python scripts/llms_txt.py
uv run pytest -q
uv run ruff check && uv run ruff format --check && uv run ty check
uv run pre-commit run --all-files
```
Expected: site builds with the pages `units/`, `timecourses/`, `api/timecourse/`; all tests pass; ruff, ty and the hooks clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -q -m "Export the data model, document units and timecourses, add the timecourses example

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 13: Full matrix and hand-off

**Files:** none new.

- [ ] **Step 1: Run the tox matrix**

Run: `uv python install 3.13 3.14 && uv run tox run-parallel`
Expected: `py3.13`, `py3.14` and `ty` succeed. If `py3.13` fails on a syntax or typing feature, fix it: the package must run on 3.13.

- [ ] **Step 2: Check the built wheel**

Run: `uvx hatch build && uvx twine check dist/* && unzip -l dist/*.whl | grep -c "pkpdutils/" && rm -rf dist`
Expected: `twine check` passes; the wheel contains `pkpdutils/__init__.py`, `units.py`, `timecourse.py`, `console.py`, `log.py`, `py.typed` and nothing from `examples/` or `tests/`.

- [ ] **Step 3: Report**

The branch `redesign` now holds phases 1 and 2. The next plan (NCA) starts from `Timecourses.times`/`values`/`dose_*` and `tests/data/reference/nca_reference.json`. Manual steps for the maintainer, not part of any plan: rename the GitHub repository to `pkpdutils`, create the PyPI project with a trusted publisher and the `pypi` environment, run `.github/rulesets/apply.sh matthiaskoenig/pkpdutils`, enable GitHub Pages (source: GitHub Actions), delete the `dependabot/bundler/*` branches on origin.

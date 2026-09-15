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
SCRIPTS: list[str] = [
    "examples.timecourses",
    "examples.nca_single",
    "examples.nca_batch",
    "examples.steady_state",
    "examples.nca_from_sbmlsim",
    "examples.group_uncertainty",
    "examples.fitting_exponential",
    "examples.emax",
    "examples.dose_proportionality",
    "examples.covariate",
]


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

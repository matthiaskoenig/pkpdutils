"""The NCA reproduces the results of pkdb_analysis 0.3.1 on its test data."""

import json
from pathlib import Path

import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse
from pkpdutils.nca import (
    AUCMethod,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    nca_single,
)
from pkpdutils.units import parse_unit

REFERENCE = Path(__file__).parent.parent / "data" / "reference" / "nca_reference.json"
CASES = json.loads(REFERENCE.read_text())

#: old parameter name -> new parameter name
MAPPING = {
    "auc": "auc_last",
    "aucinf": "auc_inf_obs",
    "tmax": "tmax",
    "cmax": "cmax",
    "kel": "lambda_z",
    "thalf": "thalf",
    "vd": "vz_f",
    "cl": "cl_f",
}

OPTIONS = NCAOptions(
    auc_method=AUCMethod.LINEAR,
    terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX),
)


def to_timecourse(case: dict) -> Timecourse:
    conc = [np.nan if v is None else v for v in case["concentration"]]
    return Timecourse(
        time=case["time"],
        value=conc,
        time_unit=case["time_unit"],
        unit=case["unit"],
        dose=Dose(amount=case["dose"], unit=case["dose_unit"], route=Route.ORAL),
        substance=case["substance"],
    )


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_reference_case(case: dict) -> None:
    result = nca_single(to_timecourse(case), options=OPTIONS)
    quantities = result.to_quantities()
    for old, new in MAPPING.items():
        expected = case["parameters"][old]
        actual = quantities[new]
        if expected["magnitude"] is None:
            assert np.isnan(actual.magnitude), f"{case['name']}: {new} should be NaN"
            continue
        converted = actual.to(expected["unit"]).magnitude
        assert converted == pytest.approx(expected["magnitude"], rel=1e-6, abs=1e-12), (
            f"{case['name']}: {new}"
        )


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_reference_regression(case: dict) -> None:
    result = nca_single(to_timecourse(case), options=OPTIONS)
    q = result.to_quantities()
    slope = case["regression"]["slope"]
    if slope is None:
        assert np.isnan(q["lambda_z"].magnitude)
        return
    assert -q["lambda_z"].to(f"1/({case['time_unit']})").magnitude == pytest.approx(
        slope, rel=1e-6
    )
    assert q["lambda_z_intercept"].magnitude == pytest.approx(
        case["regression"]["intercept"], rel=1e-6
    )
    parse_unit(case["unit"])
    parse_unit(case["time_unit"])

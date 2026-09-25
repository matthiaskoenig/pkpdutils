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
from pkpdutils.units import parse_unit, ureg

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


def expected_parameters(case: dict) -> dict:
    """The reference parameters of a case, with the area before its first sample.

    pkdb_analysis 0.3.1 started the area at the first sample. An extravascular
    single dose is 0 at the dose, so `pkpdutils` inserts a zero there when the
    first sample comes later, as Phoenix WinNonlin does: the areas gain the
    triangle from the dose to the first sample, and the clearance and the
    volume, which divide by the area to infinity, shrink by the same factor.
    """
    parameters = {name: dict(value) for name, value in case["parameters"].items()}
    t0, c0 = next(
        (t, c)
        for t, c in zip(case["time"], case["concentration"], strict=True)
        if c is not None
    )
    if t0 <= 0 or parameters["auc"]["magnitude"] is None:
        return parameters
    triangle = (
        (0.5 * t0 * c0 * ureg(case["time_unit"]) * ureg(case["unit"]))
        .to(parameters["auc"]["unit"])
        .magnitude
    )
    parameters["auc"]["magnitude"] += triangle
    aucinf = parameters["aucinf"]["magnitude"]
    if aucinf is None:
        return parameters
    parameters["aucinf"]["magnitude"] = aucinf + triangle
    for name in ("cl", "vd"):
        parameters[name]["magnitude"] *= aucinf / (aucinf + triangle)
    return parameters
    triangle = (
        (0.5 * t0 * c0 * ureg(case["time_unit"]) * ureg(case["unit"]))
        .to(parameters["auc"]["unit"])
        .magnitude
    )
    factor = parameters["aucinf"]["magnitude"] / (
        parameters["aucinf"]["magnitude"] + triangle
    )
    parameters["auc"]["magnitude"] += triangle
    parameters["aucinf"]["magnitude"] += triangle
    parameters["cl"]["magnitude"] *= factor
    parameters["vd"]["magnitude"] *= factor
    return parameters


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_reference_case(case: dict) -> None:
    result = nca_single(to_timecourse(case), options=OPTIONS)
    quantities = result.to_quantities()
    parameters = expected_parameters(case)
    for old, new in MAPPING.items():
        expected = parameters[old]
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

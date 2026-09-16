"""Absolute and relative bioavailability from two analyses."""

import numpy as np
import pytest

from pkpdutils import AUCMethod, NCAOptions, Route, Timecourses, nca
from pkpdutils.nca import bioavailability

K = 0.2
LOG = NCAOptions(auc_method=AUCMethod.LOG)
SUBJECTS = ["s1", "s2", "s3", "s4"]
#: the fraction absorbed of every subject of the test treatment
FRACTIONS = np.array([0.40, 0.50, 0.60, 0.75])
TIME = np.array([0.0, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])


def batch(amplitudes: np.ndarray, *, dose: float, route: Route) -> Timecourses:
    """Mono-exponential curves `A_i exp(-k t)` of four subjects."""
    values = amplitudes[:, None] * np.exp(-K * TIME)[None, :]
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": SUBJECTS},
        dose={"amount": np.full(len(SUBJECTS), dose), "unit": "mg"},
        route=route,
        substance="x",
    )


def intravenous() -> Timecourses:
    """The reference: a 100 mg bolus reaching 10 mg/l in every subject."""
    return batch(np.full(len(SUBJECTS), 10.0), dose=100.0, route=Route.IV_BOLUS)


def oral(fractions: np.ndarray = FRACTIONS) -> Timecourses:
    """The test: a 200 mg extravascular dose of which a fraction is absorbed."""
    return batch(20.0 * fractions, dose=200.0, route=Route.ORAL)


def test_absolute_bioavailability_of_a_crossover() -> None:
    test = nca(oral(), options=LOG)
    reference = nca(intravenous(), options=LOG)
    f = bioavailability(test, reference, dim="individual")
    # AUC / D is 10 / (k 100) intravenously and 20 F_i / (k 200) orally, so
    # the ratio of every subject is its own fraction absorbed
    assert f.name == "f_abs"
    assert f.unit == "dimensionless"
    assert f.gmr == pytest.approx(float(np.exp(np.mean(np.log(FRACTIONS)))), rel=1e-9)
    assert f.paired and f.n_test == 4 and f.n_reference == 4
    assert f.ci_level == 0.90
    assert f.ci_low < f.gmr < f.ci_high


def test_a_constant_fraction_is_reproduced_without_an_interval() -> None:
    fractions = np.full(len(SUBJECTS), 0.5)
    f = bioavailability(
        nca(oral(fractions), options=LOG),
        nca(intravenous(), options=LOG),
        dim="individual",
    )
    assert f.gmr == pytest.approx(0.5, rel=1e-9)
    # every subject has the same ratio: the interval collapses onto it
    assert f.ci_low == pytest.approx(0.5, rel=1e-9)
    assert f.ci_high == pytest.approx(0.5, rel=1e-9)


def test_an_extravascular_reference_is_a_relative_bioavailability() -> None:
    test = nca(oral(), options=LOG)
    reference = nca(oral(np.full(len(SUBJECTS), 0.5)), options=LOG)
    f = bioavailability(test, reference, dim="individual")
    assert f.name == "f_rel"
    assert f.gmr == pytest.approx(
        float(np.exp(np.mean(np.log(FRACTIONS / 0.5)))), rel=1e-9
    )


def test_the_route_of_the_reference_can_be_given() -> None:
    test = nca(oral(), options=LOG)
    reference = nca(intravenous(), options=LOG)
    f = bioavailability(test, reference, dim="individual", reference_route=Route.ORAL)
    assert f.name == "f_rel"


def test_another_exposure_and_an_unpaired_comparison() -> None:
    test = nca(oral(), options=LOG)
    reference = nca(intravenous(), options=LOG)
    f = bioavailability(
        test, reference, dim="individual", parameter="auc_last", paired=False
    )
    assert f.name == "f_abs" and not f.paired
    ratios = FRACTIONS  # auc_last is dose normalized to the same fractions
    assert f.gmr == pytest.approx(float(np.exp(np.mean(np.log(ratios)))), rel=1e-9)


def test_an_unknown_parameter_raises() -> None:
    test = nca(oral(), options=LOG)
    reference = nca(intravenous(), options=LOG)
    with pytest.raises(ValueError, match="'auc_tau' is not a parameter of the test"):
        bioavailability(test, reference, dim="individual", parameter="auc_tau")

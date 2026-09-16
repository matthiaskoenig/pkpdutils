"""Power and sample size of the two one-sided tests, pinned to PowerTOST.

Every number of `POWER_REFERENCE` and `SAMPLE_SIZE_REFERENCE` is a published
example of the R package `PowerTOST` (the reference manual of `sampleN.TOST`
and `power.TOST` and the "ABE" vignette), which is the reference
implementation of this calculation.
"""

import numpy as np
import pytest
from scipy.stats import nct

from pkpdutils.stats import power_tost, sample_size_tost
from pkpdutils.stats.power import DESIGNS, owens_q

#: `power.TOST(CV=cv, n=n, theta0=gmr, design=design)` of PowerTOST, with the
#: tolerance of the digits the source prints
POWER_REFERENCE = [
    # sampleN.TOST(CV = 0.3): n = 40, power = 0.815845, the example of the manual
    (0.30, 40, 0.95, "2x2", 0.815845, 1e-6),
    # power.TOST(CV = 0.25, n = 24) of the manual
    (0.25, 24, 0.95, "2x2", 0.7391155, 1e-7),
    # power.TOST(CV = 0.25, n = 22), the balanced comparison of the manual
    (0.25, 22, 0.95, "2x2", 0.6953401, 1e-7),
    # sampleN.TOST(CV = 0.20, theta0 = 0.92): n = 28, power = 0.822742
    (0.20, 28, 0.92, "2x2", 0.822742, 1e-6),
    # the replicate table of the vignette at CV = 0.30 and theta0 = 0.95,
    # printed with four decimals
    (0.30, 30, 0.95, "2x2x3", 0.8204, 1e-4),
    (0.30, 20, 0.95, "2x2x4", 0.8202, 1e-4),
]

#: `sampleN.TOST(CV=cv, theta0=gmr, design=design)` of PowerTOST
SAMPLE_SIZE_REFERENCE = [
    (0.30, 0.95, "2x2", 40),
    (0.20, 0.92, "2x2", 28),
    (0.30, 0.95, "2x2x3", 30),
    (0.30, 0.95, "2x2x4", 20),
]

#: the untransformed parallel example of the manual,
#: `sampleN.TOST(logscale = FALSE, theta1 = -0.2, theta0 = -0.05, CV = 0.2,
#: design = "parallel")`: n = 48 at power 0.815435. On the log scale the same
#: numbers are a standard deviation of 0.2 of the log values and limits and a
#: ratio which are their exponentials.
PARALLEL_CV = float(np.sqrt(np.expm1(0.2**2)))
PARALLEL_GMR = float(np.exp(-0.05))
PARALLEL_LIMITS = (float(np.exp(-0.2)), float(np.exp(0.2)))


@pytest.mark.parametrize(
    ("cv", "n", "gmr", "design", "expected", "tolerance"), POWER_REFERENCE
)
def test_power_reproduces_powertost(
    cv: float, n: int, gmr: float, design: str, expected: float, tolerance: float
) -> None:
    """Every published power of PowerTOST is reproduced to the digits it prints."""
    power = power_tost(cv=cv, n=n, gmr=gmr, design=design)
    assert power == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize(("cv", "gmr", "design", "expected"), SAMPLE_SIZE_REFERENCE)
def test_sample_size_reproduces_powertost(
    cv: float, gmr: float, design: str, expected: int
) -> None:
    """The sample size is the one PowerTOST reports, and the size below it misses."""
    n = sample_size_tost(cv=cv, gmr=gmr, design=design)
    assert n == expected
    assert power_tost(cv=cv, n=n, gmr=gmr, design=design) >= 0.8
    assert power_tost(cv=cv, n=n - 2, gmr=gmr, design=design) < 0.8


def test_the_parallel_design_reproduces_the_untransformed_example() -> None:
    """The parallel example of the manual, translated to the log scale."""
    power = power_tost(
        cv=PARALLEL_CV,
        n=48,
        gmr=PARALLEL_GMR,
        limits=PARALLEL_LIMITS,
        design="parallel",
    )
    assert power == pytest.approx(0.815435, abs=1e-6)
    # PowerTOST searches a parallel design in steps of two to keep the groups
    # balanced, `sample_size_tost` returns any size, and 47 already reaches 0.8
    assert (
        sample_size_tost(
            cv=PARALLEL_CV,
            gmr=PARALLEL_GMR,
            limits=PARALLEL_LIMITS,
            design="parallel",
        )
        == 47
    )


@pytest.mark.parametrize(
    ("nu", "t", "delta"), [(10.0, 1.8, 2.0), (38.0, 1.686, 2.618), (100.0, -1.66, -4.0)]
)
def test_owens_q_over_the_whole_line_is_the_noncentral_t(
    nu: float, t: float, delta: float
) -> None:
    """`Q(t, delta, 0, inf)` is the distribution function of the non-central t."""
    assert owens_q(nu, t, delta, 0.0, 60.0) == pytest.approx(
        float(nct.cdf(t, nu, delta)), abs=1e-10
    )


def test_owens_q_of_an_empty_interval_is_zero() -> None:
    """An interval which is not an interval carries no probability."""
    assert owens_q(10.0, 1.0, 0.0, 2.0, 2.0) == 0.0


def test_the_power_grows_with_the_sample_size_and_falls_with_the_variability() -> None:
    """The two monotonicities every sample size table shows."""
    sizes = [12, 24, 36, 48]
    powers = [power_tost(cv=0.3, n=n) for n in sizes]
    assert powers == sorted(powers)
    variabilities = [power_tost(cv=cv, n=24) for cv in (0.15, 0.25, 0.35, 0.45)]
    assert variabilities == sorted(variabilities, reverse=True)


def test_a_ratio_at_the_limit_needs_more_subjects() -> None:
    """The closer the assumed ratio sits at a limit, the larger the study."""
    assert sample_size_tost(cv=0.25, gmr=0.95) < sample_size_tost(cv=0.25, gmr=0.90)


def test_the_designs_carry_the_constants_of_powertost() -> None:
    """The design constant and the degrees of freedom of every design."""
    assert [(d.bk, d.df(24)) for d in DESIGNS.values()] == [
        (2.0, 22.0),  # 2x2, n - 2
        (4.0, 22.0),  # parallel, n - 2
        (1.0, 68.0),  # 2x2x4, 3n - 4
        (1.5, 45.0),  # 2x2x3, 2n - 3
    ]


def test_a_study_without_variability_is_decided_by_the_ratio_alone() -> None:
    """With `cv = 0` the interval has no width and the ratio decides."""
    assert power_tost(cv=0.0, n=12) == 1.0
    assert power_tost(cv=0.0, n=12, gmr=1.3) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cv": -0.1, "n": 12}, "must not be negative"),
        ({"cv": 0.3, "n": 12, "gmr": 0.0}, "must be positive"),
        ({"cv": 0.3, "n": 12, "limits": (1.25, 0.8)}, "must be"),
        ({"cv": 0.3, "n": 12, "alpha": 0.8}, "must lie in"),
        ({"cv": 0.3, "n": 12, "design": "3x3"}, "not a known design"),
        ({"cv": 0.3, "n": 2}, "degrees of freedom"),
    ],
)
def test_power_rejects_bad_arguments(kwargs: dict, message: str) -> None:
    """Every argument is validated instead of falling through to a default."""
    with pytest.raises(ValueError, match=message):
        power_tost(**kwargs)


def test_sample_size_rejects_a_ratio_outside_the_limits() -> None:
    """No sample size makes a study pass whose true ratio is outside the limits."""
    with pytest.raises(ValueError, match="outside the limits"):
        sample_size_tost(cv=0.3, gmr=1.3)


def test_sample_size_rejects_a_power_which_is_no_probability() -> None:
    """The target power is a probability."""
    with pytest.raises(ValueError, match="must lie in"):
        sample_size_tost(cv=0.3, target_power=1.5)


def test_sample_size_gives_up_at_max_n() -> None:
    """A study which cannot be run says so instead of searching forever."""
    with pytest.raises(ValueError, match="does not reach the power"):
        sample_size_tost(cv=0.6, gmr=0.82, max_n=40)


#: the unbalanced (one dropout) column of the replicate table of the vignette
#: and `power.TOST(CV = 0.3, n = 39)` of the manual, all of them studies with
#: one subject more in one sequence
UNBALANCED_REFERENCE = [
    (0.30, 39, "2x2", 0.8056171, 1e-7),
    (0.30, 29, "2x2x3", 0.8069, 1e-4),
    (0.30, 19, "2x2x4", 0.7992, 1e-4),
]


@pytest.mark.parametrize(
    ("cv", "n", "design", "expected", "tolerance"), UNBALANCED_REFERENCE
)
def test_an_odd_sample_size_is_the_unbalanced_study_of_powertost(
    cv: float, n: int, design: str, expected: float, tolerance: float
) -> None:
    """An odd `n` splits into the two sequences the study would really have."""
    assert power_tost(cv=cv, n=n, design=design) == pytest.approx(
        expected, abs=tolerance
    )


def test_the_unbalanced_standard_error_is_below_the_balanced_one() -> None:
    """One subject more in one sequence is worth less than a balanced pair."""
    design = DESIGNS["2x2"]
    assert design.se(0.3, 40) == pytest.approx(0.3 * np.sqrt(2.0 / 40.0))
    assert design.se(0.3, 39) > design.se(0.3, 40)
    assert design.se(0.3, 39) < design.se(0.3, 38)
    # the balanced formula would be optimistic for an odd size
    assert design.se(0.3, 39) > 0.3 * np.sqrt(2.0 / 39.0)

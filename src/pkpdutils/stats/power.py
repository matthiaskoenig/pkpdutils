r"""Power and sample size of the two one-sided tests procedure.

The question a bioequivalence study asks after it ran is whether the 90 %
interval of the geometric mean ratio lies within the acceptance limits
(`pkpdutils.stats.bioequivalence`); the question it has to answer before it
runs is how many subjects that decision needs. Both are the same procedure:
`power_tost` is the probability that the two one-sided tests of Schuirmann
(1987) both reject at the assumed ratio and the assumed within-subject
variability, and `sample_size_tost` is the smallest number of subjects which
reaches a target power.

The power is exact, not simulated: under the normal model of the log values
the two test statistics are a bivariate non-central t pair, whose probability
is Owen's Q function (Owen 1965),

$$Q_\nu(t, \delta; a, b) = \frac{1}{\Gamma(\nu/2)\,2^{(\nu-2)/2}}
\int_a^b \Phi\!\left(\frac{t x}{\sqrt{\nu}} - \delta\right) x^{\nu-1} e^{-x^2/2}\,dx,$$

evaluated here by numerical integration (`scipy.integrate.quad`) of the
logarithm of the integrand, which is the algorithm of the R package
`PowerTOST` (Labes, Schuetz & Lang). The design enters through the constant
\(b_k\) of the standard error and the degrees of freedom, both taken from
`PowerTOST`: a 2x2 crossover has \(b_k = 2\) and \(\nu = n - 2\), two parallel
groups \(b_k = 4\) and \(\nu = n - 2\), the four period full replicate
(TRTR/RTRT) \(b_k = 1\) and \(\nu = 3n - 4\) and the three period replicate
(TRT/RTR) \(b_k = 1.5\) and \(\nu = 2n - 3\), with \(n\) the total number of
subjects of the study.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, ndtr
from scipy.stats import norm
from scipy.stats import t as student_t

logger = logging.getLogger(__name__)

#: the designs of `power_tost`, the names of `PowerTOST`
DesignName = Literal["2x2", "parallel", "2x2x4", "2x2x3"]


@dataclass(frozen=True)
class TOSTDesign:
    r"""A study design of the power calculation.

    Attributes:
        name: the name of the design, as `PowerTOST` spells it
        bk: the design constant of the standard error,
            \(\mathrm{se} = \sigma\sqrt{b_k/n}\)
        df_factor: factor of the degrees of freedom, \(\nu = a n + b\)
        df_offset: offset of the degrees of freedom
        step: the step of the sample size search, 2 for a design whose
            sequences have to carry the same number of subjects
        description: how the design is spelled out in a report
    """

    name: str
    bk: float
    df_factor: float
    df_offset: float
    step: int
    description: str

    def df(self, n: int) -> float:
        """The degrees of freedom of a study of `n` subjects.

        Args:
            n: the total number of subjects.

        Returns:
            The degrees of freedom of the residual.
        """
        return float(self.df_factor * n + self.df_offset)

    def se(self, sigma: float, n: int) -> float:
        r"""The standard error of the log ratio of a study of `n` subjects.

        \(\mathrm{se} = \sigma\sqrt{b_k / n}\).

        Args:
            sigma: the standard deviation of the log values.
            n: the total number of subjects.

        Returns:
            The standard error.
        """
        return float(sigma * np.sqrt(self.bk / n))


#: the designs, with the constants of `PowerTOST::known.designs()`
DESIGNS: dict[str, TOSTDesign] = {
    "2x2": TOSTDesign(
        name="2x2",
        bk=2.0,
        df_factor=1.0,
        df_offset=-2.0,
        step=2,
        description="2x2 crossover (TR/RT)",
    ),
    "parallel": TOSTDesign(
        name="parallel",
        bk=4.0,
        df_factor=1.0,
        df_offset=-2.0,
        step=1,
        description="two parallel groups",
    ),
    "2x2x4": TOSTDesign(
        name="2x2x4",
        bk=1.0,
        df_factor=3.0,
        df_offset=-4.0,
        step=2,
        description="four period full replicate crossover (TRTR/RTRT)",
    ),
    "2x2x3": TOSTDesign(
        name="2x2x3",
        bk=1.5,
        df_factor=2.0,
        df_offset=-3.0,
        step=2,
        description="three period replicate crossover (TRT/RTR)",
    ),
}


def owens_q(nu: float, t: float, delta: float, a: float, b: float) -> float:
    r"""Owen's Q function, by numerical integration of the log integrand.

    $$Q_\nu(t, \delta; a, b) = \frac{1}{\Gamma(\nu/2)\,2^{(\nu-2)/2}}
    \int_a^b \Phi\!\left(\frac{t x}{\sqrt{\nu}} - \delta\right)
    x^{\nu-1} e^{-x^2/2}\,dx$$

    (Owen 1965), the probability of the bivariate non-central t pair the two
    one-sided tests form. \(Q_\nu(t, \delta; 0, \infty)\) is the distribution
    function of the non-central t distribution at `t` with the non-centrality
    \(\delta\), which is the regression test of this function. The integrand
    is evaluated as \(\exp((\nu-1)\ln x - x^2/2 - \ln\Gamma(\nu/2) -
    \frac{\nu-2}{2}\ln 2)\,\Phi(\cdot)\), so that neither \(x^{\nu-1}\) nor
    \(\Gamma(\nu/2)\) overflows for the large degrees of freedom of a big
    study.

    Args:
        nu: degrees of freedom, positive.
        t: the argument of the normal distribution function.
        delta: the non-centrality.
        a: lower bound of the integral, non-negative.
        b: upper bound of the integral.

    Returns:
        The value of the integral, `0` for an empty interval.

    Raises:
        ValueError: if `nu` is not positive or `a` is negative.
    """
    if not nu > 0:
        raise ValueError(f"'nu' must be positive, got {nu}")
    if a < 0:
        raise ValueError(f"'a' must not be negative, got {a}")
    if b <= a:
        return 0.0
    constant = -gammaln(nu / 2.0) - (nu / 2.0 - 1.0) * np.log(2.0)
    root = np.sqrt(nu)

    def integrand(x: float) -> float:
        if x <= 0.0:
            return 0.0
        return float(
            ndtr(t * x / root - delta)
            * np.exp((nu - 1.0) * np.log(x) - x * x / 2.0 + constant)
        )

    # the mass of `x**(nu - 1) exp(-x**2 / 2)` sits at `sqrt(nu - 1)`; naming
    # the peak keeps the adaptive quadrature from missing it on a wide interval
    peak = float(np.sqrt(max(nu - 1.0, 0.0)))
    points = [peak] if a < peak < b and np.isfinite(b) else None
    value, _ = quad(integrand, a, b, points=points, limit=200)
    return float(value)


def _check_arguments(
    cv: float, gmr: float, limits: tuple[float, float], alpha: float
) -> None:
    """Validate the arguments shared by `power_tost` and `sample_size_tost`.

    Args:
        cv: the within-subject coefficient of variation.
        gmr: the assumed geometric mean ratio.
        limits: the acceptance limits.
        alpha: the level of each one-sided test.

    Raises:
        ValueError: for a negative `cv`, a non-positive `gmr`, reversed or
            non-positive limits, or an `alpha` outside `(0, 0.5)`.
    """
    if cv < 0:
        raise ValueError(f"'cv' must not be negative, got {cv}")
    if not gmr > 0:
        raise ValueError(f"'gmr' must be positive, got {gmr}")
    if not 0 < limits[0] < limits[1]:
        raise ValueError(
            f"'limits' must be (low, high) with 0 < low < high, got {limits}"
        )
    if not 0 < alpha < 0.5:
        raise ValueError(f"'alpha' must lie in (0, 0.5), got {alpha}")


def _design(design: DesignName | str) -> TOSTDesign:
    """The design of a name.

    Args:
        design: the name of the design.

    Returns:
        The design.

    Raises:
        ValueError: if the name is not a known design.
    """
    if design not in DESIGNS:
        known = ", ".join(f"'{name}'" for name in DESIGNS)
        raise ValueError(f"'{design}' is not a known design, use one of {known}")
    return DESIGNS[design]


def power_tost(
    *,
    cv: float,
    n: int,
    gmr: float = 0.95,
    limits: tuple[float, float] = (0.8, 1.25),
    design: DesignName | str = "2x2",
    alpha: float = 0.05,
) -> float:
    r"""The exact power of the two one-sided tests procedure.

    With \(\sigma = \sqrt{\ln(1 + \mathrm{CV}^2)}\) the standard deviation of
    the log values, \(\mathrm{se} = \sigma\sqrt{b_k/n}\) the standard error of
    the log ratio of the design, \(\Delta = \ln \mathrm{GMR}\) and the log
    limits \(\ln\theta_L\), \(\ln\theta_U\), the two non-centralities are

    $$\delta_1 = \frac{\Delta - \ln\theta_L}{\mathrm{se}}, \qquad
    \delta_2 = \frac{\Delta - \ln\theta_U}{\mathrm{se}},$$

    and with \(t = t_{1-\alpha,\nu}\) and
    \(R = (\delta_1 - \delta_2)\sqrt{\nu} / (2t)\) the power is the difference
    of two Owen's Q values,

    $$1 - \beta = Q_\nu(-t, \delta_2; 0, R) - Q_\nu(t, \delta_1; 0, R),$$

    the exact probability that both one-sided tests reject (Owen 1965; the
    algorithm of `PowerTOST::power.TOST`). The study is assumed balanced, so
    an odd `n` of a crossover is the balanced approximation of the study with
    one subject more in one sequence, which is slightly optimistic.

    Args:
        cv: the within-subject coefficient of variation as a fraction (the
            total CV for a parallel design), 0.3 for 30 %.
        n: the total number of subjects of the study.
        gmr: the geometric mean ratio the study is powered for; 0.95 is the
            usual assumption of a 5 % difference of the formulations.
        limits: the acceptance limits of the ratio.
        design: the design, one of `DESIGNS`.
        alpha: the level of each one-sided test, 0.05 for a 90 % interval.

    Returns:
        The power, a probability.

    Raises:
        ValueError: for an unknown design, a negative `cv`, a non-positive
            `gmr`, reversed limits, an `alpha` outside `(0, 0.5)`, or an `n`
            which leaves no degree of freedom.
    """
    _check_arguments(cv, gmr, limits, alpha)
    resolved = _design(design)
    df = resolved.df(int(n))
    if df < 1:
        raise ValueError(
            f"a '{resolved.name}' study of {n} subjects has {df:.0f} degrees of "
            "freedom; give more subjects"
        )
    sigma = float(np.sqrt(np.log1p(cv**2)))
    se = resolved.se(sigma, int(n))
    difference = float(np.log(gmr))
    if se == 0.0:
        # no variability: the tests reject whenever the ratio lies strictly
        # within the limits
        return float(limits[0] < gmr < limits[1])
    delta_low = (difference - float(np.log(limits[0]))) / se
    delta_high = (difference - float(np.log(limits[1]))) / se
    t_value = float(student_t.ppf(1.0 - alpha, df))
    bound = (delta_low - delta_high) * float(np.sqrt(df)) / (2.0 * t_value)
    power = owens_q(df, -t_value, delta_high, 0.0, bound) - owens_q(
        df, t_value, delta_low, 0.0, bound
    )
    return float(min(max(power, 0.0), 1.0))


def sample_size_tost(
    *,
    cv: float,
    gmr: float = 0.95,
    target_power: float = 0.8,
    design: DesignName | str = "2x2",
    alpha: float = 0.05,
    limits: tuple[float, float] = (0.8, 1.25),
    max_n: int = 100000,
) -> int:
    """The smallest number of subjects which reaches a target power.

    The search walks the sample size upwards from a lower bound derived from
    the normal approximation and returns the first size whose `power_tost`
    reaches `target_power`. A crossover is searched in steps of two, so that
    the sequences carry the same number of subjects, and starts at four; a
    parallel design is searched in steps of one and starts at three. The
    number returned is the total number of subjects of the study, not the
    number per sequence or per group.

    Args:
        cv: the within-subject coefficient of variation as a fraction (the
            total CV for a parallel design).
        gmr: the geometric mean ratio the study is powered for.
        target_power: the power to reach, 0.8 or 0.9 in practice.
        design: the design, one of `DESIGNS`.
        alpha: the level of each one-sided test.
        limits: the acceptance limits of the ratio.
        max_n: the largest sample size the search looks at.

    Returns:
        The total number of subjects.

    Raises:
        ValueError: for an unknown design, a `target_power` outside
            `(0, 1)`, a `gmr` outside the limits (no sample size reaches any
            power then), or as `power_tost`; also when `max_n` subjects do
            not reach the power.
    """
    _check_arguments(cv, gmr, limits, alpha)
    if not 0 < target_power < 1:
        raise ValueError(f"'target_power' must lie in (0, 1), got {target_power}")
    if not limits[0] < gmr < limits[1]:
        raise ValueError(
            f"'gmr' {gmr} lies outside the limits {limits}; no sample size reaches "
            "a power there, the study cannot pass"
        )
    resolved = _design(design)
    minimum = 4 if resolved.step == 2 else 3

    def reached(size: int) -> bool:
        """Whether a study of `size` subjects reaches the target power."""
        if resolved.df(size) < 1:
            return False
        return (
            power_tost(
                cv=cv, n=size, gmr=gmr, limits=limits, design=design, alpha=alpha
            )
            >= target_power
        )

    # the normal approximation of the sample size is a lower bound of the exact
    # one, so the search starts there instead of at the smallest feasible size
    distance = min(
        abs(float(np.log(gmr)) - float(np.log(limits[0]))),
        abs(float(np.log(limits[1])) - float(np.log(gmr))),
    )
    z = float(norm.ppf(1.0 - alpha)) + float(norm.ppf(target_power))
    guess = int(np.ceil(resolved.bk * float(np.log1p(cv**2)) * z**2 / distance**2))
    n = max(minimum, guess)
    n += (-n) % resolved.step if resolved.step == 2 else 0
    while n <= max_n and not reached(n):
        n += resolved.step
    if n > max_n:
        raise ValueError(
            f"a '{resolved.name}' study does not reach the power {target_power} with "
            f"{max_n} subjects at CV {cv} and GMR {gmr}"
        )
    # the approximation may have started above the exact answer
    while n - resolved.step >= minimum and reached(n - resolved.step):
        n -= resolved.step
    logger.debug(
        "sample size %d for a %s study at CV %.1f %% and GMR %.3f",
        n,
        resolved.name,
        cv * 100.0,
        gmr,
    )
    return n

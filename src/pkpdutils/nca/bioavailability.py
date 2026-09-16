r"""Absolute and relative bioavailability from two non-compartmental analyses.

The fraction of a dose which reaches the systemic circulation is not measured
directly: it is the exposure of the test treatment against the exposure of a
reference treatment, both divided by their dose,

$$F = \frac{\mathrm{AUC}_\mathrm{test} / D_\mathrm{test}}
{\mathrm{AUC}_\mathrm{ref} / D_\mathrm{ref}}.$$

With an intravenous reference, whose fraction absorbed is 1 by definition, this
is the **absolute** bioavailability \(F_\mathrm{abs}\) (CDISC `FABS`); with any
other reference - another formulation, another route, a fed against a fasted
state - it is the **relative** bioavailability \(F_\mathrm{rel}\) (CDISC
`FREL`). The FDA bioavailability guidance asks for exactly this comparison, and
PKNCA computes it as `pk.calc.f`.

`bioavailability` is the geometric mean ratio of the dose normalized exposures
with its confidence interval, the same estimator a bioequivalence study uses
(`pkpdutils.stats.ratio`): the exposures are log-normal, the subjects of a
crossover are paired and the interval is a t interval on the log scale.
"""

import logging
from dataclasses import replace

from pkpdutils.nca.result import (
    DOSE_NORMALIZED_NAMES,
    DOSE_NORMALIZED_SUFFIX,
    NCAResult,
)
from pkpdutils.stats.ratio import RatioResult, ratio
from pkpdutils.timecourse import Route

logger = logging.getLogger(__name__)

#: name of the ratio with an intravenous reference (CDISC `FABS`)
ABSOLUTE_NAME: str = "f_abs"

#: name of the ratio with any other reference (CDISC `FREL`)
RELATIVE_NAME: str = "f_rel"

#: variables only an intravenous analysis reports, which is how the route of a
#: reference result is recognized when it is not given
INTRAVENOUS_VARIABLES: tuple[str, ...] = ("c0", "vss", "cl", "vz")


def is_intravenous(result: NCAResult) -> bool:
    """Whether a result comes from an intravenous analysis.

    A result carries no route of its own, but it carries the variables the
    route decides: an intravenous analysis reports `cl`, `vz` and `vss` (and
    `c0` after a bolus) where an extravascular one reports `cl_f` and `vz_f`.

    Args:
        result: the result

    Returns:
        `True` when the result carries an intravenous variable and no
        extravascular one.
    """
    variables = set(result.ds.data_vars)
    extravascular = {"cl_f", "vz_f", "cl_ss_f", "tlag", "cmax_half"} & variables
    return not extravascular and bool(set(INTRAVENOUS_VARIABLES) & variables)


def bioavailability(
    test: NCAResult,
    reference: NCAResult,
    *,
    dim: str,
    parameter: str = "auc_inf_obs",
    paired: bool | None = None,
    ci_level: float = 0.90,
    reference_route: Route | None = None,
) -> RatioResult:
    r"""Absolute or relative bioavailability of a test against a reference analysis.

    The dose normalized exposure of every subject
    (`NCAResult.dose_normalized`, \(x / D\)) of both results is compared as a
    geometric mean ratio with its confidence interval
    (`pkpdutils.stats.ratio`),

    $$F = \frac{\mathrm{AUC}_\mathrm{test} / D_\mathrm{test}}
    {\mathrm{AUC}_\mathrm{ref} / D_\mathrm{ref}},$$

    paired by the label of `dim` when both results carry the same subjects (a
    crossover) and unpaired otherwise (a parallel design, the Welch interval).
    The name of the result says which quantity it is: `f_abs` with an
    intravenous reference, whose fraction absorbed is 1, and `f_rel` with any
    other one.

    Args:
        test: the analysis of the test treatment
        reference: the analysis of the reference treatment

    Keyword Args:
        dim: the sample dimension of the subjects of both results
        parameter: the exposure to compare, `auc_inf_obs` by default;
            `auc_last` and `auc_tau` are the other usual choices
        paired: pair the subjects by label; `None` pairs when both results
            carry the same labels (`pkpdutils.stats.ratio`)
        ci_level: level of the interval, 0.90 as in bioequivalence
        reference_route: the route of the reference treatment; `None` reads it
            off the variables of the reference result (`is_intravenous`)

    Returns:
        The ratio, named `f_abs` or `f_rel`, with the unit `dimensionless`:
        the exposures are divided by their dose before they are compared.

    Raises:
        ValueError: if a result carries no dose (`NCAResult.dose_normalized`),
            if `parameter` is not a parameter of both results or if `dim` is
            not a sample dimension of both.
    """
    for name, result in (("test", test), ("reference", reference)):
        if parameter not in result.ds.data_vars:
            raise ValueError(f"'{parameter}' is not a parameter of the {name} result")
    variable = DOSE_NORMALIZED_NAMES.get(
        parameter, f"{parameter}{DOSE_NORMALIZED_SUFFIX}"
    )
    test_sample, reference_sample = (
        result.dose_normalized([parameter]).sample(variable, dim)
        for result in (test, reference)
    )
    intravenous = (
        reference_route.is_iv
        if reference_route is not None
        else is_intravenous(reference)
    )
    name = ABSOLUTE_NAME if intravenous else RELATIVE_NAME
    logger.debug("bioavailability: %s of the dose normalized '%s'", name, parameter)
    estimate = ratio(test_sample, reference_sample, ci_level=ci_level, paired=paired)
    return replace(estimate, name=name, unit="dimensionless")

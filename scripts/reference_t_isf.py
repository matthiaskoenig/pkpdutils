"""High precision reference of the Student t inverse survival function, for the kernel tests.

Usage: ``python scripts/reference_t_isf.py P DF [DF ...]`` prints ``t`` with ``P(T > t) = P``
at 60 digits for every degrees of freedom. Needs ``mpmath``, which the development extra
carries (``uv run python scripts/reference_t_isf.py 1e-300 3 10 2.5``).

The survival function is ``betainc(df/2, 1/2, 0, df/(df+t^2), regularized=True) / 2`` and is
inverted by bisection on ``log t``, which is what keeps it usable at ``p = 1e-300`` where ``t``
reaches 1e100 and a root finder on ``t`` itself does not converge.

The reference belongs to the exact ``f64`` of the input, not to the decimal literal: ``p`` and
``df`` go through ``float`` before they become ``mpf``. ``-36.79`` is not the number the kernel
sees, and a reference for the literal differs from the right one at 1e-16 relative, which is
visible in the far tail where the result changes by 1e-14 per ulp of the argument.
"""

import sys
from typing import Any

from mpmath import betainc, exp, log, mp, mpf

# mpmath creates `mpf` at runtime out of its context, so it is a value rather than a
# static type and cannot stand in an annotation: the precise numbers are `Any` here


def student_t_sf(t: Any, df: Any) -> Any:
    return betainc(df / 2, mpf(1) / 2, 0, df / (df + t * t), regularized=True) / 2


def student_t_isf(p: float, df: float) -> Any:
    mp.dps = 60
    p_exact, df_exact = mpf(float(p)), mpf(float(df))
    target = log(p_exact)
    # bracket log t between the asymptote of the tail and its double, widened until it holds
    low, high = mpf(-5), mpf(5)
    while log(student_t_sf(exp(low), df_exact)) < target:
        low -= 10
    while log(student_t_sf(exp(high), df_exact)) > target:
        high += 10
    for _ in range(400):
        middle = (low + high) / 2
        if log(student_t_sf(exp(middle), df_exact)) > target:
            low = middle
        else:
            high = middle
    return exp(low)


if __name__ == "__main__":
    probability = float(sys.argv[1])
    for df in sys.argv[2:]:
        print(
            f"df={df} p={probability!r} t={mp.nstr(student_t_isf(probability, float(df)), 20)}"
        )

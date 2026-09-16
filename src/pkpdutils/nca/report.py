r"""The tables and the methods sentence of a regulatory report.

ICH M13A (2024, section 2.2.2.2) names what the pharmacokinetic section of a
bioequivalence report carries: the summary statistics of every parameter ("n,
geometric mean, geometric coefficient of variation, median, arithmetic mean,
standard deviation, minimum and maximum"), the ratio
\(\mathrm{AUC}_{0\text{-}t}/\mathrm{AUC}_{0\text{-}\infty}\) of every
subject with the acceptance rule that the study is questioned when the ratio is
below 80 % "in more than 20% of the observations", and a description of the
methods, verbatim "the non-compartmental methods used to derive the PK
parameters from the raw data should be reported, e.g., linear trapezoidal
method for AUC and the number of data points of the terminal log-linear phase
used to estimate kel". The FDA ANDA bioequivalence guidance repeats the list.

`M13A_STATISTICS` is the first, `acceptability_table` the second and
`methods_line` the third; every number they report comes from an `NCAResult`,
so the report is assembled from the analysis rather than typed.
"""

from typing import Any

import numpy as np
import pandas as pd

from pkpdutils.nca.options import AUCMethod, NCAOptions, TerminalMethod
from pkpdutils.nca.result import NCAResult

#: the summary statistics ICH M13A (2024, 2.2.2.2) names, in its order, for
#: `pkpdutils.result.summary_table(result, dim, stats=M13A_STATISTICS)`
M13A_STATISTICS: tuple[str, ...] = (
    "n",
    "geomean",
    "geocv",
    "median",
    "mean",
    "sd",
    "min",
    "max",
)

#: how a trapezoid rule is named in the methods sentence
AUC_METHOD_NAMES: dict[AUCMethod, str] = {
    AUCMethod.LINEAR: "the linear trapezoidal method",
    AUCMethod.LINEAR_LOG: "the linear up / logarithmic down trapezoidal method",
    AUCMethod.LOG: "the logarithmic trapezoidal method",
}

#: how a terminal phase rule is named in the methods sentence
TERMINAL_METHOD_NAMES: dict[TerminalMethod, str] = {
    TerminalMethod.BEST_FIT: (
        "the points of the largest adjusted coefficient of determination"
    ),
    TerminalMethod.LAST_N: "the last points of the curve",
    TerminalMethod.ALL_AFTER_TMAX: "every point after the maximum",
    TerminalMethod.MANUAL: "the points selected by the analyst",
}


def acceptability_table(
    result: NCAResult,
    dim: str,
    *,
    threshold: float = 0.8,
    share: float = 0.2,
    include_excluded: bool = False,
    **indexers: Any,
) -> tuple[pd.DataFrame, bool]:
    r"""The acceptability of the extrapolation of every subject, and the verdict.

    One row per sample with \(\mathrm{AUC}_{0\text{-}t_\mathrm{last}}\),
    \(\mathrm{AUC}_{0\text{-}\infty}\), their ratio

    $$q = \frac{\mathrm{AUC}_{0\text{-}t_\mathrm{last}}}
    {\mathrm{AUC}_{0\text{-}\infty}}$$

    and whether it is below `threshold`. ICH M13A (2024, 2.2.2.2) questions a
    study in which \(q\) is below 80 % "in more than 20% of the observations",
    which is the verdict: `True` while the share of the samples below the
    threshold is at most `share`. Only the samples with a finite ratio are
    counted, a sample without a terminal phase being no observation of the
    rule.

    Args:
        result: the result of the individual samples
        dim: the sample dimension of the subjects

    Keyword Args:
        threshold: the smallest acceptable ratio, 0.8 of ICH M13A
        share: the share of the samples which may fall below it, 0.2 of ICH M13A
        include_excluded: count the excluded samples as well
        **indexers: coordinate label per remaining sample dimension

    Returns:
        The table with the columns of the sample dimensions, `auc_last`,
        `auc_inf_obs`, `ratio` and `below`, and the verdict of the rule.

    Raises:
        ValueError: if `dim` is not a sample dimension of the result, or if the
            result carries no `auc_last` or `auc_inf_obs`.
    """
    if dim not in result.sample_dims:
        raise ValueError(f"'{dim}' is not a sample dimension {result.sample_dims}")
    missing = [
        name for name in ("auc_last", "auc_inf_obs") if name not in result.ds.data_vars
    ]
    if missing:
        raise ValueError(f"the result carries no {missing}")
    sample = result.sample(
        "auc_last", dim, include_excluded=include_excluded, **indexers
    )
    infinite = result.sample(
        "auc_inf_obs", dim, include_excluded=include_excluded, **indexers
    )
    assert sample.values is not None and infinite.values is not None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sample.values / infinite.values
    below = np.isfinite(ratio) & (ratio < threshold)
    counted = int(np.isfinite(ratio).sum())
    frame = pd.DataFrame(
        {
            dim: sample.labels if sample.labels is not None else np.arange(ratio.size),
            "auc_last": sample.values,
            "auc_inf_obs": infinite.values,
            "ratio": ratio,
            "below": below,
        }
    )
    acceptable = counted == 0 or int(below.sum()) / counted <= share
    return frame, bool(acceptable)


def methods_line(options: NCAOptions, result: NCAResult) -> str:
    """The sentence of the methods section which names how the analysis ran.

    The trapezoid rule of the areas, the rule which selected the terminal
    log-linear phase and the number of points it used, which is what ICH M13A
    (2024, 2.2.2.2) and the FDA ANDA bioequivalence guidance ask a report to
    state.

    Args:
        options: the options of the analysis
        result: its result, for the number of terminal points

    Returns:
        The sentence, without a leading or trailing space.
    """
    area = AUC_METHOD_NAMES[options.auc_method]
    selection = TERMINAL_METHOD_NAMES[options.terminal.method]
    by_hand = (
        " The terminal window of single profiles was set by hand."
        if options.terminal.windows
        else ""
    )
    points = ""
    if "lambda_z_n_points" in result.ds.data_vars:
        counts = np.asarray(result.ds["lambda_z_n_points"].to_numpy(), dtype=np.float64)
        counts = counts[np.isfinite(counts)]
        if counts.size:
            low, high = int(counts.min()), int(counts.max())
            points = (
                f" using {low} data points"
                if low == high
                else f" using {low} to {high} data points"
            )
    return (
        f"The areas were computed with {area}. The terminal log-linear phase "
        f"was selected as {selection} and estimated by log-linear regression"
        f"{points}.{by_hand}"
    )

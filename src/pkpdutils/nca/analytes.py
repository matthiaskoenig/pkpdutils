"""Several analytes in one analysis: the metabolite to parent ratio.

A study of a parent drug and its metabolite (or of two enantiomers) is one
batch with a `substance` coordinate along a sample dimension, `analyte` by
default, and is analysed in one call of `pkpdutils.nca.nca`: every row keeps
its own substance and the result carries the coordinate. `metabolite_ratio`
then divides the exposure of the metabolite by the exposure of the parent,
subject by subject, which is the metabolite to parent ratio ICH M13A asks for
when a metabolite contributes to the effect.
"""

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from pkpdutils.result import ParameterResult

logger = logging.getLogger(__name__)

#: the parameters the metabolite to parent ratio is reported for by default
DEFAULT_RATIO_PARAMETERS: tuple[str, ...] = ("auc_inf_obs", "cmax")


def _analyte_position(result: ParameterResult, substance: str) -> tuple[str, int]:
    """The dimension and the position of one analyte of a result.

    Args:
        result: the result of the analysis of several analytes.
        substance: the substance to find in the `substance` coordinate.

    Returns:
        The name of the dimension the analytes lie along and the position of
        the substance along it.

    Raises:
        ValueError: if the result carries no `substance` coordinate, if the
            coordinate does not lie along exactly one dimension, or if the
            substance is not among its values or is there more than once.
    """
    if "substance" not in result.ds.coords:
        raise ValueError(
            "the result carries no 'substance' coordinate: analyse the "
            "analytes in one batch (Timecourses.from_timecourses of curves "
            "with different substances, or a reader with 'analytes=')"
        )
    coord = result.ds.coords["substance"]
    if len(coord.dims) != 1:
        raise ValueError(
            "the 'substance' coordinate must lie along one sample dimension, "
            f"not {tuple(str(d) for d in coord.dims)}"
        )
    dim = str(coord.dims[0])
    values = [str(value) for value in coord.to_numpy().ravel()]
    positions = [i for i, value in enumerate(values) if value == substance]
    if len(positions) != 1:
        raise ValueError(
            f"'{substance}' is {len(positions)} times among the substances "
            f"{values} of the result"
        )
    return dim, positions[0]


def metabolite_ratio(
    result: ParameterResult,
    *,
    parent: str,
    metabolite: str,
    parameters: Sequence[str] = DEFAULT_RATIO_PARAMETERS,
    molar: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    r"""The metabolite to parent ratio of every subject.

    For every parameter the ratio of the metabolite to the parent of the same
    subject,

    $$\mathrm{MPR} = \frac{X_\mathrm{metabolite}}{X_\mathrm{parent}},$$

    of the two analytes of the result, which are two positions of the sample
    dimension the `substance` coordinate lies along. With the molar masses
    (`molar`, in the same unit for both, usually g/mol) the ratio is on a molar
    basis,

    $$\mathrm{MPR}_\mathrm{molar}
    = \frac{X_\mathrm{metabolite} / M_\mathrm{metabolite}}
           {X_\mathrm{parent} / M_\mathrm{parent}}
    = \mathrm{MPR}\, \frac{M_\mathrm{parent}}{M_\mathrm{metabolite}},$$

    which is what a metabolite to parent ratio of two substances measured in
    mass concentrations means (ICH M13A 2024, 2.2.3). The parameters must have
    the same unit for both analytes, which they do when the two curves were
    measured in the same unit; the ratio is dimensionless.

    Args:
        result: the result of the analysis of both analytes, which carries the
            `substance` coordinate

    Keyword Args:
        parent: the substance of the parent, a value of the coordinate
        metabolite: the substance of the metabolite, a value of the coordinate
        parameters: the parameters to divide, `auc_inf_obs` and `cmax` by
            default
        molar: molar mass per substance, `{"parent": 300.4, "metabolite":
            316.4}`; `None` reports the ratio of the values as they are

    Returns:
        One row per subject: the labels of the remaining sample dimensions and
        one column per parameter with the ratio. `attrs` name the two
        substances and the correction factor.

    Raises:
        ValueError: if the result does not carry the two analytes
            (`_analyte_position`), if a parameter is not a variable of the
            result or does not lie along the analyte dimension, or if a molar
            mass is missing for one of the two substances.
    """
    dim, parent_index = _analyte_position(result, parent)
    _, metabolite_index = _analyte_position(result, metabolite)
    factor = 1.0
    if molar is not None:
        missing = sorted({parent, metabolite} - set(molar))
        if missing:
            raise ValueError(f"no molar mass for {missing}")
        factor = float(molar[parent]) / float(molar[metabolite])
    ds = result.ds
    sample_dims = tuple(d for d in result.sample_dims if d != dim)
    shape = tuple(int(ds.sizes[d]) for d in sample_dims)
    columns: dict[str, np.ndarray] = {}
    for name in parameters:
        if name not in ds.data_vars:
            raise ValueError(f"'{name}' is not a variable of the result")
        if dim not in ds[name].dims:
            raise ValueError(f"'{name}' does not have the analyte dimension '{dim}'")
        values = ds[name].transpose(dim, *sample_dims)
        numerator = values.isel({dim: metabolite_index}, drop=True)
        denominator = values.isel({dim: parent_index}, drop=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            columns[name] = (
                np.asarray(numerator.to_numpy(), dtype=np.float64)
                / np.asarray(denominator.to_numpy(), dtype=np.float64)
                * factor
            )
    frame = pd.DataFrame(
        {name: np.asarray(values).ravel() for name, values in columns.items()}
    )
    labels: dict[str, np.ndarray] = {}
    for position, sample_dim in enumerate(sample_dims):
        coord = (
            ds.coords[sample_dim].to_numpy()
            if sample_dim in ds.coords
            else np.arange(shape[position])
        )
        repeats = int(np.prod(shape[position + 1 :], dtype=int))
        tiles = int(np.prod(shape[:position], dtype=int))
        labels[sample_dim] = np.tile(np.repeat(coord, repeats), tiles)
    for name in reversed(list(labels)):
        frame.insert(0, name, labels[name])
    frame.attrs.update(
        {"parent": parent, "metabolite": metabolite, "molar_factor": factor}
    )
    return frame

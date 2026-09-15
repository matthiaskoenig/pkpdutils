"""Comparison of models by the corrected Akaike information criterion.

`compare_models` fits every model to the same data and ranks them per sample
by AICc (Burnham & Anderson 2002, ch. 2): `Delta_i = AICc_i - min_j AICc_j`
and the Akaike weight `w_i = exp(-Delta_i / 2) / sum_j exp(-Delta_j / 2)`, the
probability that model `i` is the best of the set given the candidates
considered. A model whose AICc is `NaN` (too few points for its number of
parameters, `n - k - 1 <= 0`) gets weight 0 and is never picked as `best`;
when every model of a sample is `NaN`, `best` is the empty string.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pkpdutils.fit.engine import fit
from pkpdutils.fit.model import Model
from pkpdutils.fit.options import FitOptions
from pkpdutils.fit.result import FitResult


@dataclass
class ModelComparison:
    """The fits of several models and their ranking by AICc.

    Attributes:
        results: model name to its `FitResult`.
        table: one row per sample and model, with the sample dims, `model`,
            `n_parameters`, `aicc`, `delta_aicc`, `akaike_weight` and `best`.
        best: name of the best model per sample, over the sample dimensions
            (empty string for a sample where no model fitted).
    """

    results: dict[str, FitResult]
    table: pd.DataFrame
    best: xr.DataArray


def compare_models(
    models: Sequence[Model],
    x: Any,
    y: Any,
    *,
    sd: Any | None = None,
    options: FitOptions | None = None,
    x_unit: str = "dimensionless",
    y_unit: str = "dimensionless",
    dims: Sequence[str] | None = None,
    coords: dict[str, Any] | None = None,
) -> ModelComparison:
    """Fit every model to the same data and rank them per sample by AICc.

    `Delta_i = AICc_i - min_j AICc_j` and the Akaike weight
    `w_i = exp(-Delta_i / 2) / sum_j exp(-Delta_j / 2)` (Burnham & Anderson
    2002, ch. 2) are computed independently for every sample, so a different
    model can be the best fit of different samples of a batch.

    Args:
        models: the candidate models, with distinct `name`s.
        x: independent variable, as for `fit`.
        y: dependent variable, as for `fit`.
        sd: standard deviations, as for `fit`.
        options: fit options shared by every model.
        x_unit: unit of `x`.
        y_unit: unit of `y`.
        dims: sample dimension names for a 2-D `y`.
        coords: coordinates of the sample dimensions.

    Returns:
        The comparison.

    Raises:
        ValueError: if two models share a `name`.
    """
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"Model names must be distinct: {names}")
    results = {
        m.name: fit(
            m,
            x,
            y,
            sd=sd,
            options=options,
            x_unit=x_unit,
            y_unit=y_unit,
            dims=dims,
            coords=coords,
        )
        for m in models
    }
    first = next(iter(results.values()))
    sample_dims = first.sample_dims
    aicc = np.stack(
        [results[name]["aicc"].to_numpy() for name in names], axis=-1
    )  # (*sample_shape, n_models)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(aicc)
        best_value = np.nanmin(np.where(finite, aicc, np.inf), axis=-1)
        delta = np.where(finite, aicc - best_value[..., None], np.nan)
        weight = np.where(finite, np.exp(-0.5 * np.where(finite, delta, 0.0)), 0.0)
        total = weight.sum(axis=-1, keepdims=True)
        weight = np.where(total > 0, weight / np.where(total > 0, total, 1.0), 0.0)
    has_any = finite.any(axis=-1)
    best_index = np.where(has_any, np.argmax(weight, axis=-1), len(names))
    best_names = np.array([*names, ""], dtype=object)[best_index]
    coords_sample = {d: first.ds[d] for d in sample_dims if d in first.ds.coords}
    best = xr.DataArray(
        best_names, dims=sample_dims, coords=coords_sample, name="best_model"
    )
    rows = []
    for index in np.ndindex(*aicc.shape[:-1]):
        labels = {
            d: (first.ds[d].values[i] if d in first.ds.coords else i)
            for d, i in zip(sample_dims, index, strict=True)
        }
        for j, name in enumerate(names):
            rows.append(
                {
                    **labels,
                    "model": name,
                    "n_parameters": int(
                        results[name]["n_parameters"].to_numpy()[index]
                    ),
                    "aicc": float(aicc[(*index, j)]),
                    "delta_aicc": float(delta[(*index, j)]),
                    "akaike_weight": float(weight[(*index, j)]),
                    "best": bool(has_any[index] and best_index[index] == j),
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            *sample_dims,
            "model",
            "n_parameters",
            "aicc",
            "delta_aicc",
            "akaike_weight",
            "best",
        ],
    )
    return ModelComparison(results=results, table=table, best=best)

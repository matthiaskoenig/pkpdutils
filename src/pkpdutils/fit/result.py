"""Result of a fit: parameters, uncertainties, statistics, data and predictions."""

from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import xarray as xr

from pkpdutils.fit.model import Model
from pkpdutils.fit.options import FitFlag
from pkpdutils.result import ParameterResult


class FitResult(ParameterResult):
    """Parameters of a fit as an `xarray.Dataset` over the sample dimensions.

    Variables: every parameter `p` with `p_se`, `p_ci_low`, `p_ci_high`,
    `p_cv`; the derived parameters likewise; the statistics `cost`, `r2`,
    `rmse`, `aic`, `aicc`, `bic`, `n_points`, `n_parameters`,
    `n_starts_converged`; the data and the prediction per point (`x_data`,
    `y_data`, `y_pred`, `residuals` over `point`); the correlation matrix over
    `(parameter, parameter_)`; and the integer `flags` (`FitFlag`). The model
    object is kept for `predict`.
    """

    flag_type: ClassVar[type[FitFlag]] = FitFlag
    discrete_parameters: ClassVar[frozenset[str]] = frozenset(
        {
            "n_points",
            "n_parameters",
            "n_starts_converged",
            "n_bootstrap",
            "flip_flop",
        }
    )

    def __init__(self, ds: xr.Dataset, model: Model) -> None:
        """Wrap the dataset of a fit of `model`.

        Args:
            ds: the result dataset of the fit.
            model: the fitted model, kept for `predict`.
        """
        super().__init__(ds)
        self.model = model

    def parameter_vector(self, **indexers: Any) -> np.ndarray:
        """The fitted parameters of one sample in the order of the model.

        Args:
            **indexers: coordinate label per sample dimension.

        Returns:
            The parameter vector, fixed parameters included.
        """
        sample = self._sample(indexers)
        return np.array(
            [float(sample[name].values) for name in self.model.parameter_names]
        )

    def predict(self, x: np.ndarray, **indexers: Any) -> np.ndarray:
        """The fitted curve of one sample at `x`.

        Args:
            x: the independent variable to predict at.
            **indexers: coordinate label per sample dimension.

        Returns:
            The predicted values at `x`.
        """
        return self.model.predict(
            np.asarray(x, dtype=np.float64), self.parameter_vector(**indexers)
        )

    def predict_all(self, x: np.ndarray) -> xr.DataArray:
        """The fitted curves of every sample at `x`, over `(*sample_dims, "x")`.

        Args:
            x: the independent variable to predict at.

        Returns:
            The predicted curves of every sample.
        """
        xs = np.asarray(x, dtype=np.float64)
        names = self.model.parameter_names
        stacked = np.stack(
            [self.ds[name].to_numpy() for name in names], axis=-1
        )  # (*sample_shape, k)
        flat = stacked.reshape(-1, len(names))
        curves = np.stack([self.model.predict(xs, p) for p in flat]).reshape(
            *stacked.shape[:-1], xs.size
        )
        coords: dict[str, Any] = {
            d: self.ds[d] for d in self.sample_dims if d in self.ds.coords
        }
        coords["x"] = xs
        return xr.DataArray(
            curves,
            dims=(*self.sample_dims, "x"),
            coords=coords,
            name="y_pred",
            attrs={"units": self.units("y_pred")},
        )

    def correlation(self, **indexers: Any) -> pd.DataFrame:
        """The correlation matrix of the fitted parameters of one sample.

        Args:
            **indexers: coordinate label per sample dimension.

        Returns:
            The correlation matrix indexed by the parameter names.
        """
        sample = self._sample(indexers)
        names = [str(name) for name in sample["parameter"].values]
        return pd.DataFrame(sample["correlation"].values, index=names, columns=names)

    def _new(self, ds: xr.Dataset) -> Self:
        """A result of the same kind (used by `summarize`), keeping the model.

        Args:
            ds: the dataset of the new result.

        Returns:
            The new result with the same model.
        """
        return type(self)(ds, self.model)

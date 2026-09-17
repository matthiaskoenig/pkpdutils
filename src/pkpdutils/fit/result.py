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
    `p_cv` (the relative standard error as a fraction, as every coefficient
    of variation of the package); the derived parameters likewise; the
    statistics `cost`, `r2`,
    `rmse`, `aic`, `aicc`, `bic`, `n_points`, `n_parameters`,
    `n_starts_converged`, `n_bootstrap` (number of successful residual
    bootstrap replicates, 0 without bootstrap; `attrs["bootstrap"]` holds the
    requested count); the data and the prediction per point (`x_data`,
    `y_data`, `sd_data` - `NaN` when the fit had no `sd` -, `y_pred`,
    `residuals` over `point`); the correlation matrix over
    `(parameter, parameter_)`; and the integer `flags` (`FitFlag`). The model
    object is kept for `predict`.

    The parameters are reported in the raw units of the data, so they are on
    the same scale as `y_data` and `y_pred`. The interval of a parameter is
    symmetric in its search space (symmetric in the logarithm for a parameter
    fitted on a log scale, so asymmetric around the estimate), while the
    interval of a derived parameter is the delta method interval
    `d +- t se(d)` and is always symmetric around `d`, even for a strongly
    non-linear function of the parameters such as a half-life. `aic`, `aicc`
    and `bic` count the residual variance as an estimated parameter,
    `K = k + 1` (Burnham & Anderson 2002, sec. 2.2, 6.9.6), while
    `n_parameters` stays `k`, the free model parameters; `aicc` is `NaN`
    when `n - K - 1 <= 0`.

    When the fit was run with `FitOptions.bootstrap > 0` (Efron & Tibshirani
    1993, ch. 9), `p_se` and the derived standard errors are the standard
    deviation of the `n_bootstrap` converged residual bootstrap replicates
    and the intervals are their percentiles at `ci_level`, so they need not
    be symmetric around the estimate; the correlation matrix is likewise
    from the replicates. Non-converged replicates are skipped, so a low
    `n_bootstrap` relative to `attrs["bootstrap"]` (the requested count)
    signals an unstable fit. Fewer than 2 converged replicates cannot
    estimate an uncertainty at all: `p_se`, the intervals and the
    correlation then fall back to the Jacobian-based ones and
    `FitFlag.BOOTSTRAP_FALLBACK` is set in `flags`. The parameter covariance
    `cov_q` of `RowFit` (not part of this dataset) is always Jacobian-based,
    bootstrap or not. Discrete derived parameters (`discrete_parameters`,
    e.g. `flip_flop`) carry no uncertainty variables at all.

    The goodness of fit and the counts are `statistic_variables`: they
    describe the fit of one sample, not a parameter of it, so `parameters`
    leaves them out and `summarize` drops them instead of averaging them over
    the samples. `to_dataframe`, which reports the individual fits, keeps
    them.
    """

    flag_type: ClassVar[type[FitFlag]] = FitFlag
    statistic_variables: ClassVar[frozenset[str]] = frozenset(
        {
            "cost",
            "r2",
            "rmse",
            "aic",
            "aicc",
            "bic",
            "n_points",
            "n_parameters",
            "n_starts_converged",
            "n_bootstrap",
        }
    )
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

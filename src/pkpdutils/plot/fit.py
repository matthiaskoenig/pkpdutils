"""Figures of fits: data and curve with residuals, goodness of fit, dose proportionality."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from pkpdutils.fit.result import FitResult
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle


def _sample(result: FitResult, indexers: dict[str, Any]) -> xr.Dataset:
    """The dataset of one sample of a result.

    Args:
        result: the fit.
        indexers: coordinate label per sample dimension, empty for a result
            without sample dimensions.

    Returns:
        The dataset reduced to one sample.

    Raises:
        ValueError: if a sample dimension has no indexer.
    """
    missing = set(result.sample_dims) - set(indexers)
    if missing:
        raise ValueError(
            f"A label for every sample dimension is needed, missing {sorted(missing)}"
        )
    return result.ds.sel(indexers)


def _parameter_text(result: FitResult, sample: xr.Dataset) -> str:
    """`name = value +- se` per parameter of the model, comma separated.

    Args:
        result: the fit (for the parameter names of the model).
        sample: the dataset of one sample.

    Returns:
        The parameter text.
    """
    parts = []
    for name in result.model.parameter_names:
        value = float(sample[name].to_numpy())
        se = float(sample[f"{name}_se"].to_numpy())
        if np.isfinite(se):
            parts.append(f"{name} = {value:.3g} ± {se:.2g}")
        else:
            parts.append(f"{name} = {value:.3g}")
    return ", ".join(parts)


def plot_fit(
    result: FitResult,
    *,
    log_x: bool = False,
    log_y: bool = False,
    n_grid: int = 200,
    title: str | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Data, fitted curve and weighted residuals of one sample.

    The upper panel draws the data points (`x_data`, `y_data`, with error
    bars from `sd_data` when the fit had `sd`), the fitted curve on a fine
    grid between the smallest and the largest finite `x` (log-spaced when
    `log_x`), and a title with the model name, the parameters
    (`name = value +- se`) and the flags of the sample. The lower panel draws
    the weighted residuals against `x` with a zero line.

    Args:
        result: the fit.
        log_x: logarithmic x axis (and a log-spaced curve grid).
        log_y: logarithmic y axis.
        n_grid: number of points of the curve grid.
        title: title, the model name by default; the parameters and the
            flags are appended.
        style: colors and markers.
        **indexers: coordinate label per sample dimension, none for a 0-D result.

    Returns:
        The figure with the fit panel and the residual panel.
    """
    sample = _sample(result, indexers)
    x = sample["x_data"].to_numpy()
    y = sample["y_data"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    ok_plot = ok & (x > 0 if log_x else True) & (y > 0 if log_y else True)
    fig, (ax, ax_res) = plt.subplots(
        nrows=2, ncols=1, figsize=(7, 6.5), height_ratios=[3, 1], sharex=True
    )
    fig.set_layout_engine("constrained")
    if "sd_data" in sample and np.any(np.isfinite(sample["sd_data"].to_numpy())):
        sd = sample["sd_data"].to_numpy()
        ax.errorbar(
            x[ok_plot],
            y[ok_plot],
            yerr=sd[ok_plot],
            marker=style.data_marker,
            linestyle="none",
            color=style.data_color,
            markersize=style.markersize,
            capsize=2,
            label="data",
        )
    else:
        ax.plot(
            x[ok_plot],
            y[ok_plot],
            marker=style.data_marker,
            linestyle="none",
            color=style.data_color,
            markersize=style.markersize,
            label="data",
        )
    p = np.array(
        [float(sample[name].to_numpy()) for name in result.model.parameter_names]
    )
    if ok.any() and np.all(np.isfinite(p)):
        lo, hi = float(x[ok].min()), float(x[ok].max())
        if lo < hi:
            grid = (
                np.geomspace(lo, hi, n_grid) if log_x else np.linspace(lo, hi, n_grid)
            )
            ax.plot(
                grid,
                result.model.predict(grid, p),
                "-",
                color=style.fit_color,
                linewidth=style.linewidth,
                label="fit",
            )
    heading = title if title is not None else result.model.name
    flags = result.decode_flags(int(sample["flags"].to_numpy()))
    text = f"{heading}: {_parameter_text(result, sample)}"
    if flags:
        text = f"{text} [{', '.join(flags)}]"
    ax.set_title(text, fontsize="small")
    ax.set_xlabel(f"x [{result.ds.attrs['x_unit']}]")
    ax.set_ylabel(f"y [{result.ds.attrs['y_unit']}]")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize="small")
    res = sample["residuals"].to_numpy()
    ok_res = ok & np.isfinite(res) & (x > 0 if log_x else True)
    ax_res.axhline(0.0, color="gray", linewidth=1)
    ax_res.plot(
        x[ok_res],
        res[ok_res],
        marker=style.data_marker,
        linestyle="none",
        color=style.data_color,
        markersize=style.markersize,
    )
    ax_res.set_xlabel(f"x [{result.ds.attrs['x_unit']}]")
    ax_res.set_ylabel("weighted residual")
    return fig


def _sample_labels(result: FitResult) -> list[str]:
    """One label per sample, in C order of the sample dimensions.

    Args:
        result: the fit.

    Returns:
        One label per sample; the model name for a 0-D result.
    """
    if not result.sample_dims:
        return [result.model.name]
    labels = []
    for index in np.ndindex(*[result.ds.sizes[d] for d in result.sample_dims]):
        parts = [
            str(result.ds[d].to_numpy()[i]) if d in result.ds.coords else str(i)
            for d, i in zip(result.sample_dims, index, strict=True)
        ]
        labels.append("|".join(parts))
    return labels


def plot_goodness_of_fit(
    result: FitResult, *, log: bool = False, style: PlotStyle = DEFAULT_STYLE
) -> Figure:
    """Predicted against observed values of every sample, with the identity line.

    Args:
        result: the fit.
        log: logarithmic axes; non-positive points are masked out to avoid a
            warning from the log scale.
        style: colors and markers.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.set_layout_engine("constrained")
    n_point = result.ds.sizes["point"]
    y = result["y_data"].to_numpy().reshape(-1, n_point)
    pred = result["y_pred"].to_numpy().reshape(-1, n_point)
    r2 = result["r2"].to_numpy().reshape(-1)
    labels = _sample_labels(result)
    cmap = plt.get_cmap(style.cmap)
    n = y.shape[0]
    for i in range(n):
        ok = np.isfinite(y[i]) & np.isfinite(pred[i])
        if log:
            ok &= (y[i] > 0) & (pred[i] > 0)
        label = f"{labels[i]} (R² = {r2[i]:.3f})" if n <= 8 else None
        color = cmap(i / max(n - 1, 1)) if n > 1 else style.data_color
        ax.scatter(
            y[i][ok], pred[i][ok], color=color, s=style.markersize**2 * 1.5, label=label
        )
    finite = np.concatenate([y[np.isfinite(y)], pred[np.isfinite(pred)]])
    if log:
        finite = finite[finite > 0]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="identity")
    unit = result.ds.attrs["y_unit"]
    ax.set_xlabel(f"observed [{unit}]")
    ax.set_ylabel(f"predicted [{unit}]")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend(fontsize="small")
    return fig


def plot_dose_proportionality(
    result: FitResult,
    *,
    test: xr.Dataset | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Log-log exposure against dose with the power fit and the acceptance bounds.

    Args:
        result: fit of `Power` (parameters `a`, `b`).
        test: the dataset of `proportionality_test`; when given, draws the
            acceptance wedge through the first data point and the verdict
            ("proportional", "inconclusive" or "not proportional") in the title.
        style: colors and markers.
        **indexers: coordinate label per sample dimension, none for a 0-D result.

    Returns:
        The figure.
    """
    sample = _sample(result, indexers)
    x = sample["x_data"].to_numpy()
    y = sample["y_data"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.set_layout_engine("constrained")
    ax.plot(
        x[ok],
        y[ok],
        marker=style.data_marker,
        linestyle="none",
        color=style.data_color,
        markersize=style.markersize,
        label="data",
    )
    a, b = float(sample["a"].to_numpy()), float(sample["b"].to_numpy())
    ci_low, ci_high = (
        float(sample["b_ci_low"].to_numpy()),
        float(sample["b_ci_high"].to_numpy()),
    )
    heading = f"b = {b:.3f}"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        heading = f"{heading} [{ci_low:.3f}, {ci_high:.3f}]"
    t = None if test is None else (test.sel(indexers) if indexers else test)
    if ok.any() and np.isfinite(a) and np.isfinite(b):
        lo, hi = float(x[ok].min()), float(x[ok].max())
        if lo < hi:
            grid = np.geomspace(lo, hi, 100)
            ax.plot(
                grid,
                a * grid**b,
                "-",
                color=style.fit_color,
                linewidth=style.linewidth,
                label=f"a·x^b, b = {b:.3f}",
            )
            if t is not None:
                bound_low = float(t["bound_low"].to_numpy())
                bound_high = float(t["bound_high"].to_numpy())
                x0, y0 = grid[0], a * grid[0] ** b
                ax.fill_between(
                    grid,
                    y0 * (grid / x0) ** bound_low,
                    y0 * (grid / x0) ** bound_high,
                    color=style.auc_color,
                    alpha=style.alpha,
                    label=f"acceptance [{bound_low:.2f}, {bound_high:.2f}]",
                )
    if t is not None:
        if bool(t["proportional"].to_numpy()):
            verdict = "proportional"
        elif bool(t["inconclusive"].to_numpy()):
            verdict = "inconclusive"
        else:
            verdict = "not proportional"
        heading = f"{heading}: {verdict}"
    ax.set_title(heading)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"x [{result.ds.attrs['x_unit']}]")
    ax.set_ylabel(f"y [{result.ds.attrs['y_unit']}]")
    ax.legend(fontsize="small")
    return fig

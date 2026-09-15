# Plotting

The figures of `pkpdutils.plot` are matplotlib figures. Every function returns the `Figure` it drew and never shows it, so a script saves it (`fig.savefig("name.png")`) and a notebook displays it; pass `ax` to draw into an existing axes. Colors and markers come from a `PlotStyle`.

## Timecourses

`plot_timecourse` draws one curve or every curve of a batch, with the standard error (or the standard deviation) as error bars when present, one color per sample, and the legend from the sample labels or from a coordinate of the batch (`by="individual"`).

```python
from pkpdutils.plot import plot_timecourse

fig = plot_timecourse(batch, log=True, by="dose")
fig.savefig("curves.png")
```

## NCA diagnostics

`plot_nca` shows what the analysis did with one curve, on a linear and a logarithmic axis: the data, the area to \(t_\mathrm{last}\), the extrapolated tail, the terminal regression line and the points it used, \(C_\mathrm{max}\)/\(t_\mathrm{max}\), \(C_0\) for a bolus, and the flags in the title. For a batch, `plot_nca_grid` draws one such panel per sample.

```python
from pkpdutils.plot import plot_nca, plot_nca_grid

fig = plot_nca(tc, nca_single(tc))
fig = plot_nca(batch.sel(individual="s2"), result, individual="s2")
fig = plot_nca_grid(batch, result, ncols=4)
```

![NCA diagnostics](images/nca_single.png)

The image is written by `examples/nca_single.py`; copy it to `docs/images/` after a change of the figure.

## Fits

`plot_fit` draws one sample of a [fit](fitting.md): the data with error bars when the fit had standard deviations, the fitted curve on a fine grid, the model name, the parameters as `name = value +- se` and the flags in the title, and the weighted residuals against \(x\) in a second panel below. `plot_goodness_of_fit` plots the predicted against the observed values of every sample with the identity line and the \(R^2\) per sample, and `plot_dose_proportionality` shows the exposure against the dose on log-log axes with the power fit, the acceptance wedge of the criterion and the verdict in the title.

```python
from pkpdutils.plot import plot_dose_proportionality, plot_fit, plot_goodness_of_fit

fig = plot_fit(result, log_y=True)  # 0-D result: no indexers
fig = plot_fit(fits, individual="s2", log_x=True)  # one sample of a batch
fig = plot_goodness_of_fit(fits, log=True)
fig = plot_dose_proportionality(
    power, test=proportionality_test(power, dose_range=(25, 400))
)
```

The images are written by `examples/fitting_exponential.py`, `examples/emax.py` and `examples/dose_proportionality.py`.

## Style

```python
from pkpdutils.plot import PlotStyle

style = PlotStyle(fit_color="tab:red", auc_color="lightgray", alpha=0.3)
fig = plot_nca(tc, result, style=style)
```

The reference of the module is in [API: plot](api/plot.md).

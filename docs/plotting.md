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

## Style

```python
from pkpdutils.plot import PlotStyle

style = PlotStyle(fit_color="tab:red", auc_color="lightgray", alpha=0.3)
fig = plot_nca(tc, result, style=style)
```

The reference of the module is in [API: plot](api/plot.md).

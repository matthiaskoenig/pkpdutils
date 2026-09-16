# Examples

Runnable examples for pkpdutils. They are **not** part of the package: they are not installed with `pip install pkpdutils`, they are read and run from a checkout of the repository.

Every example is a module of the `examples` package, so it is run from the root of the repository with

```bash
python -m examples.timecourses
```

An example writes what it creates into the current working directory. No example opens a window: matplotlib figures are saved to a file, never shown, so that the examples also run on a machine without a display.

The [gallery](https://matthiaskoenig.github.io/pkpdutils/gallery/) shows the figure and the core snippet of every example; `uv run python scripts/render_examples.py` renders the figures of all examples into `docs/images/`.

## What is where

| path | content |
| --- | --- |
| `examples/timecourses.py` | creating `Timecourse` and `Timecourses` objects from arrays, data frames and a simulation-like dataset |
| `examples/nca_single.py` | non-compartmental analysis of one curve, default and pkdb_analysis-compatible options, the diagnostic figure |
| `examples/nca_batch.py` | NCA of a `(dose, individual)` batch, results as a data frame, curve and grid figures |
| `examples/steady_state.py` | superposition of a single dose curve into a multiple dose protocol, the per-interval parameters and the steady state parameters of the last interval |
| `examples/nca_from_sbmlsim.py` | NCA of a simulation scan dataset (`Timecourses.from_dataset`), with or without sbmlsim installed |
| `examples/group_uncertainty.py` | bootstrap and delta method for a group mean curve, summary over individuals, partial AUC |
| `examples/fitting_exponential.py` | Bateman fit of an oral timecourse with weighting and bootstrap, AICc comparison of exponential models, goodness of fit |
| `examples/emax.py` | sigmoid Emax fit of a concentration-effect relationship, comparison with Emax and linear |
| `examples/dose_proportionality.py` | power model of `auc_inf_obs` against the dose from an NCA batch and the Smith criterion |
| `examples/covariate.py` | allometric scaling of a clearance against body weight with a free and a fixed exponent |
| `examples/bioequivalence.py` | average bioequivalence of a test against a reference formulation in a 2x2 crossover from two NCA batches, ratio and parameter figures |
| `examples/ddi.py` | exposure ratios with and without an inhibitor, Welch t test, FDA and EMA classification of the interaction, substrate sensitivity |
| `examples/meta_analysis.py` | fixed effect and random effects meta-analysis of published summary statistics with a forest plot |
| `examples/formats.py` | a BID batch written and read back as event records, the multiple dosing NCA of the round trip, and a CDISC ADaM ADNCA fixture read with `from_adnca` |

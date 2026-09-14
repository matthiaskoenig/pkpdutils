# Examples

Runnable examples for pkpdutils. They are **not** part of the package: they are not installed with `pip install pkpdutils`, they are read and run from a checkout of the repository.

Every example is a module of the `examples` package, so it is run from the root of the repository with

```bash
python -m examples.timecourses
```

An example writes what it creates into the current working directory. No example opens a window: matplotlib figures are saved to a file, never shown, so that the examples also run on a machine without a display.

## What is where

| path | content |
| --- | --- |
| `examples/timecourses.py` | creating `Timecourse` and `Timecourses` objects from arrays, data frames and a simulation-like dataset |
| `examples/nca_single.py` | non-compartmental analysis of one curve, default and pkdb_analysis-compatible options, the diagnostic figure |
| `examples/nca_batch.py` | NCA of a `(dose, individual)` batch, results as a data frame, curve and grid figures |
| `examples/steady_state.py` | superposition of a single dose curve and the steady state parameters of a dosing interval |
| `examples/nca_from_sbmlsim.py` | NCA of a simulation scan dataset (`Timecourses.from_dataset`), with or without sbmlsim installed |
| `examples/group_uncertainty.py` | bootstrap and delta method for a group mean curve, summary over individuals, partial AUC |

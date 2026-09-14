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

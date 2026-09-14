"""Test session configuration.

Two things are set up for the whole test session:

- **matplotlib never opens a window.** The `Agg` backend is selected before
  pyplot is imported anywhere, so a figure created by an example or by a test
  is rendered into a buffer instead of into a window.
- **the examples are importable.** They are not part of the package, they live
  in `examples/` at the root of the repository. pytest puts the directory of
  this file on `sys.path`, so that the tests can import the examples.
"""

import matplotlib

matplotlib.use("Agg", force=True)

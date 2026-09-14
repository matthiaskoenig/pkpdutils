# Installation

`pkpdutils` requires python >= 3.13 and is available from [pypi](https://pypi.python.org/pypi/pkpdutils). It is tested on Linux, macOS and Windows and is pure python; every dependency ships binary wheels, so no compiler is needed.

## With uv

[uv](https://docs.astral.sh/uv/) is the recommended way to install the package. In a project it is added as a dependency:

```bash
uv add pkpdutils
```

Into an existing virtual environment it is installed through the pip interface of uv:

```bash
uv venv --python 3.14
uv pip install pkpdutils
```

## With pip

```bash
pip install pkpdutils
```

## Development version

The current state of the `develop` branch is installed directly from GitHub:

```bash
uv add "pkpdutils @ git+https://github.com/matthiaskoenig/pkpdutils.git@develop"
```

or, with pip,

```bash
pip install git+https://github.com/matthiaskoenig/pkpdutils.git@develop
```

To work on the repository itself, with the test and documentation tooling, see [Development](development.md).

## Dependencies

| package | used for |
| --- | --- |
| [numpy](https://numpy.org), [scipy](https://scipy.org) | numerics, integration, regression, optimization and statistics |
| [xarray](https://xarray.dev), [pandas](https://pandas.pydata.org) | timecourses and results as labeled arrays and tables |
| [pint](https://pint.readthedocs.io) | units and unit conversions |
| [pydantic](https://docs.pydantic.dev) | validated data structures and options |
| [matplotlib](https://matplotlib.org) | figures |
| [rich](https://rich.readthedocs.io) | console output of scripts and examples |

## Logging

`pkpdutils` does not configure logging. It logs to loggers below the `pkpdutils` logger and leaves handlers, levels and formatting to the application:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("pkpdutils").setLevel(logging.WARNING)
```

For scripts and interactive work the rich output of the package can be turned on explicitly:

```python
from pkpdutils import log

log.enable_rich_logging()
```

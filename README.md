# ASTrIX: AeroSpace Trajectory Imaging toolboX 

_Under Construction..._

A Python tool package for planning and analysing aerospace imaging campaigns.

[![Tests](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml) [![Documentation](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/andrewjlock/astrix/graph/badge.svg?token=N781FLJEI5)](https://codecov.io/github/andrewjlock/astrix)

[View the full documentation here](https://andrewjlock.github.io/astrix/).
Source code is hosted on [GitHub](https://github.com/andrewjlock/astrix).

## Features

- Simple, object-oriented interface to reconstruct position and reference frame of moving targets, observer stations, cameras projections, and more.
- Support for NumPy and Jax backends using the [array api](https://data-apis.org/array-api/latest/) spec. Fully differentiable and JIT-compatible using Jax backend for efficient optimisation and state estimation.
- Many convenience tools for interpolation, triangulation, and frame handling in ECEF coordinates.

This toolbox has three main use cases:
1. Planning flight paths and ground station placement for aerospace observation campaigns.
2. Trajectory reconstruction and state estimation from observational data (e.g. optical tracking).
3. Aiding analysis of scientific optical measurements (e.g. spectral data).

## Install

The repository is not yet published to PyPI, so installation is via cloning the repository and installing using pip.

```bash
$ git clone https://github.com/andrewjlock/astrix.git
$ cd astrix
```

While under development editable installation is recommend

```bash
$ python3 -m pip install -e .[jax,plot]
```
The `jax` extra instals [Jax](https://github.com/jax-ml/jax) which is used for optional advanced automatic differentiation and state/parameter estimation capabilities. 
The `plot` extra install [basemap](https://matplotlib.org/basemap/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/) for geographic plotting.
For a lean install you can omit both the `[jax]` and `[plot]` extra,

```bash
$ python3 -m pip install -e .
```

For development, use the `dev` group, which includes linting and testing tools.

```bash
$ python3 -m pip install -e .[plot] --group dev
```

Alternatively, this project has been developed with, and is compliant with, the `uv` package manager.
The `dev` group is installed by default.

```bash
$ uv pip install -e .[jax,plot]
```



### Dependencies

You will need to manually install the following dependency using apt

```bash
$ sudo apt -y install libgeos-dev
```

The package requires, and is tested with Python 3.12 (default on Ubuntu 24.04 LTS).
Such a recent version of Python is required to support recent array api specification features enabling [backend switching](#backend-compatibility---numpy-and-jax).
However, type checking implements features of the Python Array API standard which require Python 3.12.
Therefore, Python 3.12 is recommended for development.

## Backend Compatibility - Numpy and Jax

This project utilises the [Array API](https://data-apis.org/array-api/) standard to enable switching between NumPy and [Jax](https://jax.readthedocs.io/en/latest/) for some core classes and functions, using their common native API. 
Numpy is the default, and superior for most use cases. 
Jax provides two extra capabilities: automatic differentiation and JIT compilation, which can be useful for optimisation and estimation problems.
However, Jax is slower than Numpy for most use cases. 

Classes and functions which support both backends have a `backend` argument, which can be set to either `'numpy'`/`'np'` or `'jax'`/`'jnp'`, or a reference to either namespace.
Those that have no need for Jax capabilities are implemented in NumPy only.
The backend is specified per-instance, as there are use cases when both may be wanted simultaneously.
Be careful not to mix NumPy- and Jax-backend objects unintentionally, as this can lead to hard-to-diagnose bugs.

To use the Jax backend, you will nees to ensure the SciPy environment environment variable is set, via Python before you import `scipy.spatial.transform.Rotation`. This is done automatically when you import `astrix`, but can also be done manually: 
```python
import os
os.environ['SCIPY_ARRAY_API'] = "True"
```
or in your shell
```bash
$ export SCIPY_ARRAY_API=True
```
Failure to set this environment variable will result in errors using the SciPy Rotation module with Jax backend.

Note that Jax backend is implemented on CPU only, and 64-bit precision is enforced on astrix import (Jax defaults to 32-bit, which often is insufficient).



## Data Conventions

This package deals primarily with time series vectors.
The following conventions are used:
- Time series data is represented as 2D arrays of shape `(N, D)`, where `N` is the number of time steps and `D` is the dimensionality of the vector (e.g., 3 for 3D position).
- Single vectors are represented as 1D arrays of shape `(D,)`.

## Tests

Tests are implemented using pytest. To run the tests:

```bash
$ pytest tests/
```
If Jax is installed, tests will be run using both Numpy and Jax backends. 
If Jax is not installed, only Numpy backend tests will be run.

## Documentation

Documentation is generated using Sphinx. To build the documentation:

1. Install dependencies: `pip install -r docs/requirements.txt`
2. Navigate to the docs directory: `cd docs`
3. Run the generate script `python generate_docs.py`
4. Build the docs: `make html`


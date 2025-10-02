# ASTrIX: AeroSpace Trajectory Imaging toolboX 

_Under Construction..._

A Python tool package for planning and analysing aerospace imaging campaigns.

[![Tests](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml) [![Documentation](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/andrewjlock/astrix/graph/badge.svg?token=N781FLJEI5)](https://codecov.io/github/andrewjlock/astrix)

[View the full documentation here](https://andrewjlock.github.io/astrix/).
Source code is hosted on [GitHub](https://github.com/andrewjlock/astrix).

## Features

- Simple, object-oriented interface to reconstruct simultaneous moving and rotating reference frames, calculate local line-of-sight vectors, and project to image planes.
- Many convenience tools for interpolation, intrinsic frame handling, triangulation, position conversion, data validation etc.
- Support for NumPy and Jax backends using the [array api](https://data-apis.org/array-api/latest/) spec. Most functionality is differentiable and JIT-compilable using Jax backend, allowing efficient optimisation and state estimation.

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
$ python3 -m pip install -e .[plot,jax]
```
The `plot` extra install [basemap](https://matplotlib.org/basemap/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/) for geographic plotting.
The `jax` extra installs [Jax](https://github.com/jax-ml/jax) which is used for optional advanced automatic differentiation and state/parameter estimation capabilities. 
For a lean install you can omit both the `[jax]` and `[plot]` extra,

```bash
$ python3 -m pip install -e .
```

For development, use the `dev` group, which includes linting and testing tools, and all optional extras,

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

The package is currently tested with Python 3.12 (default on Ubuntu 24.04 LTS).
Such a recent version of Python is required to support the recent array API spec features enabling [backend switching](#backend-compatibility---numpy-and-jax).


## Data Conventions

The following conventions are used data representation throughout the package:
- Time series data is represented as 2D arrays of shape `(N, D)`, where `N` is the number of time steps and `D` is the dimensionality of the vector (e.g., 3 for 3D position).
- Single vectors are represented as 1D arrays of shape `(D,)`.
- By default, position is stored as Cartesian ECEF coordinates represented as `(x, y, z)` in metres.
- Geodetic coordinates are represented as `(latitude, longitude, altitude)`, where latitude and longitude are in degrees and altitude is in metres.


## Backend Compatibility - NumPy and Jax

_Note: This section is only relevant for those wishing to do efficient optimisation or state estimation_

This project uses the [Array API](https://data-apis.org/array-api/) standard to enable switching between NumPy and [Jax](https://jax.readthedocs.io/en/latest/) backends for most core classes and functions, using their common native API. 
NumPy is the default, and superior for most use cases. 
Jax provides two extra capabilities: automatic differentiation and JIT compilation, which can be useful for batch processing and optimisation problems.

Classes and functions which support both backends have a `backend` argument, which can be set to either `'numpy'`/`'np'` or `'jax'`/`'jnp'`, or a reference to either namespace.
Moreover, classes also have methods to convert between backends, e.g. `to_jax()` and `to_numpy()` (including all dependent objects).
Functions/methods that have no anticipated need for Jax capabilities are implemented in NumPy only, as are those that have Jax incompatible dependencies (for example, conversion between ECEF and geodetic coordinates which uses the [`pyproj`](https://github.com/pyproj4/pyproj) package).
Users will be alerted to attempts to use Jax backend in such cases with warnings or errors.

The backend is specified per-instance and per-function call (with NumPy defaults), as there are use cases when both may be wanted simultaneously.
Although effort has been made to ensure any two interacting objects use similar/compatible backends, this is not guaranteed. 
The primary CI workflows test NumPy backends only. 

To use the Jax backend, you will need to ensure the SciPy environment environment variable is set, via Python before you import `scipy.spatial.transform.Rotation`. This is done automatically when you import `astrix`, but can also be done manually: 
```python
import os
os.environ['SCIPY_ARRAY_API'] = "True"
```
or in your shell
```bash
$ export SCIPY_ARRAY_API=True
```
Failure to set this environment variable will result in errors using the SciPy Rotation module with Jax backend.

Note that Jax backend is implemented on CPU only, and 64-bit precision is enforced on `astrix` import (Jax defaults to 32-bit, which often is insufficient for large ECEF values).
However, if you import `jax` manually before `astrix`, these environment variables can be manually set in your shell:
```bash
$ export JAX_ENABLE_X64=True
$ export JAX_PLATFORM_NAME=cpu
```
or in the header of your Python script:
```python
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
```

## Tests

Tests are implemented using pytest. To run the tests:
```bash
$ pytest tests/
```
or using `uv`:
```bash
$ uv run pytest tests/
```

If Jax is installed, tests will be run using both NumPy and Jax backends. 
If Jax is not installed, only NumPy backend tests will be run.

## Documentation

Documentation is generated using Sphinx. To build the documentation:

1. Install dependencies: `pip install -r docs/requirements.txt`
2. Navigate to the docs directory: `cd docs`
3. Run the generate script `python generate_docs.py`
4. Build the docs: `make html`
Or, to live view the docs while editing:
```bash
sphinx-autobuild . _build/html/
```


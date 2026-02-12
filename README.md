# ASTrIX · AeroSpace Trajectory Imaging toolboX

> Python-interface toolbox for planning, simulating, and analysing in-flight imaging and tracking.

[![Tests](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml) [![Documentation](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/andrewjlock/astrix/graph/badge.svg?token=N781FLJEI5)](https://codecov.io/github/andrewjlock/astrix)

[View the full documentation here](https://andrewjlock.github.io/astrix/).
Source code is hosted on [GitHub](https://github.com/andrewjlock/astrix).

## Features

- Object-oriented primitives for frames, rays, rotations, and paths with consistent ECEF/geodetic handling.
- Built-in interpolation, intrinsic frame handling, triangulation, refraction, and validation utilities.
- Dual-backend support (NumPy + JAX via [Array API](https://data-apis.org/array-api/latest/)); JIT/differentiation ready where it matters.

Core use cases:
1. Planning flight paths and ground-station placement for observation campaigns.
2. Trajectory reconstruction and state estimation from optical or mixed-sensor data.
3. Analysis workflows for scientific imaging (e.g., spectral/photometric pipelines).

## Install

The repository is not yet published to PyPI; install from source.

```bash
$ git clone https://github.com/andrewjlock/astrix.git
$ cd astrix
```

### Using pip (editable)

```bash
$ python3 -m pip install -e .[plot] --group dev --group docs
```
- `plot` installs [basemap](https://matplotlib.org/basemap/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/) for geographic plotting.
- JAX is **optional**. If you need JAX-based optimisation/differentiation, add the extra: `python3 -m pip install -e .[plot,jax] --group dev --group docs`.
- For a lean install, omit extras/groups: `python3 -m pip install -e .`.

### Using uv (recommended for dev)

```bash
$ uv python install 3.12
$ uv sync
$ uv run pytest tests/
```
To add JAX support: `uv sync --extra jax`.

To consume from another `uv` project:
```bash
$ uv add --editable ../path/to/astrix[plot]
```

### Dependencies

You will need to manually install the following dependency using apt

```bash
$ sudo apt -y install libgeos-dev
```

To use the array api capabilities of SciPy, you will need to set the environment variable
```bash
$ export SCIPY_ARRAY_API=1
```
before importing `scipy`. 
`astrix` will attempt to set this before importing `scipy`, but if you separately import `scipy` before `astrix`, you will need to set this manually.

The package is currently tested with Python 3.12 (default on Ubuntu 24.04 LTS).
Such a recent version of Python is required to support the recent array API spec features enabling [backend switching](#backend-compatibility---numpy-and-jax).


## Data Conventions

The following conventions are used data representation throughout the package:
- Time series data is represented as 2D arrays of shape `(N, D)`, where `N` is the number of time steps and `D` is the dimensionality of the vector (e.g., 3 for 3D position).
- Single vectors are represented as 1D arrays of shape `(D,)`.
- By default, position is stored as Cartesian ECEF coordinates represented as `(x, y, z)` in metres.
- Geodetic coordinates are represented as `(latitude, longitude, altitude)`, where latitude and longitude are in degrees and altitude is in metres.

In addition, all classes in the package are designed to be immutable (for Jax compatibiliy). 
Any method which modifies the state of an object will return a new instance of the object with the modified state (although child objects, such as arrays, may be shared, depending on the operation).


## Backend Compatibility - NumPy and Jax

_Note: This section is only relevant for those wishing to do efficient optimisation or state estimation_

This project uses the [Array API](https://data-apis.org/array-api/) standard to enable switching between NumPy and [Jax](https://jax.readthedocs.io/en/latest/) backends for most core classes and functions, using their common native API. 
NumPy is the default, and superior for most use cases. 
Jax provides two extra capabilities: automatic differentiation and JIT compilation, which can be useful for batch processing and optimisation problems.
If Jax is of no interest to you, you can safely ignore backend arguments altogether.

Classes and functions which support both backends have a constructor `backend` argument, which can be set to either `'numpy'`/`'np'` or `'jax'`/`'jnp'`, or a reference to either namespace.
Moreover, classes also have a `.convert_to("desired_backend")` method to convert between backends.
This conversion is recursive, and will also convert all required child objects. 
The suggested use case is therefore to create a model in NumPy, and then convery only the relevant objects to Jax when needed for JIT compilation or differentiation.
Functions/methods that have no anticipated need for Jax capabilities are implemented in NumPy only, as are those that have Jax incompatible dependencies (for example, accurate conversion between ECEF and geodetic coordinates which uses the [`pyproj`](https://github.com/pyproj4/pyproj) package).
Users will be alerted to attempts to use Jax backend in such cases with warnings or errors.

Although effort has been made to ensure any two interacting objects use similar or compatible backends, this is not guaranteed. 
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

## Contributing

Contributions are very welcome! Please submit proposed changes via pull requests on GitHub.

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

```bash
$ uv run python docs/generate_docs.py
$ uv run make -C docs html
```

For live previews while editing docs:
```bash
$ cd docs
$ sphinx-autobuild . _build/html/
```

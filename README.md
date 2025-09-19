# ASTrIX: AeroSpace Trajectory Imaging toolboX 

_Under Construction..._

A Python tool package for planning and analysing aerospace trajectories.

[![Tests](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml) [![Documentation](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml)

Full documentation is available [here](https://andrewjlock.github.io/astrix/), and the source code is hosted on [GitHub](https://github.com/andrewjlock/astrix).

This project aids three main purposes:
1. Aid flight path planning and ground station placement for aerospace observation campagins.
2. Provide a flexible, fast, and differentiable enviroment to reconstruct viewing vectors and camera projections for trakcking and state estimation problems.
3. Trace frames of vehicles and observers to compute view angles and ranges for interpreting scientific measurements (e.g. spectral data).

## Install

While under development editable installation is recommend

```bash
$ python3 -m pip install -e .[jax,plot]
```
Jax is used for advanced automatic differentiation and state/parameter estimation capabilities. 
The plot extras install [basemap](https://matplotlib.org/basemap/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/) for geographic plotting.
For a lean install you can ommit both the `[jax]` and `[plot]` extra,

```bash
$ python3 -m pip install -e .
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

Note that Jax backend is implemented on CPU only, and 64-bit precision is encorced (Jax defaults to 32-bit). 
The backend is specified per-instance, as there are use cases when both may be wanted simultaneously.
Be careful not to mix NumPy- and Jax-backend objects unintentionally, as this can lead to hard-to-diagnose bugs.

Classes and functions which support both backends have a `backend` argument, which can be set to either `'numpy'`/`'np'` or `'jax'`/`'jnp'`, or a reference to either namespace.
Those that have no need for Jax capabilities are implemented in NumPy only.


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


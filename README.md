# ASTrIX: AeroSpace Trajectory Imaging toolboX 

A Python tool package for planning and analysing aerospace remote imaging and diagnostic experiments.

[![Tests](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml) [![Documentation](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml)

Full documentation is available [here](https://andrewjlock.github.io/astrix/), and the source code is hosted on [GitHub](https://github.com/andrewjlock/astrix).

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

Package execution is tested on Python 3.10 (default on Ubuntu 22.04 LTS) and Python 3.12 (default on Ubuntu 24.04 LTS).
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


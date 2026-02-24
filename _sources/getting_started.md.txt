# ASTrIX · AeroSpace Trajectory Imaging toolboX

> Python-interface toolbox for planning, simulating, and analysing in-flight imaging and tracking.

[![Tests](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml/badge.svg)](https://github.com/andrewjlock/astrix/actions/workflows/tests.yml) [![Documentation](https://github.com/andrewjlock/astrix/actions/workflows/docs.yml/badge.svg)](https://andrewjlock.github.io/astrix/)
[![codecov](https://codecov.io/github/andrewjlock/astrix/graph/badge.svg?token=N781FLJEI5)](https://codecov.io/github/andrewjlock/astrix)

[View the full documentation here](https://andrewjlock.github.io/astrix/).
Source code is hosted on [GitHub](https://github.com/andrewjlock/astrix).

## Features

- Object-oriented framework for constructing and analysing dynamic frames, rays, rotations, and locations.
- Built-in interpolation, relative frame transformations, triangulation, atmospheric refraction correction, camera projection, and validation utilities.
- Dual-backend support (NumPy + JAX via the [Array API](https://data-apis.org/array-api/latest/)); enabling JIT compilation and automatic differentiation for optimization tasks.

### Core Use Cases
1. Planning flight paths and ground-station placement for observation campaigns.
2. Real-time pointing and tracking assistance during observation campaigns, including star and landmark tracking for orientation estimation.
3. Campaign post-processing, including trajectory reconstruction, triangulation, and view factor analysis.

## Core Primitives

The toolbox relies on several immutable spatial and temporal primitives:
- `Time` / `TimeGroup`: Manages time instances (seconds since Unix epoch) and overlapping time bounds across multiple objects.
- `Point`: Represents single or multiple spatial locations (ECEF or WGS84 Geodetic), optionally associated with specific time instances.
- `Path`: Represents a time-varying trajectory composed of sequential `Point`s, supporting interpolation, downsampling, and finite-difference velocity/acceleration derivation.
- `Frame`: Defines a 3D reference frame using a location (`Point` or `Path`) and a rotation sequence. Frames can be defined relative to one another to construct kinematic chains.
- `Ray`: Represents directional vectors with respect to a specific `Frame`. Facilitates coordinate transformations, camera projection, interpolatoin, and refraction corrections.

## Data Conventions

The following conventions are strictly adhered to throughout the package:
- **Array Shapes**: Spatial and vector data (e.g., coordinates, directions) are strictly 2D arrays of shape `(N, D)`, where `N` is the number of data points and `D` is the spatial dimensionality (e.g., 3 for 3D position). True 1D data, such as time sequences, are strictly represented as 1D arrays of shape `(N,)`.
- **Coordinate Systems**: By default, position is stored as Cartesian Earth-Centered, Earth-Fixed (ECEF) coordinates `(x, y, z)` in metres. Geodetic coordinates are represented as `(latitude, longitude, altitude)`, where latitude and longitude are in degrees and altitude is in metres.
- **Immutability**: All core classes are immutable to guarantee compatibility with JAX's functional transformations (`jit`, `grad`, `vmap`). Any method that modifies an object's state returns a new instance. Child objects, such as underlying arrays, may be shared depending on the operation.

## Backend Compatibility - NumPy and JAX

_Note: This section is relevant only for users requiring the JAX backend for automatic differentiation and JIT compilation._

This project uses the [Array API](https://data-apis.org/array-api/) standard to enable switching between NumPy and [JAX](https://jax.readthedocs.io/en/latest/) backends using their common native API. NumPy is the default and is recommended for most general planning and analysis tasks. JAX provides automatic differentiation and JIT compilation capabilities, which are highly beneficial for batch processing and optimization problems.

Classes and functions supporting both backends include a `backend` constructor argument, which accepts `'numpy'`/`'np'`, `'jax'`/`'jnp'`, or a direct reference to the namespace. Furthermore, objects provide a `.convert_to("desired_backend")` method. This conversion is recursive and applies to all required child objects. The suggested workflow is to construct the model using NumPy, and convert only the relevant objects to JAX immediately prior to executing JIT-compiled or differentiated functions.

Functions or methods with JAX-incompatible dependencies (such as accurate geodetic conversions relying on `pyproj`) are implemented solely for the NumPy backend. Attempting to use the JAX backend in these contexts will raise warnings or errors. 

### Environment Configuration

To utilize the JAX backend alongside SciPy's spatial transforms, the SciPy Array API must be enabled before importing `scipy.spatial.transform.Rotation`. `astrix` attempts to configure this automatically upon import, but manual configuration is required if `scipy` is imported beforehand:
```bash
export SCIPY_ARRAY_API=1
```

Additionally, because JAX defaults to 32-bit floating-point precision—which is insufficient for ECEF coordinate ranges—ASTrIX enforces 64-bit precision upon import. The JAX backend is currently configured for CPU execution. If you import `jax` manually before `astrix`, ensure the following environment variables are set:
```bash
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
```

## Installation

The repository is not yet published to PyPI; install from source.

```bash
git clone https://github.com/andrewjlock/astrix.git
cd astrix
```

### System Dependencies
The `GEOS` library is required for coordinate transformations:
```bash
sudo apt -y install libgeos-dev
```

### Python Environment
The package is currently tested with Python 3.12. A recent version of Python is required to support the Array API standard used for backend switching.

**Using `uv` (Recommended)**
```bash
uv python install 3.12
uv sync
uv run pytest tests/
```
To include JAX support: `uv sync --extra jax`

To consume from another `uv` project:
```bash
uv add --editable ../path/to/astrix[plot]
```

**Using `pip` (Editable)**
```bash
python3 -m pip install -e .[plot] --group dev --group docs
```
- `plot` installs `basemap` and `cartopy` for geographic plotting.
- To include JAX support: `python3 -m pip install -e .[plot,jax] --group dev --group docs`.
- For a lean installation, omit extras/groups: `python3 -m pip install -e .`.

## Git Workflow

Development should be primarily done on the `dev` branch:
```bash
git switch dev
git pull
# make changes, commit, push
git push
# when ready, open a PR on GitHub to merge into main
```

## Tests

Tests are implemented using `pytest`. If JAX is installed, tests will execute against both the NumPy and JAX backends.
```bash
uv run pytest tests/
```

## Documentation

Documentation is generated using Sphinx. To build the documentation:
```bash
uv run python docs/generate_docs.py
uv run make -C docs html
```

For live previews while editing docs:
```bash
cd docs
sphinx-autobuild . _build/html/
```

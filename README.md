# ASTrIX: AeroSpace Trajectory Imaging toolboX 

A Python tool package for planning and analysing aerospace remote imaging and diagnostic experiments.

Full documentation is available [here](https://andrewjlock.github.io/astrix/).

## Install

While under development editable installation is recommend

```bash
$ python3 -m pip install -e .[jax]
```
Jax is used for advanced automatic differentiation and state/parameter estimation capabilities. 
For a lean install you can ommit the `[jax]` extra,

```bashbash
$ python3 -m pip install -e .
```

### Array API Compatibility and Jax

This project implements most core functions using [Array API](https://data-apis.org/array-api/) standard functions to enable switching between NumPy and [Jax](https://jax.readthedocs.io/en/latest/). 
Numpy is the default, and superior for most use cases. 
Jax provides two extra capabilities: automatic differentiation and JIT compilation, which can be useful for optimisation and estimation problems.
However, Jax is slower than Numpy for most use cases. 
Note that Jax backend is implemented on CPU only, and 64-bit precision is encorced (Jax defaults to 32-bit). 


### Dependencies

You will need to manually install the following dependency using apt

```bash
$ sudo apt -y install libgeos-dev
```


## Data Conventions

## Documentation

Documentation is generated using Sphinx. To build the documentation:

1. Install dependencies: `pip install -r docs/requirements.txt`
2. Navigate to the docs directory: `cd docs`
3. Build the docs: `make html`


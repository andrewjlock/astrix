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

## Jax

Jax functionality is currently only supported on CPU backend. Given the nature of the computations, GPU/TPU backends are not expected to provide significant performance improvements and likely inferior performance due to data transfer overheads.

import numpy as np


def exmaple_util_function1(x: int) -> int:
    """A simple utility function that squares an integer."""
    return x * x


def example_util_function2(arr: np.ndarray) -> np.ndarray:
    """A utility function that normalizes a numpy array."""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

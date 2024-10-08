import numpy as np


def normalize_to_0_1(iter):
    """Normalizes to between 0 and 1"""
    x = (np.array(iter) - np.min(iter)) / (np.max(iter) - np.min(iter))
    return x.tolist()


def normalize_to_bounds(iter, bounds=1):
    """Normalizes to between -bound and +bound"""
    x = (np.array(iter) - np.min(iter)) / (np.max(iter) - np.min(iter))
    x = (x - 0.5) * (bounds / 0.5)
    return x.tolist()

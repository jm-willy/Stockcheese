import numpy as np


def normalize_to_0_1(iter):
    """Normalizes to between 0 and 1, 0.5 center"""
    if (np.max(iter) - np.min(iter)) == 0:
        return [0.5 for i in range(len(iter))]
    x = (np.array(iter) - np.min(iter)) / (np.max(iter) - np.min(iter))
    return x.tolist()


def normalize_to_bounds(iter, bounds=1):
    """Normalizes to between -bound and +bound, 0 center"""
    if (np.max(iter) - np.min(iter)) == 0:
        return [0 for i in range(len(iter))]
    x = (np.array(iter) - np.min(iter)) / (np.max(iter) - np.min(iter))
    x = (x - 0.5) * (bounds / 0.5)
    return x.tolist()


# q = [
#     1.0,
#     1.0,
# ]
# print(normalize_to_bounds(q))

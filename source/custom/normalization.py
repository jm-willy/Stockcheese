import numpy as np


def normalize_to_0_1(iter, norm_same=False):
    """
    Normalizes to between 0 and 1, 0.5 center

    norm_same: whether to normalize to 0 an iter of
    repeating identical elements
    """
    if (np.max(iter) - np.min(iter)) == 0:
        if norm_same is True:
            return [0.5 for i in range(len(iter))]
        else:
            return iter
    x = (np.array(iter) - np.min(iter)) / (np.max(iter) - np.min(iter))
    return x.tolist()


def normalize_to_bounds(iter, bounds=1, norm_same=False):
    """
    Normalizes to between -bound and +bound, 0 center

    norm_same: whether to normalize to 0 an iter of
    repeating identical elements
    """
    if (np.max(iter) - np.min(iter)) == 0:
        if norm_same is True:
            return [0 for i in range(len(iter))]
        else:
            return iter
    x = (np.array(iter) - np.min(iter)) / (np.max(iter) - np.min(iter))
    x = (x - 0.5) * (bounds / 0.5)
    return x.tolist()

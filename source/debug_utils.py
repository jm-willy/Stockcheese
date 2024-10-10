import numpy as np

from time_utils import date_time_print


def locate_NaNs(gradients, trainable_variables):
    for j, k in zip(gradients, trainable_variables):
        if j is None:
            date_time_print(k)
            continue
        if np.any(np.isnan(j[0].numpy())) is True:
            date_time_print(k)
    return

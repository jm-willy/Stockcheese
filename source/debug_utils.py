import numpy as np

from time_utils import date_time_print


def locate_NaNs(gradients, trainable_variables):
    counter = -1
    for j, k in zip(gradients, trainable_variables):
        counter += 1
        if j is None:
            print("<>" * 55)
            date_time_print("Gradient is None at step %s, layer:" % (counter))
            date_time_print(k)
            print("<>" * 55)
            continue
        if np.any(np.isnan(j[0].numpy())) is True:
            print("<>" * 55)
            date_time_print("Gradient has NaNs at step %s, layer:" % (counter))
            date_time_print(k)
            print("<>" * 55)
    return


def gradient_at_step(step, gradients, trainable_variables):
    print("<>" * 55)
    date_time_print("Gradient at step %s, layer:" % (step))
    date_time_print(gradients[step], trainable_variables[step])
    print("<>" * 55)
    return

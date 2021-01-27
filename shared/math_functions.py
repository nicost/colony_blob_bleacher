# ------------------------------------------
# FUNCTIONS for CURVE FITTING
# ------------------------------------------

import numpy as np


# single exponential
def single_exp(x, a, b):
    """
    single exponential curve for FRAP fitting
    t_half = ln(0.5)/(-b)
    :param x: input series of time points
    :param a: mobile fraction
    :param b: tau value
    :return: output series of relative intensity
    """
    return a * (1 - np.exp(-b * x))


# r_square calculation
def r_square(y, y_fit):
    """
    calculate r_squared for a given curve fitting
    :param y: values before fitting
    :param y_fit: values from fitting
    :return: r_squared
    """
    ss_res = np.sum([(a-b)**2 for a, b in zip(y, y_fit)])
    ss_tot = np.sum([(a-np.mean(y))**2 for a in y])
    r2 = 1 - (ss_res/ss_tot)
    return r2

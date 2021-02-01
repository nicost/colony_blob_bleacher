import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for CURVE FITTING
# ---------------------------------------------------------------------------------------------------

Math related:

    single_exp
        EQUATION: y = a * (1 - np.exp(-b * x))
        SYNTAX:   single_exp(x, a, b)
    
    linear
        EQUATION: y = a * x + b
        SYNTAX:   linear(x, a, b)
    
    single_exp_decay
        EQUATION: y = 1 - a * (1 - np.exp(-b * x))
        SYNTAX:   single_exp_decay(x, a, b)

Statistics:
    
    r_square
        FUNCTION: calculate r_squared for a given curve fitting
        SYNTAX:   r_square(y: list, y_fit: list)
 
Fitting related:
    
    fitting_linear
        FUNCTION: perform linear fitting
        SYNTAX:   fitting_linear(x: list, y: list)  
    
    fitting_single_exp_decay
        FUNCTION: perform single exponential decay fitting
        SYNTAX:   fitting_single_exp_decay(x: list, y: list)
    
    frap_fitting_single_exp
        FUNCTION: perform single exponential fitting for FRAP curves
        SYNTAX:   frap_fitting_single_exp(time_tseries_lst: list, frap_lst: list)
        
"""

# ---------------------------------------------------------------------------------------------------
# MATH related
# ---------------------------------------------------------------------------------------------------


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


# linear regression
def linear(x, a, b):
    """
        Linear regression function
    """

    return a * x + b


# single  exponential decay
def single_exp_decay(x, a, b):
    """
    single exponential decay curve for photobleaching fitting

    :param x: input time frames
    :param a:
    :param b:
    """
    return 1 - a * (1 - np.exp(-b * x))


# ---------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------


# r_square calculation
def r_square(y: list, y_fit: list):
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


# ---------------------------------------------------------------------------------------------------
# FITTING related
# ---------------------------------------------------------------------------------------------------


def fitting_linear(x: list, y: list):
    """
    Perform linear fitting

    y = a * x + b

    :param x:
    :param y:
    :return: y_fit:
             r2:
             a:
             b:
    """

    try:
        popt, _ = curve_fit(linear, x, y)
        a, b = popt
    except RuntimeError:
        a = np.nan
        b = np.nan

    y_fit = []
    for j in range(len(y)):
        y_fit.append(linear(j, a, b))
    r2 = r_square(y, y_fit)

    return y_fit, r2, a, b


def fitting_single_exp_decay(x: list, y: list):
    """
    Perform single exponential decay fitting

    y = 1 - a * (1 - np.exp(-b * x))

    :param x:
    :param y:
    :return: y_fit:
             r2:
             a:
             b:
    """
    try:
        popt, _ = curve_fit(single_exp_decay, x, y)
        a, b = popt
    except RuntimeError:
        a = np.nan
        b = np.nan

    y_fit = []
    for j in range(len(y)):
        y_fit.append(single_exp_decay(j, a, b))
    r2 = r_square(y, y_fit)

    return y_fit, r2, a, b


def frap_fitting_single_exp(time_tseries_lst: list, frap_lst: list):
    """
    Perform single exponential fitting for FRAP curves

    y = a * (1 - np.exp(-b * x))

    :param time_tseries_lst: list of real times
    :param frap_lst: list of FRAP curves
    :return: fit_pd: fitting table, includes 'single_exp_fit', 'single_exp_r2', 'single_exp_a',
                'single_exp_b', 'single_exp_mobile_fraction', 'single_exp_t_half'

    """
    single_exp_a = []
    single_exp_b = []
    single_exp_fit = []
    single_exp_r2 = []
    single_exp_t_half = []

    for i in range(len(frap_lst)):
        try:
            popt, _ = curve_fit(single_exp, time_tseries_lst[i], frap_lst[i])
            a, b = popt
        except RuntimeError:
            a = np.nan
            b = np.nan

        y_fit = []
        for j in range(len(time_tseries_lst[i])):
            y_fit.append(single_exp(time_tseries_lst[i][j], a, b))
        r2 = r_square(frap_lst[i], y_fit)
        t_half_fit = np.log(0.5) / (-b)
        single_exp_a.append(a)
        single_exp_b.append(b)
        single_exp_fit.append(y_fit)
        single_exp_r2.append(r2)
        single_exp_t_half.append(t_half_fit)

    fit_pd = pd.DataFrame({'single_exp_fit': single_exp_fit,
                           'single_exp_r2': single_exp_r2,
                           'single_exp_a': single_exp_a,
                           'single_exp_b': single_exp_b,
                           'single_exp_mobile_fraction': single_exp_a,
                           'single_exp_t_half': single_exp_t_half})

    return fit_pd

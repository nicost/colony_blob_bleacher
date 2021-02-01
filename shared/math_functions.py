# ------------------------------------------
# FUNCTIONS for CURVE FITTING
# ------------------------------------------

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd


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


def bg_fitting_linear(bg_int_tseries):

    try:
        popt, _ = curve_fit(linear, np.arange(0, len(bg_int_tseries), 1), bg_int_tseries)
        a, b = popt
    except RuntimeError:
        a = np.nan
        b = np.nan

    bg_fit = []
    for j in range(len(bg_int_tseries)):
        bg_fit.append(linear(j, a, b))
    r2 = r_square(bg_int_tseries, bg_fit)

    return bg_fit, r2, a, b


def pb_factor_fitting_single_exp(pb_factor_tseries):
    try:
        popt, _ = curve_fit(single_exp_decay, np.arange(0, len(pb_factor_tseries), 1), pb_factor_tseries)
        a, b = popt
    except RuntimeError:
        a = np.nan
        b = np.nan

    pb_fit = []
    for j in range(len(pb_factor_tseries)):
        pb_fit.append(single_exp_decay(j, a, b))
    r2 = r_square(pb_factor_tseries, pb_fit)

    return pb_fit, r2, a, b


def frap_fitting_single_exp(time_tseries_lst: list, frap_lst: list):
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

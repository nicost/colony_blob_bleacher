import numpy as np
from scipy.optimize import curve_fit
from scipy.special import iv
import shared.bleach_points as ble
import pandas as pd
import math
import scipy.stats.distributions as distributions
from scipy.misc import derivative

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
        EQUATION: y = a * np.exp(-b * x) + c
        SYNTAX:   single_exp_decay(x, a, b, c)
        
    soumpasis
        EQUATION: y = a * (np.exp(-tau/(2 * t)) * (iv(0, (tau/(2 * t))) + iv(1, (tau/(2 * t)))))
        SYNTAX: soumpasis(x, a, tau)
    
    double_exp
        EQUATION: y = a1 + a2 - a1 * np.exp(-b1 * t) - a2 * np.exp(-b2 * t)
        SYNTAX:   double_exp(t, a1, a2, b1, b2)

Statistics:
    
    r_square
        FUNCTION: calculate r_squared for a given curve fitting
        SYNTAX:   r_square(y: list, y_fit: list)
    
    frap_chi_square:
        FUNCTION: modified chi_square test to evaluate the goodness of FRAP curve fitting
        SYNTAX:   frap_chi_square(y: list, y_fit: list, sigma: int, imaging_length: int, fitting_model: str)
 
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
    
    frap_fitting_soumpasis
        FUNCTION: perform soumpasis fitting for FRAP curves
        SYNTAX:   frap_fitting_soumpasis(time_tseries_lst: list, frap_lst: list, sigma_lst: list)
    
    frap_fitting_double_exp
        FUNCTION: perform double exponential fitting for FRAP curves
        SYNTAX:   frap_fitting_double_exp(time_tseries_lst: list, frap_lst: list, sigma_lst: list)
        
        
"""


# ---------------------------------------------------------------------------------------------------
# MATH related
# ---------------------------------------------------------------------------------------------------


# single exponential
def single_exp(t, a, b):
    """
    Single exponential curve for FRAP analysis

    mobile fraction = a
    t_half = ln(0.5)/(-b)

    :param t: input series of time points
    :param a: mobile fraction
    :param b: tau value
    :return: output series of relative intensity

    """
    return a * (1 - np.exp(-b * t))


# linear regression
def linear(x, a, b):
    """
        Linear regression function
    """

    return a * x + b


# single exponential decay
def single_exp_decay(x, a, b, c):
    """
    single exponential decay curve for photobleaching fitting

    :param x: input time frames
    :param a:
    :param b:
    :param c:
    """
    return a * np.exp(-b * x) + c


def soumpasis(t, a, tau):
    """
    soumpasis function for diffusion dominant FRAP fitting

    mobile fraction = a
    diffusion coefficient (effective) = w^2/tau
    t_half = 0.224 * tau

    :param t:
    :param a:
    :param tau:
    :return:
    """
    return a * (np.exp(-tau / (2 * t)) * (iv(0, (tau / (2 * t))) + iv(1, (tau / (2 * t)))))


def double_exp(t, a1, a2, b1, b2):
    """
    Double exponential curve for FRAP analysis

    mobile fraction = a1+a2

    :param t:
    :param a1:
    :param a2:
    :param b1:
    :param b2:
    :return:
    """
    return a1 + a2 - a1 * np.exp(-b1 * t) - a2 * np.exp(-b2 * t)


def ellenberg(t, a, d):
    """
    Ellenberg function for FRAP analysis

    :param t:
    :param a:
    :param d:
    :return:
    """
    w = 0.5  # radius of bleach spot, unit um
    return a * (1 - (w ** 2 / (w ** 2 + 4 * math.pi * d * t)) ** 0.5)


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
    ss_res = np.sum([(a - b) ** 2 for a, b in zip(y, y_fit)])
    ss_tot = np.sum([(a - np.mean(y)) ** 2 for a in y])
    r2 = 1 - (ss_res / ss_tot)
    return r2


def frap_chi_square(y: list, y_fit: list, sigma: int,
                    imaging_length: int, fitting_model: str):
    """
    modified chi_square test to evaluate the goodness of FRAP curve fitting

                       y - y_fit
    chi_square = SUM (-----------) ^2
                         sigma

    chi-square distribution for N-M degree of freedom can be calculated using  incomplete
    gamma function. Then this distribution gives the probability Q that chi_square should
    exceed a particular chi_square by chance.

    Q(chi_square|df) = Q(chi_square/2, df/2) = 1 - P(chi_square|df) = 1 - P(chi_square/2, df/2)
    Q: probability of chi_square is larger than given value
    The larger the chi_square is, the smaller q is (equals to p value reported from most
    chi_square test

    df = N-M: number of degree of freedom
    N: number of fitted points (imaging_length)
    M: number of parameters within equation to fit (num_fitting_parameter)

    Q > 0.1     good fit
    Q > 0.01    moderately good fit
    Q < 0.01    not good fit

    Note: Q is also affected by sigma, Q could be close to 1 when noise is high

    :param y: values before fitting
    :param y_fit: values after fitting
    :param sigma: standard deviation measured from pre-bleach intensities
                 ideal is standard deviation obtained by repeating FRAP experiment (not feasible),
                 treat pre-bleach standard deviation as the constant measurement error
    :param imaging_length: number of fitted points
    :param fitting_model: determine number of parameters within equation to fit
                single_exp = 2
                double_exp = 4
                soumpasis = 2
                ellenberg = 2
    :return:
    """
    fitting = {'single_exp': 2, 'double_exp': 4, 'soumpasis': 2, 'ellenberg': 2}
    num_fitting_parameter = fitting[fitting_model]

    df = imaging_length - num_fitting_parameter
    chi_square = np.sum([((a - b) / sigma) ** 2 for a, b in zip(y, y_fit)])
    q = distributions.chi2.sf(chi_square, df)

    return chi_square, q


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

    y = c + a * np.exp(-b * x))

    :param x:
    :param y:
    :return: y_fit:
             r2:
             a:
             b:
    """
    try:
        popt, _ = curve_fit(single_exp_decay, x, y)
        a, b, c = popt
    except RuntimeError:
        a = np.nan
        b = np.nan
        c = np.nan

    y_fit = []
    for j in range(len(y)):
        y_fit.append(single_exp_decay(j, a, b, c))
    r2 = r_square(y, y_fit)

    return y_fit, r2, a, b, c


def frap_fitting_single_exp(time_tseries_lst: list, frap_lst: list, sigma_lst: list):
    """
    Perform single exponential fitting for FRAP curves

    y = a * (1 - np.exp(-b * x))

    :param time_tseries_lst: list of real times
    :param frap_lst: list of FRAP curves
    :param sigma_lst: list of sigma values calculated from pre_bleach intensities
    :return: fit_pd: fitting table

    """
    single_exp_fit = []
    single_exp_r2 = []
    single_exp_chi2 = []
    single_exp_q = []
    single_exp_a = []
    single_exp_b = []
    single_exp_t_half = []
    single_exp_slope = []

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
        t_half = np.log(0.5) / (-b)
        chi_square, q = frap_chi_square(frap_lst[i], y_fit, sigma_lst[i], len(frap_lst[i]), 'single_exp')

        def f(t):
            return a * (1 - np.exp(-b * t))
        slope = derivative(f, 0.1, dx=1e-6)

        single_exp_fit.append(y_fit)
        single_exp_r2.append(r2)
        single_exp_chi2.append(chi_square)
        single_exp_q.append(q)
        single_exp_a.append(a)
        single_exp_b.append(b)
        single_exp_t_half.append(t_half)
        single_exp_slope.append(slope)

    fit_pd = pd.DataFrame({'single_exp_fit': single_exp_fit,
                           'single_exp_r2': single_exp_r2,
                           'single_exp_chi2': single_exp_chi2,
                           'single_exp_q': single_exp_q,
                           'single_exp_a': single_exp_a,
                           'single_exp_b': single_exp_b,
                           'single_exp_mobile_fraction': single_exp_a,
                           'single_exp_t_half': single_exp_t_half,
                           'single_exp_slope': single_exp_slope})

    return fit_pd


def frap_fitting_soumpasis(time_tseries_lst: list, frap_lst: list, sigma_lst: list):
    """
    Perform soumpasis fitting for FRAP curves

    y = a * (np.exp(-tau/(2 * t)) * (iv(0, (tau/(2 * t))) + iv(1, (tau/(2 * t)))))

    :param time_tseries_lst: list of real times
    :param frap_lst: list of FRAP curves
    :param sigma_lst: list of sigma values calculated from pre_bleach intensities
    :return: fit_pd: fitting table
    """
    soumpasis_fit = []
    soumpasis_r2 = []
    soumpasis_chi2 = []
    soumpasis_q = []
    soumpasis_a = []
    soumpasis_tau = []
    soumpasis_D_eff = []
    soumpasis_t_half = []
    soumpasis_slope = []

    w = 0.5  # um

    for i in range(len(frap_lst)):
        x = time_tseries_lst[i].copy()
        x.pop(0)
        y = frap_lst[i].copy()
        y.pop(0)
        try:
            popt, _ = curve_fit(soumpasis, x, y)
            a, tau = popt
        except RuntimeError:
            a = np.nan
            tau = np.nan

        y_fit = [0]
        for j in range(len(x)):
            y_fit.append(soumpasis(x[j], a, tau))
        r2 = r_square(frap_lst[i], y_fit)
        D_eff = w ** 2 / tau
        t_half = 0.224 * tau
        chi_square, q = frap_chi_square(frap_lst[i], y_fit, sigma_lst[i], len(frap_lst[i]), 'soumpasis')

        def f(t):
            return a * (np.exp(-tau / (2 * t)) * (iv(0, (tau / (2 * t))) + iv(1, (tau / (2 * t)))))

        slope = derivative(f, 0.1, dx=1e-6)

        soumpasis_fit.append(y_fit)
        soumpasis_r2.append(r2)
        soumpasis_chi2.append(chi_square)
        soumpasis_q.append(q)
        soumpasis_a.append(a)
        soumpasis_tau.append(tau)
        soumpasis_t_half.append(t_half)
        soumpasis_D_eff.append(D_eff)
        soumpasis_slope.append(slope)

    fit_pd = pd.DataFrame({'soumpasis_fit': soumpasis_fit,
                           'soumpasis_r2': soumpasis_r2,
                           'soumpasis_chi2': soumpasis_chi2,
                           'soumpasis_q': soumpasis_q,
                           'soumpasis_a': soumpasis_a,
                           'soumpasis_tau': soumpasis_tau,
                           'soumpasis_mobile_fraction': soumpasis_a,
                           'soumpasis_t_half': soumpasis_t_half,
                           'soumpasis_D_eff': soumpasis_D_eff,
                           'soumpasis_slope': soumpasis_slope})

    return fit_pd


def frap_fitting_double_exp(time_tseries_lst: list, frap_lst: list, sigma_lst: list):
    """
    Perform double exponential fitting for FRAP curves

    y = a1 + a2 - a1 * np.exp(-b1 * t) - a2 * np.exp(-b2 * t)

    :param time_tseries_lst: list of real times
    :param frap_lst: list of FRAP curves
    :param sigma_lst: list of sigma values calculated from pre_bleach intensities
    :return: fit_pd: fitting table

    """
    double_exp_fit = []
    double_exp_r2 = []
    double_exp_chi2 = []
    double_exp_q = []
    double_exp_a1 = []
    double_exp_a2 = []
    double_exp_b1 = []
    double_exp_b2 = []
    double_exp_mobile_fraction = []
    double_exp_t_half = []
    double_exp_slope = []

    for i in range(len(frap_lst)):
        try:
            popt, _ = curve_fit(double_exp, time_tseries_lst[i], frap_lst[i])
            a1, a2, b1, b2 = popt
        except RuntimeError:
            a1 = np.nan
            a2 = np.nan
            b1 = np.nan
            b2 = np.nan

        y_fit = []
        for j in range(len(time_tseries_lst[i])):
            y_fit.append(double_exp(time_tseries_lst[i][j], a1, a2, b1, b2))
        r2 = r_square(frap_lst[i], y_fit)
        chi_square, q = frap_chi_square(frap_lst[i], y_fit, sigma_lst[i], len(frap_lst[i]), 'double_exp')
        mob = a1 + a2
        x_t = np.arange(0.001, 20, 0.001)
        y_t = [double_exp(i, a1, a2, b1, b2) for i in x_t]
        t_half = ble.get_t_half(mob / 2, y_t, x_t)

        def f(t):
            return a1 + a2 - a1 * np.exp(-b1 * t) - a2 * np.exp(-b2 * t)

        slope = derivative(f, 0.1, dx=1e-6)

        double_exp_fit.append(y_fit)
        double_exp_r2.append(r2)
        double_exp_chi2.append(chi_square)
        double_exp_q.append(q)
        double_exp_a1.append(a1)
        double_exp_a2.append(a2)
        double_exp_b1.append(b1)
        double_exp_b2.append(b2)
        double_exp_mobile_fraction.append(mob)
        double_exp_t_half.append(t_half)
        double_exp_slope.append(slope)

    fit_pd = pd.DataFrame({'double_exp_fit': double_exp_fit,
                           'double_exp_r2': double_exp_r2,
                           'double_exp_chi2': double_exp_chi2,
                           'double_exp_q': double_exp_q,
                           'double_exp_a1': double_exp_a1,
                           'double_exp_a2': double_exp_a2,
                           'double_exp_b1': double_exp_b1,
                           'double_exp_b2': double_exp_b2,
                           'double_exp_mobile_fraction': double_exp_mobile_fraction,
                           'double_exp_t_half': double_exp_t_half,
                           'double_exp_slope': double_exp_slope})

    return fit_pd


def frap_fitting_ellenberg(time_tseries_lst: list, frap_lst: list, sigma_lst: list):
    """
    Perform Ellenberg fitting for FRAP curves

    y = a * (1 - math.sqrt(w**2/(w**2 + 4*math.pi*d*t)))

    :param time_tseries_lst: list of real times
    :param frap_lst: list of FRAP curves
    :param sigma_lst: list of sigma values calculated from pre_bleach intensities
    :return: fit_pd: fitting table

    """
    ellenberg_fit = []
    ellenberg_r2 = []
    ellenberg_chi2 = []
    ellenberg_q = []
    ellenberg_a = []
    ellenberg_d = []
    ellenberg_t_half = []
    ellenberg_slope = []
    w = 0.5  # radius of bleach spot, unit um

    for i in range(len(frap_lst)):
        x = time_tseries_lst[i].copy()
        x.pop(0)
        y = frap_lst[i].copy()
        y.pop(0)
        try:
            popt, _ = curve_fit(ellenberg, x, y)
            a, d = popt
        except RuntimeError:
            a = np.nan
            d = np.nan

        y_fit = [0]
        for j in range(len(x)):
            y_fit.append(ellenberg(x[j], a, d))
        r2 = r_square(frap_lst[i], y_fit)
        t_half = 0.75 * w ** 2 / (math.pi * d)
        chi_square, q = frap_chi_square(frap_lst[i], y_fit, sigma_lst[i], len(frap_lst[i]), 'ellenberg')

        def f(t):
            return a * (1 - (w ** 2 / (w ** 2 + 4 * math.pi * d * t)) ** 0.5)

        slope = derivative(f, 0.1, dx=1e-6)

        ellenberg_fit.append(y_fit)
        ellenberg_r2.append(r2)
        ellenberg_chi2.append(chi_square)
        ellenberg_q.append(q)
        ellenberg_a.append(a)
        ellenberg_d.append(d)
        ellenberg_t_half.append(t_half)
        ellenberg_slope.append(slope)

    fit_pd = pd.DataFrame({'ellenberg_fit': ellenberg_fit,
                           'ellenberg_r2': ellenberg_r2,
                           'ellenberg_chi2': ellenberg_chi2,
                           'ellenberg_q': ellenberg_q,
                           'ellenberg_a': ellenberg_a,
                           'ellenberg_d': ellenberg_d,
                           'ellenberg_mobile_fraction': ellenberg_a,
                           'ellenberg_t_half': ellenberg_t_half,
                           'ellenberg_slope': ellenberg_slope})

    return fit_pd


def find_optimal_fitting(pointer_pd: pd.DataFrame, compare_functions: list):
    optimal_function = []
    optimal_fit = []
    optimal_r2 = []
    optimal_chi2 = []
    optimal_q = []
    optimal_mobile_fraction = []
    optimal_t_half = []
    optimal_slope = []

    for i in range(len(pointer_pd)):
        f = 'na'
        q = 0
        chi2 = 10000000
        for m in compare_functions:
            q_temp = pointer_pd[('%s_q' % m)][i]
            chi2_temp = pointer_pd[('%s_chi2' % m)][i]
            if (q_temp > q) | ((q_temp == q) & (chi2_temp < chi2)):
                q = q_temp
                chi2 = chi2_temp
                f = m

        optimal_function.append(f)
        optimal_fit.append(pointer_pd[('%s_fit' % f)][i])
        optimal_r2.append(pointer_pd[('%s_r2' % f)][i])
        optimal_chi2.append(pointer_pd[('%s_chi2' % f)][i])
        optimal_q.append(pointer_pd[('%s_q' % f)][i])
        optimal_mobile_fraction.append(pointer_pd[('%s_mobile_fraction' % f)][i])
        optimal_t_half.append(pointer_pd[('%s_t_half' % f)][i])
        optimal_slope.append(pointer_pd[('%s_slope' % f)][i])

    fit_pd = pd.DataFrame({'optimal_function': optimal_function,
                           'optimal_fit': optimal_fit,
                           'optimal_r2': optimal_r2,
                           'optimal_chi2': optimal_chi2,
                           'optimal_q': optimal_q,
                           'optimal_mobile_fraction': optimal_mobile_fraction,
                           'optimal_t_half': optimal_t_half,
                           'optimal_slope': optimal_slope})

    return fit_pd


def frap_fitting_linear(time_tseries_lst: list, frap_lst: list):
    linear_fit = []
    linear_a = []
    linear_b = []

    for i in range(len(frap_lst)):
        try:
            popt, _ = curve_fit(linear, time_tseries_lst[i][:5], frap_lst[i][:5])
            a, b = popt
        except RuntimeError:
            a = np.nan
            b = np.nan

        y_fit = []
        for j in range(len(time_tseries_lst[i])):
            y_fit.append(linear(time_tseries_lst[i][j], a, b))

        linear_fit.append(y_fit)
        linear_a.append(a)
        linear_b.append(b)

    fit_pd = pd.DataFrame({'linear_fit': linear_fit,
                           'linear_a': linear_a,
                           'linear_b': linear_b,
                           'linear_slope': linear_a})

    return fit_pd

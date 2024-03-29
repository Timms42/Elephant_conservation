"""
Author: Liam Timms 44368768
Created: 18/05/2022
Modified: 06/09/2023
Simpler model - no scavenging -> logistic growth
Convert density X into absolute number Xpop = X*A
Add line indicating where (X*,Y* ) exists
Plot the money vs money, ranger vs area contour
Turn contour plot into a function
v3: - swap axes, try to match up with Kuempel (2017)
v4: - turn slope plot into a function
    - construct the investment path from initial investment position and overlay
        on investment plot
    - converted all money values to 2012 USD to be in line with real estate data
v5: - modify M_func(), M_deriv(), implicit_curve(), and exist_cond() to compute
        the cost of area for land or ocean.
    - allow user to choose from elephant, Kuempel et al. (2018), coral trout, or
        wildebeest data sets
v6: - update carrying capacity of one cell to reflect 1km^2
    - remove "ocean" as option for area type. Only look at terrestrial species
v7: - modify M_func(), M_deriv(), implicit_curve(), and exist_cond() to compute
        the cost of area for Beverton-Holt or power function (corresp. to elephants and wildebeest)
v8: - separate exist_cond() into two functions: one for the existence condition, one for the slope
    - rename "exist_cond_slope()" to "investment_slope"
    - make the investment_slope() run without requiring inputs from the investment_plot() function
    - create function for sensitivity analysis for one parameter and compute the existence condition derivatives for each value
    - create function to run sensitivity analysis on all parameters and plot the existence cond. derivs for each param
v9: - for sensitivity analysis, plot existence condition instead of existence cond. derivatives
    - replace cF with gamma*cF, where the new cF is the actual cost of fines, old cF is expected cost
v10:- create function S_equals_0 to compute "bubble" condition, i.e. if implicit_curve() == 0 at any point for given parameters
    - Create function check_for_bubble to check S_equals_0 for all paramater combinations in sensitivity analysis
v11:- updated area-cost parameters for Zambia to reflect size of Zambia.
v12:- added option for using straight line area-cost function when using Kuempel parameters.
v13:- fixed the bubble contour plotting. Note: contour() and contourf() need X and Y specified as well as Z.
v14.0:- created a main function
   .1:- created function to calculate value of partial derivatives ( p_deriv() )
   .2:- modified check_for_bubble() to optionally calculate the proliferation rate (slope of existence condition at 0)
v15:- started a function to make numerical investment plot for a more complex system of DEs
    - added function for more complex system of DEs with nonlinear relationship between enforcement and poaching
   .2:- completed the function to make numerical investment plot for a more complex system of DEs
   .3:- switched to an ODE solver that detects and handles stiff ODEs (LSODA)
   .4:- allows user to specify simple or complex model for numerical investment plot
   .5:- speed up ODE computation with pre-calculated Jacobian and solving normal ODEs for better initial conditons.
        Added rasterizing to speed up pdf rendering of investment plot.
   .7:- implemented a combination of ODE solver and maximum step size that allows program to solve ODEs numerically for any amount of time
      - create_time_series() now works without error
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sci
import warnings as wrn
import random as rand
import winsound
import time as time
import sys


def xtot(cops_array, cells_array, nondim_list):
    """ Compute the total internal critical point MX*
        :param cells_array: (array) array of M, the number of land cells in protected area
        :param cops_array: (array) arrays of lambda, total number of cops in protected area
        :param nondim_list: (dict of floats) the nondimensional parameters as in simple model
                           [psi, delta, nu, sigma, f, Mmax, mu0]

       :return (array) M*X*(lambda, M)
           """

    par_psi, par_delta, par_nu, par_sigma, par_f, par_Mmax, par_mu0 = nondim_list.values()

    cops_density = cops_array / cells_array

    x_value = cells_array * (par_nu + par_sigma * cops_density) / (par_psi - par_delta * cops_density)

    return x_value


def M_func(par_a, par_b, money_array, dataset):
    """
    Calculate number of area cells (M) as a function of total money invest in area,
    depending on if the area is on land or ocean
    
    :param par_a: (list of floats) parameter a for Beverton-Holt/power function. Not used for Kuempel dataset
    :param par_b: (list of floats) parameter b for Beverton-Holt/ slope for Kuempel/ exponent for power function.
    :param money_array: (array) array of values for total money invested
    :param dataset: (str) "elephant", "kuempel" or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen

    :return: (array) array of M values
    """

    if dataset == 'elephant':
        # If dataset is elephant, the use Beverton-Holt model (fitted in R) for price of area
        M_val = par_a * money_array / (par_b + money_array)

    elif dataset == 'wildebeest':
        # If dataset is wildebeest, the use piecewise linear model (fitted in R) for price of area
        M_val = par_a * money_array ** par_b

    elif dataset == 'kuempel':
        M_val = par_a * money_array

    else:
        raise ValueError('dataset not in ["elephant", "kuempel", "wildebeest"].')

    return M_val


def M_deriv(par_a, par_b, money_array, dataset):
    """
    Calculate the derivative of the money curve M(mu_area) as a function of
    total money invested, depending on if the area is on land or ocean

    :param par_a: (float) parameter a for Beverton-Holt/power function. Not used for Kuempel dataset
    :param par_b: (float) parameter b for Beverton-Holt/straight line/power function
    :param money_array: (array) array of values for total money invested
    :param dataset: (str) "elephant", "kuempel", or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen

    :return: array
    """

    if dataset == 'elephant':
        # If dataset is elephant, then use use derivative of Beverton-Holt model
        M_val = par_a * par_b / (par_b + money_array) ** 2

    elif dataset == 'wildebeest':
        # If dataset is wildebeest, the use coefficients in piecewise linear function
        M_val = par_a * par_b * money_array ** (par_b - 1)

        # # par_a[1] is the intercept
        # M_val = np.piecewise(money_array, [(0 <= money_array) & (money_array < par_b[0]),
        #                                    (par_b[0] <= money_array) & (money_array < par_b[1]),
        #                                    (par_b[1] <= money_array) & (money_array < par_b[2]),
        #                                    (par_b[2] <= money_array) & (money_array < par_b[3]),
        #                                    (par_b[3] <= money_array)],
        #                      [lambda x: par_a[1], lambda x: par_a[2], lambda x: par_a[3],
        #                       lambda x: par_a[4], lambda x: par_a[5]])

    elif dataset == 'kuempel':
        # If dataset is Kuempel, slope is constant
        M_val = par_a * np.ones(len(money_array))

    else:
        raise ValueError('dataset not in ["elephant", "kuempel", "wildebeest"].')

    return M_val


def lambda_func(par_f, money_array):
    """
    Calculate number of rangers as a function of total money invested,
    assuming constant hiring cost of rangers

    :param par_f: (float) 1/cost of hiring one ranger
    :param money_array: (array) array of mu_rangers, total money invested in rangers
        
    :return: array
    """

    lambda_val = par_f * money_array

    return lambda_val


def gamma_func(num_area, num_rangers, par_gamma, par_attack, par_handle, par_power, par_denom):
    """
    Computes perceived catchability of poachers by rangers as a function of current ranger density.
    B(x) = gamma*a*x^z/(w^z + a*h*x^z), where x is ranger density. Slope is 0 at x=0, and asymptotes at gamma as x->infinity.
    Contrast this with law of mass-action, which has C2(x) = gamma. C1 <= C2 for all x.
    Return min(C1(x), 1) so that the catchability is bounded by 1.
    :param num_area: (float) current number of area cells
    :param num_rangers: (float) current number of rangers hired in total
    :param par_gamma: (float) catchability coefficient gamma from linear functional response/law of mass-action
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_handle: (float) scaling coefficient, equivalent to handling time in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param par_denom: (float) parameter w on the denominator of B(x)
    :return: (float)
    """

    # Compute density**k, where k is given in par_power
    density_pow_k = np.power(num_rangers / num_area, par_power)

    perceived_gamma = par_gamma * par_attack * density_pow_k / (par_denom**par_power + par_attack * par_handle * density_pow_k)

    # Note that this function for gamma is bounded by 0 and 1, so it can represent a proportion.
    return perceived_gamma


def simple_model(t, y, money_area, money_ranger, par_attack, par_handle, par_power, par_denom, interact, dim_list, terminate):
    """
    :param t: time. used for scipy.solve_ivp
    :param z: (float or array) state variables N, E
    :param money_area: (float) current money invested in area
    :param money_ranger: (float) current money invested in area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_handle: (float) scaling coefficient, equivalent to handling time in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param terminate: (float>0) tolerance for terminating numerical integration
    :return: array
    """
    # Unpack state variables for num elephants N and poaching effort E
    N, E = y

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Calculate current number of rangers hired and amount of area as functions of money invested
    num_rangers = lambda_func(par_f, money_ranger)
    num_area = M_func(par_Mmax, par_nonlin, money_area, 'elephant')

    # Compute the density of rangers over the protected area
    density = num_rangers / num_area

    Ndot = ((par_b - par_m) * (1 - N / par_k) - par_q * E) * N
    Edot = par_alpha * E * ((1 - par_gamma * density) * par_p0 * par_q * N - par_c0 - par_gamma * par_cf * density)

    return [Ndot, Edot]


def complex_model(t, y, money_area, money_ranger, par_attack, par_handle, par_power, par_denom, interact, dim_list, terminate):
    """
    Elephant-poacher population model over the entire protected area
    # Note: interact ['real', 'real'] -> simple model, ['perceived', 'perceived'] -> complex model
    :param t: time. used for scipy.solve_ivp
    :param y: (float or array) state variables N, E
    :param money_area: (float) current money invested in area
    :param money_ranger: (float) current money invested in area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_handle: (float) scaling coefficient, equivalent to handling time in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param interact: (list-like) [confiscation interact, fines interact]
        "real" if this model component uses the 'real' linear interact term (par_gamma),
        "perception" if the component uses the complex "perceived" interact (gamma_coeff).
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param terminate: (float>0) tolerance for terminating numerical integration
    :return: (array) value of time derivative dNdt and dEdt
    """
    # Unpack state variables for num elephants N and poaching effort E
    N, E = y

    # Unpack the interact types (type of interact for confiscation and fines model components
    confiscate_type, fines_type = interact

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Calculate current number of rangers hired and amount of area as functions of money invested
    num_rangers = lambda_func(par_f, money_ranger)
    num_area = M_func(par_Mmax, par_nonlin, money_area, 'elephant')

    # Compute the density of rangers over the protected area
    density = num_rangers / num_area

    # Compute poachers' perceived catchability by rangers, which factors into expected cost of poaching.
    complex_gamma = gamma_func(num_area, num_rangers, par_gamma, par_attack, par_handle, par_power, par_denom)
    confiscate_gamma = par_gamma if confiscate_type.lower().startswith('real') else complex_gamma
    fines_gamma = par_gamma if fines_type.lower().startswith('real') else complex_gamma

    Ndot = ((par_b - par_m) * (1 - N / par_k) - par_q * E) * N
    Edot = par_alpha * E * (
            (1 - confiscate_gamma * density) * par_p0 * par_q * N - par_c0 - fines_gamma * par_cf * density)

    return [Ndot, Edot]


def complex_model_jacobian(t, y, money_area, money_ranger, par_attack, par_handle, par_power, par_denom, interact, dim_list, tol):
    """
    Same arguments as complex_model(). Necessary for passing arguments to solve_ivp() in function numerical_investment()
    Note: interact ['real', 'real'] -> simple model, ['perceived', 'perceived'] -> complex model
    :param t: time. used for scipy.solve_ivp
    :param y: (float or array) state variables N, E
    :param money_area: (float) current money invested in area
    :param money_ranger: (float) current money invested in area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_handle: (float) scaling coefficient, equivalent to handling time in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param interact: (list-like) [confiscation interact, fines interact]
        "real" if this model component uses the 'real' linear interact term (par_gamma),
        "perception" if the component uses the complex "perceived" interact (gamma_coeff).
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param terminate: (float>0) tolerance for terminating numerical integration
    :return: (array) values of Jacobian matrix of the complex model
    """
    # Unpack state variables for num elephants N and poaching effort E
    N, E = y

    # Unpack the interact types (type of interact for confiscation and fines model components
    confiscate_type, fines_type = interact

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Calculate current number of rangers hired and amount of area as functions of money invested
    num_rangers = lambda_func(par_f, money_ranger)
    num_area = M_func(par_Mmax, par_nonlin, money_area, 'elephant')

    # Compute the density of rangers over the protected area
    density = num_rangers / num_area
    # Compute poachers' perceived catchability by rangers, which factors into expected cost of poaching.
    complex_gamma = gamma_func(num_area, num_rangers, par_gamma, par_attack, par_handle, par_power, par_denom)
    confiscate_gamma = par_gamma if confiscate_type.lower().startswith('real') else complex_gamma
    fines_gamma = par_gamma if fines_type.lower().startswith('real') else complex_gamma

    # Compute coefficients for use in Jacobian
    C = par_alpha * (1 - confiscate_gamma * density) * par_p0 * par_q
    D = par_alpha * (par_c0 + par_cf * fines_gamma * density)

    # dF1/dN
    F1_N = (b - m) - 2 * (b - m) * N / k - q * E

    # dF1/dE
    F1_E = - q * N

    # dF2/dN
    F2_N = C * E

    # dF2/dE
    F2_E = C * N - D

    # Return the Jacobian
    return np.array([[F1_N, F1_E], [F2_N, F2_E]])


def analytic_equil(money_area, money_ranger, par_attack, par_handle, par_power, par_denom, interact, dim_list):
    """
    Comnpute the analytic internal stable equilibrium (N*, P*) in one cell
    Note: interact ['real', 'real'] -> simple model, ['perceived', 'perceived'] -> complex model

    :param money_area: (float) current money invested in area
    :param money_ranger: (float) current money invested in area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_handle: (float) scaling coefficient, equivalent to handling time in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param interact: (list-like) [confiscation interact, fines interact]
        "real" if this model component uses the 'real' linear interact term (par_gamma),
        "perception" if the component uses the complex "perceived" interact (gamma_coeff).
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}

    :return: (list) equilibrium [N*, P*]
    """

    # Unpack the interact types (type of interact for confiscation and fines model components
    confiscate_type, fines_type = interact

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Calculate current number of rangers hired and amount of area as functions of money invested
    num_rangers = lambda_func(par_f, money_ranger)
    num_area = M_func(par_Mmax, par_nonlin, money_area, 'elephant')

    # Compute the density of rangers over the protected area
    density = num_rangers / num_area

    # Compute poachers' perceived catchability by rangers, which factors into expected cost of poaching.
    complex_gamma = gamma_func(num_area, num_rangers, par_gamma, par_attack, par_handle, par_power, par_denom)
    confiscate_gamma = par_gamma if confiscate_type.lower().startswith('real') else complex_gamma
    fines_gamma = par_gamma if fines_type.lower().startswith('real') else complex_gamma

    Nstar_local = (par_c0 + par_cf * confiscate_gamma * density) / (
            par_p0 * par_q * (1 - fines_gamma * density))
    Estar_local = (par_b - par_m) / par_q * (1 - Nstar_local / par_k)

    if (Estar_local > 0) and (Nstar_local > 0):
        Nstar = Nstar_local
        Estar = Estar_local

    else:  # if this equilibrium is no longer in the positive quadrant, the stable equilibrium is (k,0)
        Nstar = par_k
        Estar = 0

    return [Nstar, Estar]


def event(t, y, money_area, money_ranger, par_attack, par_handle, par_power, dim_list, terminate):
    """
    :param t: time. used for scipy.solve_ivp
    :param y: (float or array) state variables N, E
    :param money_area: (float) current money invested in area
    :param money_ranger: (float) current money invested in area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_handle: (float) scaling coefficient, equivalent to handling time in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param terminate: (float>0) tolerance for terminating numerical integration

    :return: (float) value of f(N, E), the event function. Event occurs when f = 0.
    """
    # Event function for solve_ivp. Terminate when ODEs are at equilibrium, i.e. |dNdt| + |dEdt| = epsilon << 1
    # Must return a scalar not a list, hence the summation.

    Ndot, Edot = complex_model(t, y, money_area, money_ranger, par_attack, par_handle, par_power, dim_list, terminate)

    return abs(Ndot) + abs(Edot) - terminate


def implicit_curve(money_ranger, money_area, nondim_list, dataset):
    """
        Compute implicit function
            S(lambda, M) = darea/dmu * dx/darea - f * dx/dcops
        :param money_area: (array) array of mu_area, total money invested in area
        :param money_ranger: (array) array of mu_rangers, total money invested in rangers
        :param nondim_list: (dict of floats) the nondimensional parameters as in simple model
                            f is cops conversion rate
                           [psi, delta, nu, sigma, f, M_max, mu_0]
        :param dataset: (str) "elephant", "kuempel", or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen

        :return: (array) values of S(mu_lambda, mu_area), dX/dmu_a, dX/dmu_r
   """

    # Unpack the nondimensional variables
    par_psi, par_delta, par_nu, par_sigma, par_f, par_Mmax, par_mu0 = nondim_list.values()

    # Compute the number of rangers hired for each amount in money_array
    num_rangers = lambda_func(par_f, money_ranger)

    # Compute the number of area cells purchased for each amount in money_array
    M = M_func(par_Mmax, par_mu0, money_area, dataset)

    dareadmu = M_deriv(par_Mmax, par_mu0, money_area, dataset)

    # Compute derivative of X* w.r.t. area using chain rule
    dxdarea = (-2 * M * par_delta * num_rangers * par_nu - par_delta * num_rangers ** 2 * par_sigma +
               M ** 2 * par_nu * par_psi) / (par_delta * num_rangers - M * par_psi) ** 2

    # Compute derivative of X* w.r.t. cops
    dxdcops = (M ** 2 * (par_delta * par_nu + par_sigma * par_psi)) / (par_delta * num_rangers - M * par_psi) ** 2

    # Compute implicit function S(mu_ranger, mu_area) = dx/dmu_a - dx/dmu_r
    S = dareadmu * dxdarea - par_f * dxdcops

    return S


def p_deriv(money_ranger, money_area, nondim_list, dataset):
    """
        Compute partial derivatives
            darea/dmu * dx/darea and f * dx/dcops
        :param money_area: (array) array of mu_area, total money invested in area
        :param money_ranger: (array) array of mu_rangers, total money invested in rangers
        :param nondim_list: (dict of floats) the nondimensional parameters as in simple model
                            f is cops conversion rate
                           [psi, delta, nu, sigma, f, M_max, mu_0]
        :param dataset: (str) "elephant", "kuempel", or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen

        :return: (array, array) dX/dmu_a, dX/dmu_r
   """

    # Unpack the nondimensional variables
    par_psi, par_delta, par_nu, par_sigma, par_f, par_Mmax, par_mu0 = nondim_list.values()

    # Compute the number of rangers hired for each amount in money_array
    num_rangers = lambda_func(par_f, money_ranger)

    # Compute the number of area cells purchased for each amount in money_array
    M = M_func(par_Mmax, par_mu0, money_area, dataset)

    dareadmu = M_deriv(par_Mmax, par_mu0, money_area, dataset)

    # Compute derivative of X* w.r.t. area using chain rule
    dxdarea = (-2 * M * par_delta * num_rangers * par_nu - par_delta * num_rangers ** 2 * par_sigma +
               M ** 2 * par_nu * par_psi) / (par_delta * num_rangers - M * par_psi) ** 2

    # Compute derivative of X* w.r.t. cops
    dxdcops = (M ** 2 * (par_delta * par_nu + par_sigma * par_psi)) / (par_delta * num_rangers - M * par_psi) ** 2

    # Compute implicit function S(mu_ranger, mu_area) = dx/dmu_a - dx/dmu_r
    return dareadmu * dxdarea, par_f * dxdcops


def exist_cond(money_area, nondim_list, dataset):
    """
    Compute function that determines if 0 < X* < 1,
    num_rangers < beta(num_cells).
    Define B(mu_ranger, mu_area) = beta * M(mu_area) - lambda(mu_ranger)
    Compute B = 0 contour. When  lambda(mu_ranger) > beta * M(mu_area), the stable 
    critical point is the carrying capacity, i.e. "above" the existence line

    :param money_area: (array) array of total money values invested into area M
    :param nondim_list: (dict of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma, f, M_max, mu_0]
    :param dataset: (str) "elephant", "kuempel", or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen

    :return (array, array) cell values for mu_ranger = beta(mu_area)
    """

    par_psi, par_delta, par_nu, par_sigma, par_f, par_Mmax, par_mu0 = nondim_list.values()

    # mu_area as a function of mu_rangers, solution to beta * M(mu_area) = lambda(mu_ranger)
    C = (par_psi - par_nu) / (par_sigma + par_delta)  # Constant based on parameters

    # mu_rangers as function of mu_area
    exist_contour = C / par_f * M_func(par_Mmax, par_mu0, money_area, dataset)

    return exist_contour


def proliferation(dim_list, dataset):
    """
    Compute derivative of the function that determines if 0 < X* < 1,
    num_cops < beta(num_cells), evaluated at num_cells = 0.
    Define B(mu_ranger, mu_area) = beta * M(mu_area) - lambda(mu_ranger)
    Compute slope of B = 0 contour. When  lambda(mu_ranger) > beta * M(mu_area), the stable
    critical point is the carrying capacity, i.e. "above" the existence line

    :param nondim_list: (dict of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma, f, M_max, mu_0]
    :param dataset: (str) "elephant" or "kuempel". Says which area-cost model to use,
                        depending on the dataset chosen

    :return (float) value for derivative of mu_ranger=beta(mu_area), the existence condition, at num_cells = 0
    """

    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Compute derivative of mu_ranger function
    # This tells us the ratio of investing in rangers vs area
    if dataset.lower().startswith('e'):
        prolifer = (par_p0 * par_q * par_k - par_c0) / \
                   (par_p0 * par_q * par_k + par_cf) / par_gamma / par_f * par_Mmax / par_mu0

    elif dataset.lower().startswith('k'):
        prolifer = (par_p0 * par_q * par_k - par_c0) / \
                   (par_p0 * par_q * par_k + par_cf) / par_gamma / par_f * par_Mmax

    else:
        raise ValueError('dataset not in ["elephant", "kuempel"]. Cannot use for "wildebeest".')

    return prolifer


def exist_slope(money_area, nondim_list, dataset):
    """
    Compute derivative of the function that determines if 0 < X* < 1,
    num_cops < beta(num_cells).
    Define B(mu_ranger, mu_area) = beta * M(mu_area) - lambda(mu_ranger)
    Compute slope of B = 0 contour. When  lambda(mu_ranger) > beta * M(mu_area), the stable
    critical point is the carrying capacity, i.e. "above" the existence line

    :param money_area: (array) array of total money values invested into area M
    :param nondim_list: (dict of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma, f, M_max, mu_0]
    :param dataset: (str) "elephant", "kuempel", or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen

    :return (array) values for derivative of mu_ranger=beta(mu_area), the existence condition
    """

    par_psi, par_delta, par_nu, par_sigma, par_f, par_Mmax, par_mu0 = nondim_list.values()

    # mu_rangers as a function of mu_area, solution to lambda(mu_ranger) = beta * M(mu_area)
    C = (par_psi - par_nu) / (par_sigma + par_delta)  # Constant based on parameters

    # Compute derivative of mu_ranger function
    # This tells us the ratio of investing in rangers vs area
    exist_deriv = C / par_f * M_deriv(par_Mmax, par_mu0, money_area, dataset)

    return exist_deriv


def S_equals_0(small_dim_list):
    """
    For Beverton-Holt area-money function, check the implicit_function() == 0 at any point using the condition
    mu0 < Mmax/gamma/f * c0/(c0+cF)

    For Kuempel straight line area-money function, the condition becomes (Mmax is the slope)
    1 < par_Mmax/gamma/f * c0/(c0+cF)

    :param reduced_dim_list: (list) parameters for dimensionalised model,
            [gamma, f, c0, cF, Mmax]
    :param dataset: (str) "elephant", "kuempel", or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen
    :return: (float) right hand side of S=0 condition, given the nondim parameter values.
    """

    par_gamma, par_f, par_c0, par_cf, par_Mmax = small_dim_list

    condition = par_Mmax / par_f / par_gamma * par_c0 / (par_c0 + par_cf)

    return condition


def generate_slopes(dim_range, num_it, dataset):
    """
    Compute the existence condition slope given parameter values sampled uniformly from a specified range of values.
    Repeat this num_it times and return a list of the results.

    :param dim_range: (dict of lists) the dictionary of dimensional parameters and their corresponding range of values
    :param num_it: (int) number of iterations for each parameter
    :param dataset: (str) "elephant" or "kuempel"

    :return: list of parameter combinations that result in a bubble

    Note 1: for parameters not used in the proliferation calculation, just use e.g. {'b': [0.1, 0.1]}
    Note 2: parameter ranges must satisfy p0*q*k >= c0, otherwise there's no poaching and there is no existence condition to examine
    """
    # Initialise list of slope values
    slope_values = [0] * num_it

    # Compute value of the slope num_it times
    for ii in range(num_it):
        # Generate random values for parameters from specified range of values

        dim_list = {'b': rand.uniform(*dim_range['b']),
                    'm': rand.uniform(*dim_range['m']),
                    'k': rand.uniform(*dim_range['k']),
                    'q': rand.uniform(*dim_range['q']),
                    'alpha': rand.uniform(*dim_range['alpha']),
                    'p0': rand.uniform(*dim_range['p0']),
                    'gamma': rand.uniform(*dim_range['gamma']),
                    'c0': rand.uniform(*dim_range['c0']),
                    'cF': rand.uniform(*dim_range['cF']),
                    'f': rand.uniform(*dim_range['f']),
                    'par_coeff': rand.uniform(*dim_range['par_coeff']),
                    'par_nonlin': rand.uniform(*dim_range['par_nonlin'])
                    }

        # Calculate the proliferation rate of the existence condition given the random parameter values
        slope_values[ii] = proliferation(dim_list, dataset)

    return slope_values


def check_for_bubble(dim_list, num_it, dataset):
    """
    Check if any of the parameter combinations can result in a bubble, varying parameter values by one order
    of magnitude higher and lower.

    :param dim_list: (dict of floats) the dictionary of dimensional parameters
    :param num_it: (int) number of iterations for each parameter
    :param dataset: (str) "elephant" or "kuempel"

    :return: list of parameter combinations that result in a bubble
    """
    # num_it + 1 elements. Distance between 1/10 and 10 = 100/10 - 1/10 = 99/10, so step size is (99/10)/num_it
    multiply_array = np.arange(1 / 10, 10 + 99 / 10 / num_it, 99 / 10 / num_it)

    params = dict((key, val * multiply_array) for key, val in dim_list.items())
    # Initialise list of parameter combos that result in a bubble
    bubble_params = []

    count = 0

    for f_val in params['f']:
        for c0_val in params['c0']:
            for cF_val in params['cF']:
                for Mmax_val in params['par_coeff']:

                    if dataset == 'elephant':
                        # Loop over gamma values since gamma is involved in the bubble condition
                        for gamma_val in params['gamma']:

                            # Array of bools if param combination results in a bubble (S=0) in investment plot
                            # Bubble condition for Bev-Holt area-money function is mu0 < S_equals_0()
                            bubble_array = params['par_nonlin'] < S_equals_0(
                                [gamma_val, f_val, c0_val, cF_val, Mmax_val])

                            # Check the condition for a bubble existing, i.e. S=0 at some point mu_area
                            if any(bubble_array):
                                # Find mu0 values that have a bubble
                                mu0_indices = np.where(bubble_array == True)
                                # Print the smallest and largest mu0 values that give a bubble
                                smallest_mu0 = params['par_nonlin'][mu0_indices[0][0]]
                                largest_mu0 = params['par_nonlin'][mu0_indices[0][-1]]

                                # Add the params that result in a bubble to the bubble_params list for returning
                                # Note that this will imply that
                                bubble_params.append(
                                    [gamma_val, f_val, c0_val, cF_val, Mmax_val, [smallest_mu0, largest_mu0]])

                        # Number of possible combos = num_it ** no of params = num_it**6
                        prop_bubble = len(bubble_params) / (num_it ** 6)

                    elif dataset == 'kuempel':
                        # Bubble condition for straight line area-money function is 1 < S_equals_0()
                        # Note that gamma is not involved in the S=0 condition for a straight line
                        # area-money function, so just use the original gamma value
                        if 1 < S_equals_0([dim_list['gamma'], f_val, c0_val, cF_val, Mmax_val]):
                            bubble_params.append(
                                [f_val, c0_val, cF_val, Mmax_val])

                        # Number of possible combos = num_it ** no of params = num_it**5
                        prop_bubble = len(bubble_params) / (num_it ** 5)

                    else:
                        raise ValueError('dataset not in ["elephant", "kuempel"]. Cannot use for "wildebeest".')

                    # Display loop % completion to user
                    count += 1
                    if count % 32000 == 0:
                        print("{:.2f}% completed".format(count / (len(multiply_array) ** 5) * 100))

    # Print percentage of param combinations that result in a bubble

    print(f'{prop_bubble * 100:.2f} % of param combos results in a bubble.')

    return bubble_params


def investment_path(invest_init, invest_size, num_steps, nondim_list, dataset):
    """
    Create path of investing in rangers or area from an initial investment.
    Plot path on given figure.
    :param invest_init: (list) initial investment in area and rangers
    :param invest_size: step size for investments
    :num_steps: (int) number of investment steps to perform
    :param nondim_list: (dict) parameters for nondimensionalised model, [psi, delta, nu, sigma, f, Mmax, mu0]
    :param dataset: (str) "elephant" or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen

    :return (array) two arrays, area investments and ranger investments over time
    """

    # Initialise investment lists
    invest_a = np.zeros(num_steps)  # Investment in area
    invest_r = np.zeros(num_steps)  # Investment in rangers

    # Set initial investments to the specified initial conditions
    invest_a[0] = invest_init[0]
    invest_r[0] = invest_init[1]

    # Construct investment path for specified number of steps
    for ii in range(0, num_steps - 1):
        # Compute the value of implicit function S at the current ranger and area investment
        S_val = implicit_curve(invest_r[ii], invest_a[ii], nondim_list, dataset)

        # Compute ranger investment for existence condition given current area investment
        exist_cond_val = exist_cond(invest_a[ii], nondim_list, dataset)

        # If S>0 or the investment is above the existence condition, then invest in area
        if S_val > 0 or invest_r[ii] > exist_cond_val:
            invest_a[ii + 1] = invest_a[ii] + invest_size
            invest_r[ii + 1] = invest_r[ii]  # Ranger investment doesn't change

        else:  # Otherwise invest in rangers. If we're on the existence condition,
            # or S=0, then invest in rangers as tiebreaker
            invest_a[ii + 1] = invest_a[ii]
            invest_r[ii + 1] = invest_r[ii] + invest_size

    return invest_a, invest_r


def investment_plot(ran_interval, area_interval, num_p, nondim_list, dim_list, plot_path,
                    invest_init, invest_size, num_steps, dataset,
                    lwid, clist, fontsz, save, fname, pname, ax):
    """
    Create contour plot for money into rangers vs money into area, decide which strategy is best
    :param ran_interval: (list) [ranger money lower bound, ranger money upper bound]
    :param area_interval: (list) [area money lower bound, area money upper bound]
    :param num_p: (int) number of points to plot for each axis
    :param nondim_list: (dict) parameters for nondimensionalised model, {psi, delta, nu, sigma, f, Mmax, mu0}
    :param dim_list: (dict) parameters for dimensionalised model,
            {b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param plot_path: (True/False) True if plotting optimal investment path/s. False otherwise
    :param invest_init: (list) initial investment in area and rangers
    :param invest_size: step size for investments
    :param num_steps: (int) number of investment steps to perform
    :param dataset: (str) "elephant" or "wildebeest". Says which area-cost model to use,
                        depending on the dataset chosen
    :param lwid: (float) line width, should be positive float
    :param clist (list) 2 strings, hex codes for plot colours [S contour fill, S contour line/exist cond line]
    :param fontsz: (list): list of font sizes [axes, axtick, legend]
    :param save: (True/False) True if you want to save the plot, False otherwise
    :param fname: (str) Folder location to save figure
    :param pname: (str) name for the plot, goes on end of file name when saving
    :param ax: (ax object or None) can pass an existing axis to the function, or leave as None if you want
                    to create an axis object in this function

    :return figure, with one subfigure for each value in list_cops
    """

    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    mu_area_ar = np.linspace(area_interval[0], area_interval[1], num=num_p)
    mu_ranger_ar = np.linspace(ran_interval[0], ran_interval[1], num=num_p)

    # Combine the arrays into a grid for contour plot
    [mu_a, mu_ran] = np.meshgrid(mu_area_ar, mu_ranger_ar)

    # Create figure for plotting the investment plot, with dimensions 12x12
    fig = plt.figure(figsize=(12, 12))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    if ax is None:
        ax = fig.add_subplot(111)

    # Create the contour plot of S(lambda, M)
    # Contour levels at S=0 and at 10
    # (Note: max S << 1, so this is a cheat for filling in above contour)
    # Filled in contour for S>0
    contour0_fill = ax.contourf(mu_a, mu_ran, implicit_curve(mu_ran, mu_a, nondim_list, dataset),
                                levels=np.array([0, 1]), colors=clist[0])
    print('Done with S=0 contour filled')

    # Compute existence condition and slope of existence condition
    exist_curve = exist_cond(mu_area_ar, nondim_list, dataset)

    # Plot the existence condition. Solid line with colour clist[1]. Indexed [0] since plt.plot returns a list
    exist_curve_plot = ax.plot(mu_area_ar, exist_curve,
                               color=clist[1], linewidth=lwid, linestyle='-', label='Exist. cond.')[0]

    print('Done with existence condition line')

    # Add contour line for the boundary S=0. Dashed line with colour clist[1]
    try:
        contour0_line = ax.contour(mu_a, mu_ran, implicit_curve(mu_ran, mu_a, nondim_list, dataset),
                                   levels=np.array([0]), colors=clist[1], linewidths=lwid,
                                   linestyles='dashed')

        print('Done with S=0 contour line')

        # Create proxy artist to give to legend()
        # S0_line = mlines.Line2D([], [], linestyle='dashed', color='black', label='S=0 contour')

        # plt.legend(handles=[exist_curve_plot, S0_line], fontsize=fontsz[2])

    # If the S=0 contour DNE, then don't try to plot it and move on with the program
    except UserWarning:
        print('S(mu_ran, mu_area) does not have a 0 contour.')

        # plt.legend(handles=[exist_curve_plot], fontsize=fontsz[2])

    # Shade region where (X+, Y+) = (M, 0),
    # i.e. where existence condition is negative, above existence contour
    ax.fill_between(mu_area_ar, exist_curve, y2=1e10, color=clist[0])

    # If the user wants to plot optimal investment paths
    if plot_path:
        # For each initial investment point, plot the optimal path
        for jj in range(0, len(invest_init)):
            # Add optimal investment path from initial investment of mu_a, mu_r = invest_init[jj]
            invest_area, invest_ran = investment_path(invest_init[jj], invest_size, num_steps, nondim_list, dataset)

            ax.plot(invest_area, invest_ran, markersize=2, color=clist[2])

    params_text = f'Parameters: b={par_b}, m={par_m}, k={par_k}, q={par_q},\n alpha={par_alpha}, p0={par_p0},' \
                  f' gamma={par_gamma:.5},\n c0={par_c0}, cf={par_cf:.3}, f={par_f:.2}, Mmax={par_Mmax}, mu0={par_mu0}'

    # Dataset specific plot additions
    if dataset == 'elephant':
        # Add a star for the current investment in area and rangers for Zambia.
        # See Google sheet 'model_parameters.xlsx' for source.
        plt.scatter(13389548.33, 101033450.3, s=200, c='k', marker='*')
    elif dataset == 'kuempel':
        # Add text showing the cost of area
        ax.annotate(f'{1 / (706 * par_Mmax):.2} $/km2', (0.2 * area_interval[1], 0.4 * ran_interval[1]), fontsize=24)

    # Add in parameter text to the plot
    ax.annotate(params_text, (0.1 * area_interval[1], 0.5 * ran_interval[1]), fontsize=12)

    ax.set_title(
        'Implicit curve S(mu_ranger, mu_area)\n S>0 (grey) -> invest in area\n'
        'Above solid line: (X+, Y+)=(M,0)\n Solid line: exist cond. Dashed line: S=0 contour')
    ax.set_xlabel('Current USD invested in area', fontsize=fontsz[0])
    ax.set_ylabel('Current USD invested in rangers', fontsize=fontsz[0])

    plt.axis([area_interval[0], area_interval[1], ran_interval[0], ran_interval[1]])

    plt.xticks(fontsize=fontsz[1])
    plt.yticks(fontsize=fontsz[1])

    plt.show()

    if save:
        savename = f'{fname}\\{pname}'
        fig.savefig(savename)

    return fig, ax, exist_curve


def investment_slope(ran_interval, area_interval, num_p, nondim_list, dim_list, dataset,
                     lwid, clist, fontsz, save, fname, pname):
    """
    Create plot of the slope of the existence condition to show ratio of investment
    in rangers vs in area
    :param ran_interval: (list) [ranger money lower bound, ranger money upper bound]
    :param area_interval: (list) [area money lower bound, area money upper bound]
    :param num_p: (int) number of points to plot for each axis
    :param exist_deriv: (array) values of existence condition derivative. Returned from investment_plot()
    :param nondim_list: (dict) parameters for nondimensionalised model, [psi, delta, nu, sigma, f, Mmax, mu0]
    :param dim_list: (dict) parameters for dimensionalised model,
            [b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0]
    :param dataset: (str) "elephant" or "wildebeest". Says which area-cost model to use,
                depending on the dataset chosen
    :param lwid: (float) line width, should be positive float
    :param clist (list) string, hex codes for plot line colours
    :param fontsz: (list): list of font sizes [axes, axtick, legend]
    :param save: (True/False) True if you want to save the plot, False otherwise
    :param fname: (str) Folder location to save figure
    :param pname: (str) name for the plot, goes on end of file name when saving

    :return figure, with one subfigure for each value in list_cops
    """

    par_psi, par_delta, par_nu, par_sigma, par_f, par_Mmax, par_mu0 = nondim_list.values()

    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    mu_area_ar = np.linspace(area_interval[0], area_interval[1], num_p)

    exist_deriv = exist_slope(mu_area_ar, nondim_list, dataset)

    # Create figure for plotting the slope of existence condition, with dimensions 10x10
    fig = plt.figure(figsize=(10, 10))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    ax = fig.add_subplot(111)

    # Plot the slope of the existence condition
    ax.plot(mu_area_ar, exist_deriv, linewidth=lwid, color=clist[0])

    ax.set_title(
        'Slope of existence condition mu_ranger = beta(mu_area)')
    ax.set_xlabel('Current USD invested in area', fontsize=fontsz[0])
    ax.set_ylabel('Ratio of investment in rangers vs. area', fontsize=fontsz[0])

    if dataset == 'wildebeest':
        # The existence condition curve slope decays very quickly and is vertical at the origin.
        # Limit y-axis to make the plot clearer
        ax.set_ylim([0, 0.01])

    # Create and add text for parameters
    params_text = f'Parameters: b={par_b}, m={par_m}, k={par_k}, q={par_q},\n alpha={par_alpha}, p0={par_p0},' \
                  f' gamma={par_gamma:.5},\n c0={par_c0}, cf={par_cf}, f={par_f:.5}, Mmax={par_Mmax}, mu0={par_mu0}'

    ax.annotate(params_text, (0.1 * mu_area_ar[1], 0.75 * max(exist_deriv)), fontsize=12)

    plt.xlim([area_interval[0], area_interval[1]])
    # plt.ylim([0, 1])

    plt.xticks(fontsize=fontsz[1])
    plt.yticks(fontsize=fontsz[1])

    fig.show()

    if save:
        savename = f'{fname}\\{pname}'
        fig.savefig(savename)

    return fig, ax


def sens_analysis(param_name, num_it, area_interval, num_p, dim_list, dataset):
    """
    The plan: pick a parameter to do sensitivity analysis on. For num_it iterations, multiply parameter
     by a factor between 1 and 10. Then compute nondim params and compute the slope of exist cond.
     >> sens_analysis('b', 100, [0, 1e8], 1000, dim_list, 'elephant')

    :param param_name: (str) name of parameter to perform sensitivity analysis on
    :param num_it: (int) number of different parameter values for sensitivity analysis
    :param area_interval: (list) [area money lower bound, area money upper bound]
    :param num_p: (int) number of points to plot for each axis
    :param dim_list: (dict) parameters for dimensionalised model,
            [b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0]
    :param dataset: (str) "elephant" or "wildebeest". Says which area-cost model to use,
                depending on the dataset chosen

    :return: exist_dict: (dict) dictionary of existence condition lines for each value of the parameter
    """

    # Store copy of the default parameter values
    params = dim_list.copy()

    # Initialise dictionary to store exist cond lines in
    exist_dict = {}

    # num_it + 1 elements. Distance between 1/10 and 10 = 100/10 - 1/10 = 99/10, so step size is (99/10)/num_it
    multiply_array = np.arange(1 / 10, 10 + 99 / 10 / num_it, 99 / 10 / num_it)

    # Set up array of area values to compute the derivative of existance condition
    mu_area_ar = np.linspace(area_interval[0], area_interval[1], num_p)

    # Loop over all parameter values for the given parameter (all multiplication factors)
    for p in range(num_it + 1):
        # Change parameter value for sensitivity analysis
        params[param_name] *= multiply_array[p]

        # Compute nondimensional parameters
        par_psi = params['alpha'] * params['p0'] * params['q'] * params['k'] / (params['b'] - params['m'])
        par_delta = params['alpha'] * params['p0'] * params['q'] * params['k'] * params['gamma'] / (
                params['b'] - params['m'])
        par_nu = params['alpha'] * params['c0'] / (params['b'] - params['m'])
        par_sigma = params['alpha'] * params['gamma'] * params['cF'] / (params['b'] - params['m'])

        nondim_list = {'psi': par_psi, 'delta': par_delta, 'nu': par_nu, 'sigma': par_sigma, 'f': params['f'],
                       'par_coeff': params['par_coeff'], 'par_nonlin': params['par_nonlin']}

        # Compute the existence condition
        exist_line = exist_cond(mu_area_ar, nondim_list, dataset)

        # Store results in dictionary along with the parameter value
        exist_dict[params[param_name]] = exist_line

        # Reset parameter value to default
        params[param_name] = dim_list[param_name]

    return exist_dict


def sens_analysis_full(num_it, area_interval, num_p, dim_list, nondim_list, dataset, fontsz, save, fname):
    """
        The plan: for each parameter in dim_list, for num_it iterations, multiply parameter
         by a factor between 1 and 10. Then compute nondim params and compute the slope of exist cond.
        :param num_it: (int) number of different parameter values for sensitivity analysis
        :param area_interval: (list) [area money lower bound, area money upper bound]
        :param num_p: (int) number of points to plot for each axis
        :param dim_list: (dict) parameters for dimensionalised model,
                [b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0]
        :param nondim_list: (dict) parameters for nondimensionalised model,
                [psi, nu, delta, sigma, f, Mmax, mu0]
        :param dataset: (str) "elephant" or "wildebeest". Says which area-cost model to use,
                    depending on the dataset chosen
        :param fontsz: (list): list of font sizes [axes, axtick, legend]
        :param save: (bool) True if save all sensitivity analysis plots, False otherwise
        :param fname: (str) Folder location to save figure

        :return:
    """

    # Define colormap for plotting the lines
    cmaps = plt.cm.magma(np.linspace(0, 1, num_it + 1))
    # Define the array for money invested in area for plotting
    mu_area_array = np.linspace(area_interval[0], area_interval[1], num_p)

    # For each dimensional parameter
    for par in dim_list.keys():
        # Do the sensitivity analysis and save the existence slope values in a dictionary
        sens_par = sens_analysis(par, num_it, [area_interval[0], area_interval[1]], num_p, dim_list, dataset)

        # Extract parameter values from the dictionary keys list
        sens_keys = [ki for ki in sens_par.keys()]

        # Create figure for plotting the slope of existence condition, with dimensions 10x10
        fig = plt.figure(figsize=(10, 10))
        # Add plot in figure, 1st subplot in plot with 1 row & 1 column
        ax = fig.add_subplot(111)
        # Sets the colourmap that the line plots will cycle through. Beginning at black and ending at yellow
        ax.set_prop_cycle('color', list(cmaps))

        # For each parameter value, plot the slope of existence line stored in the dictionary sens_par
        for par_val in sens_par:
            ax.plot(mu_area_array, sens_par[par_val], linewidth=1)

        # Add in default existence slope line
        ax.plot(mu_area_array, exist_cond(mu_area_array, nondim_list, dataset), 'r--', linewidth=2)

        # Add in colourbar. Since plot() makes lines that are not scalar mappable, we need to make them mappable
        sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=sens_keys[0], vmax=sens_keys[-1]))
        fig.colorbar(sm)

        title_text = 'Existence condition for {}, varying values of {}\n Colourbar shows parameter values'.format(
            dataset_name, par)
        ax.set_title(title_text, fontsize=fontsz[0])
        ax.set_xlabel('Current USD invested in area', fontsize=fontsz[0])
        ax.set_ylabel('Current USD invested in rangers', fontsize=fontsz[0])
        ax.axis([area_interval[0], area_interval[1], area_interval[0], area_interval[1]])

        fig.show()

        if save:
            savename = f'{fname}Sensitivity_analysis\\sens_analysis_{dataset}_{par}'
            fig.savefig(savename)

    return


def make_histogram(dim_range, num_it, dataset, fontsz, fname, save):
    """
        Create a histogram of no-poaching/existence condition threshold slope values using
         randomly sampled parameter sets

        :param dim_range: (dict of lists) the dictionary of dimensional parameters and their corresponding range of values
        :param num_it: (int) number of iterations for each parameter
        :param dataset: (str) "elephant" or "kuempel"
        :param fontsz: (list): list of font sizes [axes, axtick, legend]
        :param fname: (str) Folder location to save figure
        :param save: (bool) True if save the histogram plot, False otherwise

        :return: histogram plot object
        """

    slope_vals = generate_slopes(dim_range, num_it, dataset)

    # Create figure for plotting the histogram with dimensions 10x10
    fig = plt.figure(figsize=(10, 10))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    ax = fig.add_subplot(111)

    # Use Freedman-Diaconis estimator for no. of bins
    ax.hist(slope_vals, log=True, rwidth=0.75, bins='fd', label='Observations')
    # Plot vertical lines for the median and mean observed slop value
    ax.vlines(x=np.median(slope_vals), ymin=0, ymax=1000, colors='r', label=f'Median: {np.median(slope_vals):.2}')
    ax.vlines(x=np.mean(slope_vals), ymin=0, ymax=1000, colors='r', linestyles='dashed',
              label=f'Mean: {np.mean(slope_vals):.2}')

    ax.set_xlabel('Proliferation rate of the no-poaching threshold', fontsize=fontsz[0])
    ax.set_ylabel('Number of parameter sets ', fontsize=fontsz[0])
    ax.set_title(f'Histogram of no-poaching slope\n (n={num_it} parameter sets)', fontsize=fontsz[0])
    ax.legend(fontsize=fontsz[2])
    fig.show()

    # How many slopes are less than 1?
    print(f"Number of slopes <1: {sum(np.array(slope_vals) < 1)},"
          f" (i.e. {sum(np.array(slope_vals) < 1) / num_it * 100}% of {num_it} sampled parameter sets.")

    print(f"Median slope value is med = {np.median(slope_vals)}, mean is mu = {np.mean(slope_vals)}")

    print(f'Min slope value is {min(slope_vals)}, max is {max(slope_vals)}.')

    if save:
        savename = f'{fname}slope_histogram_n{num_it}.png'
        fig.savefig(savename)

    return slope_vals


def create_time_series(dim_list, mu_a, mu_r, init_cond, par_attack, par_handle, par_power, tf, interact, tol, solver,
                       max_step, fontsz, fname, save):
    """
    Create investment plot for the complex model by numerically simulating the stable equilibrium and then numerically
    approximating partial derivatives. The larger p derivative indicates if managers should invest in area or rangers.
    :param dim_list: (dict) parameters for dimensional model
    :param mu_a: (float) the current investment size in area for plotting
    :param mu_r: (float) the current investment size in rangers for plotting
    :param init_cond: (list-like) initial conditions for N and E, [N0, E0]
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param tf: (float) numerically solve ODEs to this end time
    :param interact: (list-like) [confiscation interact, fines interact]
        "real" if this model component uses the 'real' linear interact term (par_gamma),
        "perception" if the component uses the complex "perceived" interact (gamma_coeff)
    :param tol: (float>0) error tolerance for ODE solver
    :param solver: (str) name of an ODE solver supported by solve_ivp. E.g. 'LSODA', 'RK45', 'Radau'
    :param fontsz: (list): list of font sizes [axes, axtick, legend]
    :param fname: (str) Folder location to save figure
    :param save: (bool) True if save the histogram plot, False otherwise

    :return: time series object (matplotlib.figure) and corresp. axis object
    """

    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Terminate if ODEs reach equilibrium
    # event.terminal = True
    equil = sci.solve_ivp(fun=complex_model, t_span=[0, tf], y0=init_cond,
                          args=(mu_a, mu_r, par_attack, par_handle, par_power, interact, dim_list, tol),
                          method=solver, max_step=max_step)

    # Plotting the time series for one value of mu_area, mu_ranger
    fig_t = plt.figure(figsize=(12, 12))
    ax_t = fig_t.add_subplot(111)

    t_vals, N_vals, E_vals = equil.t, equil.y[0], equil.y[1]
    # Get final 10% of indices in list of time values for simulation with area investment
    late_time_index = range(int(0.10 * len(t_vals)), len(t_vals))

    # average over final 10% of N values
    late_time_mean = np.mean(N_vals[late_time_index])

    ax_t.plot(t_vals, N_vals, color='k', label='elephant')
    ax_t.plot(t_vals, E_vals, color='b', label='poacher')
    # Add horizontal line for late-time average of elephant population
    ax_t.hlines(y=late_time_mean, xmin=t_vals[0], xmax=t_vals[-1], colors='r', linestyle='-',
                linewidth=2, label=f'late time mean {late_time_mean:.4f}')
    ax_t.legend(fontsize=fontsz[0])
    ax_t.set_title(f'Time series of elephant and poacher population\nmu_a={mu_a:.2e}, mu_r={mu_r:.2e}')
    ax_t.set_xlabel('Time')
    ax_t.set_ylabel('Population')
    fig_t.show()

    # Save the plot if save==True
    if save:
        savename = f'{fname}time_series_a{par_attack}__mu_a{mu_a:.2e}_mu_r{mu_r:.2e}.pdf'
        fig_t.savefig(savename)

    return fig_t, ax_t


def numerical_investment(dim_list, mu_ran_final, mu_area_final, mu_step, par_attack, par_handle, par_power, par_denom, tf,
                         interact, num_p, tol, solver, max_step, cols, mksize, fontsz, fname, save):
    """
    Create investment plot for the complex model by numerically simulating the stable equilibrium and then numerically
    approximating partial derivatives. The larger p derivative indicates if managers should invest in area or rangers.
    :param dim_list: (dict) parameters for dimensional model
    :param mu_ran_final: (float) the largest current investment size in rangers for plotting
    :param mu_area_final: (float) the largest current investment size in area for plotting
    :param mu_step: (float) step size for finite difference derivative calculations
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_handle: (float) scaling coefficient, equivalent to handling time in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param tf: (float) numerically solve ODEs to this end time
    :param interact: (list-like) [confiscation interact, fines interact]
        "real" if this model component uses the 'real' linear interact term (par_gamma),
        "perception" if the component uses the complex "perceived" interact (gamma_coeff)
    :param num_p: (int) number of points to plot for each axis
    :param tol: (float>0) error tolerance for ODE solvers (absolute and relative)
    :param solver: (str) name of an ODE solver supported by solve_ivp. E.g. 'LSODA', 'RK45', 'Radau'
    :param max_step: (float) maximum step size for specified numerical ODE solver
    :param cols: (list of str/hex code) list of colours for markers [invest in rangers, invest in area, no-poaching threshold, original threshold]
    :param mksize: (float) marker size in points^2
    :param fontsz: (list): list of font sizes [axes, axtick, legend]
    :param fname: (str) Folder location to save figure
    :param save: (bool) True if save the histogram plot, False otherwise

    :return: investment plot object (matplotlib.figure) and corresp. axis object
    """
    print('started', flush=True)

    # Create figure for plotting the investment plot, with dimensions 12x12
    fig = plt.figure(figsize=(12, 12))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    ax = fig.add_subplot(111)

    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Create lists of the nondimensional and dimensional parameters for exist_cond()
    nondim_list = {'psi': par_alpha * par_p0 * par_q * par_k / (par_b - par_m),
                   'delta': par_alpha * par_p0 * par_q * par_k * par_gamma / (par_b - par_m),
                   'nu': par_alpha * par_c0 / (par_b - par_m),
                   'sigma': par_alpha * par_gamma * par_cf / (par_b - par_m),
                   'f': par_f, 'par_coeff': par_Mmax, 'par_nonlin': par_mu0}

    # Linspace the points from 10 up to the maximum investments specified
    mu_area_range = np.linspace(start=1, stop=mu_area_final, num=num_p)
    mu_ranger_range = np.linspace(start=1, stop=mu_ran_final, num=num_p)

    count = 0  # Initialise iteration number
    num_it = len(mu_ranger_range) * len(mu_area_range)  # Total number of iterations

    # For each investment value in area and rangers, numerically solve ODE for long time
    # to approximate the stable equilibrium.
    # Create a list of all points corresp to investing in area (one list of x coords, one of y), then plot them in grey.
    # x and y coords of the points where you should invest in area
    plot_point_x = []
    plot_point_y = []
    
    # Create a list of all points corresp to eliminating poaching\no-poaching (one list of x coords, one of y), then plot them in red.
    threshold_x = []
    threshold_y = []

    equil_diff = []  # Initialise list to store differences between equilibrium values for area/ranger investments

    for mu_a in mu_area_range:
        for ii in range(len(mu_ranger_range)):
            mu_r = mu_ranger_range[ii]  # The current value of ranger investment

            # Set initial conditions close to the stable equilibrium for faster convergence.
            # Don't use the exact equilibrium values in case  this equilibrium is not stable (pop can't change if set at equilibrium value).
            # ICs for investment in area and investment in rangers
            ICs_moneyr = analytic_equil(mu_a, mu_r+mu_step, par_attack, par_handle, par_power, par_denom, interact, dim_list)
            ICs_moneya = analytic_equil(mu_a+mu_step, mu_r, par_attack, par_handle, par_power, par_denom, interact, dim_list)

            try:
                # Compute the equilibrium population with an increase in area investment
                equil_more_area = sci.solve_ivp(fun=complex_model, t_span=[0, tf], y0=ICs_moneya,
                                                args=(mu_a + mu_step, mu_r, par_attack, par_handle, par_power, par_denom,
                                                      interact, dim_list, tol),
                                                method=solver, jac=complex_model_jacobian, max_step=max_step)

                # Compute the equilibrium population with an increase in ranger investment
                equil_more_ranger = sci.solve_ivp(fun=complex_model, t_span=[0, tf], y0=ICs_moneyr,
                                                  args=(mu_a, mu_r + mu_step, par_attack, par_handle, par_power, par_denom,
                                                        interact, dim_list, tol),
                                                  method=solver, jac=complex_model_jacobian, max_step=max_step)
            except RuntimeWarning:
                raise RuntimeError(f'Runtime warning at mu_a={mu_a}, mu_r={mu_r}, tf={tf}')

            # Time and population values for area investment equilibrium
            t_vals_a, N_vals_a, P_vals_a = equil_more_area.t, equil_more_area.y[0], equil_more_area.y[1]

            # Time and population values for ranger investment equilibrium
            t_vals_r, N_vals_r, P_vals_r = equil_more_ranger.t, equil_more_ranger.y[0], equil_more_ranger.y[1]

            # Convert population sizes to *total* population sizes by multiplying by amount of area
            N_vals_a *= M_func(par_Mmax, par_mu0, mu_a + mu_step, 'elephant')
            P_vals_a *= M_func(par_Mmax, par_mu0, mu_a + mu_step, 'elephant')
            
            N_vals_r *= M_func(par_Mmax, par_mu0, mu_a, 'elephant')
            P_vals_r *= M_func(par_Mmax, par_mu0, mu_a, 'elephant')

            # Get final 10% of indices in list of time values for simulation with area investment
            late_time_index_a = range(int(0.9 * len(t_vals_a)), len(t_vals_a))
            late_time_mean_a = np.mean(N_vals_a[late_time_index_a])  # average over final 10% of N values

            # Do the same for simulation with ranger investment
            try:
                late_time_index_r = range(int(0.9 * len(t_vals_r)), len(t_vals_r))
                late_time_mean_r = np.mean(N_vals_r[late_time_index_r])
            except IndexError:
                raise IndexError(f'List indexout of range for late_time_mean_r.'
                                 f'\n len(N) = {len(N_vals_r)}, late indices = {late_time_index_r}')

            # Calculate the improvement in equilibrium population (averaged over final 10% of values to account
            # for slow oscillations, since equilibrium is a stable spiral) based on investing in rangers or area.
            # Improvement > 0 if rangers is better, <0 if area is better
            diff = late_time_mean_r - late_time_mean_a

            # only append for values in bottom right quadrant
            equil_diff.append(diff)
            
            # average over final 10% of P values for area investment
            # CHANGE THIS if the threshold looks left-shifted
            if abs(np.mean(P_vals_a[late_time_index_a])) < 1e-4:
                # If poaching is eliminated at this point, add the coordinates (investment values) to threshold_x and threshold_y
                threshold_x.append(mu_a)
                threshold_y.append(mu_r)

            elif diff < 0:
                # If invest in area at this point, add the coordinates (investment values) to area_x and area_y
                plot_point_x.append(mu_a)
                plot_point_y.append(mu_r)

            # Progress bar
            # \r so print returns to previous line, overwrites it
            print(f'\r{count / num_it * 100:.2f}% complete', end='')

            if count == num_it:
                print('Complete\n', end='\r')
            else:
                count += 1
                

        # Finish ranger investment loop
    # Finish area loop

    # Plot the points where you should invest in area. Rasterise to speed up rendering time
    ax.scatter(plot_point_x, plot_point_y, s=mksize, c=cols[1], marker='s', rasterized=True, label='area')
    
    # Plot the points where poaching is eliminated. Rasterise to speed up rendering time
    ax.scatter(threshold_x, threshold_y, s=mksize, c=cols[2], marker='s', rasterized=True, label='no-poaching')

    # Compute existence condition and slope of existence condition
    no_poaching = exist_cond(mu_area_range, nondim_list, 'elephant')
    # Plot the existence condition. Solid line with colour clist[1].
    ax.plot(mu_area_range, no_poaching, color=cols[3], linewidth=2, linestyle='-', label='Original threshold')

    # Add a star for the current investment in area and rangers for Zambia.
    # See Google sheet 'model_parameters.xlsx' for source.
    ax.scatter(x=13389548.33, y=101033450.3, s=200, c='k', marker='*', label='Zambia')

    # Add nice text labels to the plot
    title_text = (
        f'Numerical investment plot\n Zambia elephants with {interact[0]} confiscation and {interact[1]} fines')
    ax.set_title(title_text, fontsize=fontsz[0])
    ax.set_xlabel('Current USD invested in area', fontsize=fontsz[0])
    ax.set_ylabel('Current USD invested in rangers', fontsize=fontsz[0])
    ax.axis([0, mu_area_final, 0, mu_ran_final])
    ax.legend(loc='upper left', fontsize=fontsz[2], facecolor='#f0f0f0')

    # Add in parameter text to the plot
    params_text = (f'C(lambda/n): a={par_attack}, h={par_handle}, z={par_power}, w={par_denom},'
                   f'\nPoints={num_p}x{num_p}, tfinal={tf}, delta_mu={mu_step}'
                   f'\ngrey->invest in area, white->invest in rangers'
                   f'\nSolver={solver}, maxstep={max_step}')
    ax.annotate(params_text, (0.3 * mu_area_final, 0.3 * mu_ran_final), fontsize=12)

    # fig.show()

    # Save the plot if save==True
    if save:  # e.g. path\money_ranger_area_numerical_a1_p1000_2e8_tf1000_real_perceived.png
        savename = f'{fname}numerical_w{par_denom}_z{par_power}_p{num_p}_{mu_area_final:.0e}_tf{tf}_' + '{}_{}.pdf'.format(
            *interact)
        fig.savefig(savename)

    return fig, ax, equil_diff, plot_point_x, plot_point_y


def gamma_plot(ran_density, params, fontsz, fname, save):
    """
    :param ran_density: (list-like) list of ranger densities for plotting
    :param params: (list-like) list of tuples (w, z) for plotting
    :param fontsz: (list): list of font sizes [axes, axtick, legend]
    :param fname: (str) Folder location to save figure
    :param save: (bool) True if save the histogram plot, False otherwise
    :return: figure
    """
    # Plot gamma function against ranger density
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

    nonlinear_lines = []

    # Plot gamma function with varying ranger densities, given w and z. Also plot constant C(lambda/n) = lambda/n
    # use different linetype for each line
    for par, linestyle in zip(params, linestyles):

        par_denom, par_power = par
        # plot_colour = plt.cm.Greys(a_val/10)  # Can cycle colour for each line
        plot_colour = 'grey'

        # By setting num_area=1, ranger_density/num_area = ranger_density, and we can input these into gamma_func
        gamma_line = ax.plot(ran_density, gamma_func(1, ran_density, par_gamma=1, par_attack=1,
                                                       par_handle=1,  par_power=par_power, par_denom=par_denom),
                                color=plot_colour, linewidth=5, linestyle=linestyle, label=f'{par}')[0]

        nonlinear_lines.append(gamma_line)

    line_constant = ax.plot(ran_density, np.ones(len(ran_density)),
                            color='k', linewidth=5, label='constant')[0]  # Constant gamma

    ax.set_title("Poachers' believed catchability by rangers\n against ranger density.",
                 fontsize=fontsz[0])
    ax.set_xlabel('Ranger density, \u03BB/n', fontsize=fontsz[0])
    ax.set_ylabel("Catchability, B(\u03BB/n)", fontsize=fontsz[0])
    ax.legend(handles=[line_constant, *nonlinear_lines], fontsize=fontsz[2], loc='upper right')

    ax.annotate(f'C(x) = x^z/(w^z+x^z),\n(w,z)\u2208{params}', (0.25 * max(ran_density), 0.75),
                fontsize=fontsz[0] / 2)

    fig.show()
    if save:
        fig.savefig(f'{fname}money_ranger_area_gamma_function.pdf')

    return fig, ax


def create_hist_equilibrium(differ, fontsz):
    """
    :param differ: (list-like) list of differences between dX/dmu_r and dX/dmu_a, obtained from numerical_investment()
    :param fontsz: (int) font size for text
    :return: None
    """
    fig_hist = plt.figure(figsize=(12, 12))
    ax_hist = fig_hist.add_subplot(111)

    diffs_cap = [min(abs(x), 500) for x in differ]  # Round any observations above 500 to 500.

    ax_hist.hist(diffs_cap, log=True, rwidth=0.75, bins='fd', label='Observations')
    # Plot vertical lines for the median and mean observed slop value
    ax_hist.vlines(x=np.median(diffs_cap), ymin=0, ymax=1000, colors='r', label=f'Median: {np.median(diffs_cap):.2}')
    ax_hist.vlines(x=np.mean(diffs_cap), ymin=0, ymax=1000, colors='r', linestyles='dashed',
                   label=f'Mean: {np.mean(diffs_cap):.2}')

    ax_hist.legend(fontsize=fontsz)
    ax_hist.set_title('Histogram of difference between dX/dmu_r and dX/dmu_a', fontsize=fontsz)
    ax_hist.set_xlabel('Absolute value of difference', fontsize=fontsz)
    fig_hist.show()

    return fig_hist, ax_hist


def main(dim_list, mu_ran_final, mu_area_final, num_p, dataset, ax, fontsz, fileloc, pname, plot_slope, save_plot):
    """
    Creates and displays the investment plot and investment slope plot
    :param dim_list: (dict) parameters for dimensional model
    :param mu_ran_final: (float) largest current investment size in rangers for plotting
    :param mu_area_final: (float) largest current investment size in area for plotting
    :param num_p: (int) number of points to plot for each axis
    :param dataset: (str) name of the dataset. Should be either 'elephant', 'kuempel', or 'wildebeest'
    :param fileloc: (str) folder location to save plot
    :param ax: (ax object or None) can pass an existing axis to the function, or leave as None if you want
                    to create an axis object in the investment_plot function
    :param fontsz: (list): list of font sizes [axes, axtick, legend]
    :param plot_slope: (bool) True if you want to plot the investment line derivative, False otherwise.
    :param save_plot: (bool) True if you want to save the investment plot, False otherwise.
    :param pname: (str) name for the plot, goes on end of file name when saving

    :return: investment plot object
    """

    # Unpack dimensional parameter values
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Compute nondimensional parameters
    psi = par_alpha * par_p0 * par_q * par_k / (par_b - par_m)
    delta = par_alpha * par_p0 * par_q * par_k * par_gamma / (par_b - par_m)
    nu = par_alpha * par_c0 / (par_b - par_m)
    sigma = par_alpha * par_gamma * par_cf / (par_b - par_m)

    # Create lists of the nondimensional and dimensional parameters
    nondim_list = {'psi': psi, 'delta': delta, 'nu': nu, 'sigma': sigma, 'f': par_f,
                   'par_coeff': par_Mmax, 'par_nonlin': par_mu0}

    # ----------------------------- MAKING THE PLOT -----------------------------

    # For plotting investment paths. Only used if plot_path=True
    # List of tuples, each tuple is separate IC
    invest_IC = [(1000, 100)]

    invest_amount = 10000  # Step size for optimal investment path
    # Number of steps to take when computing the optimal investment path with investment_path()
    invest_steps = int(1e3)

    # Graphical parameters
    plot_colours = ['#bdbdbd', 'k']  # Plot colours [S contour fill, S contour line/exist cond line]

    # Make the investment plot
    fig1, ax1, exist_line = investment_plot([0, mu_ran_final], [1, mu_area_final], num_p,
                                            nondim_list, dim_list, plot_path=False,
                                            invest_init=invest_IC, invest_size=invest_amount,
                                            num_steps=invest_steps, dataset=dataset_name,
                                            lwid=5, clist=plot_colours, fontsz=fontsz, save=save_plot,
                                            fname=fileloc, pname=pname, ax=ax)

    # Make the existence cond. derivative plot
    if plot_slope:
        fig2, ax2 = investment_slope([0, mu_ran_final], [1, mu_area_final], num_p, nondim_list, dim_list,
                                     dataset=dataset, lwid=5, clist=plot_colours,
                                     fontsz=fontsz, save=False, fname=fileloc, pname=pname)

    return fig1, ax1, nondim_list


# ------------- THE MAIN PROGRAM -------------
# File location
filename = 'C:\\Users\\TIMMSLF\\Downloads\\'

# Catch the following warnings as errors, so they work with try/exceptd
wrn.simplefilter('error', UserWarning)
wrn.simplefilter('error', RuntimeWarning)

# Set random seed
rand.seed(42)

# ------------- Model parameters -------------
fontsizes = [28, 28, 22]  # Font sizes for axis labels, axis ticks, legend

alpha = 1e-5  # poacher effort adjustment rate (Source: Holden & Lockyer, 2021)

# Parameters for nonlinear price of land area
cell_size = 706  # Size of area cells in km2. Based on circle of radius 15km  (Source: Hofer, 2000)

dataset_name = 'elephant'
real_or_bubble = 'real'
run_investment = False
run_sensitivity = False
run_histogram = False
run_numerical = False
run_difference = False
run_timeseries = False
run_gamma = True
make_heatmap = False

if dataset_name.lower().startswith('e'):
    # Use elephant parameter set and run the program
    # All money is in 2012 USD
    # ZMK_inflation = 7852.278  # 1 ZMK in 1985 = 7852.278 ZMK in 2012
    # ZMK_to_USD = 0.0002024  # 2012 ZMK t0 2012 USD
    # # Note: 1ZMK in 1985 = ZMK_inflation * ZMK_to_USD USD in 2012

    dataset_name = 'elephant'  # Turn dataset_name into the full word

    total_size = 753000  # Size of Zambia (km2) (Source: UN Statistics Department)
    LVNP_size = 40000
    # Total number of elephants LVNP can support
    k_total = 100000

    b = 0.33  # natural per capita birth rate (Source: Lopes, 2015)
    m = 0.27  # natural per capita death rate (Source: Lopes, 2015)
    k = k_total / LVNP_size * cell_size  # Based on cell size of 706 = 15^2pi km2
    q = 2.56e-3  # catchability (Source: MG&LW, 1992)
    p0 = 3158.76  # max price paid for poached goods (Source: Messer, 2010)

    if real_or_bubble.lower().startswith('r'):
        c0 = 1.911  # opportunity cost (Source: Lopes, 2015)
        cF = 1037.7276  # cost of fines per poacher * prob of conviction (Source: Holden et al., 2018)
        gamma = 0.043  # arrest * conviction probability (Source: Holden et al., 2018)
        f = 1 / 94900  # Lifetime discounted cost, based on $20USD/day from Holden et al. 2018

        # Beverton-Holt area-cost curve parameters - need to be in list for M_val() function
        par_coeff = 1105.811  # M_max parameter. Source: R nls() fit to duplicated real estate data
        par_nonlin = 19339344  # mu0 parameter. Source: as above

        # Max values for mu_rangers and mu_area for the contour plot
        mu_ran_max = 2e9
        mu_area_max = 1e8  # Change x-axis limit


    elif real_or_bubble.lower().startswith('b'):
        # Theoretical parameter values that make a bubble
        c0 = 1.911
        cF = 1037.7276
        gamma = 0.043
        f = 1 / 94900
        par_coeff = 1105.811
        par_nonlin = 193393.44

        # Max values for mu_rangers and mu_area for the contour plot
        mu_ran_max = 2e9
        mu_area_max = 1e8  # Use smaller x-axis limit

    else:
        print("Did not enter a valid choice for 'real' or bubble'")

elif dataset_name.lower().startswith('k'):
    # Use the Kuempel et al. 2018 parameter set and run the program
    dataset_name = 'kuempel'

    # Max values for mu_rangers and mu_area for the contour plot
    mu_ran_max = 1e4
    mu_area_max = 1e4

    # These parameters result in an S=0 contour
    b = 1
    m = 0.1
    q = 0.7
    p0 = 1
    c0 = 0.018
    gamma = 0.2
    cF = 0.01 / gamma  # Different from parameter value in Kuempel. gamma so cF*gamma gives desired value of 0.01
    k = 1000
    f = 1 / 300

    par_coeff = 0.003  # cells/$
    par_nonlin = 0  # Not used since the area-money line goes through (0,0)

elif dataset_name.lower().startswith('w'):
    # Use wildebeest parameter set and run the program
    # All money is in 2012 USD

    dataset_name = 'wildebeest'

    # Max values for mu_rangers and mu_area for the contour plot
    mu_ran_max = 2e9
    mu_area_max = 2e9

    # # Area-cost power function parameters -> M = a*mu^b
    par_coeff = 0.3327  # Source: R nls() fit to duplicated real estate data
    par_nonlin = 0.2211  # Source: as above

    park_size = 14763  # Size of Serengeti national park in km2
    k_total = 1.21e6  # Total number of wildebeest across Serengeti (Source: Mduma, Sinclair, Hillborn, 1975)

    b = 0.25  # (Source: Mduma et al., 1999) Net natural growth rate
    m = 0.0
    k = k_total / park_size * cell_size
    q = 0.00256  # catchability (Source: Fryxell et al., 2007)
    p0 = 12.8986  # max price paid for poached goods (Source: Ndibalema and Songorwa, 2008)
    c0 = 0.637  # opportunity cost (Source: Holden et al., 2018)
    cF = 19.8744  # cost of fines per poacher * prob of conviction (Source: Holden et al., 2018)
    gamma = 0.0007  # arrest * conviction probability (Source: Holden et al., 2018)

    # f = 1 / (20 * 0.91)  # (Source: Holden et al., 2018)
    f = 1 / 94900  # Lifetime discounted cost, based on $20USD/day from Holden et al. 2018

elif dataset_name.lower().startswith('q'):
    print("Successfully quit")

else:
    # Do not run the program
    print("Dataset name not in 'elephant', 'kuempel', or 'wildebeest'. Weird.")

dim_params = {'b': b, 'm': m, 'k': k, 'q': q, 'alpha': alpha, 'p0': p0, 'gamma': gamma,
              'c0': c0, 'cF': cF, 'f': f, 'par_coeff': par_coeff, 'par_nonlin': par_nonlin}
#
# # ------------- Run the investment plot program -------------
# if run_investment:
#     num_points = 100  # Number of points in each array, used for investment plot (needs to be the same)
#
#     fig_invest, ax_invest, nondim_params = main(dim_list=dim_params, mu_ran_final=mu_ran_max, mu_area_final=mu_area_max,
#                                                 num_p=num_points, dataset=dataset_name, ax=None, fontsz=fontsizes,
#                                                 fileloc=filename,
#                                                 pname=f'money_mu{mu_ran_max:.2}_{dataset_name}_usd_test.pdf',
#                                                 plot_slope=False, save_plot=True)
#
# # ------------- Run the sensitivity analysis -------------
# if run_sensitivity:
#     num_sim = 100
#     sens_analysis_full(num_sim, [1, mu_area_max], num_points, dim_params, nondim_params,
#                        dataset_name, fontsizes, save=False, fname=filename)
#
#     # Check what p. derivative values are in the bubble region
#     area_val = 0.3e6
#     ranger_val = 1e8
#     derivs = p_deriv(ranger_val, area_val, nondim_params, 'elephant')
#     cells = M_func(par_coeff, par_nonlin, area_val, 'elephant')
#     rangers = lambda_func(f, ranger_val)
#     # elephants = prop. of carry cap * carry cap/cell * cells
#     print(f"Change in elephants = {[i * k * cells for i in derivs]}")
#     print(f"ranger/cell = {rangers / cells}")
#
# if run_histogram:
#     # Histogram of the existence condition slope values
#     dim_param_range = {'b': [0.33, 0.33],
#                        'm': [0.27, 0.27],
#                        'k': [1000, 100000],
#                        'q': [0.00256, 0.51],
#                        'alpha': [1e-5, 1e-5],
#                        'p0': [200, 5000],  # Need p0>200 and k> 1000 so that p0*q*k > c0 for all parameter combinations.
#                        'gamma': [0.02, 0.5],  # otherwise, there's no poaching and the model doesn't make sense
#                        'c0': [5, 5000],
#                        'cF': [10, 10000],
#                        'f': [1 / 1067683.571, 1 / 94900],
#                        'par_coeff': [2.31e-09, 5e-03],
#                        'par_nonlin': [1, 1]  # Approx. Mmax/mu0 with constant price of area
#                        }
#
#     slope_dict = make_histogram(dim_param_range, num_it=50000, dataset='elephant',
#                                 fontsz=[28, 28, 20], fname=filename, save=False)
#
# # ------------- Numerical investment plot with complex catchability -------------
# num_points = 100  # 1000
# # The change in investment (mu_step) should be big enough that it steps into the next "chunk"/discretised section
# # of the investment plot. Size of each chunk (on area axis) = mu_area_max/num_points.
# # Make mu_step = 1/2 of this chunk size
# delta_mu = mu_area_max / num_points / 2
# attack_rate = 1
# handle_time = 1
# # power_z = 2
# tfinal = 100000
# interaction = ['real', 'perceived']
# tolerance = 1e-5
# the_solver = 'LSODA'
# the_step = 1



# if run_numerical:
#     for power_z in [1, 3, 10]:
#         for param_w in [1, 2, 5, 10]:

#             # Marker size to cover a d x d plot (in points) with a marker width of sqrt(s), given n markers, is s = (d/n)^2
#             fig_numeric, ax_numeric, differences, area_x, area_y = numerical_investment(dim_list=dim_params,
#                                                                                         mu_ran_final=mu_ran_max,
#                                                                                         mu_area_final=mu_area_max,
#                                                                                         mu_step=delta_mu,
#                                                                                         par_attack=attack_rate,
#                                                                                         par_handle=handle_time,
#                                                                                         par_power=power_z,
#                                                                                         par_denom=param_w,
#                                                                                         tf=tfinal,
#                                                                                         interact=interaction,
#                                                                                         num_p=num_points,
#                                                                                         tol=tolerance,
#                                                                                         solver=the_solver,
#                                                                                         max_step=the_step,
#                                                                                         cols=['#ffffff', '#bdbdbd', 'r', 'k'],
#                                                                                         # white, grey, and black
#                                                                                         mksize=(12 * 72 / num_points) ** 2,
#                                                                                         fontsz=fontsizes,
#                                                                                         fname=filename,
#                                                                                         save=True)

#     # winsound.PlaySound('SystemHand', winsound.SND_ALIAS)

#     # Make a histogram of the differences between equilibria?
#     if run_difference:
#         create_hist_equilibrium(differences, fontsz=20)

# if run_timeseries:
#     # Original initial conditions
#     test_val = (0.2 * mu_area_max, 0 * mu_ran_max)

#     ICs = analytic_equil(test_val[0], test_val[1], attack_rate,
#                          power_k, interaction_list[0], dim_params)

#     fig_time, ax_time = create_time_series(dim_params,  # mu_r=mu_r_val, mu_a=mu_a_val,
#                                            mu_a=test_val[0], mu_r=test_val[1],
#                                            # Start near but not on the stable equilibrium
#                                            init_cond=[0.99 * x for x in ICs], par_attack=attack_rate, par_power=power_k,
#                                            tf=tfinal, interact=interaction_list[0], tol=tolerance, solver=the_solver,
#                                            max_step=the_step, fontsz=fontsizes, fname=filename, save=False)

# # ------------- Plot the gamma functions -------------
if run_gamma:
    ranger_density = np.arange(start=0, stop=15, step=0.01)

    fig_gamma, ax_gamma = gamma_plot(ranger_density, params=[(1, 1), (1,10), (10,1), (10,10)],fontsz=fontsizes, fname=filename,
                                     save=True)


# if make_heatmap == True:
#     fig_heat = plt.figure(figsize=(12, 12))
#     ax_heat = fig_heat.add_subplot(111)

#     array_a = np.arange(1000, mu_area_max+1, mu_area_max/1000)
#     array_r = np.arange(1000, mu_ran_max+1, mu_ran_max/1000)

#     ran_density = np.zeros((np.size(array_r), np.size(array_a)))

#     for jj in range(len(array_r)):
#         for kk in range(len(array_a)):
#             ran_density[jj][kk] = lambda_func(f, array_r[jj])/M_func(par_coeff, par_nonlin, array_a[kk], 'elephant')


#     im = ax_heat.imshow(ran_density, cmap="YlGn", vmax=50, origin='lower')
#     cbar = ax_heat.figure.colorbar(im, ax=ax_heat)
    
#     xtick_loc, ytick_loc = np.arange(0,len(array_a)+1, 200), np.arange(0,len(array_r)+1, 200)
#     xtick_lab, ytick_lab = np.arange(1000, mu_area_max+1, mu_area_max/len(xtick_loc)), np.arange(1000, mu_ran_max+1, mu_ran_max/len(ytick_loc))
    
#     # ax_heat.set_xticks(np.linspace(1000, mu_area_max, 10))
#     # ax_heat.set_yticks(np.linspace(1000, mu_ran_max, 10))
#     ax_heat.set_xticks(xtick_loc, labels=[f'{y:.2e}' for y in xtick_lab])
#     ax_heat.set_yticks(ytick_loc, labels=[f'{y:.2e}' for y in ytick_lab])
#     ax_heat.set_xlabel('$ in area (array element)')
#     ax_heat.set_ylabel('$ in rangers')
#     # fig_heat.show()
    
#     savename = f'{filename}ranger_density.pdf'
#     fig_heat.savefig(savename)
    
"""
Author: Liam Timms (UQ 44368768/uqltimm1)
Created: 11/09/2023
Program just to do numerical optimisation of the complex ODE system w.r.t. the proportion of money
invested in rangers.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.integrate as sci
from scipy.optimize import minimize_scalar, shgo
import random as rand
import warnings as wrn
import time as time


def M_func(par_a, par_b, money_array):
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
    # If dataset is elephant, the use Beverton-Holt model (fitted in R) for price of area
    M_val = par_a * money_array / (par_b + money_array)

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


def gamma_func(num_area, num_rangers, par_gamma, par_attack, par_power):
    """
    Computes perceived catchability of poachers by rangers as a function of current ranger density.
    C1(x) = gamma*a*x**k/(1 + ax**k), where x is ranger density. Slope is 0 at x=0, and asymptotes at gamma as x->infinity.
    Contrast this with law of mass-action, which has C2(x) = gamma. C1 <= C2 for all x.
    Return min(C1(x), 1) so that the catchability is bounded by 1.
    :param num_area: (float) current number of area cells
    :param num_rangers: (float) current number of rangers hired in total
    :param par_gamma: (float) catchability coefficient gamma from linear functional response/law of mass-action
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :return: (float)
    """

    # Compute density**k, where k is given in par_power
    density_pow_k = np.power(num_rangers / num_area, par_power)

    perceived_gamma = par_gamma * par_attack * density_pow_k / (1 + par_attack * density_pow_k)

    # Note that this function for gamma is bounded by 0 and 1, so it can represent a proportion.
    return perceived_gamma


def equilibrium(mu_a, mu_r, dim_list):
    """
    Compute the analytic equilibrium for the simple model (total population over area)
    :param mu_a: (float) current money invested in area
    :param mu_r: (float) current money invested in area
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}

    :return (array) n * Nstar(lambda, n) the equilibrium population across the entire protected area
    """

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    num_area = M_func(par_Mmax, par_mu0, mu_a)
    num_rangers = lambda_func(par_f, mu_r)

    # # Original equation
    # equil_pop = num_area * (par_c0 + par_cf * par_gamma * density) / (par_p0 * par_q * (1 - par_gamma * density))

    # Equivalent equation that is defined for n=0
    equil_pop = (num_area ** 2 * par_c0 + par_cf * par_gamma * num_rangers * num_area) / (
            par_p0 * par_q * (num_area - par_gamma * num_rangers))

    return equil_pop


# TESTING ONLY
def exist_cond(money_area, dim_list):
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

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    par_psi = dim_list['alpha'] * dim_list['p0'] * dim_list['q'] * dim_list['k'] / (dim_list['b'] - dim_list['m'])
    par_delta = dim_list['alpha'] * dim_list['p0'] * dim_list['q'] * dim_list['k'] * dim_list['gamma'] / (
            dim_list['b'] - dim_list['m'])
    par_nu = dim_list['alpha'] * dim_list['c0'] / (dim_list['b'] - dim_list['m'])
    par_sigma = dim_list['alpha'] * dim_list['gamma'] * dim_list['cF'] / (dim_list['b'] - dim_list['m'])

    # mu_area as a function of mu_rangers, solution to beta * M(mu_area) = lambda(mu_ranger)
    C = (par_psi - par_nu) / (par_sigma + par_delta)  # Constant based on parameters

    # mu_rangers as function of mu_area
    exist_contour = C / par_f * M_func(par_Mmax, par_mu0, money_area)

    return exist_contour


def proportion_from_curve(mu_a, mu_r, dim_list):
    """
    Compute the analytic optimal proportion for a given total amount of money invested for the simple model,
    derived from the no-poaching threshold
    :param mu_a: (array-like) current money invested in area
    :param mu_r: (array-like) current money invested in area
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}

    :return (array-like) optimal proportion of money to invest in rangers for given total investment
    """

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    factor1 = 1 / par_f * (par_p0 * par_q * par_k - par_c0) / (par_gamma * (par_cf + par_p0 * par_q * par_k))

    factor2 = par_Mmax * mu_a / (par_mu0 + mu_a) / (mu_a + mu_r)

    return factor1 * factor2


def proportion_analytic(mu, dim_list):
    """
    Compute the analytic optimal proportion for a given total amount of money invested for the simple model,
    derived from maximising the equilibrium population as a function of money invested
    :param mu: (array-like) total money invested, mu = mu_a + mu_r
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}

    :return (array-like) optimal proportion of money to invest in rangers for given total investment
    """

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    A = mu * (par_c0 * par_Mmax + par_cf * par_f * par_gamma * par_mu0)
    B = par_c0 * par_Mmax * (-3 * mu + par_mu0) - par_cf * par_f * par_gamma * par_mu0 * (mu + par_mu0)
    C = 3 * par_c0 * par_Mmax * mu
    D = -par_c0 * par_Mmax * (mu + par_mu0)

    # TRY NUMEIRCALLY SOLVING THE RHO CUBIC
    # ALSO WE NEED THE DENOMINATOR OF THE dN/drho TO BE NONZERO

    cubic = poly.Polynomial([D, C, B, A], domain=[-1e7, 1e7])

    return poly.polyroots(cubic)


def simple_model(t, y, money_area, money_ranger, par_attack, par_power, dim_list, model):
    """
    :param t: time. used for scipy.solve_ivp
    :param z: (float or array) state variables N, E
    :param money_area: (float) current money invested in area
    :param money_ranger: (float) current money invested in area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param model: 'simple' or 'complex'
    :return: array
    """

    N, E = y

    # Number of rangers
    num_r = lambda_func(dim_list['f'], money_ranger)
    num_a = M_func(dim_list['par_coeff'], dim_list['par_nonlin'], money_area)

    density = num_r / num_a

    dNdt = (dim_list['b'] - dim_list['m']) * (1 - N / dim_list['k']) * N - dim_list['q'] * N * E
    dEdt = dim_list['alpha'] * (1 - dim_list['gamma'] * density) * dim_list['p0'] * dim_list['q'] * N * E \
           - dim_list['alpha'] * (dim_list['c0'] + dim_list['cF'] * dim_list['gamma'] * density) * E

    return [dNdt, dEdt]


def complex_model(t, y, money_area, money_ranger, par_attack, par_power, dim_list, model):
    """
    :param t: time. used for scipy.solve_ivp
    :param y: (float or array) state variables N, E
    :param money_area: (float) current money invested in area
    :param money_ranger: (float) current money invested in area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param model: 'simple' or 'complex'
    :return: (array) value of time derivative dNdt and dEdt
    """
    # Unpack state variables for num elephants N and poaching effort E
    N, E = y

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Calculate current number of rangers hired and amount of area as functions of money invested
    num_rangers = lambda_func(par_f, money_ranger)
    num_area = M_func(par_Mmax, par_mu0, money_area)

    # Compute the density of rangers over the protected area
    density = num_rangers / num_area

    # Compute poachers' perceived catchability by rangers, which factors into expected cost of poaching.
    gamma_coeff = gamma_func(num_area, num_rangers, par_gamma, par_attack, par_power)

    Ndot = ((par_b - par_m) * (1 - N / par_k) - par_q * E) * N
    Edot = par_alpha * E * ((1 - par_gamma * density) * par_p0 * par_q * N - par_c0 - gamma_coeff * par_cf * density)

    return [Ndot, Edot]


def model_jacobian(t, y, money_area, money_ranger, par_attack, par_power, dim_list, model):
    """
    Same arguments as complex_model(). Necessary for passing arguments to solve_ivp()
    :param t:
    :param y:
    :param money_area:
    :param money_ranger:
    :param par_attack:
    :param par_power:
    :param dim_list:
    :param model: 'simple' or 'complex'
    :return:
    """
    # Unpack state variables for num elephants N and poaching effort E
    N, E = y

    # Unpack dimensional parameters
    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    # Calculate current number of rangers hired and amount of area as functions of money invested
    num_rangers = lambda_func(par_f, money_ranger)
    num_area = M_func(par_Mmax, par_nonlin, money_area)

    # Compute the density of rangers over the protected area
    density = num_rangers / num_area
    if model == 'simple':
        gamma_coeff = par_gamma
    elif model == 'complex':
        gamma_coeff = gamma_func(num_area, num_rangers, par_gamma, par_attack, par_power)
    else:
        raise ValueError("Oops something's gone wrong in model_jacobian")

    # Compute coefficients for use in Jacobian
    C = par_alpha * (1 - gamma_coeff * density) * par_p0 * par_q
    D = par_alpha * (par_c0 + par_cf * gamma_coeff * density)

    # dF1/dN
    F1_N = (par_b - par_m) - 2 * (par_b - par_m) * N / par_k - par_q * E

    # dF1/dE
    F1_E = - par_q * N

    # dF2/dN
    F2_N = C * E

    # dF2/dE
    F2_E = C * N - D

    # Return the Jacobian
    return np.array([[F1_N, F1_E], [F2_N, F2_E]], dtype=float)


def create_time_series(prop, money, par_attack, par_power, tf, init_cond, dim_list, model, **kwargs):
    """
    :param prop: (float) proportion of money invested in rangers
    :param money: (float) total money invested across rangers and area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param tf: (float) stopping time for ODE numerical integration
    :param init_cond: (list) list of initial conditions [N0, P0]
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param model: 'simple' or 'complex'
    :param kwargs: optional arguments 'solver' and 'max_step' for numerical ODE solver

    :return: (figure, axis) plot of the population time series
    """

    # Create figure for plotting the investment plot, with dimensions 12x12
    fig = plt.figure(figsize=(12, 12))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    ax = fig.add_subplot(111)

    # Default solver for ODE integration. Should be LSODA for complex and Radau for simple
    solver = 'LSODA' if model == 'complex' else 'Radau'
    # Default maximum step size for chosen ODE solver
    max_step = 1000 if model == 'complex' else 1e5

    # If the user specifies an ODE solver and a max. step size, use these
    for key in kwargs:
        if key == 'solver':
            solver = kwargs[key]
        elif key == 'max_step':
            max_step = kwargs[key]
        else:
            raise TypeError(f'{key} is an invalid keyword argument for create_time_series()')

    # Function handle for ODE system to simulate
    func = complex_model if model == 'complex' else simple_model

    mu_a = (1 - prop) * money
    mu_r = prop * money
    # Calculate current total area
    num_area = M_func(dim_list['par_coeff'], dim_list['par_nonlin'], mu_a)

    equil = sci.solve_ivp(fun=func, t_span=[0, tf], y0=init_cond,
                          args=(mu_a, mu_r, par_attack, par_power, dim_list, model),
                          jac=model_jacobian,
                          method=solver, max_step=max_step)

    # Multiply by total area to get total population
    t_vals, N_vals, E_vals = equil.t, num_area * equil.y[0], num_area * equil.y[1]

    # Average over final 25% of N population values
    time_25percent = range(int(0.75 * len(t_vals)), len(t_vals))
    mean_25 = np.mean(N_vals[time_25percent])

    # Average over final 10% of N values
    time_10percent = range(int(0.9 * len(t_vals)), len(t_vals))
    mean_10 = np.mean(N_vals[time_10percent])

    ax.plot(t_vals, N_vals, 'k', label='Elephants')
    ax.plot(t_vals, E_vals, 'r', label='Poachers')
    ax.hlines(mean_25, t_vals[0], t_vals[-1], '#d95f02', linestyles='dashed', label=f"25% mean={mean_25:.1f}")
    ax.hlines(mean_10, t_vals[0], t_vals[-1], '#1b9e77', linestyles='dashed', label=f"10% mean={mean_10:.1f}")

    if model == 'simple':
        # If this is the simple model, plot the analytic equilibrium as a horizontal line
        analytic_equil = equilibrium(mu_a, mu_r, dim_list)
        ax.hlines(analytic_equil, t_vals[0], t_vals[-1], '#7570b3', linestyles='dashed',
                  label=f"Equil.={analytic_equil:.1f}")

        print(f'% difference between 10% late-time mean and analytic equilibrium ='
              f' {abs(mean_10 - analytic_equil) / analytic_equil * 100:.2f}%')
        ax.set_ylim([0, 5 * analytic_equil])

    # If this is the complex model, use long-time mean as the y-axis limit
    ax.set_ylim([0, 10 * mean_10])

    ax.set_title(f'Time series of total population {"simple" if model == "simple" else "complex"} model'
                 f'\n with prop. {prop}, attack rate {par_attack}, tf={tf}, solver={solver}'
                 f'\nmax_step={max_step}')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Total population across PA, N(t) and E(t)')
    ax.legend()
    fig.show()

    return fig, ax, t_vals, N_vals, E_vals


def objective(prop, *args):
    """
    Create investment plot for the complex model by numerically simulating the stable equilibrium and then numerically
    approximating partial derivatives. The larger p derivative indicates if managers should invest in area or rangers.
    :param prop: (float) objective parameter in [0,1]
    :param args: see below
    :param model: (str) 'simple' or 'complex' for which ODE system the user wants to simulate
    :param mu_r: (float) the current investment size in rangers for plotting
    :param mu_a: (float) the current investment size in area for plotting
    :param init_cond: (list-like) initial conditions for N and E, [N0, E0]
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param tf: (float) numerically solve ODEs to this end time
    :param dim_list: (dict) parameters for dimensional model
    :param kwargs: optional arguments 'solver' and 'max_step' for numerical ODE solver

    :return: long-time average population (float)
    """

    model, money, init_cond, par_attack, par_power, tf, solver, max_step, dim_list = args

    # TESTING ONLY
    # solver = 'LSODA' if model == 'complex' else 'Radau'
    # max_step = 1000 if model == 'complex' else 1e5
    # # If the user specifies an ODE solver and a max. step size, use these
    # for key in kwargs:
    #     if key == 'solver':
    #         solver = kwargs[key]
    #     elif key == 'max_step':
    #         max_step = kwargs[key]
    #     else:
    #         raise TypeError(f'{key} is an invalid keyword argument for objective()')

    mu_a = (1 - prop) * money
    mu_r = prop * money

    num_area = M_func(dim_list['par_coeff'], dim_list['par_nonlin'], mu_a)

    # If the user specifies complex_model, use LSODA solver with a max_step of 1000 so we can
    # simulate it for long time with stability. For simple_model, use Radu and and max_step is fine.
    func = complex_model if model == 'complex' else simple_model

    equil = sci.solve_ivp(fun=func, t_span=[0, tf], y0=init_cond,
                          args=(mu_a, mu_r, par_attack, par_power, dim_list, model),
                          # jac=model_jacobian,
                          method=solver, max_step=max_step)

    t_vals, N_vals, E_vals = equil.t, equil.y[0], equil.y[1]
    # Average over final 10% of N values
    time_10percent = range(int(0.9 * len(t_vals)), len(t_vals))
    mean_10 = np.mean(N_vals[time_10percent])

    return -mean_10 * num_area


def create_population_plot(money, par_attack, par_power, tf, init_cond, num_it, dim_list, model, **kwargs):
    """
    Plot the equilibrium population size as a function of proportion of money invested in rangers
    :param money: (float) total money invested across rangers and area
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param tf: (float) stopping time for ODE numerical integration
    :param init_cond: (list) list of initial conditions [N0, P0]
    :param num_it: (int) number of proportions in [0,1] to simulate
    :param dim_list: (dict) parameters for dimensionalised model,{b, m, k, q, alpha, p0, a, gamma, c0, cf, f, Mmax, mu0}
    :param model: 'simple' or 'complex'
    :param kwargs: optional arguments 'solver' and 'max_step' for numerical ODE solver

    :return: (figure, axis) plot of the equilibrium population against proportion
    """

    # Default solver for ODE integration. Should be LSODA for complex and Radau for simple
    solver = 'LSODA' if model == 'complex' else 'Radau'
    # Default maximum step size for chosen ODE solver
    max_step = 1000 if model == 'complex' else 1e5

    # If the user specifies an ODE solver and a max. step size, use these
    for key in kwargs:
        if key == 'solver':
            solver = kwargs[key]
        elif key == 'max_step':
            max_step = kwargs[key]
        else:
            raise TypeError(f'{key} is an invalid keyword argument for create_population_plot()')

    epsilon = 1e-3  # Small number since 1 is a bad value
    # Create an array of the proportions between 0 and 1, and initialise array for the population values
    # Have half as many points in [0, 0.2] as in [0.2, 0.4] and [0.4,0.6], etc. -> N/9 points in [0, 0.2]

    pvals_low = np.linspace(0, 0.2, int(np.ceil(num_it / 9)))
    pvals_high = np.linspace(0.2, 1 - epsilon, int(np.floor(8 * num_it / 9)))
    pvals = np.append(pvals_low, pvals_high)
    yvals = np.zeros(np.size(pvals))

    # Sped twice as long simulating population for low proportions for simple model, and 5 times for complex model,
    # since small proportions make the populations do really slow oscillations (O(1e6) time between spikes)
    extra_time = 2*tf if model.startswith('s') else 5*tf

    for ii in range(len(pvals)):
        # For each proportion, compute the long-time population value

        if pvals[ii] <= 0.4:
            # Use coarser grid for rho<0.2 but spend twice as long simulating the equilibrium.
            # The late-time population vs rho is roughly linear for low rho in the simple model
            # Note: objective returns the negative population for minimisation
            yvals[ii] = -objective(pvals[ii], model, money, init_cond,
                                   par_attack, par_power, extra_time, solver, max_step, dim_list)
        else:
            # Note: objective returns the negative population for minimisation
            yvals[ii] = -objective(pvals[ii], model, money, init_cond,
                                   par_attack, par_power, tf, solver, max_step, dim_list)

        # Print progress counter in steps of 0.1%
        print(f'{ii / num_it * 100:.1f}% complete', end='\r')
        if ii == num_points:
            print('Complete\n', end='\r')

    # Create figure for plotting the investment plot, with dimensions 12x12
    fig = plt.figure(figsize=(12, 12))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    ax = fig.add_subplot(111)

    if model == 'simple':
        ax.plot(pvals, equilibrium((1 - pvals) * money, pvals * money, dim_list), 'b--', label='Analytic equil.')

    ax.plot(pvals, yvals, 'k', label='Equil. pop.')
    ax.set_title('Late-time pop. average against proportion of money invested in rangers'
                 f'\nattack={attack_rate}, tf={tfinal}, money={money}, {the_model} model'
                 f'\nsolver={solver}, maxstep={max_step}')
    ax.set_xlabel('Proportion p')
    ax.set_ylabel('Late-time population size')
    # ax.set_ylim([0, max(yvals)])
    ax.legend()
    fig.show()
    # Return list of p values where solver didn't converge (ie final pop is negative)
    noncoverge = [pvals[ii] for ii in range(len(pvals)) if yvals[ii] < 0]
    return fig, ax, noncoverge


def optimal_proportions(model, max_money, init_cond, num_p, par_attack, par_power, tf, dim_list, **kwargs):
    """
    Create an array of optimal proportions, given an array of investment values and a number of points
    :param model: (str) 'simple' or 'complex' for which ODE system the user wants to simulate
    :param max_money: (float) the current investment size in rangers for plotting
    :param init_cond: (list-like) initial conditions for N and E, [N0, E0]
    :param num_p: (int) number of points to compute
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param tf: (float) numerically solve ODEs to this end time
    :param dim_list: (dict) parameters for dimensional model
    :param kwargs: optional arguments 'solver' and 'max_step' for numerical ODE solver

    :return: array of money values, array of optimal proportion values
    """

    # Default solver for ODE integration. Should be LSODA for complex and Radau for simple
    solver = 'LSODA' if model == 'complex' else 'Radau'
    # Default maximum step size for chosen ODE solver
    max_step = 1000 if model == 'complex' else 1e5

    # If the user specifies an ODE solver and a max. step size, use these
    for key in kwargs:
        if key == 'solver':
            solver = kwargs[key]
        elif key == 'max_step':
            max_step = kwargs[key]
        else:
            raise TypeError(f'{key} is an invalid keyword argument for optimal_proportion()')

    # Array of total money values
    money_array = np.linspace(1, max_money, num_p)

    # Initialise array for optimal proportion values
    pstar = np.zeros(np.size(money_array))

    times = np.zeros(np.size(money_array))

    for ii in range(len(money_array)):
        it_start = time.perf_counter()  # Start the iteration timer
        # TESTING ONLY
        epsilon = 1e-3  # Small number to subtract from p=1 since 1 is a bad value
        # Use a lambda function here to specify the extra arguments of objective() since shgo() currently has a bug
        # sol = shgo(
        #     func=lambda p: objective(p, model, money_array[ii], init_cond, par_attack, par_power, tf, dim_list, **kwargs),
        #     bounds=[(0, 1 - epsilon)],
        #     n=100,  # Number of sampling points
        #     sampling_method='halton'
        # )

        sol = minimize_scalar(fun=objective,
                              bounds=[0, 1 - epsilon],
                              method='bounded',
                              args=(
                                  model, money_array[ii], init_cond, par_attack, par_power, tf, solver, max_step,
                                  dim_list),
                              options=dict(disp=False)
                              )

        times[ii] = time.perf_counter() - it_start  # Time taken for 1 iteration

        # Store the optimal proportion in pstar
        pstar[ii] = sol.x

        print(f'{ii / num_p * 100:.1f}% complete', end='\r')
        if ii == num_points:
            print('Complete\n', end='\r')

    return money_array, pstar, times


def optimal_proportion_plot(model, init_cond, num_p, par_attack, par_power, tf, dim_list, **kwargs):
    """
    Create an array of optimal proportions, given an array of investment values and a number of points
    :param model: (str) 'simple' or 'complex' for which ODE system the user wants to simulate
    :param init_cond: (list-like) initial conditions for N and E, [N0, E0]
    :param num_p: (int) number of points to compute
    :param par_attack: (float) scaling coefficient, equivalent to attack rate in Holling type III functional response
    :param par_power: (float) raise density to the power of par_power
    :param tf: (float) numerically solve ODEs to this end time
    :param dim_list: (dict) parameters for dimensional model
    :param kwargs: optional arguments 'solver' and 'max_step' for numerical ODE solver

    :return: figure, axis, array of optimal proportion values, time taken (s) for each iteration of optimisation
    """

    # Default solver for ODE integration. Good ones are LSODA for complex and Radau for simple
    solver = 'LSODA' if model == 'complex' else 'Radau'
    # Default maximum step size for chosen ODE solver
    max_step = 1000 if model == 'complex' else 1e5

    lwd = 1
    fontsz = 14

    # If the user specifies an ODE solver and a max. step size, use these
    for key in kwargs:
        if key == 'solver':
            solver = kwargs[key]
        elif key == 'max_step':
            max_step = kwargs[key]
        elif key == 'lwd':  # Plot line width
            lwd = kwargs[key]
        elif key == 'fontsz':
            fontsz = kwargs[key]  # Plot font size
        else:
            raise TypeError(f'{key} is an invalid keyword argument for optimal_proportion()')

    # Generate total money invested from original plot
    mua_array = np.arange(1, 2e8, 10)
    mur_array = exist_cond(mua_array, dim_list)
    money_max = max(mua_array + mur_array)

    # Create figure for plotting the investment plot, with dimensions 12x12
    fig = plt.figure(figsize=(12, 12))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    ax = fig.add_subplot(111)

    if model.startswith('s'):
        # For the simple model, plot the optimal proportions calculated from no-poaching threshold in the original simple investment plot
        pstar_simple = mur_array / (
                mua_array + mur_array)  # Taken straight from numerical output of the no-poaching threshold
        ax.plot(mua_array + mur_array, pstar_simple, linestyle='solid', color='#d95f02', label='y/(x+y)',
                linewidth=lwd / 2)

    # Plot the optimal proportion against total money for both simple and complex models
    for mod in ["complex", "simple"]:

        # Optimise the proportion numerically from the numerical pop equilibrium
        mu_array, p_array, iter_time = optimal_proportions(mod, money_max, init_cond, num_p,
                                                           par_attack, par_power, tf, dim_list, solver=solver,
                                                           max_step=max_step)
        ltype = 'dashed' if mod.startswith('s') else 'solid'
        lcolour = '#d95f02' if mod.startswith('s') else 'k'

        ax.plot(mu_array, p_array, linestyle=ltype, color=lcolour, label=mod, linewidth=lwd)

    ax.set_title(f'Optimal proportion of money invested in rangers, vs. total money invested'
                 f'\nModel: {model}, num_p={num_p}, attack={par_attack}, tf={tf}'
                 f'\nmaxstep={max_step}, solver={solver}', fontsize=fontsz)
    ax.set_xlabel('Total money invested', fontsize=fontsz)
    ax.set_ylabel('Optimal proportion p*', fontsize=fontsz)
    ax.axis([0, money_max, 0, 1])
    ax.tick_params(labelsize=fontsz)
    ax.legend(fontsize=fontsz)
    fig.show()

    return fig, ax, p_array, iter_time


rand.seed(42)

# Catch the following warnings as errors, so they work with try/exceptd
wrn.simplefilter('error', UserWarning)
wrn.simplefilter('error', RuntimeWarning)

# ------------------------------------------
# PARAMETERS - population model
cell_size = 706  # Size of area cells in km2. Based on circle of radius 15km  (Source: Hofer, 2000)
total_size = 753000  # Size of Zambia (km2) (Source: UN Statistics Department)
LVNP_size = 40000
# Total number of elephants LVNP can support
k_total = 100000

alpha = 1e-5
b = 0.33  # natural per capita birth rate (Source: Lopes, 2015)
m = 0.27  # natural per capita death rate (Source: Lopes, 2015)
k = k_total / LVNP_size * cell_size  # Based on cell size of 706 = 15**2pi km2
q = 2.56e-3  # catchability (Source: MG&LW, 1992)
p0 = 3158.76  # max price paid for poached goods (Source: Messer, 2010)

c0 = 1.911  # opportunity cost (Source: Lopes, 2015)
cF = 1037.7276  # cost of fines per poacher * prob of conviction (Source: Holden et al., 2018)
gamma = 0.043  # arrest * conviction probability (Source: Holden et al., 2018)
f = 1 / 94900  # Lifetime discounted cost, based on $20USD/day from Holden et al. 2018

# Beverton-Holt area-cost curve parameters - need to be in list for M_val() function
par_coeff = 1105.811  # M_max parameter. Source: R nls() fit to duplicated real estate data
par_nonlin = 19339344  # mu0 parameter. Source: as above

dim_params = {'b': b, 'm': m, 'k': k, 'q': q, 'alpha': alpha, 'p0': p0, 'gamma': gamma,
              'c0': c0, 'cF': cF, 'f': f, 'par_coeff': par_coeff, 'par_nonlin': par_nonlin}

# PARAMETERS - complex model
attack_rate = 1  # 1e8
power_k = 2
tfinal = 100000
total_money = 2e8
proportion = 0.4
ICs = [k / 2, 2]
the_model = 'complex'
num_points = 1000
the_step = 1

# ------------------------------------------
# BLACK BOX OPTIMISATION

if input('Create population time series (y/n)').lower().startswith('y'):
    # Plot time series for random proportion value
    fig_time, ax_time, t, x, y = create_time_series(proportion, total_money,
                                                    attack_rate, power_k, tfinal, ICs, dim_params, the_model,
                                                    solver='LSODA', max_step=the_step)

# ------------------------------------------
if input('Plot equilibrium pop. against proportion (y/n)').lower().startswith('y'):
    fig_pop, ax_pop, bad_ps = create_population_plot(total_money, attack_rate, power_k, tfinal, ICs,
                                                     num_it=100, dim_list=dim_params, model=the_model, solver='LSODA',
                                                     max_step=the_step)

    # Find the argmax of the population-proportion results and plot a red line
    start = time.perf_counter()
    result = minimize_scalar(fun=objective,
                             bounds=[0, 1],
                             method='bounded',
                             args=(
                                 the_model, total_money, ICs, attack_rate, power_k, tfinal, 'LSODA', the_step,
                                 dim_params),
                             options=dict(disp=True)
                             )
    finish = time.perf_counter() - start

    ax_pop.vlines(result.x, 0, ax_pop.get_ylim()[1], 'r', label='Argmax of pop.')
    ax_pop.legend()
    fig_pop.show()

# ------------------------------------------
if input('Plot optimal proportion against money (y/n)').lower().startswith('y'):
    fig_opt, ax_opt, p_opt, times = optimal_proportion_plot(the_model, ICs, num_p=20,
                                                            par_attack=attack_rate, par_power=power_k,
                                                            tf=tfinal, dim_list=dim_params,
                                                            solver='LSODA', max_step=the_step, lwd=5, fontsz=20)

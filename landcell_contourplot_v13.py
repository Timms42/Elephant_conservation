"""
Author: Liam Timms 44368768
Created: 18/05/2022
Modified: 01/11/2022
Lockyer model redux
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
"""

import matplotlib.lines as mlines
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import warnings as wrn


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

    if dataset == ('elephant' or 'kuempel'):
        # If dataset is elephant, the use Beverton-Holt model (fitted in R) for price of area
        M_val = par_a * money_array / (par_b + money_array)

    elif dataset == 'wildebeest':
        # If dataset is wildebeest, the use piecewise linear model (fitted in R) for price of area
        M_val = par_a * money_array ** par_b

        # # Piecewise function (variable, [conditions], [values])
        # M_val = np.piecewise(money_array,
        #                      [(0 <= money_array) & (money_array < par_b[0]),
        #                       (par_b[0] <= money_array) & (money_array < par_b[1]),
        #                       (par_b[1] <= money_array) & (money_array < par_b[2]),
        #                       (par_b[2] <= money_array) & (money_array < par_b[3]),
        #                       (par_b[3] <= money_array)],
        #                      [lambda x: par_a[0] + par_a[1] * x,
        #                       lambda x: par_a[0] + par_a[1] * par_b[0] + par_a[2] * (x - par_b[0]),
        #                       lambda x: par_a[0] + par_a[1] * par_b[0] + par_a[2] * (par_b[1] - par_b[0]) +
        #                                 par_a[3] * (x - par_b[1]),
        #                       lambda x: par_a[0] + par_a[1] * par_b[0] + par_a[2] * (par_b[1] - par_b[0]) +
        #                                 par_a[3] * (par_b[2] - par_b[1]) + par_a[4] * (x - par_b[2]),
        #                       # Match value for x>par_b[3] with final value for parb[2]<=x<parb[3]
        #                       lambda x: par_a[0] + par_a[1] * par_b[0] + par_a[2] * (par_b[1] - par_b[0]) +
        #                                 par_a[3] * (par_b[2] - par_b[1]) + par_a[4] * (par_b[3] - par_b[2])]
        #                      )
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

    if dataset == ('elephant' or 'kuempel'):
        # If dataset is elephant, then use use derivative of Beverton-Holt model
        M_val = par_a * par_b / (par_b + money_array) ** 2

    elif dataset == 'wildebeest':
        # If dataset is wildebeest, the use coefficients in piecewise linear function
        M_val = par_a * par_b * money_array ** (par_b-1)

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
    :param money_array: (array) array of mu_area, total money invested in area
        
    :return: array
    """

    lambda_val = par_f * money_array

    return lambda_val


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

        :return (array) values of S(lambda, M)
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

    # Compute implicit functio S(mu_ranger, mu_area)
    S = dareadmu * dxdarea - par_f * dxdcops

    return S


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

    par_gamma, par_f, par_c0, par_cF, par_Mmax = small_dim_list

    condition = par_Mmax / par_f / par_gamma * par_c0 / (par_c0 + par_cF)

    return condition


def check_for_bubble(dim_list, num_it, dataset):
    """
    Check if any of the parameter combinations can result in a bubble, varying parameter values by one order
    of magnitude higher and lower.

    :param dim_list:
    :param num_it:
    :param dataset:

    :return: list of parameter combinations that result in a bubble
    """
    # num_it + 1 elements. Distance between 1/10 and 10 = 100/10 - 1/10 = 99/10, so step size is (99/10)/num_it
    multiply_array = np.arange(1 / 10, 10 + 99 / 10 / num_it, 99 / 10 / num_it)

    params = dict((key, val * multiply_array) for key, val in dim_list.items())

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
                            bubble_array = params['par_nonlin'] < S_equals_0([gamma_val, f_val, c0_val, cF_val, Mmax_val])

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
                    # count += 1
                    # if count % 32000 == 0:
                    #     print("{:.2f}% completed".format(count / (len(multiply_array) ** 5) * 100))

    # Print percentage of param combinations that result in a bubble

    print(f'{prop_bubble*100} % of param combos results in a bubble.')

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
                    lwid, clist, fontsz, save, fname, pname):
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

    :return figure, with one subfigure for each value in list_cops
    """

    par_psi, par_delta, par_nu, par_sigma, par_f, par_Mmax, par_mu0 = nondim_list.values()

    par_b, par_m, par_k, par_q, par_alpha, par_p0, par_gamma, par_c0, par_cf, par_f, par_Mmax, par_mu0 = dim_list.values()

    mu_area_ar = np.linspace(area_interval[0], area_interval[1], num=num_p)
    mu_ranger_ar = np.linspace(ran_interval[0], ran_interval[1], num=num_p)

    # Combine the arrays into a grid for contour plot
    [mu_a, mu_ran] = np.meshgrid(mu_area_ar, mu_ranger_ar)

    # Create figure for plotting the investment plot, with dimensions 10x10
    fig = plt.figure(figsize=(12, 12))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
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
    exist_curve_plot = ax.plot(mu_ranger_ar, exist_curve,
                               color=clist[1], linewidth=lwid, linestyle='-', label='Exist. cond.')[0]

    print('Done with existence condition line')

    # Add contour line for the boundary S=0. Dashed line with colour clist[1]
    try:
        contour0_line = ax.contour(mu_a, mu_ran, implicit_curve(mu_ran, mu_a, nondim_list, dataset),
                                   levels=np.array([0]), colors=clist[1], linewidths=lwid,
                                   linestyles='dashed')

        # Create proxy artist to give to legend()
        S0_line = mlines.Line2D([], [], linestyle='dashed', color='black', label='S=0 contour')

        # plt.legend(handles=[exist_curve_plot, S0_line], fontsize=fontsz[2])

    # If the S=0 contour DNE, then don't try to plot it and move on with the program
    except UserWarning:
        print('S(mu_ran, mu_area) does not have a 0 contour.')

        # plt.legend(handles=[exist_curve_plot], fontsize=fontsz[2])

    print('Done with S=0 contour line')

    # Shade region where (X+, Y+) = (M, 0),
    # i.e. where existence condition is negative, above existence contour
    ax.fill_between(mu_ranger_ar, exist_curve, y2=1e10, color=clist[0])

    # If the user wants to plot optimal investment paths
    if plot_path:
        # For each initial investment point, plot the optimal path
        for jj in range(0, len(invest_init)):
            # Add optimal investment path from initial investment of mu_a, mu_r = invest_init[jj]
            invest_area, invest_ran = investment_path(invest_init[jj], invest_size, num_steps, nondim_list, dataset)

            ax.plot(invest_area, invest_ran, markersize=2, color=clist[2])

    params_text = f'Parameters: b={par_b}, m={par_m}, k={par_k}, q={par_q},\n alpha={par_alpha}, p0={par_p0},'\
                  f' gamma={par_gamma:.5},\n c0={par_c0}, cf={par_cf}, f={par_f:.5}, Mmax={par_Mmax}, mu0={par_mu0}'

    # Add a star for the current investment in area and rangers for Zambia.
    # See Google sheet 'model_parameters.xlsx' for source.
    if dataset == 'elephant':
        plt.scatter(13389548.33, 101033450.3, s=200, c='k', marker='*')

    # Add in parameter text to the plot
    ax.annotate(params_text, (0.3 * area_interval[1], 0.5e5), fontsize=12)

    ax.set_title(
        'Implicit curve S(mu_ranger, mu_area)\n S>0 (grey) -> invest in area\n'
        'Above solid line: (X+, Y+)=(M,0)\n Solid line: exist cond. Dashed line: S=0 contour')
    ax.set_xlabel('Current USD invested in area', fontsize=fontsz[0])
    ax.set_ylabel('Current USD invested in rangers', fontsize=fontsz[0])

    if dataset == 'elephant':
        plt.axis([area_interval[0], 1e8, ran_interval[0], ran_interval[1]])
    else:
        plt.axis([area_interval[0], area_interval[1], ran_interval[0], ran_interval[1]])

    plt.xticks(fontsize=fontsz[1])
    plt.yticks(fontsize=fontsz[1])

    plt.show()

    if save:
        savename = '{}Implicit_cop_area\\{}'.format(fname, pname)
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
        savename = '{}Implicit_cop_area\\{}'.format(fname, pname)
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

        # Compute the derivative of existance condition
        exist_line = exist_cond(mu_area_ar, nondim_list, dataset)

        # Store results in dictionary along with the parameter value
        exist_dict[params[param_name]] = exist_line

        # Reset parameter value to default
        params[param_name] = dim_list[param_name]

    return exist_dict


def sens_analysis_full(num_it, area_interval, num_p, dim_list, nondim_list, dataset, fontsz, save, fname):
    """
        The plan: pick a parameter to do sensitivity analysis on. For num_it iterations, multiply parameter
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
        sens_keys = [k for k in sens_par.keys()]

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
            savename = '{}Sensitivity_analysis\\{}_{}_{}'.format(fname, 'sens_analysis', dataset, par)
            fig.savefig(savename)

    return


# ------------- THE MAIN PROGRAM -------------
# File location
filename = r'Z:\\Elephant_project\\Code\\Plots\\'

wrn.simplefilter('error', UserWarning)

# Specify font type and size
# font_args = {'family': 'Verdana',
#              'weight': 'normal',
#              'size': 24}

# Font sizes for axis labels, axis ticks, legend
# fontsizes = [24, 22, 18]
fontsizes = [28, 28, 28]

# Plot colours [S contour fill, S contour line/exist cond line]
# Green: '#1b9e77', orange: '#d95f02', grey: '#bdbdbd'
plot_colours = ['#bdbdbd', 'k']

# Min mu_area value for the contour plot
mu_area_min = 1  # Can't plot at (0,0) since the denominator is 0 in S(lambda, M). So take M_min < M < M_max

num_points = 1000  # int(mu_max)  # Number of points in each array (needs to be the same)

# Initial investments in area and rangers, used for plotting optimal investment path with investment_path()
# List of tuples, each tuple is separate IC
invest_IC = [(1000, 100)]

invest_amount = 10000  # Step size for optimal investment path
# Number of steps to do when computing the optimal investment path with investment_path()
invest_steps = int(1e3)

# ------------- Model parameters -------------
# All population data is for Luangwa Valley LVNP
alpha = 1e-5  # poacher effort adjustment rate (Source: Holden & Lockyer, 2021)

# Parameters for nonlinear price of land area
cell_size = 706  # Size of area cells in km2. Based on circle of radius 15km  (Source: Hofer, 2000)

parameter_set = input('Choose a parameter set ("e": elephant, "k": kuempel, or "w": wildebeest), or type "q" to quit: ')
while parameter_set not in ['e', 'k', 'w']:
    print('Input must be one of "e", "k", or "w". Please try again, or type "quit" to quit')
    parameter_set = input('Choose a parameter set ("e": elephant, "k": kuempel, or "w": wildebeest): ')

if parameter_set == 'e':
    # Use elephant parameter set and run the program
    # All money is in 2012 USD
    # ZMK_inflation = 7852.278  # 1 ZMK in 1985 = 7852.278 ZMK in 2012
    # ZMK_to_USD = 0.0002024  # 2012 ZMK t0 2012 USD
    # # Note: 1ZMK in 1985 = ZMK_inflation * ZMK_to_USD USD in 2012

    # Beverton-Holt area-cost curve parameters - need to be in list for M_val() function
    par_coeff = 1105.811  # M_max parameter. Source: R nls() fit to duplicated real estate data
    par_nonlin = 19339344  # mu0 parameter. Source: as above

    park_size = 753000  # Size of Zambia (km2) (Source: UN Statistics Department)
    # Total number of elephants Zambia can support (carrying capacity of LVNP extended to all of Zambia)
    k_total = 100000/40000 * park_size

    b = 0.33  # natural per capita birth rate (Source: Lopes, 2015)
    m = 0.27  # natural per capita death rate (Source: Lopes, 2015)
    k = k_total / park_size * cell_size  # Based on cell size of 706 = 15^2pi km2
    q = 2.56e-3  # catchability (Source: MG&LW, 1992)
    p0 = 3158.76  # max price paid for poached goods (Source: Messer, 2010)
    c0 = 1.911  # opportunity cost (Source: Lopes, 2015)
    cF = 1037.7276  # cost of fines per poacher * prob of conviction (Source: Holden et al., 2018)
    gamma = 0.043  # arrest * conviction probability (Source: Holden et al., 2018)

    # How much to employ ranger in Zambia
    # f = 1 / (20 * 0.91)  # (Source: Holden et al., 2018)
    f = 1/94900  # Lifetime discounted cost, based on $20USD/day from Holden et al. 2018

    dataset_name = 'elephant'

    # Max values for mu_rangers and mu_area for the contour plot
    mu_max = 2e9

    run_program = True

elif parameter_set == 'k':
    # Use the Kuempel et al. 2018 parameter set and run the program
    # These parameters result in an S=0 contour
    b = 1
    m = 0.1
    q = 0.7
    p0 = 1
    c0 = 0.018
    gamma = 0.2
    cF = 0.01/gamma # Different from parameter value in Kuempel. gamma so cF*gamma gives desired value of 0.01
    k = 1000
    f = 1 / 300

    par_coeff = 0.003
    par_nonlin = 0  # Not used since the area-money line goes through (0,0)

    dataset_name = 'kuempel'

    # Max values for mu_rangers and mu_area for the contour plot
    mu_max = 1e4

    run_program = True

elif parameter_set == 'w':
    # Use wildebeest parameter set and run the program
    # All money is in 2012 USD

    # # Area-cost power function parameters -> M = a*mu^b
    par_coeff = 0.3327  # Source: R nls() fit to duplicated real estate data
    par_nonlin = 0.2211  # Source: as above

    # Area-cost piecewise linear function parameters -> M = a_i * mu + b_i
    # Source: R nls() fit to duplicated Tanzania real estate data
    # 5.7486e-07 is final slope estimate but set to 0 to simulate finite area
    # First element is the intercept
    # par_coeff = [1680.674, 9.7526e-03, 8.6789e-04, 1.7621e-04, 3.8797e-05, 0]
    # par_nonlin = [233290, 3786769, 24845746, 128008240]

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
    f = 1/94900   # Lifetime discounted cost, based on $20USD/day from Holden et al. 2018
    dataset_name = 'wildebeest'

    # Max values for mu_rangers and mu_area for the contour plot
    mu_max = 2e9

    run_program = True

else:
    # Do not run the program
    run_program = False
    print("Something weird happened.")

# ------------- Run the investment plot program -------------
# If the user did not quit the program, proceed with plotting
if run_program:

    # Compute nondimensional parameters
    psi = alpha * p0 * q * k / (b - m)
    delta = alpha * p0 * q * k * gamma / (b - m)
    nu = alpha * c0 / (b - m)
    sigma = alpha * gamma * cF / (b - m)

    # Create lists of the nondimensional and dimensional parameters
    nondim_params = {'psi': psi, 'delta': delta, 'nu': nu, 'sigma': sigma, 'f': f,
                     'par_coeff': par_coeff, 'par_nonlin': par_nonlin}

    dim_params = {'b': b, 'm': m, 'k': k, 'q': q, 'alpha': alpha, 'p0': p0, 'gamma': gamma,
                  'c0': c0, 'cF': cF, 'f': f, 'par_coeff': par_coeff, 'par_nonlin': par_nonlin}

    # ----------------------------- MAKING THE PLOT -----------------------------
    # Set name for saving the plot
    plotname = f'money_mu{mu_max:.2e}_{dataset_name}_usd.pdf'

    # Make the investment plot
    fig1, ax1, exist_line = investment_plot([0, mu_max], [mu_area_min, mu_max], num_points,
                                            nondim_params, dim_params, plot_path=False,
                                            invest_init=invest_IC, invest_size=invest_amount,
                                            num_steps=invest_steps, dataset=dataset_name,
                                            lwid=5, clist=plot_colours, fontsz=fontsizes, save=False,
                                            fname=filename, pname=plotname)

    # # Make the existence cond. derivative plot
    # fig2, ax2 = investment_slope([0, mu_max], [mu_area_min, mu_max], num_points, nondim_params, dim_params,
    #                              dataset=dataset_name, lwid=5, clist=plot_colours,
    #                              fontsz=fontsizes, save=False, fname=filename, pname=plotname)

else:
    print("So long and thanks for all the fish!")

# ------------- Run the sensitivity analysis -------------
# num_sim = 100
# sens_analysis_full(num_sim, [mu_area_min, mu_max], num_points, dim_params, nondim_params,
#                    dataset_name, fontsizes, save=False, fname=filename)

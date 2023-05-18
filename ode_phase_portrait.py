"""Created 12/05/2021
Author: Liam Timms
Modified: 03/11/2021
Version 7
Modifications: Lockyer model w confiscation.
Constant price, no scavenging -> logistic growth
Plot nondimensional system phase portraits
Plot bifurcation diagram
Plot phase portraits as one figure for thesis - modification to phase_portrait() function
v7: use final Zambia parameters
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sci


def odefun(y, t, num_cells, num_cops, nondim_list):
    """
    Create ODE function to pass into odeint() solver
    :param num_cells: (float, positive) M the number of land cells in protected area
    :param num_cops: (float, non-negative) total number of cops in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]

    :return (list) derivative at X, Y, t (xplus, yplus)
    """

    # Set up the main variables
    X, Y = y

    # Unpack the nondimensional parameters
    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    X_dash = (1 - X) * X - X * Y

    Y_dash = ((par_psi - par_delta * num_cops / num_cells) * X - par_nu - par_sigma * num_cops / num_cells) * Y

    return [X_dash, Y_dash]


def odesol(x0, t_int, num_cells, num_cops, nondim_list):
    """
    Numerically solve the ODE system, given initial condition x0 and time interval t_int
    :param x0: (list) the initial condition x0 = [X0, Y0]
    :param t_int: (array) the time interval [t0, ..., tmax]
    :param num_cells: (float, positive) M the number of land cells in protected area
    :param num_cops: (float, non-negative) total number of cops in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]

    :return sol the numerical ODE solution
    """

    sol = sci.odeint(odefun, x0, t_int, args=(num_cells, num_cops, nondim_list))

    return sol


def internal_crit(num_cells, num_cops, nondim_list):
    """
    Compute the internal critical point for model with const. price, no scavenging, logistic growth.
    :param num_cells: (float, positive) M the number of land cells in protected area
    :param num_cops: (float, non-negative) total number of cops in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]

    :return (list) internal crit point (xplus, yplus)
        """
    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    xplus = (par_nu + par_sigma * num_cops / num_cells) / (par_psi - par_delta * num_cops / num_cells)
    yplus = 1 - xplus

    return xplus, yplus


def spiral_bif(nondim_list, num_cells):
    """
    Find out where the spiral/node bifurcation 4B is, as a function of lambda
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]
    :param num_cells: (float, positive) M the number of land cells in protected area

    :return: bif_point: (float) the lambda value where the spiral-node bifurcation of X* occurs
    """

    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    # fourB = 4 * (par_psi - par_delta * num_cops / num_cells - par_nu - par_sigma * num_cops / num_cells)

    # The solution to X*=4B that corresponds to 0<X<1
    bif_point = (
                        -4 * num_cells * par_delta * par_nu + num_cells * par_sigma + 8 * num_cells * par_delta * par_psi + 4 * num_cells * par_sigma * par_psi
                        - num_cells * np.sqrt(16 * par_delta ** 2 * par_nu + 16 * par_delta ** 2 * par_nu ** 2 +
                                              8 * par_delta * par_nu * par_sigma + par_sigma ** 2 + 16 * par_delta * par_sigma * par_psi +
                                              32 * par_delta * par_nu * par_sigma * par_psi + 8 * par_sigma ** 2 * par_psi +
                                              16 * par_sigma ** 2 * par_psi ** 2)) / (
                        8 * (par_delta ** 2 + par_delta * par_sigma))

    return bif_point


def bifurc_diagram(num_cells, nondim_list, N_points, tstep, tmax, max_cops):
    """
    Create the bifurcation diagram by finding the ODE solution at long times for several ICs and parameter values
    :param num_cells: (float, positive) M the number of land cells in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]
    :param N_points: (int) simulate N x N initial conditions (X0,Y0) in [0, 2] x [0, 2]
    :param tstep (float): time step size for numerical integral
    :param tmax: (float) numerically solve trajectories from t = 0 to t=tmax
    :param max_cops: (float) do bifurcation diagram from lambda0 = 0 to lambda0 = max_cops

    :return: lambda_list, soln_list (list, list): list of no. of cops and
                corresponding final trajectory values for varied ICs
    """
    # Min and max values for X and Y, sets up the list of initial conditions
    min_val, max_val = 0, 2
    step = (max_val - min_val) / N_points

    # Array of lambda values to plot
    cops_range = np.arange(0, max_cops, 1)

    # List of initial conditions
    IC_list = [[x0, y0] for x0 in np.arange(min_val, max_val, step) for y0 in np.arange(min_val, max_val, step)]

    # Inital time and time step size
    t0 = 0

    # List of num_cops values and list of trajectory value at t=tmax
    # Used to plot the bifurcation diagram
    lambda_list = []
    soln_list = []

    for num_cops in cops_range:
        if num_cops < cops_range[-1]/10:
            # In the first 10% of lambda values, use 2xtime steps. This is because these orbits take longer to converge
            t_int = np.arange(t0, tmax, tstep/3)
        else:
            t_int = np.arange(t0, tmax, tstep)

        for x0 in IC_list:
            soln = odesol(x0, t_int, num_cells, num_cops, nondim_list)

            # Get the final elephant population value at t=tmax
            xfinal = soln[-1][0]

            # Add the final population and the no. of cops to the respective lists
            soln_list.append(xfinal)
            lambda_list.append(num_cops)

    return lambda_list, soln_list


def bifurcation_plot(num_cells, nondim_list, N_points, tstep, tmax, max_cops, mksize, mkcol, save, fname, pname):
    """
    Make the bifurcation plot
    :param num_cells: (float, positive) M the number of land cells in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]
    :param N_points: (int) simulate N x N initial conditions (X0,Y0) in [0, 2] x [0, 2]
    :param tstep (float): time step size for numerical integral
    :param tmax: (float) numerically solve trajectories from t = 0 to t=tmax
    :param max_cops: (float) do bifurcation diagram from lambda0 = 0 to lambda0 = max_cops
    :param mksize: (float) marker size in scatter plot
    :param mkcol: (string) marker colour in scatter plot, HEX code
    :param save: (True/False) true if you want to save the figure, false otherwise
    :param fname: (string) file location to save figure
    :param pname: (string) file name to save figure

    :return: figure, axis, float (location of spiral-node bifurcation)
    """

    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    lambda0_list, X_list = bifurc_diagram(num_cells, nondim_list, N_points, tstep, tmax, max_cops)

    # Create figure for plotting, with dimensions 10x10
    fig = plt.figure(figsize=(10, 10))
    # Add plot in figure, 1st subplot in plot with 1 row & 1 column
    ax = fig.add_subplot(111)

    ax.scatter(lambda0_list, X_list, s=mksize, c=mkcol)
    plt.axis([0, max_cops, 0, 1])
    plt.xlabel("\u03BB (total rangers)")
    plt.ylabel("X* = N*/k (equilibrium elephant pop.)")
    plt.title("Bifurcation diagram for (X',Y')")

    # Add in point where X* undergoes spiral/node bifurcation, X*=4B
    spiral_bif_lambda = spiral_bif(nondim_list, num_cells)  # lambda coordinate of bifurcation
    # X*, Y* coordinate of bifurcation
    spiral_bif_x, spiral_bif_y = internal_crit(num_cells, spiral_bif_lambda, nondim_list)
    ax.plot(spiral_bif_lambda, spiral_bif_x, 'X', markersize=10, color='#d95f02')

    fig.legend(['Spiral-node bif.', 'Critical point'],
               loc='upper right', fontsize=12
               )

    plt.show()

    if save:
        savename = '{}{}'.format(fname, pname)
        fig.savefig(savename)

    return fig, ax, spiral_bif_lambda


def beta(num_cells, nondim_list):
    """ Compute function that determines if 0 < X* < 1,
     and the stability of crit point at (1,0). num_cops < beta(num_cells)

    :param num_cells: (float, positive) M the number of land cells in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]

    :return (float) beta_M, the maths function beta(num_cops)
    """
    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    beta_M = (par_psi - par_nu) / (par_sigma + par_delta) * num_cells

    return beta_M


def nullclines(xarray, yarray, num_cells, num_cops, nondim_list, lwidth, lcolour, ax):
    """
    Compute the dX/dT and dY/dT nullclines and plot them on the current figure

    :param xarray: (array) lower and upper bounds on x values, [xmin, xmax]
    :param yarray: (array) lower and upper bounds on y values, [ymin, ymax]
    :param num_cells: (float, positive) M the number of land cells in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]
    :param lwidth: (float) width for nullcline line
    :param lcolour: (str) hex code for nullcline line colour, e.g. '#1b9e77'
    :param ax: (matplotlib.axis object) the current axis to plot on

    :return None
    """

    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    # Create arrays for the nontrivial nullclines
    # Nullcline dX/dT = 0 (only relevant for positive values, line goes from Y=1 to Y=0. Use X values [0,1]
    dXdTnull = 1 - np.array(xarray)

    # Nullcline dY/dT = 0
    dYdTnull = (par_nu + par_sigma * num_cops / num_cells) / (par_psi - par_delta * num_cops / num_cells)

    # Plot the nullclines
    ax.plot(xarray, dXdTnull, '--', color=lcolour, linewidth=lwidth)

    # dYdT nullcline (the nullcline is a vertical line)
    # Only plot if it's a physical number
    if 0 <= dYdTnull <= 1.1:
        ax.plot([dYdTnull, dYdTnull], [yarray[0], yarray[1]], ':',
                color=lcolour, linewidth=lwidth)

    return


def carry_cap_stability(num_cells, num_cops, nondim_list):
    """ Determine the stability of crit point at (1,0).

    :param num_cells: (float, positive) M the number of land cells in protected area
    :param num_cops: (float, non-negative) total number of cops in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]

    :return None
    """

    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    beta_M = beta(num_cells, nondim_list)

    print("The carrying capacity crit pt. (1, 0) is ", end='')

    if num_cops < beta_M:
        print("a saddle. lambda1 = -1, lambda2 > 0.")

    elif num_cops == beta_M:
        print("nonhyperbolic. A bifurcation occurs here.")

    elif num_cops == beta_M + num_cells / (par_sigma + par_delta):
        print("a stable (improper) node. lambda1 = -1, multiplicity 2.")

    elif num_cops > beta_M and num_cops != beta_M + -1:
        print("a stable (proper) node. lambda1 = -1, lambda2<0, !=-1.")

    else:
        raise Exception(
            "num_cops is apparently neither greater nore less than beta_M. \n Check conditional block for errors.")

    return


def int_crit_stability(num_cells, num_cops, nondim_list, int_crit):
    """ Determine the stability of internal crit point (X+, Y+). Only works for simple model.

    :param num_cells: (float, positive) M the number of land cells in protected area
    :param num_cops: (float, non-negative) total number of cops in protected area
    :param nondim_list: (list of floats) the nondimensional parameters as in simple model
                        [psi, delta, nu, sigma]
    :param int_crit: (list of floats, length 2) the internal critical point (X+, Y+)

    :return (float, array, array) discriminant, array of Jacobian eigenvalues, and array of eigenvectors
    """

    # Unpack interior critical point values
    xval = int_crit[0]
    yval = int_crit[1]

    # Unpack parameters
    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    """ For the Jacobian at the internal critical point, tr<0 and det>0, so (X*, Y*) is either
    a stable spiral or node. The type is determined by the sign of tr^2-4det. """

    # Compute Jacobian elements
    J11 = -xval
    J12 = -xval
    J21 = par_psi - par_nu - (par_delta + par_sigma) * num_cops / num_cells
    J22 = 0

    jacobian = np.array([[J11, J12],
                         [J21, J22]])

    evals, evecs = np.linalg.eig(jacobian)

    # Compute the discriminant tr^2-4det
    discrim = xval ** 2 - 4 * xval * (par_psi - par_nu - (par_delta + par_sigma) * num_cops / num_cells)

    print("The internal critical point (X*,Y*) is a ", end='')

    if discrim > 0:
        print("stable proper node.")
    elif discrim == 0:
        print("stable improper node")
    elif discrim < 0:
        print("stable spiral.")
    else:
        raise Exception(
            "tr^2-4det for internal critical point is apparently not a real number."
            " \n Check conditional block for errors.")

    return discrim, evals, evecs


def phase_portrait(x_interval, y_interval, x_nump, y_nump, num_cells, list_cops, nondim_list,
                   lspec, colour_list, save, fname, pname):
    """
    Create phase portrait plot for elephant-poacher system w constant money
    :param x_interval: (list) [x lower bound, x upper bound]
    :param y_interval: (list) [y lower bound, y upper bound]
    :param x_nump: (int) number of x points to plot
    :param y_nump: (int) number of y points to plot
    :param num_cells: (float, positive) M the number of land cells in protected area
    :param list_cops: (list of floats, non-negative) list of total ranger values, plot portrait for each
    :param nondim_list: (list) parameters for nondimensionalised model, [psi, delta, nu, sigma]
    :param lspec: (list) [line width, marker size, density of lines, arrow size] all should be positive floats
    :param colour_list (list) 3 string, hex codes for plot colours [trajectories, nullclines, crit. points]
    :param save: (True/False) True if you want to save the plot, False otherwise
    :param fname: (str) Folder location to save figure
    :param pname: (str) name for the plot, goes on end of file name when saving

    :return figure, with one subfigure for each value in list_cops
    """

    par_psi = nondim_list[0]
    par_delta = nondim_list[1]
    par_nu = nondim_list[2]
    par_sigma = nondim_list[3]

    # Number of phase portraits to create - one for each lambda value in list_cops
    n_lambda = len(list_cops)

    # Create array of X, Y, U points to plot
    x = np.arange(x_interval[0], x_interval[1], 1 / x_nump)
    y = np.arange(y_interval[0], y_interval[1], 1 / y_nump)

    # Create grid for plotting
    X, Y = np.meshgrid(x, y)

    # Create n_lambda number of subplots, arranged horizontally, with dimensions 12x4. Keep the same X and Y axes
    fig_len = 10 * n_lambda
    fig, ax = plt.subplots(nrows=1, ncols=n_lambda, figsize=(fig_len, 10), sharex='all', sharey='all')

    # Make phase portrait in each subplot, one for each value in list_cops
    for ii in range(0, n_lambda):

        # Is there is more than one subplot, then ax is an array and we have to index it
        if type(ax) == np.ndarray:
            axis = ax[ii]
        # If there is only one subplot, then ax is a matplotlib.axes object and is not subscriptable
        else:
            axis = ax

        # The ODE system for this number of rangers
        X_dash, Y_dash = odefun((X, Y), 0, num_cells, list_cops[ii], nondim_list)

        # Create streamplot of the ODE system and add to current subplot
        axis.streamplot(X, Y, X_dash, Y_dash, linewidth=lspec[0], color=colour_list[0],
                        density=lspec[2], arrowsize=lspec[3])

        # Add the nullclines to the current subplot
        # nullclines([x_interval[0], 1], y_interval, num_cells, list_cops[ii],
        #            nondim_list, lspec[0], colour_list[1], axis)

        # Add markers at extinction and  carrying capacity critical points
        axis.plot(0, 0, '*', markersize=lspec[1], color=colour_list[2])
        axis.plot(1, 0, '*', markersize=lspec[1], color=colour_list[2])
        # Add marker at internal critical point if it exists
        if list_cops[ii] < beta(num_cells, nondim_list) and par_psi > par_nu:
            xstar, ystar = internal_crit(num_cells, list_cops[ii], nondim_list)

            axis.plot(xstar, ystar, '*', markersize=lspec[1], color=colour_list[2])

        # Add title text for each subplot detailing the value of lambda
        title_text = f'\u03BB={round(list_cops[ii], 3)}'
        axis.set_title(title_text, fontsize=40)

        # End phase portrait for loop

    # Specify the axis labels
    # Is there is more than one subplot, then ax is an array and we have to index it
    if type(ax) == np.ndarray:
        # Set Y axis label for 1st plot, and X axis label for middle plot
        ax[0].set_ylabel('Y = qE/(b-m)', fontsize=30)
        ax[1].set_xlabel('X = N/k', fontsize=30)
    # If there is only one subplot, then ax is a matplotlib.axes object and that plot gets all the labels
    else:
        ax.set_ylabel('Y = qE/(b-m)', fontsize=30)
        ax.set_xlabel('X = N/k', fontsize=30)

    params_text = f'Parameters: psi={par_psi:.1}, delta={par_delta:.1}, nu={par_nu:.1e},' \
                  f' sigma={par_sigma:.1e}, M={num_cells:.1f}'
    # Add the text in bottom right
    plt.text(-0.1 * x_interval[1], -0.2 * y_interval[1], params_text, fontsize=14)
    plt.show()

    if save:
        savename = '{}Phase_portraits\\{}'.format(fname, pname)
        fig.savefig(savename)

    return fig, ax


# --------------------------------------------------------------------------
# ------------------- INTIALISE PARAMETERS AND VARIABLES -------------------
# File location
filename = r'Z:\elephant_project\Code\Plots\\'

# Specify font type and size
font_args = {'family': 'Verdana',
             'weight': 'normal',
             'size': 22}

plt.rc('font', **font_args)

# Specify width/size of lines, markers, density of lines, arrows
linespec = [3, 30, 0.4, 4]

# Plot colours [trajectories, nullclines, crit. points]
# plot_colours = ['#7570b3', 'k', '#d95f02']
plot_colours = ['#636363', 'k', 'k']

# Model parameters
park_size = 40000   # Size of LVNP = 40000km2
cell_size = 706

parameter_set = input('Choose a parameter set ("e": elephant or "k": kuempel), or type "q" to quit: ')
while parameter_set not in ['e', 'k']:
    print('Input must be one of "e" or "k". Please try again, or type "quit" to quit')
    parameter_set = input('Choose a parameter set ("e": elephant or "k": kuempel')


if parameter_set == 'e':
    # Model parameters - Zambia elephants
    alpha = 1e-5
    b = 0.33  # natural per capita birth rate (Source: Lopes, 2015)
    m = 0.27  # natural per capita death rate (Source: Lopes, 2015)
    q = 2.56e-3  # catchability (Source: MG&LW, 1992)
    k = 100000 / park_size * cell_size  # Based on cell size of 706 = 15^2pi km2
    p0 = 3158.76  # max price paid for poached goods (Source: Messer, 2010)
    c0 = 1.911  # opportunity cost (Source: Lopes, 2015)
    cF = 1037.7276  # cost of fines per poacher * prob of conviction (Source: Holden et al., 2018)
    gamma = 0.043  # arrest * conviction probability (Source: Holden et al., 2018)

    lambda0 = [park_size / 300]  # number of law enforcement individuals (Source: LW, 1990 - 1ranger/300km2)
    M = park_size / cell_size  # Number of land cells (LVNP size/cell size of 706km2), or 10 for Kuempel et al.

elif parameter_set == 'k':
    # Use the Kuempel et al. 2018 parameter set and run the program
    # These parameters result in an S=0 contour
    alpha = 0.5
    b = 1
    m = 0.1
    q = 0.7
    k = 1000
    p0 = 1
    c0 = 0.018
    gamma = 0.2
    cF = 0.01 / gamma  # Different from parameter value in Kuempel. /gamma so cF*gamma gives desired value

    lambda0 = [49.99]#[49.9, 49.97, 49.99]  # Lambda values for Kuempel et al.
    M = 10

# Initialise plotting parameters
# Elephant population N/(k_m*A_max) GUIDELINE: Carrying capacity is 1, so Xmax should be ~1 - 2
x_int = [0, 1.5]
x_points = 100  # num points to plot

# Poacher effort qE/(b-m) GUIDELINE: Holden & Lockyer use Ymax ~1 - 1.5
y_int = [0, 2]
y_points = 100  # num points to plot

# ------ FOR THE FULL MODEL -------
# Compute nondimensionalised parameters
psi = alpha * p0 * q * k / (b - m)
delta = alpha * p0 * q * k * gamma / (b - m)
nu = alpha * c0 / (b - m)
sigma = alpha * cF / (b - m)

nondim_params = [psi, delta, nu, sigma]

# ------ CREATE THE PHASE PORTRAIT -------
phase = True
if phase:
    pname_phase = f'phase_kuempel_carry.pdf'
    the_plot, axes = phase_portrait(x_int, y_int, x_points, y_points, M, lambda0, nondim_params,
                                    linespec, plot_colours, save=True, fname=filename,
                                    pname=pname_phase)

# ------ CREATE THE BIFURCATION DIAGRAM -------
bif = False
if bif:
    pname_bif = 'bif_diagram1.pdf'
    time_step = 0.025
    # Set size and colour of scatter points
    scattersize = 1
    scattercol = 'k'
    fig_bif, ax_bif, bif_loc = bifurcation_plot(M, nondim_params, N_points=10, tstep=time_step, tmax=200, max_cops=100,
                                                mksize=scattersize, mkcol=scattercol, save=False, fname=filename,
                                                pname=pname_bif)

# ------------------- STABILITY ANALYSIS -------------------
stability_check = True
# If we want to see some stability analysis for the simple model
if stability_check:
    # Compute the internal critical point

    x_plus, y_plus = internal_crit(M, lambda0[0], nondim_params)

    print(f'(X*, Y*) = {(round(x_plus, 3), round(y_plus, 3))}')

    # Compute stability of carrying capacity critical point
    carry_cap_stability(M, lambda0[0], nondim_params)

    # Compute stability of internal critical point, if it exists
    if 0 < x_plus < 1:
        discr, evals, evecs = int_crit_stability(M, lambda0[0], nondim_params, (x_plus, y_plus))

        # print("The eigenvalues of the Jacobian at (X+, Y+) are ", (eval1, eval2))

# ------------------- ERROR CHECKING -------------------
error_check = False
if error_check:
    # DEBUGGING: print nondim parameter values
    print("Numerical values of nondim. parameters:")
    print('psi=', psi, '\n delta=', delta, '\n nu=', nu, '\n sigma=', sigma, '\n lambda=', lambda0, '\n M=', M)

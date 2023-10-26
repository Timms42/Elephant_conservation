# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:07:31 2023

@author: timmslf
"""

import numpy as np
import random as rand
from multiprocessing import Pool

import conservation_model_analysis as model


filename = 'C:\\Users\\timmslf\\Documents\\official_plots\\'

# Set random seed
rand.seed(42)

# ------------- Model parameters -------------
fontsizes = [28, 28, 22]  # Font sizes for axis labels, axis ticks, legend

alpha = 1e-5  # poacher effort adjustment rate (Source: Holden & Lockyer, 2021)

# Parameters for nonlinear price of land area
cell_size = 706  # Size of area cells in km2. Based on circle of radius 15km  (Source: Hofer, 2000)
total_size = 753000  # Size of Zambia (km2) (Source: UN Statistics Department)
LVNP_size = 40000
# Total number of elephants LVNP can support
k_total = 100000

b = 0.33  # natural per capita birth rate (Source: Lopes, 2015)
m = 0.27  # natural per capita death rate (Source: Lopes, 2015)
k = k_total / LVNP_size * cell_size  # Based on cell size of 706 = 15^2pi km2
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

# Max values for mu_rangers and mu_area for the contour plot
mu_ran_max = 2e9
mu_area_max = 1e8  # Change x-axis limit

num_points = 100  # 1000
# The change in investment (mu_step) should be big enough that it steps into the next "chunk"/discretised section
# of the investment plot. Size of each chunk (on area axis) = mu_area_max/num_points.
# Make mu_step = 1/2 of this chunk size
delta_mu = mu_area_max / num_points / 2
attack_rate = 1
handle_time = 1
# power_z = 2
tfinal = 100000
tolerance = 1e-5
the_solver = 'LSODA'
the_step = 1

param_combos = [(z, w, inter) for z in [1, 3, 5, 10] for w in [0.5, 1, 2, 10] for inter in [['perceived', 'perceived']]]

def parallel_model(*pars):
    par_power, par_denom, interaction = pars[0] # Need to unpack a 2nd time because of how map passes arguments

    # Define numerical_investment() that only requires parameters for gamma function par_power and par_denom.
    model.numerical_investment(dim_list=dim_params,
                                         mu_ran_final=mu_ran_max,
                                         mu_area_final=mu_area_max,
                                         mu_step=delta_mu,
                                         par_attack=attack_rate,
                                         par_handle=handle_time,
                                         par_power=par_power,
                                         par_denom=par_denom,
                                         tf=tfinal,
                                         interact=interaction,
                                         num_p=num_points,
                                         tol=tolerance,
                                         solver=the_solver,
                                         max_step=the_step,
                                         cols=['#ffffff', '#bdbdbd', 'r', 'k'],
                                         # white, grey, and black
                                         mksize=(12 * 72 / num_points) ** 2,
                                         fontsz=fontsizes,
                                         fname=filename,
                                         save=True)

if __name__ == '__main__':
    with Pool(16) as p:
        p.map(parallel_model, param_combos)
    

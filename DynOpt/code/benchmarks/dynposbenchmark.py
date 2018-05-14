'''
Contains the function to create the dynamic optimization problems for the GECCO 
2018 paper. Afterwards extended to re-create the EvoStar 2018 data (14.3.18).

Only the position of the fitness landscape is changed but not the fitnesss 
level. The data set values are stored per change.

Contains functionality to create a data set as well as computing the 
fitness during the runtime. 
Additionally, this module contains functions to test and plot new movements.

Note: in predictor_comparison.py are the data modified so that they have one
entry for each generation. TODO ist das noch so?

Created on Jan 17, 2018

@author: ameier
'''

import copy
import math
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.pardir))

import matplotlib.pyplot as plt
import numpy as np
from utils.fitnessfunctions import sphere, rosenbrock, rastrigin,\
    get_original_global_opt_pos_and_fit
from utils.utils_print import get_current_day_time


def create_str_problems():
    '''
    Computes for each change the global optimum position. The new position
    is computed by adding a movement vector that depends on the change number.
    '''
    day, time = get_current_day_time()
    # -------------------------------------------------------------------------
    # TODO(dev) parameters to adjust
    n_chg_periods = 10000
    dims = [2, 5, 10, 20, 50, 100]
    functions = [sphere, rosenbrock, rastrigin]
    pos_chng_types = ['pch-linear', 'pch-sine']
    fit_chng_type = 'fch-none'
    # "EvoStar_2018" or "GECCO_2018" (must be equivalent to directory)
    conference = "GECCO_2018"
    # -------------------------------------------------------------------------

    # severity of change (for linear movement)
    if conference == "GECCO_2018":
        linear_movement_factor = 5
    elif conference == "EvoStar_2018":
        linear_movement_factor = 2

    # path to data set directory to store data sets there
    # ".../DynOptimization/DynOpt/code"
    splitted_path = os.path.abspath(os.pardir).split('/')
    # ".../DynOptimization/DynOpt"
    path_to_dynopt = '/'.join(splitted_path[:-1])
    # create data sets
    for func in functions:
        # same seed for different functions so that the movement is the same
        np_rand_gen = np.random.RandomState(234012)  # TODO(ueberdenken?) 8.Mai
        for dim in dims:
            func_name = func.__name__
            orig_global_opt_position, _ = get_original_global_opt_pos_and_fit(
                func, dim)
            folder_path = path_to_dynopt + "/datasets/" + conference + "/" + func_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            folder_path = folder_path + "/"

            if fit_chng_type == "fch-none":
                # store global optimum fitness (stays same for all changes
                global_opt_fit = np.array(n_chg_periods * [0])
            else:
                # TODO(dev) implement if desired
                pass

            # compute optimum movement
            for pos_chng_type in pos_chng_types:
                opts = []
                opts.append(copy.copy(orig_global_opt_position))
                if pos_chng_type == 'pch-sine':
                    if conference == "GECCO_2018":
                        # initialize sine-parameters randomly (stay unchanged
                        # during all change periods
                        amplitudes = np_rand_gen.randint(5, 50, dim)
                        width_factors = np_rand_gen.rand(dim)
                        for chg_period in range(1, n_chg_periods):
                            # compute movement in all dimensions
                            movement = np.zeros(dim)
                            for d in range(dim):
                                # computing schema as follows:
                                #new_opt[d] = 30 * np.sin(0.25 * c) + 30 + c
                                movement[d] = amplitudes[d] * \
                                    np.sin(width_factors[d] *
                                           chg_period) + amplitudes[d] + chg_period
                            # new optimum position
                            new_opt = orig_global_opt_position + movement
                            opts.append(copy.copy(new_opt))
                    elif conference == "EvoStar_2018":
                        for chg_period in range(1, n_chg_periods):
                            step = chg_period * linear_movement_factor
                            x = orig_global_opt_position
                            new_opt = copy.copy(x)
                            # move first dimension
                            new_opt[0] = new_opt[0] + step
                            # move second dimension (make amplitude larger (30
                            # instead of 1) and make sinus wider (with
                            # multiplication with 0.25).
                            # If the sinus is not made wider, the algorithm will
                            # find a "double curve".
                            new_opt[1] = new_opt[1] + \
                                30.0 * math.sin(0.25 * step)
                            opts.append(copy.copy(new_opt))
                elif pos_chng_type == 'pch-linear':
                    if conference == "GECCO_2018" or conference == "EvoStar_2018":
                        for chg_period in range(1, n_chg_periods):
                            movement = np.array(
                                dim * [chg_period * linear_movement_factor])
                            new_opt = orig_global_opt_position + movement
                            opts.append(copy.copy(new_opt))
                opts = np.array(opts)
                # save optimum
                ds_file_name = folder_path + func_name + "_d-" + \
                    str(dim) + "_chgperiods-" + str(n_chg_periods) + "_" + pos_chng_type + "_" + \
                    fit_chng_type + "_" + day + '_' + time + ".npz"
                np.savez(ds_file_name, global_opt_fit_per_chgperiod=global_opt_fit,
                         global_opt_pos_per_chgperiod=opts, orig_global_opt_pos=orig_global_opt_position)


# TODO löschen (sollte nicht mehr gebraucht werden, weil man in dynea/dynpso
# das globale Optimum nicht benötigt. Metriken werden getrennt berechnet,
# sodass dannn die Datensatzdateien ausgelesen werden können.
#
# TODO wird für Erweiterung mit Konfidenzintervall das globale Optimum
# während der Optimierung benötigt? (sollte eigentlich nicht, denn das Wissen
# über Umwelt darf nicht bekannt sein.)
# def get_global_optimum(gen, global_opt_fit):
# TODO umbenennen in get_global_optimum_fit
#    return global_opt_fit[gen]


# def get_global_optimum_position(gen, global_opt_pos):
#    return global_opt_pos[gen]


def original_fitness(x, problem):
    '''
    TODO warum ist problem als String übergeben??? (Funktionen werden in diesem
    Modul ja eh bereits importiert.

    Computes fitness for this individual.
    Assumes that the individual/fitness function is not moved.
    '''
    if problem == "sphere":
        return sphere(x)
    elif problem == "rosenbrock":
        return rosenbrock(x)
    elif problem == "rastrigin":
        return rastrigin(x)
    else:
        msg = "original_fitness(): unknown problem " + problem
        warnings.warn(msg)


def compute_fitness(x, gen, problem, global_opt_pos_per_gen, orig_opt_pos):
    '''
    Computes the fitness of the passed individual. Depends on the current
    generation.
    @param x: individual
    @param gen: current generation
    @param problem: name of fitness function: sphere, rosenbrock, rastrigin
    @param global_opt_pos_per_gen: 2d numpy array, contains the global optimum
    position for each generation
    @param orig_opt_pos: global optimum position of unmoved fitness landscape 
    '''
    # compute optimum movement
    # (since new optimum position was computed by adding the movement to the
    # original one, backwards the movement can be computed by substraction)
    # TODO should be per CHANGE
    optimum_movement = global_opt_pos_per_gen[gen] - orig_opt_pos
    # move individual, so that its fitness can be computed with the original
    # function
    moved_x = x - optimum_movement
    return original_fitness(moved_x, problem)

###############################################################################
# only for testing new movements
###############################################################################


def create_and_plot_different_movements():
    '''
    Creates different movements and plots them. Only for testing new movements.
    '''
    dim = 2
    n_chg_periods = 3000
    opts = []
    for p in range(n_chg_periods):
        # movement same in all dimensions
        new_opt = np.zeros(dim)
        for d in range(dim):
            new_opt[d] = 30 * np.sin(0.25 * p) + 30 + p

        # different movement for each dimension
        new_opt = np.zeros(dim)
        new_opt[1] = 30 * np.sin(0.25 * p) + 30 + p
        # sinus linear nach oben
        new_opt[0] = p
        # sinus als Sättigungskurve nach oben
        new_opt[0] = p + (0.1 * p**2)
        # 8er-Kurve linear nach oben
        new_opt[0] = 15 * np.sin(0.5 * p) + 15 + p
        # 8er-Kurve als Sättigungskurve nach oben
        new_opt[0] = 15 * np.sin(0.5 * p) + 15 + (0.1 * p**2)

        opts.append(copy.copy(new_opt))

    opts = np.array(opts)
    plot_scatter(opts)


def create_and_plot_random_sine_movement():
    '''
    Global optimum is moved with a random sine-function in each dimension. 
    '''
    dim = 2
    n_chg_periods = 3000  # 100
    opts = []
    aplitudes = np.random.randint(5, 50, dim)
    width_factors = np.random.rand(dim)
    for p in range(n_chg_periods):
        new_opt = np.zeros(dim)
        for d in range(dim):
            new_opt[d] = aplitudes[d] * \
                np.sin(width_factors[d] * p) + aplitudes[d] + p

        opts.append(copy.copy(new_opt))

    opts = np.array(opts)
    plot_scatter(opts)


def plot_scatter(points):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = range(len(points))
    ax.scatter(points[:, 0], points[:, 1],
               marker='x', c=color)

    plt.title('Optimum position during time')
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.show()
###############################################################################


if __name__ == '__main__':
    # create_and_plot_different_movements()
    # create_and_plot_random_sine_movement()
    create_str_problems()
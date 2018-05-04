'''
Benchmarks where only the position of the fitness landscape is changed but not
the fitness level.
 
Created on May 4, 2018

@author: ameier
'''
'''
Contains the function to create the dynamic optimization problems for the
Predictorcomparison (GECCO: PSO with prediction)

In addition: some functions to test and plot new movements

Created on Jan 17, 2018

@author: ameier
'''
#from dynamicopt.utils.utils_print import get_current_day_time
#from main.fitnessfunctions import sphere, rosenbrock, rastrigin

import copy
import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from utils.fitnessfunctions import sphere, rosenbrock, rastrigin
from utils.utils_print import get_current_day_time


def create_and_plot_different_movements():
    '''
    Creates different movements and plots them. Only for testing new movements.
    '''
    dim = 2
    n_changes = 3000
    opts = []
    for c in range(n_changes):
        # movement same in all dimensions
        new_opt = np.zeros(dim)
        for d in range(dim):
            new_opt[d] = 30 * np.sin(0.25 * c) + 30 + c

        # different movement for each dimension
        new_opt = np.zeros(dim)
        new_opt[1] = 30 * np.sin(0.25 * c) + 30 + c
        # sinus linear nach oben
        new_opt[0] = c
        # sinus als Sättigungskurve nach oben
        new_opt[0] = c + (0.1 * c**2)
        # 8er-Kurve linear nach oben
        new_opt[0] = 15 * np.sin(0.5 * c) + 15 + c
        # 8er-Kurve als Sättigungskurve nach oben
        new_opt[0] = 15 * np.sin(0.5 * c) + 15 + (0.1 * c**2)

        opts.append(copy.copy(new_opt))

    opts = np.array(opts)
    plot_scatter(opts)


def create_and_plot_random_sine_movement():
    '''
    Global optimum is moved with a random sine-function in each dimension. 
    '''
    dim = 2
    n_changes = 3000  # 100
    opts = []
    aplitudes = np.random.randint(5, 50, dim)
    width_factors = np.random.rand(dim)
    for c in range(n_changes):
        new_opt = np.zeros(dim)
        for d in range(dim):
            new_opt[d] = aplitudes[d] * \
                np.sin(width_factors[d] * c) + aplitudes[d] + c

        opts.append(copy.copy(new_opt))

    opts = np.array(opts)
    plot_scatter(opts)


def plot_scatter(points):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = range(len(points))
    ttt = ax.scatter(points[:, 0], points[:, 1],
                     marker='x', c=color)

    plt.title('Optimum position during time')
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.show()


def create_str_problems():
    '''
    Call this function to create and save benchmark functions with different settings
    For each change the position of the global optimum is stored. 

    Computes for each change the global optimum position. The new position
    is computed by adding a movement vector that depends on the change number.

    For the Predictor Comparison (GECCO 2018, PSO with prediction).
    Afterwards extended to re-create the EvoStar data (14.3.18)

    Note: in predictor_comparison.py are the data modified so that they have
    for each generation an entry.

    18.1.18
    '''
    # file name of this data set
    day, time = get_current_day_time()

    # the new optimum is reached by adding "linear_movement_factor" to the old
    # one

    n_changes = 10000
    experiment_name = 'str'
    dims = [2, 5, 10, 20, 50, 100]
    functions = ['sphere', 'rosenbrock', 'rastrigin']
    pos_chng_types = ['linear_pos_ch', 'sine_pos_ch']
    fit_chng_type = 'none_fit_ch'
    conference = "evostar_2018"  # "evostar_2018" or "gecco_2018"
    if conference == "gecco_2018":
        linear_movement_factor = 5
    elif conference == "evostar_2018":
        linear_movement_factor = 2

    folder_path = os.path.expanduser(
        "~/Documents/Promotion/GITs/datasets/Predictorvergleich/" + experiment_name + "/")

    for dim in dims:
        orig_opt_positions = {'sphere': np.array(dim * [0]),
                              'rosenbrock': np.array(dim * [1]),
                              'rastrigin': np.array(dim * [0])}
        for func in functions:
            # store global optimum fitness (stays same because "none_fit_ch"
            global_opt_fit = np.array(n_changes * [0])

            # compute optimum movement
            for pos_chng_type in pos_chng_types:
                if pos_chng_type == 'sine_pos_ch':
                    if conference == "gecco_2018":
                        opts = []
                        np_rand_gen = np.random.RandomState(234012)
                        amplitudes = np_rand_gen.randint(5, 50, dim)
                        width_factors = np_rand_gen.rand(dim)
                        for chg in range(n_changes):
                            # compute movement in all dimensions
                            movement = np.zeros(dim)
                            for d in range(dim):
                                #new_opt[d] = 30 * np.sin(0.25 * c) + 30 + c
                                movement[d] = amplitudes[d] * \
                                    np.sin(width_factors[d] *
                                           chg) + amplitudes[d] + chg
                            # new optimum position
                            new_opt = orig_opt_positions[func] + movement
                            opts.append(copy.copy(new_opt))
                    elif conference == "evostar_2018":
                        opts = []
                        for chg in range(n_changes):
                            step = chg * linear_movement_factor
                            x = orig_opt_positions[func]
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
                elif pos_chng_type == 'linear_pos_ch':
                    if conference == "gecco_2018" or conference == "evostar_2018":
                        opts = []
                        for chg in range(n_changes):
                            movement = np.array(
                                dim * [chg * linear_movement_factor])
                            new_opt = orig_opt_positions[func] + movement
                            opts.append(copy.copy(new_opt))
                opts = np.array(opts)
                # save optimum
                ds_file_name = folder_path + experiment_name + "_" + func + "_d-" + \
                    str(dim) + "_chgs-" + str(n_changes) + "_" + pos_chng_type + "_" + \
                    fit_chng_type + "_" + day + '_' + time + ".npz"
                np.savez(ds_file_name, global_opt_fit=global_opt_fit,
                         global_opt_pos=opts, orig_opt_pos=orig_opt_positions[func])


def get_global_optimum(gen, global_opt_fit):
    return global_opt_fit[gen]


def get_global_optimum_position(gen, global_opt_pos):
    return global_opt_pos[gen]


def original_fitness(x, problem):
    '''
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
    optimum_movement = global_opt_pos_per_gen[gen] - orig_opt_pos
    # move individual, so that its fitness can be computed with the original
    # function
    moved_x = x - optimum_movement
    return original_fitness(moved_x, problem)


if __name__ == '__main__':
    # create_and_plot_different_movements()
    # create_and_plot_random_sine_movement()
    create_str_problems()

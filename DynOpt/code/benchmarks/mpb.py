'''
The moving peaks benchmark following the description in the paper:
Branke, J.: Memory enhanced evolutionary algorithms for changing optimization
problems. In: Congress on Evolutionary Computation (CEC). pp. 1875–1882 (1999)

The data set values are stored per change. TODO anderes anpassen, weil es vorher
 per GENERATIion war? (8.5.18)
Contains functionality to create a MPB data set as well as computing the 
fitness during the runtime.

With "start_creating_problem()" the MPB-Benchmark data sets are created.

Created on Oct 10, 2017

@author: ameier
'''
import copy
import os
import warnings

from dynamicopt.utils.utils_ea import gaussian_mutation
from dynamicopt.utils.utils_print import get_current_day_time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def __create_vector(dimensionality, len_vector, np_random_generator, noise=None):
    '''
    Creates a random vector with specified length (i.e. Euclidean norm).

    https://stackoverflow.com/questions/37577803/generating-random-vectors-of-euclidean-norm-1-in-python (10.10.17)
    @param dimensionality: number of entries within the vector
    @param len_vector: desired length (Euclidean norm) of vector
    @param np_random_generator: random number generator
    @param noise: if noise is not None, instead of random movement the peaks' 
    positions are moved linearly with random noise. "noise" specifies its strength.
    '''
    if noise is None:
        # normal case as defined in the paper
        # the initial random vector
        rnd_vec = np_random_generator.uniform(-1, 1, dimensionality)
        initial_length = np.linalg.norm(rnd_vec)
        # the scaling factors
        scale_f = np.array([initial_length])
        # the random vectors in R^d -> vector has length one
        rnd_vec = rnd_vec / scale_f
        # change vector so that it has desired length
        rnd_vec = rnd_vec * len_vector
        return rnd_vec
    else:
        # linear movement (with noise)
        direction = np.ones(dimensionality)  # linear movement vector
        noisy_direction = gaussian_mutation(
            direction, 0, noise, np_random_generator)
        return noisy_direction

        # =========
        # other tested approaches for movement
        #
        # no movement
        # return np.zeros(dimensionality)
        #


def __create_and_save_mpb_problem__(n_gens, n_dims, n_peaks, len_chg_period, len_movement_vector,
                                    np_random_generator, path_to_file, noise=None):
    '''
    Creates mpb data set for a specific setting.
    '''
    # 2d list: for each peak one row containing a list that contains for each
    # generation the peak's height
    heights = []
    # 2d list: similar to heights
    widths = []
    # 3d list: for each peak one row, containing a list that contains the
    # multi-dimensional position of the peak for that generation
    positions = []
    # 1d list: for each generation the optimum fitness
    global_opt_fit = []
    # 2d list: for each generation: position corresponding to optimal fitness
    global_opt_pos = []

    heigth_severity = 7  # as in initial paper by Branke
    width_severity = 0.1  # as in initial paper by Branke

    # bound for initialization of peak positions
    min_bound = 0
    max_bound = 100
    for gen in range(n_gens):
        if gen == 0:  # first generation
            # initialize position etc.
            init_height = 50.0  # as in initial paper by Branke
            init_width = 0.1  # as in initial paper by Branke

            max_fit = np.finfo(np.float).min  # min. possible float value
            best_position = None
            for peak in range(n_peaks):
                init_position = np_random_generator.uniform(
                    min_bound, max_bound, n_dims)
                heights.append([init_height])
                widths.append([init_width])
                positions.append([copy.deepcopy(init_position)])

                # test whether this peak is the global optimum
                curr_fit = __compute_mpb_fitness(
                    init_position, init_height, init_width,  init_position)
                if curr_fit > max_fit:
                    max_fit = curr_fit
                    best_position = init_position
            global_opt_fit.append(-max_fit)  # minimization problem
            global_opt_pos.append(copy.deepcopy(best_position))
        elif gen % len_chg_period == 0:  # change occured
            max_fit = np.finfo(np.float).min  # min. possible float value
            for peak in range(n_peaks):
                # parameter values of previous change period
                old_height = heights[peak][gen - 1]
                old_width = widths[peak][gen - 1]
                old_position = positions[peak][gen - 1]

                # compute parameter values of current change period
                position_movement = __create_vector(
                    n_dims, len_movement_vector, np_random_generator, noise)
                min_heigth = 1.0
                curr_height = max(min_heigth, old_height + heigth_severity *
                                  np_random_generator.normal(loc=0.0, scale=1.0))
                min_width = 1.0  # should not become smaller than 0
                curr_width = max(min_width, old_width + width_severity *
                                 np_random_generator.normal(loc=0.0, scale=1.0))
                curr_position = old_position + position_movement

                # update problem parameters
                heights[peak].append(curr_height)
                widths[peak].append(curr_width)
                positions[peak].append(copy.deepcopy(
                    curr_position))

                # test whether this peak is the global optimum
                curr_fit = __compute_mpb_fitness(
                    curr_position, curr_height, curr_width,  curr_position)
                if curr_fit > max_fit:
                    max_fit = curr_fit
                    best_position = curr_position
            global_opt_fit.append(-max_fit)
            global_opt_pos.append(copy.deepcopy(best_position))
        else:  # no change
            for peak in range(n_peaks):
                heights[peak].append(heights[peak][-1])
                widths[peak].append(widths[peak][-1])
                positions[peak].append(copy.deepcopy(
                    positions[peak][-1]))
            global_opt_fit.append(global_opt_fit[-1])
            global_opt_pos.append(copy.deepcopy(global_opt_pos[-1]))
    # convert lists to numpy arrays
    heights = np.array(heights)
    widths = np.array(widths)
    positions = np.array(positions)
    global_opt_fit = np.array(global_opt_fit)
    global_opt_pos = np.array(global_opt_pos)
    np.savez(path_to_file, heights=heights, widths=widths, positions=positions,
             global_opt_fit=global_opt_fit, global_opt_pos=global_opt_pos)


def start_creating_problem():
    '''
    Call this function to create and save MPB-functions with different settings
    (generates data for each generation (not only for the changes))
    '''
    folder = os.path.expanduser(
        "~/Documents/Promotion/GITs/datasets/Predictorvergleich/GECCO_2018/mpb/")
    day, time = get_current_day_time()

    # ==================================
    # settings for experiments "mpbrand"
    problem_name = "mpbrand"
    gens = [6000]
    dims = [2, 5, 10, 20, 50, 100]
    peaks = [10]
    noise_strengths = [None]
    lens_movement_vector = [0.6]
    lens_chg_period = [20]
    # ==================================
    # settings for experiments "mpbnoisy"
    #problem_name = "mpbnoisy"
    #gens = [6000]
    #dims = [2, 5, 10, 20, 50, 100]
    #peaks = [10]
    #noise_strengths = [0.0, 0.1, 1.0, 10.0]
    #lens_movement_vector = [0.6]
    #lens_chg_period = [20]
    # ==================================
    for n_gens in gens:
        for n_dims in dims:
            for len_movement_vector in lens_movement_vector:
                for n_peaks in peaks:
                    for noise in noise_strengths:
                        for len_chg_period in lens_chg_period:
                            mpb_np_random_generator = np.random.RandomState(
                                234572)

                            file_name = problem_name + "_d-" + str(n_dims) + "_gen-" + \
                                str(n_gens) + "_lenchgperiod-" + str(len_chg_period) + \
                                "_veclen-" + str(len_movement_vector) + "_peaks-" + \
                                str(n_peaks) + "_noise-" + str(noise) + "_" + day + \
                                "_" + time + ".npz"
                            path_to_file = folder + file_name
                            __create_and_save_mpb_problem__(n_gens, n_dims, n_peaks,
                                                            len_chg_period, len_movement_vector,
                                                            mpb_np_random_generator,
                                                            path_to_file, noise)


def __compute_mpb_fitness(x, height, width, position):
    '''
    Only for internal use.
    Compute value for one specific peak. != fitness

    @param x: individual which fitness should be computed
    @param height,width,position: properties of the specific peak 
    '''
    # compute fitness regarding this peak
    diff = x - position
    try:
        tmp = np.sum(np.square(diff))
        # https://stackoverflow.com/questions/17208567/how-to-find-out-where-a-python-warning-is-from
        warnings.filterwarnings(
            'error', message='overflow encountered in square')
    except RuntimeWarning:
        print("moving-peaks-benchmarks: caught warning", flush=True)
        # compute lowest and largest value that can be feed into np.square
        # without throwing an exception
        max_float = np.finfo(np.float).max
        sqrt_max_float = np.sqrt(max_float)
        min_float = np.finfo(np.float).min
        sqrt_min_float = np.sqrt(-min_float)  # make value positive

        # entries that are too large are made smaller
        indices = np.argwhere(diff > sqrt_max_float)
        diff[indices] = sqrt_max_float
        # entries that are too small are made larger
        indices = np.argwhere(diff < -sqrt_min_float)
        diff[indices] = sqrt_min_float
        # next try for computing square
        tmp = np.sum(np.square(diff))
    divisor = (1 + width * tmp)
    peak_result = height / divisor
    return peak_result


def get_global_mpb_optimum_fitness(gen, global_opt_fit):
    '''
    For global access.
    '''
    return global_opt_fit[gen]


def get_global_mpb_optimum_position(gen, global_opt_pos):
    '''
    For global access.
    '''
    return global_opt_pos[gen]


def compute_fitness(x, gen, heights, widths, positions):
    '''
    For global access.

    Computes fitness for individual in specific generation.
    @param x: individual
    @param gen: current generation
    @param heigts, widths, positions: information loaded from the mpb data

    '''
    peak_results = [__compute_mpb_fitness(x, height[gen], width[gen], pos[gen])
                    for height, width, pos in zip(heights, widths, positions)]
    max_fitness = np.max(peak_results)
    return -max_fitness  # invert mbp, because dynea and dynpso minimize


if __name__ == '__main__':
    start_creating_problem()

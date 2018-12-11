'''
The moving peaks benchmark following the description in the paper:
Branke, J.: Memory enhanced evolutionary algorithms for changing optimization
problems. In: Congress on Evolutionary Computation (CEC). pp. 1875–1882 (1999)

The data set values are stored per change. 
 
Contains functionality to create a MPB data set as well as computing the 
fitness during the runtime.

With "start_creating_problem()" the MPB-Benchmark data sets are created. There 
the desired parameters have to be specified.

Created on Oct 10, 2017

@author: ameier
'''
import copy
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.pardir))

import numpy as np
from utils.utils_ea import gaussian_mutation
from utils.utils_files import get_current_day_time


def __create_vector(dimensionality, len_vector, np_random_generator, noise=None, use_correlation=False, old_movement=None):
    '''
    Creates a random vector with specified length (i.e. Euclidean norm).

    https://stackoverflow.com/questions/37577803/generating-random-vectors-of-euclidean-norm-1-in-python (10.10.17)
    @param dimensionality: number of entries within the vector
    @param len_vector: desired length (Euclidean norm) of vector
    @param np_random_generator: random number generator
    @param noise: if noise is not None, instead of random movement the peaks' 
    positions are moved linearly with random noise. "noise" specifies its strength.
    '''
    if use_correlation and old_movement is not None:
        # newer verion of MPB, like listed in
        #    - CEC tutorial: http://ieee-tf-ecidue.cug.edu.cn/Yang-CEC2017-Tutorial-ECDOP.pdf
        #    - publication of e.g. Irene Moser: "Dynamic Function Optimization: The Moving Peaks Benchmark"

        # convert noise to correlation (in order to have formulas like in the
        # paper
        correlation_factor = 1 - noise
        # the initial random vector
        rnd_vec = np_random_generator.uniform(-1, 1, dimensionality)
        denominator = np.linalg.norm(rnd_vec + old_movement)
        fraction = len_vector / denominator
        rnd_factor = (1 - correlation_factor) * rnd_vec
        deterministic_part = correlation_factor * old_movement
        return fraction * rnd_factor + deterministic_part
    elif noise is None or (use_correlation and old_movement is None):
        # normal case as defined in the paper (or actually the "correlation"-variant
        # is desired but it is only the second point for which no previous
        # movement exists (only from the third on)

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


def __create_and_save_mpb_problem__(min_range, max_range,
                                    n_chg_periods, n_dims, n_peaks, len_movement_vector,
                                    np_random_generator, mpb_peaks_np_random_generator,
                                    path_to_file, noise=None, use_correlation=False):
    '''
    Creates mpb data set for a specific setting.
    '''
    # 2d list: for each peak one row containing a list that contains for each
    # change period the peak's height
    heights = []
    # 2d list: similar to heights
    widths = []
    # 3d list: for each peak one row, containing a list that contains the
    # multi-dimensional position of the peak for that change period
    positions = []
    # 1d list: for each change period the optimum fitness
    global_opt_fit = []
    # 2d list: for each change period: position corresponding to optimal
    # fitness
    global_opt_pos = []

    heigth_severity = 7  # as in initial paper by Branke
    width_severity = 0.1  # as in initial paper by Branke

    # (unequal to dynposbenchmark.py) position of first global optimum
    orig_global_opt_position = []
    # saves for each peak the previous position movement (required for
    # correlation-based MPB variant
    old_movement_per_peak = np.array([None] * n_peaks)
    for chg_period in range(n_chg_periods):
        if chg_period == 0:  # first change period
            # initialize position etc.
            init_height = 50.0  # as in initial paper by Branke
            init_width = 0.1  # as in initial paper by Branke

            max_fit = np.finfo(np.float).min  # min. possible float value
            best_position = None
            for peak in range(n_peaks):
                init_position = np_random_generator.uniform(
                    min_range, max_range, n_dims)
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
            orig_global_opt_position = copy.deepcopy(best_position)
        else:
            max_fit = np.finfo(np.float).min  # min. possible float value
            for peak in range(n_peaks):
                # parameter values of previous change period
                old_height = heights[peak][chg_period - 1]
                old_width = widths[peak][chg_period - 1]
                old_position = positions[peak][chg_period - 1]

                # compute parameter values of current change period
                # height
                min_heigth = 1.0
                curr_height = max(min_heigth, old_height + heigth_severity *
                                  mpb_peaks_np_random_generator.normal(loc=0.0, scale=1.0))
                # width
                min_width = 1.0  # should not become smaller than 0
                curr_width = max(min_width, old_width + width_severity *
                                 mpb_peaks_np_random_generator.normal(loc=0.0, scale=1.0))
                # position
                position_movement = __create_vector(
                    n_dims, len_movement_vector, np_random_generator, noise, use_correlation, old_movement=old_movement_per_peak[peak])
                curr_position = old_position + position_movement
                # if position outside the range, move in opposite direction
                too_small_idcs = curr_position < min_range
                too_large_idcs = curr_position > max_range
                position_movement[too_small_idcs] = position_movement[too_small_idcs] * -1
                position_movement[too_large_idcs] = position_movement[too_large_idcs] * -1
                old_movement_per_peak[peak] = position_movement

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
                    best_position = copy.deepcopy(curr_position)
            global_opt_fit.append(-max_fit)
            global_opt_pos.append(copy.deepcopy(best_position))

    # convert lists to numpy arrays
    heights = np.array(heights)
    widths = np.array(widths)
    positions = np.array(positions)
    global_opt_fit = np.array(global_opt_fit)
    global_opt_pos = np.array(global_opt_pos)
    np.savez(path_to_file, heights=heights, widths=widths, positions=positions,
             global_opt_fit_per_chgperiod=global_opt_fit,
             global_opt_pos_per_chgperiod=global_opt_pos,
             orig_global_opt_pos=orig_global_opt_position)


def start_creating_problem(func_name=None, output_dir_path=None):
    '''
    Call this function to create and save MPB-functions with different settings)
    @param func_name: "mpbnoisy" or "mpbrand" or "mpbcorr"
    '''
    np.random.seed(4)
    # ==================================
    # TODO(exp)
    # "EvoStar_2018" or "GECCO_2018" or "ESANN_2019" (must be equivalent to directory)
    conference = "GECCO_2019"
    min_range = 100
    max_range = 200
    func_name = "mpbcorr" if func_name is None else func_name
    if func_name == "mpbrand":
        # settings for experiments "mpbrand"
        chg_periods = [10000]
        dims = [2, 5, 10, 20, 50, 100]
        dims = [2, 50]
        peaks = [10]
        noise_strengths = [None]
        lens_movement_vector = [0.6]
    elif func_name == "mpbnoisy":
        # settings for experiments "mpbnoisy"
        chg_periods = [10000]
        dims = [2, 5, 10, 20, 50, 100]
        dims = [2, 50]
        peaks = [10]
        noise_strengths = [0.0, 0.1, 1.0, 10.0]
        noise_strengths = [0.0]
        lens_movement_vector = [0.6]
    elif func_name == "mpbcorr":
        # settings for experiments "mpbcorr" (with correlation factor)
        chg_periods = [10000]
        dims = [2, 10]
        peaks = [10]
        lens_movement_vector = [0.6]
        use_correlation = True  # TODO adapt file name
        correlation_factors = [0.95]  # in range [0,1]
        # convert correlation to noise
        noise_strengths = np.subtract(1, correlation_factors)
    # ==================================

    if output_dir_path is None:
        # create output folder for data set if not existing
        splitted_path = os.path.abspath(
            os.pardir).split('/')  # ".../DynOpt/code"
        path_to_dynopt = '/'.join(splitted_path[:-1])

        output_dir_path = path_to_dynopt + "/datasets/" + conference + "/" + func_name
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        output_dir_path = output_dir_path + "/"

    day, time = get_current_day_time()

    for n_periods in chg_periods:
        mpb_np_random_generator = np.random.RandomState(2342)
        mpb_peaks_np_random_generator = np.random.RandomState(928)

        for n_dims in dims:
            for len_movement_vector in lens_movement_vector:
                for n_peaks in peaks:
                    for noise in noise_strengths:
                        file_name = func_name + "_d-" + str(n_dims) + "_chgperiods-" + \
                            str(n_periods) + "_veclen-" + str(len_movement_vector) + \
                            "_peaks-" + str(n_peaks) + "_noise-" + str(noise).lower() +\
                            "_" + day + "_" + time + ".npz"
                        path_to_file = output_dir_path + file_name
                        __create_and_save_mpb_problem__(min_range, max_range,
                                                        n_periods, n_dims, n_peaks,
                                                        len_movement_vector,
                                                        mpb_np_random_generator,
                                                        mpb_peaks_np_random_generator,
                                                        path_to_file, noise, use_correlation)


def __compute_mpb_fitness(x, height, width, position):
    '''
    Only for internal use.
    Compute value for one specific peak. != fitness (must be multiplied by -1 
    to have a minimization problem instead of a maximization problem)

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


def compute_fitness(x, gen, heights, widths, positions):
    '''
    For global access.

    Computes fitness for individual in specific generation.
    @param x: individual
    @param gen: current generation
    @param heigts, widths, positions: properties per generation

    '''
    peak_results = [__compute_mpb_fitness(x, height[gen], width[gen], pos[gen])
                    for height, width, pos in zip(heights, widths, positions)]
    max_fitness = np.max(peak_results)
    return -max_fitness  # invert mbp, because dynea and dynpso minimize


if __name__ == '__main__':
    start_creating_problem()

'''
Dynamic Sine Benchmark: optimum position follows in each dimension a separate
function that is composed by multiplying randomly parameterized sine functions.

The complexity of the benchmark can be determined by
    - velocity: medium distance between function values
    - curviness: number of extremes within base interval [0,2pi) 
    
Created on Jan 10, 2019

@author: ameier
'''
import math
import sys

import numpy as np


def generate_sine_fcts_for_multiple_dimensions(dims, n_chg_periods, seed,
                                               l_bound, u_bound, desired_curv,
                                               desired_med_vel):
    '''
    Generates the optimum movement separately for each dimension.

    @return 2d array: for each change period the optimum position (each row 
    contains one position)
    '''
    np.random.seed(seed)

    values_per_dim = []
    for d in range(dims):
        print("\n\nd: ", d)
        values, fcts_params = generate_sine_fcts_for_one_dimension(
            n_chg_periods, desired_curv, desired_med_vel, l_bound, u_bound)
        values_per_dim.append(values)

    data = np.transpose(np.array(values_per_dim))
    return data


def generate_sine_fcts_for_one_dimension(n_data, desired_curv,
                                         desired_med_vel, l_bound, u_bound):
    '''
    variables used in the following:
    a: amplitude
    b: frequency
    c: phase shift
    '''

    do_print = True  # plot data?

    #============================================
    assert l_bound < u_bound

    # number of functions to multiply
    min_n_functions = 1
    max_n_functions = 4  # TODO (exe) adapt if desired
    n_functions = np.random.randint(min_n_functions, max_n_functions)
    print("n_functions: ", n_functions)
    assert max_n_functions < desired_curv

    # maximum amplitude must match value range (apply root to compute max_a)
    value_range = u_bound - l_bound
    max_a = math.floor(math.pow(value_range / 2, 1 / n_functions))
    print("max_a: ", max_a)

    # horizontal movement only positive, since positive and negative values
    # have the same effect
    min_c = 0
    max_c = 2 * math.pi

    # probabilities that a or b, respectively, are in [0,1) (chosen
    # arbitrarily)
    a_prob_smaller_one = 0.2
    b_prob_smaller_one = 0.5

    # y-movement for the composed function (not for the single ones)
    # will be set in correct_range()
    y_movement = 0

    #============================================
    # number sampling points in base interval [0,2pi)
    n_base_time_points = 100
    # equidistant points
    step_size = (2 * math.pi) / n_base_time_points
    # according to Shannon-Nyquist
    max_possible_b = 1 / (2 * step_size)
    max_b = desired_curv / 2  # according to desired curvature
    assert max_b < max_possible_b
    # max number extremes in an interval with length 2pi
    max_possible_curv = 2 * max_possible_b
    assert desired_curv < max_possible_curv
    if desired_curv == 0:
        # TODO implement this case
        print("case desired_curv==0 not yet implemented")
        sys.exit()

    # define sampling points
    max_time_point = math.ceil(n_data * step_size)
    time = np.arange(0, max_time_point, step_size)
    time = time[:n_data]

    #============================================

    # 2d array: each row consists of 5 values: the values for parameters a,b,c
    # of the base function: a*sin(bx+c), and the fourth value is the vertical
    # movement of the composed function, and the fifth value is the overall
    # scaling of the composed function.
    fcts = []
    a_idx = 0
    b_idx = 1  # b is second parameter in the "fcts"-array
    c_idx = 2
    y_movement_idx = 3
    scaling_idx = 4

    overall_scale = 1  # default; will be updated in correct_velocity
    compos_curv = 0  # curviness of composition function
    for i in range(n_functions):
        print("\n function: ", i)
        a = 0
        b = 0
        # number of functions that also have to be constructed
        n_remaining_fcts = (n_functions - i) - 1
        while a == 0:  # must not be zero otherwise whole function is zero
            a = get_a_or_b(max_a, a_prob_smaller_one)
        while b == 0:  # must not be zero otherwise function has constant value
            # ...compute the maximum possible frequency (for each remaining
            # function at least one extreme must remain; division by 2 to
            # convert curviness to frequency)
            curr_max_b = max_b - n_remaining_fcts / 2 - compos_curv / 2
            assert curr_max_b >= 0, "curr_max_b: " + str(curr_max_b)
            if i == n_functions - 1:
                # last time; b must be chosen so that requirements are
                # fulfilled
                b = curr_max_b
            elif curr_max_b == 0 and i != n_functions:
                # frequency reached but still functions that have to be created
                print("frequency reached but still functions that have to be created")
                sys.exit()
            else:
                b = get_a_or_b(curr_max_b, b_prob_smaller_one)

        # horizontal movement
        c = np.random.uniform(min_c, max_c)
        # store (preliminary) values, b probably is corrected afterwards
        fcts.append([a, b, c, y_movement, overall_scale])
        # correct b
        compos_curv, fcts = correct_frequency(fcts, time, n_base_time_points,
                                              desired_curv, n_remaining_fcts,
                                              a_idx, b_idx, c_idx, do_print)

    fcts = np.array(fcts)

    # correct velocity
    _, vel_fcts = correct_velocity(fcts, time, desired_med_vel, scaling_idx)
    # correct range
    final_vals, final_fcts = correct_range(vel_fcts, time, l_bound, u_bound,
                                           y_movement_idx)

    #============================================
    # test number data
    assert len(final_vals) == n_data, "len(final_vals):" + \
        str(len(final_vals)) + " n_data: " + str(n_data)
    # test range
    assert np.min(final_vals) >= l_bound, "min: " + str(np.min(final_vals))
    assert np.max(final_vals) <= u_bound, "max: " + str(np.max(final_vals))

    # test curviness
    min_vel, max_vel, med_vel, _ = compute_velocity_analytically(final_vals)
    final_base_vals = final_vals[:n_base_time_points]
    curr_curv = compute_curviness_analytically(final_base_vals)
    assert desired_curv == curr_curv, "curv: " + str(curr_curv)

    if do_print:
        import matplotlib.pyplot as plt
        print()
        # curviness
        print("curr_curv: ", curr_curv)
        # min, max, med velocity
        print("min, max, med: ", (min_vel, max_vel, med_vel))
        # function parameters
        print("final_fcts: ")
        print(final_fcts)
        # function values within base interval
        # plt.plot(final_base_vals)
        #plt.title("for base time")
        # plt.show()
        # all function values
        # plt.plot(final_vals)
        #plt.title("for all time steps")
        # plt.show()

    return final_vals, final_fcts


def get_a_or_b(max_val, perc_smaller_one):
    '''
    Choose random value for parameter (a or b). 

    @param perc_smaller_one: percentage with that a value in [0,1) should be 
    returned. With probability 1-perc_smaller_one the value is in [1,max_val).
    '''
    min = 0  # neither a nor b should be smaller than 0
    if False:  # TODO
        # if max_val > 1:
        if np.random.rand() < perc_smaller_one:
            # param in [0,1)
            return np.random.rand()
        else:
            min_val = 1
            return np.random.uniform(min_val, max_val + 1)

    else:
        return np.random.rand() * (max_val - min) + min


def compute_single_curviness(a, b, c, time, n_base_time_points):
    # compute current function values for single function
    single_vals = a * np.sin(b * time + c)
    # compute curviness of single function
    single_base_vals = single_vals[:n_base_time_points]
    single_curv = compute_curviness_analytically(single_base_vals)
    return single_curv


def compute_composition_curviness(fcts, time, n_base_time_points):
    compos_vals = compute_vals_for_fct(fcts, time)
    compos_base_vals = compos_vals[:n_base_time_points]
    return compute_curviness_analytically(compos_base_vals)


def compute_curviness_analytically(values):
    '''
    @param values: for each time step the function values
    '''
    diff_vals = values[1:] - values[:-1]
    signs = np.sign(diff_vals)
    sign_chgs = 0
    n_comparisons = 0
    for i in range(1, len(signs)):
        n_comparisons += 1
        if (signs[i] == -1 and signs[i - 1] == 1) or \
                (signs[i] == 1 and signs[i - 1] == -1):
            sign_chgs += 1
    return sign_chgs


def compute_velocity_analytically(values):
    # difference between succeeding points
    diff_vals = values[1:] - values[:-1]
    abs_diffs = np.abs(diff_vals)
    min_vel = np.min(abs_diffs)
    max_vel = np.max(diff_vals)
    med_vel = np.median(abs_diffs)
    return min_vel, max_vel, med_vel, diff_vals


def compute_vals_for_fct(fcts, time):
    '''
    Computes product of component sine-functions with specified parameters

    @param fct: 2d array: for each component function one row containing five
    values: first: a, second: b, third: c (for base function a*sin(bx+c))
    fourth: vertical movement, fifth: overall scaling
    @return 1d array for each time step the function value
    '''
    values = np.ones(len(time))

    for f in fcts:
        values *= f[0] * np.sin(f[1] * time + f[2])
    return values * f[4] + f[3]


def correct_velocity(fcts, time, desired_med_vel, scaling_idx):
    '''
    Corrects median velocity.

    @param fcts: params for already specified functions
    @param desired_min_vel: desired minimum velocity of overall function
    '''
    # current function values
    values = compute_vals_for_fct(fcts, time)
    # current minimum/medium velocity
    _, _, med_vel, _ = compute_velocity_analytically(values)
    # desired factor for scaling the composition function
    missing_med_vel_ratio = desired_med_vel / med_vel
    # store scaling factor
    fcts[:, scaling_idx] = missing_med_vel_ratio
    # compute corrected function values and velocity
    corrected_values = compute_vals_for_fct(fcts, time)
    _, _, curr_med, _ = compute_velocity_analytically(corrected_values)
    # test
    assert abs(desired_med_vel - curr_med) < 0.01,  "curr_med: " + str(
        curr_med) + " desired_med_vel: " + str(desired_med_vel)

    return corrected_values, fcts


def correct_range(fcts, time, l_bound, u_bound, y_movement_idx):
    '''
    Sets parameter for y_movement (vertical movement) to comply with the 
    specified range.
    Might not be possible if desired velocity is very high. Then a larger range 
    has to be chosen.
    '''
    # current function values and range
    values = compute_vals_for_fct(fcts, time)
    curr_min = np.min(values)
    curr_max = np.max(values)
    curr_range = curr_max - curr_min
    # maximum possible range
    possible_range = u_bound - l_bound
    if curr_range > possible_range:
        # nothing possible to do
        print("curr_min: ", curr_min)
        print("curr_max: ", curr_max)
        raise Exception("Current range is larger than possible range. No" +
                        " correction possible, otherwise it is not possible" +
                        " to comply with the velocity.")
    elif curr_range == possible_range:
        # no vertical movement necessary
        fcts[:, y_movement_idx] = 0
    else:
        # place function in the middle of the possible range
        middle_of_range = l_bound + possible_range / 2
        curr_middle = curr_min + curr_range / 2
        movement = middle_of_range - curr_middle
        fcts[:, y_movement_idx] = movement
    # compute corrected function value
    corrected_values = compute_vals_for_fct(fcts, time)
    # test
    assert np.min(corrected_values) >= l_bound
    assert np.max(corrected_values) <= u_bound

    return corrected_values, fcts


def correct_frequency(fcts, time, n_base_time_points,
                      desired_curv, n_remaining_fcts, a_idx, b_idx, c_idx, do_print):
    '''
    Corrects frequency of first generated single function if the curviness of
    the composition function of all functions generated so far is too large.
    Also it is corrected when the last single function was generated but the 
    curviness is still too small.
    '''
    # index of function of which the frequency is adapted to correct the
    # frequency of the composition function
    f_idx = 0

    # compute curviness of composition function
    compos_curv = compute_composition_curviness(fcts, time, n_base_time_points)

    if do_print:
        print("compos_curv:", compos_curv)
        # compos_curv is different to sum of frequencies (next line)
        sum_of_freqcs = np.sum(fcts, axis=0)[b_idx] * 2
        print("sum_of_freqcs:", sum_of_freqcs)

    # correct b if curvature of composition function already is too large
    n_loops = 0
    while compos_curv > desired_curv or (n_remaining_fcts == 0 and compos_curv < desired_curv):
        # correction term for b (division by 2 to convert curvature into
        # frequency)
        correction_term = (desired_curv - (compos_curv + n_remaining_fcts)) / 2
        old_b = fcts[f_idx][b_idx]
        # correct b in nearly arbitrary function
        if correction_term != 0:
            # convert list to array to use numpy operations (slicing)
            tmp_fcts = np.array(fcts)
            if correction_term < 0:  # decrease b
                # search for functions that have a large enough b so that the
                # correction produces no negative b
                # 2-d array: [1, #functions that comply with condition]
                f_idcs = np.argwhere(
                    tmp_fcts[:, b_idx] > abs(correction_term))
                # choose an arbitrary function
                f_idx = f_idcs[0, np.random.randint(len(f_idcs[0]))]
            else:  # increase b
                f_idx = np.random.randint(len(fcts))
            fcts[f_idx][b_idx] += correction_term

        if do_print:
            print("old_b: ", old_b)
            print("correction: ", correction_term)
            print("new b: ", fcts[f_idx][b_idx])

        assert fcts[f_idx][b_idx] > 0
        compos_curv = compute_composition_curviness(
            fcts, time, n_base_time_points)
        n_loops += 1
        if n_loops > 20:
            print("sine_generator.correct_frequency(): DSB seems not be able to" +
                  " correct the frequency. Therefore the process is terminated.")
            sys.exit()

    if do_print:
        print("new compos_curv: ", compos_curv)
    return compos_curv, fcts


def start_generation():
    seed = 4  # None  # 53
    dims = 13
    n_data = math.ceil(2 * math.pi * 10 * 100)
    # 10 extremes in base interval [0, pi], ten in [0,2pi]
    desired_curv = 15
    desired_med_vel = 0.5
    l_bound = 0
    u_bound = 300

    _ = generate_sine_fcts_for_multiple_dimensions(dims, n_data, seed,
                                                   l_bound, u_bound, desired_curv,
                                                   desired_med_vel)


if __name__ == '__main__':
    start_generation()

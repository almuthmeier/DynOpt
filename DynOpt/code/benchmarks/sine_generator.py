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
import numpy as np


def generate_sine_fcts_for_multiple_dimensions(dims, n_chg_periods, seed,
                                               min_val, max_val, desired_curv,
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
            n_chg_periods, desired_curv, desired_med_vel, min_val, max_val)
        values_per_dim.append(values)

    data = np.transpose(np.array(values_per_dim))
    return data


def generate_sine_fcts_for_one_dimension(n_data, desired_curv,
                                         desired_med_vel, min_val, max_val):
    '''
    variables used in the following:
    a: amplitude
    b: frequency
    c: phase shift
    '''

    do_print = False  # plot data?

    #============================================
    assert min_val < max_val

    # number of functions to multiply
    min_n_functions = 1
    max_n_functions = 4  # TODO (exe) adapt if desired
    n_functions = np.random.randint(min_n_functions, max_n_functions)
    print("n_functions: ", n_functions)

    # maximum amplitude must match max_val (apply root to compute max_a)
    value_range = max_val - min_val
    max_a = math.floor(math.pow(value_range / 2, 1 / n_functions))
    print("max_a: ", max_a)

    # allowed frequency depends only on curviness
    max_b = desired_curv / 2

    # only positive, since positive and negative horizontal movement are the
    # same
    min_c = 0
    max_c = 2 * math.pi

    # probabilities that a or b, respectively, are in [0,1) (chosen
    # arbitrarily)
    a_prob_smaller_one = 0.2
    b_prob_smaller_one = 0.5

    # y-movement for the composed function (not for the single ones)
    # Is chosen so that min_val really is the minimum value
    # "math.pow(max_a, n_functions)" maximally possible amplitude
    y_movement = min_val + math.pow(max_a, n_functions)
    print("y_movement: ", y_movement)
    assert math.pow(max_a, n_functions) <= max_val

    #============================================
    # determine step size with Nyquist–Shannon sampling theorem
    # (step size must match curvature (and frequency))
    step_size = 1 / (2 * max_b)
    # step size must me lower than the computed boundary (here it is 1/8th
    # smaller (chosen arbitrarily))
    step_size = step_size - (1 / 8) * step_size
    print("step_size: ", step_size)

    # define sampling points
    max_time_point = math.ceil(n_data * step_size)
    time = np.arange(0, max_time_point, step_size)
    time = time[:n_data]

    # number of time points in base interval [0, 2*pi)
    n_base_time_points = len(time[time < 2 * math.pi])
    print("n_base_time_points: ", n_base_time_points)

    #============================================

    # 2d array: each row consists of 5 values: the values for parameters a,b,c
    # of the base function: a*sin(bx+c), and the fourth value is the vertical
    # movement of the composed function, and the fifth value is the overall
    # scaling of the composed function.
    fcts = []
    b_idx = 1  # b is second parameter in the "fcts"-array

    overall_scale = 1  # default; will be updated in correct_params
    for i in range(n_functions):
        a = 0
        b = 0
        while a == 0:  # must not be zero otherwise whole function is zero
            a = get_a_or_b(max_a, a_prob_smaller_one)
        while b == 0:  # must not be zero otherwise function has constant value
            # compute current curviness in order to...
            summed_frequency = 0 if fcts == [] else np.sum(fcts, axis=0)[b_idx]
            # ...compute the maximum possible frequency
            curr_max_b = max_b - summed_frequency
            print("summed_frequency: ", summed_frequency)
            print("curr_max_b: ", curr_max_b)
            if i == n_functions - 1:
                # last time; b must be chosen so that requirements are
                # fulfilled
                b = curr_max_b
            else:
                b = get_a_or_b(curr_max_b, b_prob_smaller_one)
            print("chosen b: ", b)

        c = np.random.uniform(min_c, max_c)
        fcts.append([a, b, c, y_movement, overall_scale])
        if do_print:
            # compute current function values
            tmp_vals = a * np.sin(b * time + c)
            min_vel, max_vel, med_vel, _ = compute_velocity_analytically(
                tmp_vals)
            print("min_vel, max_vel, med_vel: ", (min_vel, max_vel, med_vel))
            print("a, b, c: ", a, ", ", b, ", ", c)

            tmp_base_vals = tmp_vals[:n_base_time_points]
            # plt.plot(tmp_vals)
            tmp_curv = compute_curviness_analytically(tmp_base_vals)
            print("tmp_curv: ", tmp_curv)

    fcts = np.array(fcts)

    # correct velocity (after that "fcts" must not be used, since the
    # parameters could not be corrected, only the function values)
    final_vals, final_fcts = correct_params(fcts, time, desired_med_vel)

    #============================================
    # test number data
    assert len(final_vals) == n_data, "len(final_vals):" + \
        str(len(final_vals)) + " n_data: " + str(n_data)
    # test range
    assert np.min(final_vals) >= min_val, "min: " + str(np.min(final_vals))
    assert np.max(final_vals) <= max_val, "max: " + str(np.max(final_vals))
    # test curviness
    min_vel, max_vel, med_vel, _ = compute_velocity_analytically(final_vals)
    #base_time = np.arange(0, 2 * math.pi, 0.1)
    final_base_vals = final_vals[:n_base_time_points]
    curr_curv = compute_curviness_analytically(final_base_vals)
    assert desired_curv == curr_curv, "curv: " + str(curr_curv)

    if do_print:
        print()
        import matplotlib.pyplot as plt
        # curviness
        print("curr_curv: ", curr_curv)
        # min, max, med velocity
        print("min, max, med: ", (min_vel, max_vel, med_vel))
        # function parameters
        print("final_fcts: ")
        print(final_fcts)
        # function values within base interval
        plt.plot(final_base_vals)
        plt.title("for base time")
        plt.show()
        # all function values
        plt.plot(final_vals)
        plt.title("for all time steps")
        plt.show()

    return final_vals, final_fcts


def get_a_or_b(max_val, perc_smaller_one):
    '''
    Choose random value for parameter (a or b). 

    @param perc_smaller_one: percentage with that a value in [0,1) should be 
    returned. With probability 1-perc_smaller_one the value is in [1,max_val).
    '''
    if np.random.rand() > perc_smaller_one:
        # param in [0,1)
        return np.random.rand()
    else:
        min_val = 1
        return np.random.uniform(min_val, max_val + 1)


def compute_curviness_analytically(values):
    '''
    @param values: for each time step the function values
    '''
    import matplotlib.pyplot as plt
    plt.plot(values)
    plt.show()

    diff_vals = values[1:] - values[:-1]
    signs = np.sign(diff_vals)
    sign_chgs = 0
    n_comparisons = 0
    for i in range(1, len(signs)):
        n_comparisons += 1
        if (signs[i] == -1 and signs[i - 1] == 1) or \
                (signs[i] == 1 and signs[i - 1] == -1):
            sign_chgs += 1
    print("n_comparisons: ", n_comparisons)
    print("len(signs): ", len(signs))
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


def correct_params(fcts, time, desired_med_vel):
    '''
    Corrects medium velocity (also minimum vel. would be possible)

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
    fcts[:, -1] = missing_med_vel_ratio
    # compute corrected function values and velocity
    corrected_values = compute_vals_for_fct(fcts, time)
    _, _, curr_med, _ = compute_velocity_analytically(corrected_values)
    # print and test
    assert abs(desired_med_vel - curr_med) < 0.01,  "curr_med: " + str(
        curr_med) + " desired_med_vel: " + str(desired_med_vel)

    return corrected_values, fcts


def start_generation():
    seed = 4  # None  # 53
    dims = 2
    n_data = math.ceil(2 * math.pi * 10 * 100)
    # 10 extremes in base interval [0, pi], ten in [0,2pi]
    desired_curv = 10
    desired_med_vel = 0.5
    min_val = 0
    max_val = 100

    _ = generate_sine_fcts_for_multiple_dimensions(dims, n_data, seed,
                                                   min_val, max_val, desired_curv,
                                                   desired_med_vel)


if __name__ == '__main__':
    start_generation()

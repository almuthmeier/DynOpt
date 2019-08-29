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
                                               desired_min_vel, desired_med_vel):
    '''
    Generates the optimum movement separately for each dimension.

    @return 2d array: for each change period the optimum position (each row 
    contains one position)
    '''
    np.random.seed(seed)

    values_per_dim = []
    for d in range(dims):
        print("d: ", d)
        values, fcts_params = generate_sine_fcts_for_one_dimension(
            n_chg_periods, desired_curv, desired_min_vel,
            desired_med_vel, min_val, max_val)
        values_per_dim.append(values)

    data = np.transpose(np.array(values_per_dim))
    return data


def generate_sine_fcts_for_one_dimension(n_data, desired_curv, desired_min_vel,
                                         desired_med_vel, min_val, max_val):
    '''
    variables used in the following:
    a: amplitude
    b: frequency
    c: phase shift
    '''

    do_print = False  # plot data?
    #============================================
    step_size = math.pi / 30  # TODO (exe) must fit to curvature
    # define sampling points
    max_time_point = math.ceil(n_data * step_size)
    time = np.arange(0, max_time_point, step_size)
    time = time[:n_data]

    #============================================

    # number of functions to multiply
    min_n_functions = 1
    max_n_functions = 4  # TODO (exe) adapt if desired
    n_functions = np.random.randint(min_n_functions, max_n_functions)
    print("n_functions: ", n_functions)

    # TODO (exe) must match max_val
    max_a = 4

    # depends only on curviness
    max_b = desired_curv / 2

    # only positive, since positive and negative movement are the same
    min_c = 0
    max_c = 2 * math.pi

    # probabilities that a or b, respectively, are in [0,1)
    a_prob_smaller_one = 0.2
    b_prob_smaller_one = 0.5

    # y-movement for the composed function (not for the single ones)
    # Is chosen so that min_val really is the minimum value
    # "max(...)" because of oscillating function the minimum value of the
    # function might be the negative (maximally possible) amplitude
    # "math.pow(max_a, n_functions)" maximally possible amplitude
    y_movement = max(min_val, math.pow(max_a, n_functions))
    assert math.pow(max_a, n_functions) <= max_val

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
            if i == n_functions - 1:
                # last time; b must be chosen so that requirements are
                # fulfilled
                b = curr_max_b
            else:
                b = get_a_or_b(curr_max_b, b_prob_smaller_one)

        c = np.random.uniform(min_c, max_c)
        fcts.append([a, b, c, y_movement, overall_scale])
        # compute current function values
        vals = a * np.sin(b * time + c)
        if do_print:
            min_vel, max_vel, med_vel, _ = compute_velocity_analytically(
                vals)
            print("curv: ", compute_curviness_analytically(vals))
            print("min, max, med: ", (min_vel, max_vel, med_vel))
            print("a, b, c: ", a, ", ", b, ", ", c)

    fcts = np.array(fcts)

    if do_print:
        import matplotlib.pyplot as plt
        print("\nfor base time:")
        base_time = np.arange(0, 2 * math.pi, 0.1)
        vals = compute_vals_for_fct(fcts, base_time)
        min_vel, max_vel, med_vel, _ = compute_velocity_analytically(vals)
        print("\ncurv: ", compute_curviness_analytically(vals))
        print("min, max, med: ", (min_vel, max_vel, med_vel))
        plt.plot(vals)
        plt.title("for base time")
        plt.show()

        print("\nfor all time steps")
        vals = compute_vals_for_fct(fcts, time)
        min_vel, max_vel, med_vel, _ = compute_velocity_analytically(vals)
        print("\ncurv: ", compute_curviness_analytically(vals))
        print("min, max, med: ", (min_vel, max_vel, med_vel))
        plt.plot(vals)
        plt.title("for all time steps")
        plt.show()

    # correct velocity (after that "fcts" must not be used, since the
    # parameters could not be corrected, only the function values)

    final_vals, new_fcts = correct_params(
        fcts, time, desired_min_vel, desired_med_vel)
    if do_print:
        print("\nfor all time steps (after correction)")
        min_vel, max_vel, med_vel, _ = compute_velocity_analytically(
            final_vals)
        base_time = base_time = np.arange(0, 2 * math.pi, 0.1)
        final_base_vals = final_vals[:len(base_time)]
        print("\ncurv (base): ", compute_curviness_analytically(final_base_vals))
        print("min, max, med: ", (min_vel, max_vel, med_vel))
        plt.plot(final_vals)
        plt.title("all time steps (after correction)")
        plt.show()

    assert len(final_vals) == n_data, "len(final_vals):" + \
        str(len(final_vals)) + " n_data: " + str(n_data)
    return final_vals, new_fcts


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
        return np.random.uniform(min_val, max_val)


def compute_curviness_analytically(values):
    '''
    @param values: for each time step the function values
    '''
    diff_vals = values[1:] - values[:-1]
    signs = np.sign(diff_vals)
    sign_chgs = 0
    for i in range(1, len(signs)):
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


def correct_params(fcts, time, desired_min_vel, desired_med_vel):
    '''
    Corrects medium velocity (also minimum vel. would be possible)

    @param fcts: params for already specified functions
    @param desired_min_vel: desired minimum velocity of overall function
    '''
    correct_min = False
    values = compute_vals_for_fct(fcts, time)

    # current minimum/medium velocity
    min_vel, _, med_vel, diff_vals = compute_velocity_analytically(values)

    # correct min vel
    if correct_min:
        # should not be used in most cases since function values will be
        # overly large afterwards.
        missing_min_vel = desired_min_vel - min_vel
        missing_min_vel_ratio = desired_min_vel / min_vel
        if missing_min_vel > 0:
            # to small velocity
            diff_vals *= missing_min_vel_ratio

        # testing
        curr_min = np.min(np.abs(diff_vals))
        assert abs(desired_min_vel - curr_min) < 0.01,  "curr_min: " + str(
            curr_min) + " desired_min_vel: " + str(desired_min_vel)
    else:  # correct median velocity (may be too large or too small)

        # desired factor for scaling the composition function
        missing_med_vel_ratio = desired_med_vel / med_vel

        if False:  # correct differences (more complicated, used until 29.8.19)
            diff_vals *= missing_med_vel_ratio  # adapt all differences accordingly
            # update function values
            corrected_values = update_function_values_from_diff_values(
                values, diff_vals)
            # testing
            curr_med = np.median(np.abs(diff_vals))
            curr_min = np.min(np.abs(diff_vals))
            curr_max = np.max(np.abs(diff_vals))
        else:  # scale function values directly
            fcts[:, -1] = missing_med_vel_ratio
            corrected_values = compute_vals_for_fct(fcts, time)
            curr_min, curr_max, curr_med, _ = compute_velocity_analytically(
                corrected_values)

        print("curr_min/max/med: ", curr_min, ", ", curr_max, ", ", curr_med)
        assert abs(desired_med_vel - curr_med) < 0.01,  "curr_med: " + str(
            curr_med) + " desired_med_vel: " + str(desired_med_vel)

    return corrected_values, fcts


def update_function_values_from_diff_values(old_values, new_diff_vals):
    n_vals = len(old_values)
    new_vals = np.zeros(n_vals)
    # first value is equal to old value (because it is the starting point)
    new_vals[0] = old_values[0]
    # add differences to function values (beginning from starting point)
    for i in range(1, n_vals):
        new_vals[i] = new_vals[i - 1] + new_diff_vals[i - 1]

    # testing (differences should be equal before and after correction of vel.)
    test_diffs = new_vals[1:] - new_vals[:-1]
    assert np.sum(np.abs(test_diffs - new_diff_vals)) < 0.1
    return new_vals


def start_generation():
    seed = 4  # None  # 53
    dims = 2
    n_data = math.ceil(2 * math.pi * 10 * 100)
    desired_curv = 10  # five extremes in base interval [0, pi], ten in [0,2pi]
    desired_min_vel = 0.5  # no longer used
    desired_med_vel = 0.5
    min_val = 0
    max_val = 100

    _ = generate_sine_fcts_for_multiple_dimensions(dims, n_data, seed,
                                                   min_val, max_val, desired_curv,
                                                   desired_min_vel, desired_med_vel)


if __name__ == '__main__':
    start_generation()

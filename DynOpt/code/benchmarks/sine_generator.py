'''
Benchmark for sine functions: one function is composed by multiplying 
sine functions parametrized with random parameter values.

Difficulty of benchmark can be determined by
    - minimum acceleration (i.e. minimum distance between function values)
    - number of extremes (within base interval [0,2pi)
    
Created on Jan 10, 2019

@author: ameier
'''
import math

import matplotlib.pyplot as plt
import numpy as np


def generate_sine_fcts_for_one_dimension(n_data, desired_curv, desired_min_acc):
    do_print = False
    #============================================
    print("n_data: ", n_data)
    step_size = math.pi / 30
    print("step_size: ", step_size)
    max_time_point = math.ceil(n_data * step_size)
    print("max_time_point: ", max_time_point)
    time = np.arange(0, max_time_point, step_size)
    time = time[:n_data]

    #============================================
    # indices for amplitude, frequency and movement in "fcts"-array
    a_idx = 0
    b_idx = 1
    c_idx = 2
    #============================================

    # np.random.seed()
    max_n_functions = 3
    max_a = 2
    max_b = desired_curv / 2
    # only positive, since positive and negative movement are the same
    min_c = 0
    max_c = 10
    # probabilities that a or b, respectively, are in [0,1)
    a_prob_smaller_one = 0.2
    b_prob_smaller_one = 0.5

    # 2d array: each row consists of 3 values: the values for parameters a,b,c
    # of the base function: a*sin(bx+c)
    fcts = []
    for i in range(max_n_functions):
        a = 0
        b = 0
        while a == 0:  # must not be zero otherwise is whole function zero
            a = get_a_or_b(max_a, a_prob_smaller_one)
        while b == 0:  # must not be zero otherwise function has constant value
            # compute current curvature in order to...
            summed_frequency = 0 if fcts == [] else np.sum(fcts, axis=0)[b_idx]
            # ...compute the maximum possible frequency
            curr_max_b = max_b - summed_frequency
            if i == max_n_functions - 1:
                # last time; b must be chosen so that requirements are
                # fulfilled
                b = curr_max_b
            else:
                b = get_a_or_b(curr_max_b, b_prob_smaller_one)

        c = np.random.uniform(min_c, max_c)
        fcts.append([a, b, c])
        # compute current function values
        vals = a * np.sin(b * time + c)
        if do_print:
            min_acc, max_acc, med_acc, _ = compute_acceleration_analytically(
                vals)
            print("curv: ", compute_curvature_analytically(vals))
            print("min, max, med: ", (min_acc, max_acc, med_acc))
            print("a, b, c: ", a, ", ", b, ", ", c)

    fcts = np.array(fcts)

    if do_print:
        print("\nfor base time:")
        base_time = np.arange(0, 2 * math.pi, 0.1)
        vals = compute_vals_for_fct(fcts, base_time)
        min_acc, max_acc, med_acc, _ = compute_acceleration_analytically(vals)
        print("\ncurv: ", compute_curvature_analytically(vals))
        print("min, max, med: ", (min_acc, max_acc, med_acc))
        plt.plot(vals)
        plt.title("base time")
        plt.show()

        print("\nfor all time steps")
        vals = compute_vals_for_fct(fcts, time)
        min_acc, max_acc, med_acc, _ = compute_acceleration_analytically(vals)
        print("\ncurv: ", compute_curvature_analytically(vals))
        print("min, max, med: ", (min_acc, max_acc, med_acc))
        plt.plot(vals)
        plt.title("all time steps")
        plt.show()

    # correct acceleration (after that "fcts" must not be used, since the
    # parameters could not be corrected, only the function values)

    final_vals = correct_parms(fcts, time, desired_min_acc)
    if do_print:
        print("\nfor all time steps (after correction)")
        vals = compute_vals_for_fct(fcts, time)
        min_acc, max_acc, med_acc, _ = compute_acceleration_analytically(vals)
        print("\ncurv: ", compute_curvature_analytically(vals))
        print("min, max, med: ", (min_acc, max_acc, med_acc))
        plt.plot(vals)
        plt.title("all time steps (after correction)")
        plt.show()

    assert len(final_vals) == n_data, "len(final_vals):" + \
        str(len(final_vals)) + " n_data: " + str(n_data)
    return final_vals


def get_a_or_b(max_val, perc_smaller_one):
    '''
    Choose random value for parameter (a or b). 

    @param perc_smaller_one: percentage with that a value in [0,1) should be 
    returned. With probability 1- perc_smaller-one the value is in [1,max_val).
    '''
    if np.random.rand() > perc_smaller_one:
        # param in [0,1)
        return np.random.rand()
    else:
        min_val = 1
        return np.random.uniform(min_val, max_val)


def compute_curvature_analytically(values):
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


def compute_acceleration_analytically(values):
    diff_vals = values[1:] - values[:-1]
    abs_diffs = np.abs(diff_vals)
    min_acc = np.min(abs_diffs)
    max_acc = np.max(diff_vals)
    med_acc = np.average(abs_diffs)
    return min_acc, max_acc, med_acc, diff_vals


def compute_vals_for_fct(fcts, time):
    '''
    Computes product of component sine-functions with specified parameters

    @param fct: 2d array: for each component function one row containing three
    values: first: a, second: b, third: c (for base function a*sin(bx+c))
    @return 1d array for each time step the function value
    '''
    values = np.ones(len(time))

    for f in fcts:
        values *= f[0] * np.sin(f[1] * time + f[2])
    return values


def correct_parms(fcts, time, desired_min_acc):
    '''
    Correct minimum acceleration.

    @param fcts: params for already specified functions
    @param desired_min_acc: desired minimum acceleration of overall function
    '''

    values = compute_vals_for_fct(fcts, time)

    # current minimum acceleration
    min_acc, _, _, diff_vals = compute_acceleration_analytically(values)

    # correct min acc.
    missing_min_acc = desired_min_acc - min_acc
    missing_min_acc_ratio = desired_min_acc / min_acc
    if missing_min_acc > 0:
        # to small acceleration
        diff_vals *= missing_min_acc_ratio

    # update function values
    min_acc_corrected_values = update_function_values_from_diff_values(
        values, diff_vals)

    # testing
    curr_min = np.min(np.abs(diff_vals))
    assert abs(desired_min_acc - curr_min) < 0.01,  "curr_min: " + str(
        curr_min) + " desired_min_acc: " + str(desired_min_acc)

    return min_acc_corrected_values


def update_function_values_from_diff_values(old_values, new_diff_vals):
    n_vals = len(old_values)
    new_vals = np.zeros(n_vals)
    # first value is equal to old value (because it is the starting point)
    new_vals[0] = old_values[0]
    # add differences to function values (beginning from starting point)
    for i in range(1, n_vals):
        new_vals[i] = new_vals[i - 1] + new_diff_vals[i - 1]

    test_diffs = new_vals[1:] - new_vals[:-1]
    #print("test_diffs: ", test_diffs)
    #print("new_diff_vals: ", new_diff_vals)
    print("sum: ", np.sum(np.abs(test_diffs - new_diff_vals)))
    return new_vals


def start_generation():
    n_data = math.ceil(2 * math.pi * 10 * 100)
    desired_curv = 10  # five extremes in base interval [0, pi]
    desired_min_acc = 4
    generate_sine_fcts_for_one_dimension(n_data, desired_curv, desired_min_acc)


if __name__ == '__main__':
    start_generation()

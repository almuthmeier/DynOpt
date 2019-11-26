'''
Defines different metrics to quantify the quality of dynamic optimization 
algorithms.

Created on Oct 6, 2017

@author: ameier
'''
import sys

import sklearn.metrics

import numpy as np


def rmse(true_vals, current_vals):
    # https://stackoverflow.com/questions/17197492/root-mean-square-error-in-python/17221808
    # (16.1.19)
    return np.sqrt(sklearn.metrics.mean_squared_error(true_vals, current_vals))


def arr(generations_of_chgperiods, global_opt_fit_per_chgperiod, best_found_fit_per_gen,
        only_for_preds, first_chgp_idx_with_pred, with_abs):
    '''
    ARR (absolute recovery rate) from the paper "Evolutionary dynamic 
    optimization: A survey of the state of the art", Trung Thanh Nguyen et al. 2012.

    @param generations_of_chgperiods:  dictionary containing for each change 
    period (the real ones, not only those the EA has detected) a list with the 
    generation numbers
    @param global_opt_fit_per_chgperiod: 1d numpy array: for each change period
    the global global optimum fitness (for all changes stored in the dataset 
    file, not only for those used in the experiments, i.e. 
    len(global_opt_fit_per_chgperiod) may be larger than len(generations_of_chgperiods)
    @param best_found_fit_per_gen: 1d numpy array: best found fitness for each
    generation
    @param with_abs: True if absolute values of differences should be used
    @return scalar: ARR
    '''
    # =========================================================================
    # for test purposes
    n_gens = 0
    for _, generations in generations_of_chgperiods.items():
        n_gens += len(generations)
    assert n_gens == len(best_found_fit_per_gen), "inconsistency: n_gens: " + str(
        n_gens) + " but len(best_found_fit_per_gen): " + str(len(best_found_fit_per_gen))

    # =========================================================================

    n_chgperiods = len(generations_of_chgperiods)

    chgp_idxs = np.arange(n_chgperiods)
    if only_for_preds:
        # index of change periods for that a prediction was made
        chgp_idxs = chgp_idxs[first_chgp_idx_with_pred:]

    sum_over_changes = 0
    for i in chgp_idxs:
        first_fitness = best_found_fit_per_gen[generations_of_chgperiods[i][0]]
        optimum_fitness = global_opt_fit_per_chgperiod[i]

        if first_fitness == optimum_fitness:
            # best case
            sum_over_changes += 1
        else:
            sum_over_generations = 0
            if with_abs:
                diff_optimum_first = abs(optimum_fitness - first_fitness)
            else:
                diff_optimum_first = (optimum_fitness - first_fitness)
            # number of generations during this change period
            n_generations_of_change = len(generations_of_chgperiods[i])
            for j in range(n_generations_of_change):
                until_now_best_fitness = best_found_fit_per_gen[generations_of_chgperiods[i][j]]
                if with_abs:
                    diff_now_first = abs(
                        until_now_best_fitness - first_fitness)
                else:
                    diff_now_first = (until_now_best_fitness - first_fitness)
                sum_over_generations += diff_now_first
                #print("    until_now_best_fitness: ", until_now_best_fitness)
                # if until_now_best_fitness > first_fitness:
                #    print("now larger")
            divisor = n_generations_of_change * diff_optimum_first
            summand = sum_over_generations / divisor

            # if divisor < sum_over_changes:
            #    print("chgp: ", i)
            #    print("    sum_over_generations: ", sum_over_generations)
            #    print("    divisor: ", divisor)
            #    print("    opt_fit: ", optimum_fitness)
            #    print("    first fit: ", first_fitness)
            sum_over_changes = sum_over_changes + summand
    n_summed_chgps = len(chgp_idxs)
    sum_over_changes = sum_over_changes / n_summed_chgps
    # if sum_over_changes > 1:
    #    from matplotlib import pyplot as plt
    #    plt.plot(best_found_fit_per_gen)
    #    plt.show()
    #    print("----------------- ARR > 1")
    return sum_over_changes


def avg_bog_for_one_run(best_found_fit_per_gen, only_for_preds, first_gen_idx_with_pred):
    '''
    Averages the best found fitness values over all generations for one run.

    Is not the real average bog.

    @param best_found_fit_per_gen: 1d numpy array: best found fitness for each
    generation
    @return scalar
    '''
    if only_for_preds:
        best_found_fit_per_gen = best_found_fit_per_gen[first_gen_idx_with_pred:]
    return np.average(best_found_fit_per_gen)


def __best_of_generation(best_found_fit_per_gen_and_run):
    '''
    Internal function. Called by avg_best_of_generation.

    Best-of-generation measure according to description in the paper 
    "Evolutionary dynamic optimization: A survey of the state of the art"
    by Trung Thanh Nguyen et al. 2012.

    @param best_found_fit_per_gen_and_run: 2d numpy array containing for each run 
    one row that contains per generation the best achieved fitness value
    @return: 1d numpy array: for each generation the average best fitness
    '''

    # average the fitness over the runs (for each generation), i.e. column-wise
    return np.average(best_found_fit_per_gen_and_run, axis=0)


def avg_best_of_generation(best_found_fit_per_gen_and_run):
    '''
    Average Best-of-generation measure according to description in the paper 
    "Evolutionary dynamic optimization: A survey of the state of the art"
    by Trung Thanh Nguyen et al. 2012.

    Averages the Best-of-generation over all generations resulting in a scalar.

    @param best_found_fit_per_gen_and_run: 2d numpy array containing for each run 
    one row that contains per generation the best achieved fitness value
    @return: Tupel of two scalar values: (avg_bog, standard deviation of bog)
    '''

    bog_per_gen = __best_of_generation(best_found_fit_per_gen_and_run)
    return np.average(bog_per_gen), np.std(bog_per_gen)


def normalized_bog(avg_bog_per_alg_and_problem):
    '''
    Normalized score according to description in the paper 
    "Evolutionary dynamic optimization: A survey of the state of the art"
    by Trung Thanh Nguyen et al. 2012.

    Computes the quality of the tested algorithms relatively to the other tested 
    algorithms and over all test problems so that different algorithms can be
    compared regarding their overall quality.

    @param avg_bog_per_alg_and_problem: dictionary containing for each algorithm 
    a dictionary containing the average BOG per problem instance.  
    @return: dictionary containing one score for each algorithm:
        Score 0 -> worst algorithm
        Score 1 -> best algorithm
    '''
    # =======================
    # identify best and worst BOG per problem achieved by any algorithm

    # initialize dictionaries; key: problem, value: best/worst BOG
    best_bog_per_problem = {}
    worst_bog_per_problem = {}
    problems = list(avg_bog_per_alg_and_problem.values())[0].keys()
    for p in problems:
        best_bog_per_problem[p] = sys.maxsize
        worst_bog_per_problem[p] = -sys.maxsize - 1
    # search for best/worst BOGs
    for alg, bog_per_problem in avg_bog_per_alg_and_problem.items():
        for prob, bog in bog_per_problem.items():
            if bog > worst_bog_per_problem[prob]:
                worst_bog_per_problem[prob] = bog
            if bog < best_bog_per_problem[prob]:
                best_bog_per_problem[prob] = bog

    # =========================
    # Compute the normalized score

    norm_bog_per_alg = {}
    for alg, bog_per_problem in avg_bog_per_alg_and_problem.items():
        n_probs = len(bog_per_problem)  # number of test problems
        sum_of_norm_bog = 0
        for prob, bog in bog_per_problem.items():
            worst = worst_bog_per_problem[prob]
            best = best_bog_per_problem[prob]
            worst_best_diff = abs(worst - best)
            if worst_best_diff != 0:
                sum_of_norm_bog += abs(worst - bog) / worst_best_diff
            else:
                # score 1 is for best algorithm
                sum_of_norm_bog += 1
        norm_bog_per_alg[alg] = sum_of_norm_bog / n_probs

    return norm_bog_per_alg


def best_error_before_change(generations_of_chgperiods, global_opt_fit_per_chgperiod,
                             best_found_fit_per_gen, only_for_preds,
                             first_chgp_idx_with_pred):
    '''
    Best-error-before-change according to description in the paper 
    "Evolutionary dynamic optimization: A survey of the state of the art"
    by Trung Thanh Nguyen et al. 2012.

    Computes for each change period the difference between the best fitness
    achieved during that change period and the actual optimal fitness. 
    Afterwards these differences are averaged.
    The smaller the better, 0 is best.

    @param generations_of_chgperiods:  dictionary containing for each change 
    period (the real ones, not only those the EA has detected) a list with the 
    generation numbers
    @param global_opt_fit_per_chgperiod: 1d numpy array: for each change period
    the global optimum fitness
    @param best_found_fit_per_gen: 1d numpy array: best found fitness for each
    generation
    @return scalar: bebc
    '''
    # =========================================================================
    # for test purposes
    n_gens = 0
    for _, generations in generations_of_chgperiods.items():
        n_gens += len(generations)
    assert n_gens == len(best_found_fit_per_gen), "inconsistency: n_gens: " + str(
        n_gens) + " but len(best_found_fit_per_gen): " + str(len(best_found_fit_per_gen))

    # =========================================================================
    sum_of_errors = 0
    n_summed_chgps = 0
    for chgperiod, generations in generations_of_chgperiods.items():
        if only_for_preds and chgperiod < first_chgp_idx_with_pred:
            continue
        n_summed_chgps += 1
        best_found_fit = np.min(best_found_fit_per_gen[generations])
        sum_of_errors += abs((best_found_fit) -
                             (global_opt_fit_per_chgperiod[chgperiod]))
    return sum_of_errors / n_summed_chgps


def rel_conv_speed(generations_of_chgperiods, global_opt_fit_per_chgperiod, best_found_fit_per_gen_and_alg,
                   only_for_preds, first_chgp_idx_with_pred_per_alg, with_abs):
    '''
    Measure of relative convergence speed of algorithms for one run of a 
    specific problem. Depends on the worst fitness value any algorithm achieved.
    Disadvantage: results not comparable to results in other papers if other 
    algorithms are employed.

    Proposed in: Almuth Meier and Oliver Kramer: "Prediction with Recurrent 
    Neural Networks in Evolutionary Dynamic Optimization", EvoApplications 2018.

    Function is called in statistical_tests.py

    @param generations_of_chgperiods:  dictionary containing for each change 
    period (the real ones, not only those the EA has detected) a list with the 
    generation numbers
    @param global_opt_fit_per_chgperiod: 1d numpy array: for each change period
    the global optimum fitness (for all changes stored in the dataset 
    file, not only for those used in the experiments, i.e. 
    len(global_opt_fit_per_chgperiod) may be larger than len(generations_of_chgperiods)
    @param best_found_fit_per_gen_and_alg: dictionary containing for each 
    algorithm a 1d numpy array that contains the best found fitness of each
    generation 
    @return: dictionary containing one score for each algorithm:
        Score 0 -> best algorithm
        Score 1 -> worst algorithm
        None: if the respective run was not exexuted for the algorithm
    '''
    n_chgperiods = len(generations_of_chgperiods)
    # assert n_chgperiods == len(global_opt_fit_per_chgperiod), "inconsistency: n_chgperiods: " + str(
    # n_chgperiods) + " but len(global_opt_fit_per_chgperiod): " +
    # str(len(global_opt_fit_per_chgperiod))

    # -------------------------------------------------------------------------
    # compute worst fitness per change achieved by any algorithm

    # 2d list: one row for each algorithm
    evals_per_alg_list = list(best_found_fit_per_gen_and_alg.values())
    # worst fitness per generation achieved by any algorithm
    worst_fit_evals = np.max(evals_per_alg_list, axis=0)
    # compute worst fitness per change
    worst_fit_per_chgperiod = {}
    for chgperiod_nr, gens in generations_of_chgperiods.items():
        worst_fit_per_chgperiod[chgperiod_nr] = np.max(worst_fit_evals[gens])
    # test whether worst fitness is larger than global best fitness
    # assert len(global_opt_fit_per_chgperiod) == len(list(worst_fit_per_chgperiod.values())), "len-opt: " + \
    #    str(len(global_opt_fit_per_chgperiod)) + " len-worst: " + \
    #    str(len(list(worst_fit_per_chgperiod.values())))
    try:
        all_idcs = np.arange(n_chgperiods)
        bools = np.array(global_opt_fit_per_chgperiod)[all_idcs]
        assert np.all(list(worst_fit_per_chgperiod.values(
        )) >= bools), "global fitness worse than worst fitness"
    except Exception as e:
        print(e, flush=True)
        print("worst-fit-per-change-period: ")
        print(list(worst_fit_per_chgperiod.values()))
        print()
        print("global-opt-fit-per-change-period: ")
        print(global_opt_fit_per_chgperiod)
        raise  # throw the exception

    # -------------------------------------------------------------------------
    # compute convergence speed for each algorithm
    speed_per_alg = {}
    algs = list(best_found_fit_per_gen_and_alg.keys())
    for alg in algs:
        speed_per_alg[alg] = __convergence_speed__(generations_of_chgperiods,
                                                   global_opt_fit_per_chgperiod,
                                                   best_found_fit_per_gen_and_alg[alg],
                                                   worst_fit_per_chgperiod,
                                                   only_for_preds, first_chgp_idx_with_pred_per_alg[alg], with_abs)
    return speed_per_alg


def __convergence_speed__(generations_of_chgperiods,
                          global_opt_fit_per_chgperiod,
                          best_found_fit_per_gen,
                          worst_fit_per_chgperiod,
                          only_for_preds, first_chgp_idx_with_pred, with_abs):
    '''
    Internal method, called by rel_conv_speed().

    Computes convergence speed for one specific algorithm.

    Works only for minimization problems. (this implementation; but the measure 
    as formally defined should be able to handle maximization problems as well)
    Between 0 and 1. Best case: 0, worst case: 1.

    @param generations_of_chgperiods:  dictionary containing for each change 
    period (the real ones, not only those the EA has detected) a list with the 
    generation numbers
    @param global_opt_fit_per_chgperiod: 1d numpy array: for each change period
    the global optimum fitness (for all changes stored in the dataset 
    file, not only for those used in the experiments, i.e. 
    len(global_opt_fit_per_chgperiod) may be larger than len(generations_of_chgperiods)
    @param best_found_fit_per_gen: 1d numpy array containing for each generation 
    the best fitness value achieved by this algorithm.
    @param worst_fit_per_chgperiod: dictionary containing for each change period 
    the worst fitness value achieved by any algorithm.
    @return: scalar: convergence speed for this algorithm
             None: if for the respective algorithm this run was not executed
    '''
    sum_norm_areas = 0
    n_summed_chgps = 0
    for chgperiod_nr, gens in generations_of_chgperiods.items():
        if only_for_preds and chgperiod_nr < first_chgp_idx_with_pred:
            continue
        n_summed_chgps += 1

        optimal_fit = global_opt_fit_per_chgperiod[chgperiod_nr]
        worst_fit = worst_fit_per_chgperiod[chgperiod_nr]
        if with_abs:
            best_worst_fit_diff = abs(worst_fit - optimal_fit)
        else:
            best_worst_fit_diff = (worst_fit - optimal_fit)
        # compute area for this change
        area_for_change = 0
        max_area_for_change = 0
        gen_in_chg = 0
        for gen in gens:
            found_fit = best_found_fit_per_gen[gen]
            if found_fit is None:
                return None
            assert optimal_fit <= found_fit, "opt-fit " + str(
                optimal_fit) + " fit " + str(found_fit)
            if with_abs:
                diff = abs(found_fit - optimal_fit)
            else:
                diff = (found_fit - optimal_fit)
            area_for_change += (gen_in_chg + 1) * diff  # +1, otherwise first 0
            max_area_for_change += (gen_in_chg + 1) * best_worst_fit_diff
            gen_in_chg += 1

        if max_area_for_change == 0:
            # means RCS=0 for this change period since all algorithms always
            # had the global optimum fitness
            pass
        else:
            # normalize area so that it lies between 0 and 1
            norm_area_for_change = area_for_change / max_area_for_change
            sum_norm_areas += norm_area_for_change

    return sum_norm_areas / n_summed_chgps

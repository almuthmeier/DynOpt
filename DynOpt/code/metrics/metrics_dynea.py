'''
Created on Oct 6, 2017

@author: ameier
'''
import sys
import numpy as np


def arr(generations_of_chgperiods, global_opt_fit_per_chgperiod, best_found_fit_per_gen):
    '''
    TODO umbenennen optima_of_changes muss optima_of_generations heißen.
    Nein!?!? soll pro CHANGE (so wie Programmcode aussieht), aber bekommt pro
    GENERATION?

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
    @return scalar: ARR
    '''
    # =========================================================================
    # for test purposes TODO delete this?
    n_gens = 0
    for _, generations in generations_of_chgperiods.items():
        n_gens += len(generations)
    assert n_gens == len(best_found_fit_per_gen), "inconsistency: n_gens: " + str(
        n_gens) + " but len(best_found_fit_per_gen): " + str(len(best_found_fit_per_gen))

    # =========================================================================

    n_chgperiods = len(generations_of_chgperiods)

    sum_over_changes = 0
    for i in range(n_chgperiods):
        first_fitness = best_found_fit_per_gen[generations_of_chgperiods[i][0]]
        optimum_fitness = global_opt_fit_per_chgperiod[i]

        if first_fitness == optimum_fitness:
            # best case
            sum_over_changes += 1
        else:
            sum_over_generations = 0
            diff_optimum_first = abs(optimum_fitness - first_fitness)
            # number of generations during this change period
            n_generations_of_change = len(generations_of_chgperiods[i])
            for j in range(n_generations_of_change):
                until_now_best_fitness = best_found_fit_per_gen[generations_of_chgperiods[i][j]]
                diff_now_first = abs(until_now_best_fitness - first_fitness)
                sum_over_generations += diff_now_first

            divisor = n_generations_of_change * diff_optimum_first
            summand = sum_over_generations / divisor
            sum_over_changes = sum_over_changes + summand
    sum_over_changes = sum_over_changes / n_chgperiods

    return sum_over_changes


def __best_of_generation(best_of_generation_per_run):
    '''
    Internal function. Called by avg_best_of_generation.

    Best-of-generation measure according to description in the paper 
    "Evolutionary dynamic optimization: A survey of the state of the art"
    by Trung Thanh Nguyen et al. 2012.

    @param best_of_generation_per_run: 2d numpy array containing for each run 
    one row that contains per generation the best achieved fitness value
    @return: 1d numpy array: for each generation the best (i.e. minimum) fitness
    '''

    # average the fitness over the runs (for each generation), i.e. column-wise
    return np.average(best_of_generation_per_run, axis=0)


def avg_best_of_generation(best_of_generation_per_run):
    '''
    Average Best-of-generation measure according to description in the paper 
    "Evolutionary dynamic optimization: A survey of the state of the art"
    by Trung Thanh Nguyen et al. 2012.

    Averages the Best-of-generation over all generations resulting in a scalar.

    @param best_of_generation_per_run: 2d numpy array containing for each run 
    one row that contains per generation the best achieved fitness value
    @return: Tupel of two scalar values: (avg_bog, standard deviation of bog)
    '''

    bog = __best_of_generation(best_of_generation_per_run)
    return np.average(bog), np.std(bog)


def normalized_bog(avg_bog_per_alg_and_problem):
    '''
    Normalized score according to description in the paper 
    "Evolutionary dynamic optimization: A survey of the state of the art"
    by Trung Thanh Nguyen et al. 2012.

    Computes the quality of the tested algorithms relative to the other tested 
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
                             best_found_fit_per_gen):
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
    # for test purposes TODO delete this?
    n_gens = 0
    for _, generations in generations_of_chgperiods.items():
        n_gens += len(generations)
    assert n_gens == len(best_found_fit_per_gen), "inconsistency: n_gens: " + str(
        n_gens) + " but len(best_found_fit_per_gen): " + str(len(best_found_fit_per_gen))

    # =========================================================================
    n_chgperiods = len(generations_of_chgperiods)
    sum_of_errors = 0
    for chgperiod, generations in generations_of_chgperiods.items():
        best_found_fit = np.min(best_found_fit_per_gen[generations])
        sum_of_errors += abs((best_found_fit) -
                             (global_opt_fit_per_chgperiod[chgperiod]))
    return sum_of_errors / n_chgperiods


def conv_speed(gens_of_chgs, optima_of_changes, best_fit_evals_per_alg):
    '''
    Measure of relative convergence speed of algorithms for one run of a 
    specific problem. Depends on the worst fitness value any algorithm achieved.
    Disadvantage: results not comparable to results in other papers if other 
    algorithms are employed.

    Proposed in: Almuth Meier and Oliver Kramer: "Prediction with Recurrent 
    Neural Networks in Evolutionary Dynamic Optimization", EvoApplications 2018.

    Function is called in statistical_tests.py

    @param gens_of_chgs_per_alg: dictionary containing a dictionary that 
    contains the generation number for every change 
    @param optima_of_gens: 1d numpy array containing for each generation the 
    optimum fitness TODO falsch benannt: ist optimum per change
    @param best_fit_evals_per_alg: dictionary containing for each algorithm a 
    1d numpy array that contains the best fitness evaluation of each generation
    @return: dictionary containing one score for each algorithm:
        Score 0 -> best algorithm
        Score 1 -> worst algorithm
    '''
    n_changes = len(gens_of_chgs)
    assert n_changes == len(optima_of_changes), "inconsistency: n_changes: " + str(
        n_changes) + " but len(optima_of_changes): " + str(len(optima_of_changes))

    # -------------------------------------------------------------------------
    # compute worst fitness per change achieved by any algorithm

    # 2d list: one row for each algorithm
    evals_per_alg_list = list(best_fit_evals_per_alg.values())
    # worst fitness per generation achieved by any algorithm
    worst_fit_evals = np.max(evals_per_alg_list, axis=0)
    # compute worst fitness per change
    worst_fit_per_change = {}
    # dictionary containing the fitness of the global optimum for each change
    global_opt_fit_per_change = {}
    for chg_nr, gens in gens_of_chgs.items():
        worst_fit_per_change[chg_nr] = np.max(worst_fit_evals[gens])
        # all generations with indices in "gens" have same global opt. fitness
        #global_opt_fit_per_change[chg_nr] = optima_of_gens[gens[0]]
        global_opt_fit_per_change[chg_nr] = optima_of_changes[chg_nr]
    # test whether worst fitness is larger than global best fitness
    assert len(list(global_opt_fit_per_change.values())) == len(list(worst_fit_per_change.values())), "len-opt: " + \
        str(len(list(global_opt_fit_per_change.values()))) + " len-worst: " + \
        str(len(list(worst_fit_per_change.values())))
    try:
        assert list(worst_fit_per_change.values()) >= list(
            global_opt_fit_per_change.values()), "global fitness worse than worst fitness"
    except Exception as e:
        print(e, flush=True)
        print("worst-fit-per-change: ")
        print(list(worst_fit_per_change.values()))
        print()
        print("global-opt-fit: ")
        print(list(global_opt_fit_per_change.values()))

    # -------------------------------------------------------------------------
    # compute convergence speed for each algorithm
    speed_per_alg = {}
    algs = list(best_fit_evals_per_alg.keys())
    for alg in algs:
        speed_per_alg[alg] = __convergence_speed__(gens_of_chgs,
                                                   global_opt_fit_per_change,
                                                   best_fit_evals_per_alg[alg],
                                                   worst_fit_per_change)
    return speed_per_alg


def __convergence_speed__(generations_of_changes, optima_of_changes, best_fitness_evals, worst_fit_per_change):
    '''
    Internal method, called by conv_speed().

    Computes convergence speed for one specific algorithm.

    Works only for minimization problems. (this implementation; but the measure 
    as formally defined should be able to handle maximization problems as well)
    Between 0 and 1. Best case: 0, worst case: 1.

    @param best_fitness_evals: 1d numpy array containing for each generation the
    best fitness value achieved by this algorithm.
    @param worst_fit_per_change: dictionary containing for each change period 
    the worst fitness value achieved by any algorithm.
    @return: scalar: convergence speed for this algorithm
    '''
    sum_norm_areas = 0
    for chg_nr, gens in generations_of_changes.items():
        optimal_fit = optima_of_changes[chg_nr]
        worst_fit = worst_fit_per_change[chg_nr]
        best_worst_fit_diff = abs(optimal_fit - worst_fit)
        # compute area for this change
        area_for_change = 0
        max_area_for_change = 0
        gen_in_chg = 0
        for gen in gens:
            assert optimal_fit <= best_fitness_evals[gen], "opt-fit " + str(
                optimal_fit) + " fit " + str(best_fitness_evals[gen])
            diff = abs(optimal_fit - best_fitness_evals[gen])
            area_for_change += (gen_in_chg + 1) * diff  # +1, otherwise first 0
            max_area_for_change += (gen_in_chg + 1) * best_worst_fit_diff
            gen_in_chg += 1

        if max_area_for_change == 0:
            pass
        else:
            # normalize area so that it lies between 0 and 1
            norm_area_for_change = area_for_change / max_area_for_change
            sum_norm_areas += norm_area_for_change

    return sum_norm_areas / len(generations_of_changes)
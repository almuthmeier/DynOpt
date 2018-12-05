'''
Computes variances for repeated change periods
    - variance of best found position among runs
    - variance of positions of final population 
Created on Nov 30, 2018

@author: ameier
'''

import copy

from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np


def compute_variance_between_runs(in_full_name, out_full_name):
    '''
    Only for one file (later the implementation can be extended to summarize
    the variance results for different experiments).
    '''
    # -------------------------------------------------------------------------
    # pre-works (file handling,...)
    # -------------------------------------------------------------------------
    # load file
    in_file = np.load(in_full_name)
    # 4d list [runs, chgperiods, parents, dims]
    final_pop_per_run_per_chgperiod = in_file['final_pop_per_run_per_chgperiod']
    # 3d list [runs, chgperiods, parents]
    final_pop_fitness_per_run_per_changeperiod = in_file['final_pop_fitness_per_run_per_changeperiod']
    in_file.close()

    # set parameters
    n_change_periods = final_pop_per_run_per_chgperiod.shape[1]
    n_chgp_runs = final_pop_per_run_per_chgperiod.shape[0]

    # column names
    cols = ["variance-type", "run"]
    cols_for_chgperiods = ["cp" + str(i) for i in range(n_change_periods)]
    cols = cols + cols_for_chgperiods

    # -------------------------------------------------------------------------
    # compute variance
    # -------------------------------------------------------------------------
    # determine best position per run (for each change period
    # 3d [runs, chgperiods, dims]
    best_positions = [[] for _ in range(n_chgp_runs)]
    # 2d [runs, chgperiods]
    best_fitnesses = [[] for _ in range(n_chgp_runs)]
    for r in range(n_chgp_runs):
        for chgp in range(n_change_periods):
            # get population and its fitness for run and chgp
            population = final_pop_per_run_per_chgperiod[r][chgp]
            population_fitness = final_pop_fitness_per_run_per_changeperiod[r][chgp]
            # determine best individual and its fitness
            min_fitness_index = np.argmin(population_fitness)
            best_fitnesses[r].append(copy.deepcopy(
                population_fitness[min_fitness_index]))
            best_positions[r].append(
                copy.deepcopy(population[min_fitness_index]))
    best_positions = np.array(best_positions)
    best_fitnesses = np.array(best_fitnesses)

    # compute variance over runs
    # [chgperiods, dims]
    stddev_among_runs_per_chgp = np.std(best_positions, axis=0)
    mean_among_runs_per_chgp = np.average(best_positions, axis=0)
    min_val_among_runs_per_chgp = np.min(best_positions, axis=0)
    max_val_among_runs_per_chgp = np.max(best_positions, axis=0)

    # compute variance over population
    # [runs, chgperiods, dims]
    stddev_within_pop_per_run_per_chgp = np.std(
        final_pop_per_run_per_chgperiod, axis=2)

    # -------------------------------------------------------------------------
    # real optima
    # -------------------------------------------------------------------------
    # plot real values of optimum per change period
    benchmark_folder = "/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/datasets/GECCO_2019/"
    benchmark_file = "sphere/sphere_d-2_chgperiods-10000_pch-sine_fch-none_2018-11-30_12:21.npz"
    #benchmark_file = "sphere/sphere_d-10_chgperiods-10000_pch-sine_fch-none_2018-11-30_12:21.npz"
    #benchmark_file = "rastrigin/rastrigin_d-2_chgperiods-10000_pch-sine_fch-none_2018-11-30_12:21.npz"
    #benchmark_file = "rastrigin/rastrigin_d-10_chgperiods-10000_pch-sine_fch-none_2018-11-30_12:21.npz"

    # mixture
    benchmark_file = "sphere/sphere_d-2_chgperiods-10000_pch-mixture_fch-none_2018-11-30_12:40.npz"
    benchmark_file = "sphere/sphere_d-10_chgperiods-10000_pch-mixture_fch-none_2018-11-30_12:40.npz"
    #benchmark_file = "rastrigin/rastrigin_d-2_chgperiods-10000_pch-mixture_fch-none_2018-11-30_12:40.npz"
    #benchmark_file = "rastrigin/rastrigin_d-10_chgperiods-10000_pch-mixture_fch-none_2018-11-30_12:40.npz"

    benchmark_full_name = benchmark_folder + benchmark_file
    b_file = np.load(benchmark_full_name)
    global_opt_pos_per_chgperiod = b_file['global_opt_pos_per_chgperiod']
    b_file.close()
    # -------------------------------------------------------------------------
    # plot (https://stackoverflow.com/questions/7744697/how-to-show-two-figures-using-matplotlib)
    # -------------------------------------------------------------------------

    dims = [0]
    #dims = range(10)

    plot_shade = True
    # variance between runs (around mean) and real optimum
    f1 = plt.figure(1)
    for d in dims:
        stddev = stddev_among_runs_per_chgp[:, d]
        # plt.plot(stddev)
        plt.plot(global_opt_pos_per_chgperiod[:n_change_periods, d])
        if plot_shade:
            mean_val = mean_among_runs_per_chgp[:, d]
            min_val = min_val_among_runs_per_chgp[:, d]
            max_val = max_val_among_runs_per_chgp[:, d]
            plt.plot(mean_val)
            x = np.arange(len(stddev_among_runs_per_chgp))
            plt.fill_between(x, mean_val + stddev,
                             mean_val - stddev, alpha=0.4)
            #plt.fill_between(x, max_val, min_val, alpha=0.2)
    plt.title("real opt, mean & stddev of found")
    f1.show()

    # variance within population and best found position
    f2 = plt.figure(2)
    for d in dims:
        stddev = stddev_among_runs_per_chgp[:, d]
        plt.plot(stddev)
        # for r in range(n_chgp_runs):
        #    plt.plot(stddev_within_pop_per_run_per_chgp[r, :, d])
    plt.title("stddev among runs, stddev within pop")
    f2.show()

    # deviation to real optimum
    f3 = plt.figure(3)
    for r in range(n_chgp_runs):
        for d in dims:
            plt.plot(
                abs(best_positions[r, :, d] - global_opt_pos_per_chgperiod[:n_change_periods, d]))
    plt.title("deviation to real opt")
    f3.show()

    plt.show()


def main():
    in_path = "/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/output/GECCO_2019/sphere/ersterTest/ea_no/arrays/"
    in_file_name = "no_sphere_d-2_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-sine_fch-none_2018-11-30_12:21_00.npz"
    #in_file_name = "no_sphere_d-10_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-sine_fch-none_2018-11-30_12:21_00.npz"
    #in_file_name = "no_rastrigin_d-2_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-sine_fch-none_2018-11-30_12:21_00.npz"
    #in_file_name = "no_rastrigin_d-10_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-sine_fch-none_2018-11-30_12:21_00.npz"

    # mixture
    in_file_name = "no_sphere_d-2_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-mixture_fch-none_2018-12-04_16:02_00.npz"
    in_file_name = "no_sphere_d-10_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-mixture_fch-none_2018-12-04_16:02_00.npz"
    #in_file_name = "no_rastrigin_d-2_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-mixture_fch-none_2018-11-30_12:40_00.npz"
    #in_file_name = "no_rastrigin_d-10_chgperiods-50_lenchgperiod-20_ischgperiodrandom-False_pch-mixture_fch-none_2018-11-30_12:40_00.npz"

    in_full_name = in_path + in_file_name
    out_path = "/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/output/GECCO_2019/sphere/ersterTest/ea_no/metrics/"
    out_file_name = "variances.csv"
    out_full_name = out_path + out_file_name
    compute_variance_between_runs(in_full_name, out_full_name)


if __name__ == '__main__':
    main()


# Erkenntnisse (30.11.18)
# - Standardabweichung innerhalb der Population ist sehr gering
# - Standardabweichung zwischen bester gefundener Lösung verschiedener runs ist größer
#     - scheint Abweichung zum realen Optimum widerzuspiegeln: 2*stddev = Abweichung (bei 2 Dimensionen)

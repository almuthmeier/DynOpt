'''
Created on May 15, 2018

@author: ameier
'''
import os
from os.path import isdir, join
from posix import listdir
import re

from metrics.metrics_dynea import best_error_before_change, arr,\
    avg_best_of_generation, conv_speed
import numpy as np
import pandas as pd
from utils.utils_dynopt import convert_chgperiods_for_gens_to_dictionary
from utils.utils_files import select_experiment_files,\
    get_array_file_names_for_experiment_file_name, get_info_from_array_file_name


class MetricCalculator():
    def __init__(self):
        # are set outside
        self.algorithms = None
        self.benchmarkfunctions = None
        self.benchmark_folder_path = None
        self.output_dir_path = None
        self.pos_chg_types = None
        self.fit_chg_types = None
        self.dims = None
        self.noises = None

    def compute_metrics(self, best_found_fit_per_gen,
                        real_chgperiods_for_gens,
                        global_opt_fit_per_chgperiod,
                        global_opt_pos_per_chgperiod,
                        gens_of_chgperiods,
                        orig_global_opt_pos):

        # bebc
        bebc = best_error_before_change(
            gens_of_chgperiods, global_opt_fit_per_chgperiod,  best_found_fit_per_gen)
        print("bebc: ", bebc)

        # arr
        arr_value = arr(gens_of_chgperiods,
                        global_opt_fit_per_chgperiod, best_found_fit_per_gen)
        print("arr: ", arr_value)
        # TODO metric values of different runs are same!?! (-> seeds?)
        return bebc, arr_value

    def compute_and_save_all_metrics(self):
        # order of columns should be meaningless!!!
        column_names = ['function', 'algparams', 'alg',
                        'dim', 'periods', 'lenperiods', 'periodsrandom', 'poschg', 'fitchg', 'noise',
                        'run', 'bog', 'bebc', 'rcs', 'arr']

        df = pd.DataFrame(columns=column_names)

        for benchmarkfunction in self.benchmarkfunctions:
            # load experiment files for the benchmark function to get
            # information about real global optimum
            print()
            print("benchmark: ", benchmarkfunction)
            experiment_files = select_experiment_files(self.benchmark_folder_path + benchmarkfunction,
                                                       benchmarkfunction,
                                                       self.poschgtypes,
                                                       self.fitchgtypes,
                                                       self.dims, self.noises)
            for exp_file_name in experiment_files:

                # load experiment data from file
                exp_file_path = self.benchmark_folder_path + \
                    benchmarkfunction + "/" + exp_file_name
                exp_file = np.load(exp_file_path)
                global_opt_fit_per_chgperiod = exp_file['global_opt_fit_per_chgperiod']
                global_opt_pos_per_chgperiod = exp_file['global_opt_pos_per_chgperiod']
                orig_global_opt_pos = exp_file['orig_global_opt_pos']
                exp_file.close()

                dim = len(orig_global_opt_pos)
                # =============================================================
                # find output files of all algorithms for this experiment

                # get output
                output_dir_for_benchmark_funct = self.output_dir_path + benchmarkfunction + "/"
                print(output_dir_for_benchmark_funct)
                # different alg settings
                direct_cild_dirs = [d for d in listdir(output_dir_for_benchmark_funct) if (
                    isdir(join(output_dir_for_benchmark_funct, d)))]
                print(direct_cild_dirs)

                # algorithm parameter settings, e.g. "c1c2c3_1.49"
                for subdir in direct_cild_dirs:
                    subdir_path = output_dir_for_benchmark_funct + subdir + "/"
                    # different alg types/predictors
                    alg_types = [d for d in listdir(subdir_path) if (
                        isdir(join(subdir_path, d)))]
                    # dictionary: for each algorithm  a list of 1d numpy arrays
                    # (for each run the array of best found fitness values)
                    best_found_fit_per_gen_and_run_and_alg = {
                        key: [] for key in alg_types}

                    # algorithms with predictor types, e.g. "ea_no"
                    for alg in alg_types:
                        # data structure for metrics (for writing into files)
                        bebc_per_run = []
                        bog_per_gen = []
                        arr_per_run = []

                        print("    alg: ", alg)
                        # read all array files for the runs of the experiment
                        arrays_path = subdir_path + alg + "/arrays/"
                        array_names = get_array_file_names_for_experiment_file_name(exp_file_name,
                                                                                    arrays_path)

                        # get general info from one arbitrary array file
                        (predictor, benchmark, d, chgperiods, lenchgperiod,
                            ischgperiodrandom, veclen, peaks, noise, poschg,
                            fitchg,  date, time, run) = get_info_from_array_file_name(array_names[0])
                        assert benchmarkfunction == benchmark, "benchmark names unequal; benchmarkfunction: " + \
                            str(benchmarkfunction) + \
                            " and benchmark: " + str(benchmark)
                        assert dim == d, "dimensionality unequal; dim: " + \
                            str(dim) + " and d: " + str(d)

                        for array_file_name in array_names:
                            run = int(
                                re.search("_\d+.npz", array_file_name).group())
                            run = run.replace("_", "")
                            run = run.replace(".npz", "")

                            file = np.load(arrays_path + array_file_name)
                            best_found_fit_per_gen = file['best_found_fit_per_gen']
                            best_found_pos_per_gen = file['best_found_pos_per_gen']
                            best_found_fit_per_chgperiod = file['best_found_fit_per_chgperiod']
                            best_found_pos_per_chgperiod = file['best_found_pos_per_chgperiod']
                            pred_opt_fit_per_chgperiod = file['pred_opt_fit_per_chgperiod']
                            pred_opt_pos_per_chgperiod = file['pred_opt_pos_per_chgperiod']
                            detected_chgperiods_for_gens = file['detected_chgperiods_for_gens']
                            real_chgperiods_for_gens = file['real_chgperiods_for_gens']
                            file.close()
                            gens_of_chgperiods = convert_chgperiods_for_gens_to_dictionary(
                                real_chgperiods_for_gens)

                            # arr, bebc
                            bebc, arr_value = self.compute_metrics(best_found_fit_per_gen,
                                                                   real_chgperiods_for_gens,
                                                                   global_opt_fit_per_chgperiod,
                                                                   global_opt_pos_per_chgperiod,
                                                                   gens_of_chgperiods,
                                                                   orig_global_opt_pos)
                            # todo: auch dim, len_chg, ... im Datensatz speichern, sodass man es nicht aus dem Dateinamen auslesen muss??? (23.5.18)
                            # unabhängig von Spaltenanordnung
                            data = {'function': benchmarkfunction, 'predictor': predictor,
                                    'algparams': subdir, 'alg': alg, 'dim': dim,
                                    'chgperiods': chgperiods, 'len_c_p': lenchgperiod,
                                    'ischgperiodrandom': ischgperiodrandom,
                                    'veclen': veclen, 'peaks': peaks, 'noise': noise,
                                    'poschg': poschg, 'fitchg': fitchg, 'run': run,
                                    'bog': None, 'bebc': bebc, 'rcs': None, 'arr': arr_value}
                            df.at[len(df)] = data
                            # TODO wie kann man jetzt DAten für BOG, RCS nachträglich hier hineinpacken???
                            # store data for bog
                            best_found_fit_per_gen_and_run_and_alg[alg].append(
                                best_found_fit_per_gen)
                        # bog
                        bog_avg, bog_dev = avg_best_of_generation(
                            best_found_fit_per_gen_and_run_and_alg[alg])
                        print("bog: ", bog_avg)
                        # rcs
                    keys = list(best_found_fit_per_gen_and_run_and_alg.keys())
                    runs = len(best_found_fit_per_gen_and_run_and_alg[keys[0]])

                    # compute RCS
                    for run in range(runs):
                        # convert dict to new one that contains for each
                        # algorithm only one 1d numpy array (the best found
                        # fitness per generation)
                        new_dict = {}
                        for alg in keys:
                            new_dict[alg] = best_found_fit_per_gen_and_run_and_alg[alg][run]

                        rcs_per_alg = conv_speed(
                            gens_of_chgperiods, global_opt_fit_per_chgperiod, new_dict)
                        print("rcs_per_alg ", rcs_per_alg)

                    # write into files
                    # TODO
        # for each experiment (in benchmarkfolder):
        #    - load benchmark files to get information about real global optimum
        #
        #    - for each run
        #         - for each algorithm:
        #             - load array file
        #             - compute metrics
        #             - store data that are necessary for RCS
        #         - compute RCS
        #     - average metrics, do other statistical stuff, like box plots (no statistical tests here)
        # compute normBOG here?


def init_metric_calculator():
    # TODO(dev) set parameters as required
    calculator = MetricCalculator()
    # TODO nein. es wird ja nicht (unbedingt) über Optimierungsalgorithmen,
    # sondern über Prediktoren verglichen

    # TODO woher weiß man Zuordnung zw. Benchmark und Ausgabepfad (Dict?)

    # path to "..../DynOpt/code"
    path_to_code = os.path.abspath(os.pardir)
    path_to_datasets = '/'.join(path_to_code.split('/')[:-1]) + "/datasets/"
    path_to_output = '/'.join(path_to_code.split('/')[:-1]) + "/output/"

    calculator.algorithms = []
    calculator.benchmarkfunctions = ["sphere"]  # sphere, rosenbrock
    calculator.benchmark_folder_path = path_to_datasets + "EvoStar_2018/"
    calculator.output_dir_path = path_to_output + "EvoStar_2018/"
    calculator.poschgtypes = ["linear", "sine"]
    calculator.fitchgtypes = ["none"]
    calculator.dims = [50]
    calculator.noises = [0.0]
    return calculator


if __name__ == '__main__':
    calculator = init_metric_calculator()
    calculator.compute_and_save_all_metrics()

# BOG has values for generations
# other metrics have values for runs

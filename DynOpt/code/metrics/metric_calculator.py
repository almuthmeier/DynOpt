'''
Computes the metric after the optimization by reading in the files generated
by the optimization algorithms and the data set files of the corresponding
experiments.
 
Created on May 15, 2018

@author: ameier
'''
import os
from os.path import isdir, join
from posix import listdir
import warnings

from metrics.metrics_dynea import best_error_before_change, arr,\
    rel_conv_speed, avg_bog_for_one_run, rmse
import numpy as np
import pandas as pd
from utils.utils_dynopt import convert_chgperiods_for_gens_to_dictionary
from utils.utils_files import get_array_names_for_ks_and_filters
from utils.utils_files import select_experiment_files,\
    get_sorted_array_file_names_for_experiment_file_name, \
    get_info_from_array_file_name, get_run_number_from_array_file_name
from utils.utils_prediction import get_first_chgp_idx_with_pred
from utils.utils_prediction import get_first_generation_idx_with_pred


class MetricCalculator():
    def __init__(self, path_to_datasets=None, path_to_output=None,
                 benchmarkfunctions=None, poschgtypes=None, fitchgtypes=None,
                 dims=None, noises=None, path_addition=None, metric_filename=None,
                 only_for_preds=None):
        '''
        Initialize paths, properties of the experiments, etc.
        '''
        # TODO(exp) set parameters as required

        # output and benchmark path
        if path_to_output is None:  # assumes that  all input arguments are none
            # path to "..../DynOpt/code"
            path_to_code = os.path.abspath(os.pardir)
            path_to_datasets = '/'.join(path_to_code.split('/')
                                        [:-1]) + "/datasets/"
            path_to_output = '/'.join(path_to_code.split('/')
                                      [:-1]) + "/output/"
            self.output_dir_path = path_to_output + \
                "GECCO_2019/"
            self.output_dir_path = "/home/ameier/Documents/Promotion/Ausgaben/DynCMA/Ausgaben/output_2019-09-11_DSB_vel-0.5/tests_for_examination/"
            self.output_dir_path = "/home/ameier/Documents/Promotion/Ausgaben/DynCMA/Ausgaben/output_2019-09-11_DSB_vel-0.5/"
            self.benchmark_folder_path = path_to_datasets + "GECCO_2019/"
            self.benchmark_folder_path = "/home/ameier/Documents/Promotion/Ausgaben/DynCMA/Ausgaben/data_2019-09-11_newDSB/vel-0.5/"
            # , "rosenbrock", "rastrigin"]  # sphere, rosenbrock, mpbnoisy,griewank
            self.benchmarkfunctions = ["sphere"]
            # ["linear", "sine", "circle"]
            self.poschgtypes = ["sinefreq"]
            self.fitchgtypes = ["none"]
            self.dims = [2]
            self.noises = [0.0]
            self.path_addition = "architecture/"
            self.metric_filename = "metric_db.csv"
            self.only_for_preds = True

        else:
            self.output_dir_path = path_to_output
            self.benchmark_folder_path = path_to_datasets
            self.benchmarkfunctions = benchmarkfunctions
            self.poschgtypes = poschgtypes
            self.fitchgtypes = fitchgtypes
            self.dims = dims
            self.noises = noises
            self.path_addition = path_addition  # for further subdirectories
            self.metric_filename = metric_filename
            # True if metrics are computed only for change periods where
            # predictions where made
            self.only_for_preds = only_for_preds

    def compute_rmses(self, global_opt_per_chgperiod, best_found_per_chgperiod,
                      pred_opt_per_chgperiod, first_chgp_idx_with_pred):
        '''
        Computes three RMSEs:
            - prediction - truth (-> truepred_rmse)
            - found - truth (-> ea_rmse)
            - prediction - found (-> foundpred_rmse)

        Is either for fitness values or for positions (depending on the input).
        '''
        # RMSEs (do not compute it when prediction is None (e.g, for "no")
        n_found = len(best_found_per_chgperiod)
        relevant_glob_opt_per_chgperiod = global_opt_per_chgperiod[
            first_chgp_idx_with_pred:n_found]
        ea_rmse = rmse(relevant_glob_opt_per_chgperiod,
                       best_found_per_chgperiod[first_chgp_idx_with_pred:])
        print("                    ea-rmse: ", ea_rmse)
        foundpred_rmse = None
        truepred_rmse = None
        if not len(pred_opt_per_chgperiod) == 0:
            # assume that all changes are detected
            n_predictions = len(pred_opt_per_chgperiod)
            # compute rmse only for chgperiods where something was predicted
            assert n_predictions <= n_found, "n_predictions: " + \
                str(n_predictions) + " n_found:" + str(n_found)
            foundpred_rmse = rmse(best_found_per_chgperiod[first_chgp_idx_with_pred:],
                                  pred_opt_per_chgperiod)
            truepred_rmse = rmse(relevant_glob_opt_per_chgperiod,
                                 pred_opt_per_chgperiod)
        print("                    truepred-rmse: ", truepred_rmse)
        print("                    foundpred-rmse: ", foundpred_rmse)
        return ea_rmse, foundpred_rmse, truepred_rmse

    def compute_metrics(self, best_found_fit_per_gen, gens_of_chgperiods,
                        global_opt_fit_per_chgperiod, global_opt_pos_per_chgperiod,
                        pred_opt_pos_per_chgperiod, pred_opt_fit_per_chgperiod,
                        best_found_pos_per_chgperiod, best_found_fit_per_chgperiod,
                        first_chgp_idx_with_pred):
        '''
        Computes BEBC and ARR
        '''
        # bebc
        bebc = best_error_before_change(gens_of_chgperiods, global_opt_fit_per_chgperiod,
                                        best_found_fit_per_gen, self.only_for_preds,
                                        first_chgp_idx_with_pred)
        print("                    bebc: ", bebc)

        # arr
        arr_value = arr(gens_of_chgperiods,
                        global_opt_fit_per_chgperiod, best_found_fit_per_gen,
                        self.only_for_preds, first_chgp_idx_with_pred)
        print("                    arr: ", arr_value)

        # fitness RMSEs
        fit_ea_rmse, fit_foundpred_rmse, fit_truepred_rmse = self.compute_rmses(
            global_opt_fit_per_chgperiod, best_found_fit_per_chgperiod,
            pred_opt_fit_per_chgperiod, first_chgp_idx_with_pred)

        # position RMSEs
        pos_ea_rmse, pos_foundpred_rmse, pos_truepred_rmse = self.compute_rmses(
            global_opt_pos_per_chgperiod, best_found_pos_per_chgperiod,
            pred_opt_pos_per_chgperiod, first_chgp_idx_with_pred)

        return (bebc, arr_value, fit_ea_rmse, fit_foundpred_rmse, fit_truepred_rmse,
                pos_ea_rmse, pos_foundpred_rmse, pos_truepred_rmse)

    def compute_and_save_all_metrics(self):
        '''
        Computes all metrics for all algorithms and experiments and stores them
        in one file.
        '''
        # order of columns should is meaningless
        column_names = ['function', 'dim', 'predictor',
                        'algparams', 'alg',
                        'ks', 'kernels', 'lr', 'epochs', 'bs', 'traindrop', 'testdrop',
                        'chgperiods', 'len_c_p',
                        'ischgperiodrandom', 'veclen', 'peaks', 'noise',
                        'poschg', 'fitchg', 'run', 'bog-for-run', 'bebc', 'rcs', 'arr',
                        'fit-ea-rmse', 'fit-foundpred-rmse', 'fit-truepred-rmse',
                        'pos-ea-rmse', 'pos-foundpred-rmse', 'pos-truepred-rmse',
                        'expfilename', 'arrayfilename']

        df = pd.DataFrame(columns=column_names)

        for benchmarkfunction in self.benchmarkfunctions:
            # load experiment files for the benchmark function to get
            # information about real global optimum
            print("", flush=True)
            print("\n\n\nbenchmark: ", benchmarkfunction, flush=True)
            experiment_files = select_experiment_files(self.benchmark_folder_path + benchmarkfunction,
                                                       benchmarkfunction,
                                                       self.poschgtypes,
                                                       self.fitchgtypes,
                                                       self.dims, self.noises)
            for exp_file_name in experiment_files:
                print("exp_file_name: ", exp_file_name, flush=True)
                # load experiment data from file
                exp_file_path = self.benchmark_folder_path + \
                    benchmarkfunction + "/" + exp_file_name
                exp_file = np.load(exp_file_path)
                global_opt_fit_per_chgperiod = exp_file['global_opt_fit_per_chgperiod']
                global_opt_pos_per_chgperiod = exp_file['global_opt_pos_per_chgperiod']
                # position of unmoved base function (not necessarily the global
                # optimum in first change period)
                orig_global_opt_pos = exp_file['orig_global_opt_pos']
                exp_file.close()

                dim = len(orig_global_opt_pos)

                # =============================================================
                # find output files of all algorithms for this experiment

                # get output
                output_dir_for_benchmark_funct = self.output_dir_path + \
                    benchmarkfunction + "/" + self.path_addition
                print("    output_dir_for_benchmark_funct:",
                      output_dir_for_benchmark_funct, flush=True)
                # different alg settings
                direct_cild_dirs = [d for d in listdir(output_dir_for_benchmark_funct) if (
                    isdir(join(output_dir_for_benchmark_funct, d)) and not listdir(output_dir_for_benchmark_funct + d) == [])]
                print("    direct_child_dirs: ", direct_cild_dirs, flush=True)

                # algorithm parameter settings, e.g. "c1c2c3_1.49"
                for subdir in direct_cild_dirs:
                    # for ks in [2, 3, 4, 5, 6, 7]: # TODO only for evaluation of ks and filters
                    #    for filters in [27, 16, 8]:
                    df = self.evaluate_subdirs(subdir, output_dir_for_benchmark_funct,
                                               df, exp_file_name, benchmarkfunction,
                                               dim, global_opt_fit_per_chgperiod,
                                               global_opt_pos_per_chgperiod,
                                               None, None)
        # save data frame into file (index=False --> no row indices)
        df.to_csv(self.output_dir_path + self.metric_filename, index=False)

    def evaluate_subdirs(self, subdir, output_dir_for_benchmark_funct, df,
                         exp_file_name, benchmarkfunction, dim,
                         global_opt_fit_per_chgperiod, global_opt_pos_per_chgperiod,
                         ks=None, filters=None):
        print("        subdir: ", subdir, flush=True)
        if subdir == "steps_100":
            return
        subdir_path = output_dir_for_benchmark_funct + subdir + "/"
        # different alg types/predictors
        alg_types = [d for d in listdir(subdir_path) if (
            isdir(join(subdir_path, d)))]
        # if "logs" in alg_types and "metrics" in alg_types and "arrays" in alg_types:
        #    # no special types for the current algorithm
        #    alg_types = [""]
        # dictionary: for each algorithm  a list of 1d numpy arrays
        # (for each run the array of best found fitness values)
        best_found_fit_per_gen_and_run_and_alg = {
            key: [] for key in alg_types}
        array_file_names_per_run_and_alg = {
            key: [] for key in alg_types}

        print("        alg_types: ", alg_types, flush=True)

        # algorithms with predictor types, e.g. "ea_no"
        for alg in alg_types:
            print("            \n\nalg: ", alg, flush=True)
            # read all array files for the runs of the experiment
            arrays_path = subdir_path + alg + "/arrays/"
            array_names = get_sorted_array_file_names_for_experiment_file_name(exp_file_name,
                                                                               arrays_path)
            if ks is not None and filters is not None:
                array_names = get_array_names_for_ks_and_filters(
                    array_names, ks, filters)
            print("                array_names: ", array_names, flush=True)

            for array_file_name in array_names:
                print("                    \nresults for array_name: ",
                      array_file_name)
                (predictor, benchmark, d, chgperiods, lenchgperiod,
                 ischgperiodrandom, veclen, peaks, noise, poschg,
                 fitchg,  _, _, run, kernel_size, n_kernels, l_rate,
                 n_epochs, batch_size, train_drop, test_drop) = get_info_from_array_file_name(array_file_name)

                assert benchmarkfunction == benchmark, "benchmark names unequal; benchmarkfunction: " + \
                    str(benchmarkfunction) + \
                    " and benchmark: " + str(benchmark)
                assert dim == d, "dimensionality unequal; dim: " + \
                    str(dim) + " and d: " + str(d)

                # run = get_run_number_from_array_file_name(
                #    array_file_name)

                file = np.load(arrays_path + array_file_name)
                best_found_fit_per_gen = file['best_found_fit_per_gen']
                real_chgperiods_for_gens = file['real_chgperiods_for_gens']
                pred_opt_pos_per_chgperiod = file['pred_opt_pos_per_chgperiod']
                pred_opt_fit_per_chgperiod = file['pred_opt_fit_per_chgperiod']
                best_found_pos_per_chgperiod = file['best_found_pos_per_chgperiod']
                best_found_fit_per_chgperiod = file['best_found_fit_per_chgperiod']
                file.close()

                gens_of_chgperiods = convert_chgperiods_for_gens_to_dictionary(
                    real_chgperiods_for_gens)
                n_preds = len(pred_opt_pos_per_chgperiod)
                n_chgps = len(best_found_pos_per_chgperiod)
                first_chgp_idx_with_pred = get_first_chgp_idx_with_pred(
                    n_chgps, n_preds)
                first_gen_idx_with_pred = get_first_generation_idx_with_pred(
                    n_chgps, n_preds, gens_of_chgperiods)

                # arr, bebc
                (bebc, arr_value,
                 fit_ea_rmse, fit_foundpred_rmse, fit_truepred_rmse,
                 pos_ea_rmse, pos_foundpred_rmse, pos_truepred_rmse) = self.compute_metrics(best_found_fit_per_gen,
                                                                                            gens_of_chgperiods,
                                                                                            global_opt_fit_per_chgperiod,
                                                                                            global_opt_pos_per_chgperiod,
                                                                                            pred_opt_pos_per_chgperiod,
                                                                                            pred_opt_fit_per_chgperiod,
                                                                                            best_found_pos_per_chgperiod,
                                                                                            best_found_fit_per_chgperiod,
                                                                                            first_chgp_idx_with_pred)
                # averaged bog for this run (not the average bog as
                # defined) (should not be used other than as average
                # over all runs)
                bog_for_run = avg_bog_for_one_run(best_found_fit_per_gen,
                                                  self.only_for_preds,
                                                  first_gen_idx_with_pred)
                data = {'expfilename': exp_file_name,
                        'arrayfilename': array_file_name,
                        'function': benchmarkfunction, 'predictor': predictor,
                        'algparams': subdir, 'alg': alg, 'dim': dim,
                        'ks': kernel_size, 'kernels': n_kernels, 'lr': l_rate,
                        'epochs': n_epochs, 'bs': batch_size,
                        'traindrop': train_drop, 'testdrop': test_drop,
                        'chgperiods': chgperiods, 'len_c_p': lenchgperiod,
                        'ischgperiodrandom': ischgperiodrandom,
                        'veclen': veclen, 'peaks': peaks, 'noise': noise,
                        'poschg': poschg, 'fitchg': fitchg, 'run': run,
                        'bog-for-run': bog_for_run, 'bebc': bebc, 'rcs': None, 'arr': arr_value,
                        'fit-ea-rmse': fit_ea_rmse, 'fit-foundpred-rmse': fit_foundpred_rmse, 'fit-truepred-rmse': fit_truepred_rmse,
                        'pos-ea-rmse': pos_ea_rmse, 'pos-foundpred-rmse': pos_foundpred_rmse, 'pos-truepred-rmse': pos_truepred_rmse}
                df.at[len(df)] = data
                #print("len: ", len(df), flush=True)
                #print(df.columns, flush=True)
                # store data for bog and rcs
                best_found_fit_per_gen_and_run_and_alg[alg].append(
                    best_found_fit_per_gen)
                array_file_names_per_run_and_alg[alg].append(
                    array_file_name)
            # bog (as defined)
            # bog_avg, bog_dev = avg_best_of_generation(
            #    best_found_fit_per_gen_and_run_and_alg[alg])
            #print("bog: ", bog_avg,flush=True)

        # rcs
        keys = list(best_found_fit_per_gen_and_run_and_alg.keys())
        # TODO should be the largest value any algorithm has
        runs = len(best_found_fit_per_gen_and_run_and_alg[keys[0]])

        # compute RCS (per run)
        for run in range(runs):
            # convert dict to new one that contains for each
            # algorithm only one 1d numpy array (the best found
            # fitness per generation)
            new_dict = {}
            for alg in keys:
                try:
                    new_dict[alg] = best_found_fit_per_gen_and_run_and_alg[alg][run]
                except IndexError:
                    # IndexError: list index out of range
                    # this error is thrown, if for one algorithm not the
                    # maximum number of runs was executed
                    #new_dict[alg] = None
                    pass
                    # this algorithm needs not to be considered for RCS
            rcs_per_alg = rel_conv_speed(
                gens_of_chgperiods, global_opt_fit_per_chgperiod, new_dict,
                self.only_for_preds, first_chgp_idx_with_pred)
            print("                    rcs_per_alg ", rcs_per_alg, flush=True)

            # store RCS data
            for alg in keys:
                try:
                    # write line only if run was executed for this alg.
                    df.loc[(df['arrayfilename'] == array_file_names_per_run_and_alg[alg][run]),
                           ['rcs']] = rcs_per_alg[alg]
                except KeyError:
                    # KeyError: 'dynea_autoregressive_predDEV'
                    # this error is thrown in case the respective run was not
                    # executed for this algorithm (see above except-case)
                    pass

        return df


def start_computing_metrics(benchmarkfunctionfolderpath=None, outputpath=None,
                            benchmarkfunctions=None, poschgtypes=None,
                            fitchgtypes=None, dims=None, noises=None,
                            path_addition=None, metric_filename=None, only_for_preds=None):
    calculator = MetricCalculator(benchmarkfunctionfolderpath, outputpath,
                                  benchmarkfunctions, poschgtypes, fitchgtypes,
                                  dims, noises, path_addition, metric_filename,
                                  only_for_preds)
    calculator.compute_and_save_all_metrics()
    print("saved metric database", flush=True)


if __name__ == '__main__':
    warnings.simplefilter("always")  # prints every warning
    start_computing_metrics()

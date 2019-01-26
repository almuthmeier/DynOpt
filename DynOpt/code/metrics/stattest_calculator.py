'''
Computes Mann-Whitney-U tests for each possible algorithm pair.
Requires the file created by metric_calculator.py.

Created on May 18, 2018
Implemented on June 14, 2018

@author: ameier
'''
import itertools
import math
import os

from scipy import stats

import numpy as np
import pandas as pd
from utils.utils_files import print_line, print_to_file


class StatisticalTestsCalculator():

    def __init__(self):
        # TODO(dev) set parameters as required
        # path to "..../DynOpt/code"
        path_to_code = os.path.abspath(os.pardir)
        path_to_output = '/'.join(path_to_code.split('/')[:-1]) + "/output/"
        self.metric_path = path_to_output + "GECCO_2019/"

        #self.metric_path = "/home/ameier/Documents/Promotion/Ausgaben/Uncertainty/Ausgaben/output_2019-01-21_sigmas_zusammengefuehrt/"
        #self.metric_file_path = self.metric_path + "metric_db_sigmas_2019-01-22.csv"
        #self.stattest_dir_path = self.metric_path + "stattests/"

        self.metric_path = "/home/ameier/Documents/Promotion/Ausgaben/Uncertainty/Ausgaben/output_2019-01-25_alle_reini_zusammen/"
        self.metric_file_path = self.metric_path + \
            "metric_db_2019-01-25_reinitialization.csv"
        self.stattest_dir_path = self.metric_path + "stattests/"

    def select_rows_for_alg(self, df, alg, exp):
        '''
        Selects for the specified algorithm the rows for all runs of the 
        desired experiment.

        @param df: csv-file containing all results for all algorithms
        @param alg: list containing 2 strings:
            alg[0] corresponds to "algparams" column in the metric file
            alg[1] corresponds to "alg" column in the metric file
        @param exp: (string) name of the file containing the fitness function data set 
        @return dataframe: the selected rows
        '''
        return df.loc[(df["alg"].isin([alg])) &
                      (df["expfilename"].isin([exp]))]

    def get_experiment_properties(self, selected_rows):
        '''
        Extracts the properties of the experiment that belongs to the 
        selected_rows
        @param selected_rows: (dataframe) from select_rows_for_alg()
        @return dictionary: containing for each property the corresponding value (int, float, or str)
        '''
        # the following columns have unique values
        names_of_unique_columns = ["function", "dim", "chgperiods",
                                   "len_c_p", "ischgperiodrandom", "veclen", "peaks",
                                   "noise", "poschg", "fitchg", "expfilename"]
        first_row = selected_rows.iloc[0]  # select arbitrary row
        values_of_unique_columns = {}
        for name in names_of_unique_columns:
            value = first_row[name]
            if name == "poschg":
                # print(type(value))
                pass
            if (type(value) is np.float64 or type(value) is float) and math.isnan(value):
                # replace nan-values (i.e. empty cells) with empty string
                value = ''
            values_of_unique_columns[name] = value

        return values_of_unique_columns

    def values_for_runs_of_alg(self, selected_rows, metric):
        '''
        Extracts from the passed selected_rows the value of the specified 
        metric for all runs.

        @param selected_rows: (dataframe) result of select_rows_for_alg();
        @param metric: (string) name of the metric, must be the same as the 
        corresponding column name in the metric file
        '''
        return selected_rows[metric].values

    def create_test_result_file(self, df, result_file_name):
        '''
        Creates one file for results of the pairwise test two algorithms.

        @param df: csv-file containing all results for all algorithms
        @param result_file_name: name of the result file
        '''
        if not os.path.isfile(result_file_name):
            # construct column header (remove columns that are unnecessary
            first_metrics_file_line = list(df.columns.values)
            first_metrics_file_line.remove("algparams")
            first_metrics_file_line.remove("alg")
            first_metrics_file_line.remove("arrayfilename")
            first_metrics_file_line.remove("run")
            first_metrics_file_line.remove("predictor")
            first_metrics_file_line.remove("ks")
            first_metrics_file_line.remove("kernels")
            first_metrics_file_line.remove("lr")
            first_metrics_file_line.remove("epochs")
            first_metrics_file_line.remove("bs")
            first_metrics_file_line.remove("traindrop")
            first_metrics_file_line.remove("testdrop")

            # skip first column that only contains the row indices
            # is no longer the index, but the function name
            # first_metrics_file_line = first_metrics_file_line[1:]
            # make one string where the entries are comma-separated
            first_metrics_file_line_onestring = ','.join(
                first_metrics_file_line)
            first_metrics_file_line_onestring += "\n"

            print_line(result_file_name,
                       first_metrics_file_line_onestring)

    def compute_and_save_all_stattests(self, alternative, alg_combinations):
        '''
        Computes for all desired algorithm combinations statistical tests  for
        all problems and metrics.
        Creates for each algorithm pair one file with test results for all 
        experiments.
        '''
        metric_names = ['bog-for-run', 'bebc', 'rcs', 'arr',
                        'fit-ea-rmse', 'fit-foundpred-rmse', 'fit-truepred-rmse',
                        'pos-ea-rmse', 'pos-foundpred-rmse', 'pos-truepred-rmse']

        # load result file containing results for all algorithms and problems
        df = pd.read_csv(self.metric_file_path, ',')

        # executed algorithms (alg and params)
        algorithms = df.drop_duplicates(
            ['alg', 'algparams'])[['algparams', 'alg']].values

        # conducted experiments
        experiments = df['expfilename'].unique()
        for i, j in alg_combinations:
            # create file for test results
            result_file_name = self.stattest_dir_path + "whitney_pairwise_" + alternative + "_" + \
                ''.join(i) + "-" + ''.join(j) + ".csv"
            self.create_test_result_file(df, result_file_name)

            for exp in experiments:
                # select rows in metric-result file for the algorithms
                rows_alg1 = self.select_rows_for_alg(df, i, exp)
                rows_alg2 = self.select_rows_for_alg(df, j, exp)
                if rows_alg1.empty or rows_alg2.empty:
                    # this experiment was not executed with at least one of
                    # the algorithms, so no test can be conducted
                    continue

                # properties of experiment (should be same for both algs)
                props1 = self. get_experiment_properties(rows_alg1)
                props2 = self. get_experiment_properties(rows_alg2)
                assert props1 == props2

                p_values = {}
                for m in metric_names:
                    if not metric_computable_for_no(m) and ("_no_"in i or "_no_" in j):
                        p_value = "''"
                    else:
                        # metric value for each run
                        runs_alg1 = self.values_for_runs_of_alg(rows_alg1, m)
                        runs_alg2 = self.values_for_runs_of_alg(rows_alg2, m)
                        # execute test
                        _, p_value = stats.mannwhitneyu(
                            runs_alg1, runs_alg2, alternative=alternative)
                    p_values[m] = p_value

                # construct and print output line
                metric_values_to_print = [props1["function"],
                                          props1["dim"], props1["chgperiods"],
                                          props1["len_c_p"], props1["ischgperiodrandom"],
                                          props1["veclen"], props1["peaks"],
                                          props1["noise"], props1["poschg"],
                                          props1["fitchg"], p_values["bog-for-run"],
                                          p_values["bebc"], p_values["rcs"],
                                          p_values["arr"], p_values["fit-ea-rmse"],
                                          p_values["fit-foundpred-rmse"],
                                          p_values["fit-truepred-rmse"],
                                          p_values["pos-ea-rmse"],
                                          p_values["pos-foundpred-rmse"],
                                          p_values["pos-truepred-rmse"],
                                          props1["expfilename"]]
                print_to_file(result_file_name, metric_values_to_print)


def metric_computable_for_no(metric):
    '''
    Returns True if the metric was measured for EA without prediction model.
    '''
    not_computable_metrics = ['fit-ea-rmse', 'fit-foundpred-rmse', 'fit-truepred-rmse',
                              'pos-ea-rmse', 'pos-foundpred-rmse', 'pos-truepred-rmse']
    return metric not in not_computable_metrics


def define_test_combinations():
    combs = []
    for i in range(len(row_algs)):
        ra = row_algs[i]
        for ca in col_algs[i:]:
            combs.append((ra, ca))
    return combs


if __name__ == "__main__":
    calculator = StatisticalTestsCalculator()
    alternatives = ["less", "two-sided", "greater"]

    algs = ["dynea_tcn_auto_dynsig",
            "dynea_tcn_auto_00-8",
            "dynea_tcn_auto_08-0",
            "dynea_tcn_auto_36-2",
            "dynea_tcn_auto",
            "dynea_tcn_auto_90-0",
            "dynea_tcn_auto_95-4",
            "dynea_tcn",
            "dynea_autoregressive",
            "dynea_no"
            ]

    algs = ["dynea_no_noRND",
            "dynea_no_noVAR",
            "dynea_no_noPRE",
            "dynea_autoregressive_predRND",
            "dynea_autoregressive_predDEV",
            "dynea_tcn_predRND",
            "dynea_tcn_predDEV",
            "dynea_kalman_predRND",
            "dynea_kalman_predDEV",
            "dynea_kalman_predUNC",
            "dynea_kalman_predKAL",
            "dynea_tcn_auto_predRND",
            "dynea_tcn_auto_predDEV",
            "dynea_tcn_auto_predUNC",
            "dynea_tcn_auto_predKAL"
            ]
    col_algs = [a for a in algs[1:]]
    row_algs = [a for a in algs[:-1]]

    alg_combs = define_test_combinations()
    for altn in alternatives:
        calculator.compute_and_save_all_stattests(altn, alg_combs)

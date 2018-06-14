'''
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

        self.metric_path = path_to_output + "EvoStar_2018/"
        self.metric_file_path = self.metric_path + "metric_db.csv"
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
        return df.loc[(df["algparams"].isin([alg[0]])) &
                      (df["alg"].isin([alg[1]])) &
                      (df["expfilename"].isin([exp]))]

    def get_experiment_properties(self, selected_rows):
        '''
        Extracts the properties of the experiment that belongs to the 
        selected_rows
        @param selected_rows: (dataframe) from select_rows_for_alg()
        @return dictionary: containing for each property the corresponding value (int, float, or str)
        '''
        # the following columns have unique values
        names_of_unique_columns = ["function", "predictor", "dim", "chgperiods",
                                   "len_c_p", "ischgperiodrandom", "veclen", "peaks",
                                   "noise", "poschg", "fitchg", "expfilename"]
        first_row = selected_rows.iloc[0]  # select arbitrary row
        values_of_unique_columns = {}
        for name in names_of_unique_columns:
            value = first_row[name]
            if name == "poschg":
                print(type(value))
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
            # skip first column that only contains the row indices
            first_metrics_file_line = first_metrics_file_line[1:]
            # make one string where the entries are comma-separated
            first_metrics_file_line_onestring = ','.join(
                first_metrics_file_line)
            first_metrics_file_line_onestring += "\n"

            print_line(result_file_name,
                       first_metrics_file_line_onestring)

    def compute_and_save_all_stattests(self):
        '''
        Computes for all desired algorithm combinations statistical tests  for
        all problems and metrics.
        Creates for each algorithm pair one file with test results for all 
        experiments.
        '''
        metric_names = ['bog-for-run', 'bebc', 'rcs', 'arr']

        # load result file containing results for all algorithms and problems
        df = pd.read_csv(self.metric_file_path, ',')

        # executed algorithms (alg and params)
        algorithms = df.drop_duplicates(
            ['alg', 'algparams'])[['algparams', 'alg']].values

        # conducted experiments
        experiments = df['expfilename'].unique()
        for i, j in itertools.combinations(algorithms, 2):
            # create file for test results
            result_file_name = self.stattest_dir_path + "whitney_pairwise_" + \
                ''.join(i) + "-" + ''.join(j) + ".csv"
            self.create_test_result_file(df, result_file_name)

            for exp in experiments:
                # select rows in metric-result file for the algorithms
                rows_alg1 = self.select_rows_for_alg(df, i, exp)
                rows_alg2 = self.select_rows_for_alg(df, j, exp)
                # properties of experiment (should be same for both algs)
                props1 = self. get_experiment_properties(rows_alg1)
                props2 = self. get_experiment_properties(rows_alg2)
                assert props1 == props2

                p_values = {}
                for m in metric_names:
                    # metric value for each run
                    runs_alg1 = self.values_for_runs_of_alg(rows_alg1, m)
                    runs_alg2 = self.values_for_runs_of_alg(rows_alg2, m)
                    # execute test
                    _, p_value = stats.mannwhitneyu(
                        runs_alg1, runs_alg2, alternative='two-sided')
                    p_values[m] = p_value

                # construct and print output line
                metric_values_to_print = [props1["function"], props1["predictor"],
                                          props1["dim"], props1["chgperiods"],
                                          props1["len_c_p"], props1["ischgperiodrandom"],
                                          props1["veclen"], props1["peaks"],
                                          props1["noise"], props1["poschg"],
                                          props1["fitchg"], p_values["bog-for-run"],
                                          p_values["bebc"], p_values["rcs"],
                                          p_values["arr"], props1["expfilename"]]
                print_to_file(result_file_name, metric_values_to_print)


if __name__ == "__main__":
    calculator = StatisticalTestsCalculator()
    calculator.compute_and_save_all_stattests()

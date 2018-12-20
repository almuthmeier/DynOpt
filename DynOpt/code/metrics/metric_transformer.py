'''
Is based on the metric database produced by metrics_calculator.py

Created on Nov 21, 2018

@author: ameier
'''
import copy
import os

import numpy as np
import pandas as pd


class MetricTransformer():
    def __init__(self, path_to_output=None,
                 benchmarkfunctions=None, poschgtypes=None, fitchgtypes=None,
                 dims=None, noises=None, metric_filename=None, output_file_name=None):
        '''
        Initialize paths, properties of the experiments, etc.
        '''
        # TODO(exp) set parameters as required

        self.metrics = ["bog-for-run", "bebc", "rcs", "arr",
                        "ea-rmse", "foundpred-rmse", "truepred-rmse"]

        # output and benchmark path
        if path_to_output is None:  # assumes that  all input arguments are none
            # path to "..../DynOpt/code"
            path_to_code = os.path.abspath(os.pardir)
            path_to_datasets = '/'.join(path_to_code.split('/')
                                        [:-1]) + "/datasets/"
            path_to_output = '/'.join(path_to_code.split('/')
                                      [:-1]) + "/output/"
            #self.output_dir_path = path_to_output + "ESANN_2019/"
            self.output_dir_path = path_to_output = "/home/ameier/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/output_2018-11-20_ohneDrop_mitNOundARR/"
            self.output_dir_path = path_to_output = "/home/ameier/Documents/Promotion/Ausgaben/Uncertainty/Robustness/output_2018-12-11/"
            # , "rosenbrock", "rastrigin"]  # sphere, rosenbrock, mpbnoisy,griewank
            self.benchmarkfunctions = [
                "rastrigin", "sphere", "mpbcorr"]
            # ["linear", "sine", "circle"]
            self.poschgtypes = ["sine"]  # , "linear"]
            self.fitchgtypes = ["none"]
            self.dims = [1, 5, 10, 50]
            self.noises = [0.0, 0.2]
            self.metric_filename = "metric_db_noiseevalation_2018-12-14_rmses.csv"
            self.output_file_name = "avg_metric_db_2018-12-11_rmses.csv"
        else:
            self.output_dir_path = path_to_output
            self.benchmarkfunctions = benchmarkfunctions
            self.poschgtypes = poschgtypes
            self.fitchgtypes = fitchgtypes
            self.dims = dims
            self.noises = noises
            self.metric_filename = metric_filename
            self.output_file_name = output_file_name

    def make_table_with_selected_data(self):
        path_transformed_db = "/home/ameier/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/output_2018-11-23/"
        path_transformed_db = "/home/ameier/Documents/Promotion/Ausgaben/Uncertainty/Robustness/output_2018-12-11/"
        full_input_name = path_transformed_db + "avg_metric_db_2018-12-11_rmses.csv"
        all_data = pd.read_csv(full_input_name)

        # functions that are combined into one file
        # , "tftlrnndense"]
        preds = ["no", "autoregressive", "tfrnn", "tftlrnn", "tftlrnndense"]
        preds = ["no", "autoregressive", "tfrnn"]
        functions = ["sphere", "mpbcorr", "rastrigin"]
        algparams = ['steps_50', 'nonoise']
        poschgs = ['mixture', 'sine']
        dims = [2, 10]
        noises = [0.05]
        #steps = 50
        # "bog-for-run"  # , "bebc", "rcs", "arr" "ea-rmse","foundpred-rmse","truepred-rmse"
        metric = "truepred-rmse"

        full_output_file_name = path_transformed_db + "selection_fcts-" + \
            str(functions) + "metric-" + str(metric) + "_rmses.csv"

        # column names for average metric values (element-wise string
        # concatenation)
        avg_cols = list(np.core.defchararray.add(preds, ["_avg"] * len(preds)))
        std_cols = list(np.core.defchararray.add(preds, ["_std"] * len(preds)))
        header_line = ["function", "algparams", "dim",
                       "poschg"] + avg_cols + std_cols

        # get values for experiments
        data = []
        for f in functions:
            print("f: ", f)
            for d in dims:
                print("    d: ", d)
                is_first_mpb = True
                for poschg in poschgs:
                    if is_first_mpb and f == "mpbcorr":
                        is_first_mpb = False
                    elif f == "mpbcorr":
                        break
                    for param in algparams:
                        print("            param: ", param)
                        row_prefix = [f, param, d,
                                      poschg if f != "mpbcorr" else ""]
                        avg_values = []
                        std_values = []
                        for p in preds:
                            if param == 'nonoise' and p != "tfrnn":
                                avg_values.append("-")
                                std_values.append("-")
                            print("                p: ", p)
                            if f == "mpbcorr":
                                v = all_data.loc[(all_data["function"] == f) &
                                                 (all_data["dim"] == d) &
                                                 (all_data["predictor"] == p) &
                                                 (all_data["algparams"] == param), metric]
                            elif f == "sphere" or f == "rastrigin":  # no noise available
                                v = all_data.loc[(all_data["function"] == f) &
                                                 (all_data["dim"] == d) &
                                                 (all_data["predictor"] == p) &
                                                 (all_data["poschg"] == poschg) &
                                                 (all_data["algparams"] == param), metric]

                            # requires that the avg row is before the std
                            # row!!
                            if v.values != []:
                                avg_values.append(v.iloc[0])
                                std_values.append(v.iloc[1])
                        row = row_prefix + avg_values + std_values
                        data.append(row)
                        # if not self.function_has_noise_param(f):
                        # do not execute again otherwise duplicate rows would
                        # appear
                        #    break
        df = pd.DataFrame(data, columns=header_line)
        df.to_csv(full_output_file_name, index=False)

    def convert_buffer_to_rows(self, buffer, all_cols, col_idx_exp_filename,
                               col_idx_run, col_idx_metrics_dict, data, line):
        print("predictor: ", buffer[-1][1])
        buffered_df = pd.DataFrame(buffer, columns=all_cols)
        # compute average/stddev column-wise
        averages = buffered_df[self.metrics].mean(axis=0)
        print("avg: ", averages[0])
        stddevs = buffered_df[self.metrics].std(axis=0)
        print("stddev: ", stddevs[0])
        # cut experiment and array file name
        avg_row = copy.deepcopy(buffer[-1][:col_idx_exp_filename])
        std_row = copy.deepcopy(buffer[-1][:col_idx_exp_filename])
        # insert label
        avg_row[col_idx_run] = "avg"
        std_row[col_idx_run] = "std"
        # insert average/stddev values
        for m in self.metrics:
            avg_row[col_idx_metrics_dict[m]] = averages[m]
            std_row[col_idx_metrics_dict[m]] = stddevs[m]

        # empty the buffer and start filling it again
        data.append(avg_row)
        data.append(std_row)

        # last time the function is called line is None
        if line is not None:
            buffer = []
            buffer.append(line.values.flatten())
            return buffer

    def compute_avg_and_stddev(self):
        '''
        Step 1)

        Transforms metric_db into file containing avg and std as rows.
        '''

        full_path = self.output_dir_path + self.metric_filename
        whole_file = pd.read_csv(full_path)
        all_cols = list(whole_file.columns.values)
        col_idx_arrayfilename = whole_file.columns.get_loc('arrayfilename')
        col_idx_exp_filename = whole_file.columns.get_loc('expfilename')
        col_idx_run = whole_file.columns.get_loc('run')
        col_idx_metrics_dict = {
            m: whole_file.columns.get_loc(m) for m in self.metrics}

        data = []
        # rows for all repetitions of one experiment for one algorithm
        buffer = []
        is_header = True
        for line in pd.read_csv(full_path, chunksize=1, header=None, index_col=False, encoding='utf-8'):
            if is_header:
                is_header = False
                continue
            if len(buffer) == 0:  # first run of experiment
                buffer.append(line.values.flatten())
            else:
                # compare array file name to see whether the current line is
                # a further run of the same experiment
                curr_array_f_name = line[col_idx_arrayfilename].values[0]
                prev_array_f_name = buffer[-1][col_idx_arrayfilename]
                # cut repetition and file ending
                curr_array_f_name = curr_array_f_name.split("_")[:-1]
                prev_array_f_name = prev_array_f_name.split("_")[:-1]

                if curr_array_f_name == prev_array_f_name:  # further run for exp
                    buffer.append(line.values.flatten())
                else:  # all repetitions saved, so compute average + stddev
                    buffer = self.convert_buffer_to_rows(buffer, all_cols, col_idx_exp_filename,
                                                         col_idx_run, col_idx_metrics_dict, data, line)

        self.convert_buffer_to_rows(buffer, all_cols, col_idx_exp_filename,
                                    col_idx_run, col_idx_metrics_dict, data, None)
        df = pd.DataFrame(data, columns=all_cols[:col_idx_exp_filename])
        df.to_csv(self.output_dir_path + self.output_file_name, index=False)

    def function_has_noise_param(self, f):
        return f == "mpbcorr"


def start_computing_avgs_stddevs(path_to_output=None,
                                 benchmarkfunctions=None, poschgtypes=None,
                                 fitchgtypes=None, dims=None, noises=None, metric_filename=None,
                                 output_file_name=None):
    calculator = MetricTransformer(path_to_output, benchmarkfunctions, poschgtypes, fitchgtypes,
                                   dims, noises, metric_filename, output_file_name)
    # Step 1)
    # calculator.compute_avg_and_stddev()
    #print("saved metric database", flush=True)

    # Step 2)
    calculator.make_table_with_selected_data()


if __name__ == '__main__':
    start_computing_avgs_stddevs()

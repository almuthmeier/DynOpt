'''
Is based on the metric database produced by metrics_calculator.py

Created on Nov 21, 2018

@author: ameier
'''
import os
import pandas as pd


class MetricTransformer():
    def __init__(self, path_to_output=None,
                 benchmarkfunctions=None, poschgtypes=None, fitchgtypes=None,
                 dims=None, noises=None, metric_filename=None, output_file_name=None):
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
            #self.output_dir_path = path_to_output + "ESANN_2019/"
            self.output_dir_path = path_to_output = "/home/ameier/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/output_2018-11-20_ohneDrop_mitNOundARR/"
            # , "rosenbrock", "rastrigin"]  # sphere, rosenbrock, mpbnoisy,griewank
            self.benchmarkfunctions = [
                "rastrigin", "griewank", "sphere", "mpbcorr"]
            # ["linear", "sine", "circle"]
            self.poschgtypes = ["mixture"]  # , "linear"]
            self.fitchgtypes = ["none"]
            self.dims = [1, 5, 10, 50]
            self.noises = [0.0, 0.2, 0.4]
            self.metric_filename = "metric_db_stepevaluation.csv"
            self.output_file_name = "avg_metric_db.csv"
        else:
            self.output_dir_path = path_to_output
            self.benchmarkfunctions = benchmarkfunctions
            self.poschgtypes = poschgtypes
            self.fitchgtypes = fitchgtypes
            self.dims = dims
            self.noises = noises
            self.metric_filename = metric_filename
            self.output_file_name = output_file_name

    def compute_avg_and_stddev(self):
        metrics = ["bog-for-run", "bebc", "rcs", "arr"]
        full_path = self.output_dir_path + self.metric_filename
        whole_file = pd.read_csv(full_path)
        all_cols = list(whole_file.columns.values)
        col_idx_arrayfilename = whole_file.columns.get_loc('arrayfilename')
        col_idx_exp_filename = whole_file.columns.get_loc('expfilename')
        col_idx_run = whole_file.columns.get_loc('run')
        col_idx_metrics_dict = {
            m: whole_file.columns.get_loc(m) for m in metrics}

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
                    buffered_df = pd.DataFrame(buffer, columns=all_cols)
                    # compute average/stddev column-wise
                    averages = buffered_df[metrics].mean(axis=0)
                    stddevs = buffered_df[metrics].std(axis=0)
                    # cut experiment and array file name
                    avg_row = line.values[0][:col_idx_exp_filename]
                    std_row = line.values[0][:col_idx_exp_filename]
                    # insert label
                    avg_row[col_idx_run] = "avg"
                    std_row[col_idx_run] = "std"
                    # insert average/stddev values
                    for m in metrics:
                        avg_row[col_idx_metrics_dict[m]] = averages[m]
                        std_row[col_idx_metrics_dict[m]] = stddevs[m]

                    # empty the buffer and start filling it again
                    data.append(avg_row)
                    data.append(std_row)
                    buffer = []
                    buffer.append(line.values.flatten())

        df = pd.DataFrame(data, columns=all_cols[:col_idx_exp_filename])
        df.to_csv(self.output_dir_path + self.output_file_name, index=False)


def start_computing_avgs_stddevs(path_to_output=None,
                                 benchmarkfunctions=None, poschgtypes=None,
                                 fitchgtypes=None, dims=None, noises=None, metric_filename=None,
                                 output_file_name=None):
    calculator = MetricTransformer(path_to_output, benchmarkfunctions, poschgtypes, fitchgtypes,
                                   dims, noises, metric_filename, output_file_name)
    calculator.compute_avg_and_stddev()
    print("saved metric database", flush=True)


if __name__ == '__main__':
    start_computing_avgs_stddevs()

'''
Functionality to print the result files.
'''
import datetime
from os.path import isfile, join
from posix import listdir
import re
import warnings


def print_to_file(file_name, values):
    '''
    Prints a list of values as one line into a csv file.
    @param values: list of scalar values containing the values for this metrics (in this specific order):
    file_name, problem, dim, pos_chng_type, fit_chng_type, avg_bog, arr_avg, arr_stddev, bebc_avg, bebc_stddev
    @param file_name: file name including its path and format
    '''
    # convert list of values to string of comma separated values
    # https://stackoverflow.com/questions/44778/how-would-you-make-a-comma-separated-string-from-a-list
    # (16.10.17)
    line = ','.join(map(str, values)) + "\n"
    print_line(file_name, line)


def print_line(file_name, line):
    '''
    Appends line to result csv-file.
    @param file_name: file name including its path and format
    '''
    # https://stackoverflow.com/questions/31948879/using-explict-predefined-validation-set-for-grid-search-with-sklearn
    file = open(file_name, 'a')
    file.write(line)
    file.close()


def get_current_day_time():
    current_date = datetime.datetime.now()
    day = str(current_date.year) + '-' + \
        str(current_date.month).zfill(2) + '-' + str(current_date.day).zfill(2)
    time = str(current_date.hour).zfill(2) + ':' + \
        str(current_date.minute).zfill(2)
    return day, time


def get_metrics_file_name(metrics_file_path, predictor_name, benchmarkfunction, day, time):
    return metrics_file_path + predictor_name + "_" + benchmarkfunction + "_" + \
        day + '_' + time + ".csv"


def get_logs_file_name(logs_file_path, predictor_name, benchmarkfunction, day, time):
    return logs_file_path + predictor_name + "_" + benchmarkfunction + "_" + \
        day + '_' + time + ".txt"


def convert_exp_to_arrays_file_name(predictor, exp_file_name, day, time, repetition_ID, chgperiods):
    # generate array file name from the exeriment file name with some
    # replacements:
    # append predictor name at the beginning
    arrays_file_name = predictor + "_" + exp_file_name
    # replace date and time with start date and time;
    # \d is the same as [0-9]
    # {2} means that the previous regular expression must occour exactly 2 times
    arrays_file_name = re.sub(
        "_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}.npz", "_" + day + "_" + time + ".npz", arrays_file_name)
    # append the repetition number at the end before the file ending
    arrays_file_name = arrays_file_name.replace(
        ".npz", "_" + str(repetition_ID) + ".npz")
    # insert the correct number of change periods
    # https://stackoverflow.com/questions/16720541/python-string-replace-regular-expression
    # (15.5.18)
    arrays_file_name = re.sub(
        "_chgperiods-[0-9]+_", "_chgperiods-" + str(chgperiods) + "_", arrays_file_name)
    return arrays_file_name


def get_array_file_names_for_experiment_file_name(exp_file_name, arrays_path):
    '''
    Get all file names in the arrays_path and select only those corresponding 
    to the exp_file_name.
    '''
    # abhängig, ob mpb oder sphere
    splitted_benchmark_file_name = exp_file_name.split('_')
    function = splitted_benchmark_file_name[0]
    dim = splitted_benchmark_file_name[1].split('-')[1]
    #chgperiods = splitted_benchmark_file_name[2].split('-')[1]

    if function == "sphere" or function == "rastrigin" or function == "rosenbrock":
        pch = splitted_benchmark_file_name[3].split('-')[1]
        fch = splitted_benchmark_file_name[4].split('-')[1]
        selected_files = [f for f in listdir(arrays_path) if (isfile(join(
            arrays_path, f)) and f.endswith('.npz') and function in f
            and ("_d-" + str(dim) + "_") in f and ("_pch-" + pch + "_")in f
            and ("_fch-" + fch + "_") in f)]
    elif function == "mpbnoisy" or function == "mpbrandom":
        veclen = splitted_benchmark_file_name[3].split('-')[1]
        peaks = splitted_benchmark_file_name[4].split('-')[1]
        noise = splitted_benchmark_file_name[5].split('-')[1]
        selected_files = [f for f in listdir(arrays_path) if (isfile(join(
            arrays_path, f)) and f.endswith('.npz') and function in f
            and ("_d-" + str(dim) + "_") in f and ("_veclen-" + str(veclen) + "_")in f
            and ("_peaks-" + peaks + "_") in f and ("_noise-" + str(noise) + "_") in f)]
    else:
        warnings.warn("unknown benchmark function")
    return selected_files


def select_experiment_files(benchmark_path, benchmarkfunction, poschgtypes,
                            fitchgtypes, dims, noises):
    '''
    Selects some experiment files for a benchmark function to run only these experiments.

    @param all_experiment_files: list of all filenames (absolute paths) 
    being in the benchmark function directory
    '''

    all_experiment_files = [f for f in listdir(benchmark_path) if
                            (isfile(join(benchmark_path, f)) and
                             f.endswith('.npz') and
                             benchmarkfunction in f)]
    # TODO(dev) add further benchmarks
    selected_exp_files = None
    if benchmarkfunction == "sphere" or \
            benchmarkfunction == "rastrigin" or \
            benchmarkfunction == "rosenbrock":
        selected_exp_files = [f for f in all_experiment_files if (
                              any(("_d-" + str(dim) + "_") in f for dim in dims) and
                              any(("_pch-" + poschgtype) + "_" in f for poschgtype in poschgtypes) and
                              any(("_fch-" + fitchgtype) + "_" in f for fitchgtype in fitchgtypes))]
    elif benchmarkfunction == "mpbnoisy" or \
            benchmarkfunction == "mpbrandom":
        selected_exp_files = [f for f in all_experiment_files if (
                              any(("_d-" + str(dim) + "_" in f for dim in dims) and
                                  ("_noise-" + str(noise) + "_") in f for noise in noises))]
    return selected_exp_files

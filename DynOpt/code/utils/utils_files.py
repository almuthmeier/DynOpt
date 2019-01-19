'''
Functionality to print the result files.
'''
import datetime
from os.path import isfile, join
from posix import listdir
import re
import warnings

import numpy as np


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


def get_logs_file_name(logs_file_path, predictor_name, benchmarkfunction, dims,
                       poschgtype,
                       day, time, noise,
                       ks, n_kernels, lr, n_epochs, bs, train_drop, test_drop):

    beginning = logs_file_path + predictor_name + "_" + benchmarkfunction + \
        "_d-" + str(dims) + "_pch-" + str(poschgtype) + \
        "_noise-" + "{:.2f}".format(noise) + "_"
    between = "ks-" + str(ks) + "_kernels-" + str(n_kernels) + "_lr-" + "{:.3f}".format(lr) + \
        "_epochs-" + str(n_epochs) + "_bs-" + str(bs) + "_traindrop-" + "{:.3f}".format(train_drop) + \
        "_testdrop-" + "{:.3}".format(test_drop) + "_"
    ending = day + '_' + time.replace(":", "-") + ".txt" % noise

    return beginning + between + ending


def convert_exp_to_arrays_file_name(predictor, exp_file_name, day, time,
                                    repetition_ID, chgperiods, lenchgperiod,
                                    ischgperiodrandom,
                                    kernel_size, n_kernels, lr, epochs,
                                    batchsize, traindropout, testdropout):
    '''
        generate array file name from the experiment file name with some
        replacements
    '''
    # append predictor name at the beginning
    arrays_file_name = predictor + "_" + exp_file_name
    # replace date and time with start date and time;
    # \d is the same as [0-9]
    # {2} means that the previous regular expression must occur exactly 2 times
    arrays_file_name = re.sub(
        "_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}.npz", "_" + day + "_" + time + ".npz", arrays_file_name)
    # append the repetition number at the end before the file ending
    arrays_file_name = arrays_file_name.replace(
        ".npz", "_" + str(repetition_ID).zfill(2) + ".npz")
    # substitute the number of change periods by the correct number, insert
    # after that the period length and whether the changes are random
    # https://stackoverflow.com/questions/16720541/python-string-replace-regular-expression
    # (15.5.18)
    n_periods = "_chgperiods-" + str(chgperiods)
    len_periods = "_lenchgperiod-" + str(lenchgperiod)
    israndom = "_ischgperiodrandom-" + str(ischgperiodrandom)
    arrays_file_name = re.sub(
        "_chgperiods-[0-9]+_", n_periods + len_periods + israndom + "_", arrays_file_name)

    # append network specification (used for TCN)
    if predictor == 'tcn':
        # insert tcn_string before the date
        date_start_idx = re.search(
            "_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}", arrays_file_name).start()
        tcn_string = "_ks-" + str(kernel_size) + "_kernels-" + str(n_kernels) + \
            "_lr-" + str(lr) + "_epochs-" + str(epochs) + "_bs-" + str(batchsize) + \
            "_traindrop-" + str(traindropout) + "_testdrop-" + str(testdropout)

        arrays_file_name = arrays_file_name[:date_start_idx] + \
            tcn_string + arrays_file_name[date_start_idx:]
        #kernel_size, n_kernels, lr, epochs,
        #batchsize, traindropout, testdropout
    return arrays_file_name


def get_run_number_from_array_file_name(array_file_name):
    run = re.search("_\d+.npz", array_file_name).group()
    run = run.replace("_", "")
    run = int(run.replace(".npz", ""))
    return run


def get_info_from_array_file_name(array_file_name):
    '''
    Extracts information from an output array file name and converts it to its actual 
    data type.

    Example file names:
    rnn_mpbnoisy_d-2_chgperiods-10_lenchgperiod-20_ischgperiodrandom-False_veclen-0.6_peaks-10_noise-0.0_2018-05-24_10:22_00.npz
    gdbg_f-0_t-1_d-5_chgperiods-100_peaks-10_2018-07-03_12:25
    rnn_rosenbrock_d-2_chgperiods-10_lenchgperiod-20_ischgperiodrandom-False_pch-linear_fch-none_2018-05-24_10:27_00.npz

    tcn_sphere_d-2_chgperiods-19_lenchgperiod-20_ischgperiodrandom-False_pch-sinefreq_fch-none_ks-3_kernels-16_lr-0.002_epochs-80_bs-32_traindrop-0.1_testdrop-0.0_2019-01-18_15:06_00.npz
    @return tupel, the extracted information
    '''

    #

    dim = int(re.search('d-[0-9]+', array_file_name).group().split('-')[1])
    chgperiods = int(re.search('chgperiods-[0-9]+',
                               array_file_name).group().split('-')[1])
    lenchgperiod = int(
        re.search('lenchgperiod-[0-9]+', array_file_name).group().split('-')[1])
    ischgperiodrandom_string = re.search(
        'ischgperiodrandom-[False|True]+', array_file_name).group().split('-')[1]
    ischgperiodrandom = ischgperiodrandom_string == 'True'
    try:
        veclen = float(re.search('veclen-[0-9]+\.[0-9]+',
                                 array_file_name).group().split('-')[1])
    except AttributeError:
        # AttributeError: 'NoneType' object has no attribute 'group'
        veclen = None
    try:
        peaks = int(re.search('peaks-[0-9]+',
                              array_file_name).group().split('-')[1])
    except AttributeError:
        peaks = None
    try:
        noise = float(re.search('noise-[0-9]+\.[0-9]+',
                                array_file_name).group().split('-')[1])
    except AttributeError:
        noise = None
    try:
        poschg = re.search('pch-[a-z]+', array_file_name).group().split('-')[1]
    except AttributeError:
        poschg = None
    try:
        fitchg = re.search('fch-[a-z]+', array_file_name).group().split('-')[1]
    except AttributeError:
        fitchg = None
    try:
        kernel_size = int(re.search(
            '_ks-[0-9]+', array_file_name).group().split('-')[1])
    except AttributeError:
        kernel_size = None
    try:
        n_kernels = int(re.search(
            '_kernels-[0-9]+', array_file_name).group().split('-')[1])
    except AttributeError:
        n_kernels = None
    try:
        l_rate = float(re.search('_lr-[0-9]+\.[0-9]+',
                                 array_file_name).group().split('-')[1])
    except AttributeError:
        l_rate = None
    try:
        n_epochs = int(re.search(
            '_epochs-[0-9]+', array_file_name).group().split('-')[1])
    except AttributeError:
        n_epochs = None
    try:
        batch_size = int(re.search(
            '_bs-[0-9]+', array_file_name).group().split('-')[1])
    except AttributeError:
        batch_size = None
    try:
        train_drop = float(re.search(
            '_traindrop-[0-9]+\.[0-9]+', array_file_name).group().split('-')[1])
    except AttributeError:
        train_drop = None
    try:
        test_drop = float(re.search(
            '_testdrop-[0-9]+\.[0-9]+', array_file_name).group().split('-')[1])
    except AttributeError:
        test_drop = None

    # get further info (without keys): predictor, benchmark, date, time, run
    # TODO(dev) here the order of the info in the file name is important
    predictor, benchmark = array_file_name.split('_')[0:2]
    date = re.search("\d{4}-\d{2}-\d{2}", array_file_name).group()
    time = re.search("_\d{2}:\d{2}_", array_file_name).group()
    run = get_run_number_from_array_file_name(array_file_name)

    return (predictor, benchmark, dim, chgperiods, lenchgperiod,
            ischgperiodrandom, veclen, peaks, noise, poschg, fitchg, date, time, run,
            kernel_size, n_kernels, l_rate, n_epochs, batch_size, train_drop, test_drop)


def get_sorted_array_file_names_for_experiment_file_name(exp_file_name, arrays_path):
    '''
    Get all file names in the arrays_path and select only those corresponding 
    to the exp_file_name.
    Names are sorted.
    '''
    # abhängig, ob mpb oder sphere
    splitted_benchmark_file_name = exp_file_name.split('_')
    function = splitted_benchmark_file_name[0]
    dim = splitted_benchmark_file_name[1].split('-')[1]
    #chgperiods = splitted_benchmark_file_name[2].split('-')[1]

    if function == "sphere" or function == "rastrigin" or function == "rosenbrock" or \
            function == "griewank":
        pch = splitted_benchmark_file_name[3].split('-')[1]
        fch = splitted_benchmark_file_name[4].split('-')[1]
        selected_files = [f for f in listdir(arrays_path) if (isfile(join(
            arrays_path, f)) and f.endswith('.npz') and function in f
            and ("_d-" + str(dim) + "_") in f and ("_pch-" + pch + "_")in f
            and ("_fch-" + fch + "_") in f)]
    elif function == "mpbnoisy" or function == "mpbrand" or function == "mpbcorr":
        veclen = splitted_benchmark_file_name[3].split('-')[1]
        peaks = splitted_benchmark_file_name[4].split('-')[1]
        noise = splitted_benchmark_file_name[5].split('-')[1]
        selected_files = [f for f in listdir(arrays_path) if (isfile(join(
            arrays_path, f)) and f.endswith('.npz') and function in f
            and ("_d-" + str(dim) + "_") in f and ("_veclen-" + str(veclen) + "_")in f
            and ("_peaks-" + peaks + "_") in f and ("_noise-" + "{:.2f}".format(noise) + "_") in f)]
    else:
        warnings.warn("unknown benchmark function")
    return np.sort(selected_files)


def get_array_names_for_ks_and_filters(array_names, ks, n_filters):
    return [f for f in array_names if(("_ks-" + str(ks)) in f and ("_kernels-" + str(n_filters)) in f)]


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
            benchmarkfunction == "rosenbrock" or \
            benchmarkfunction == "griewank":
        selected_exp_files = [f for f in all_experiment_files if (
                              any(("_d-" + str(dim) + "_") in f for dim in dims) and
                              any(("_pch-" + poschgtype + "_") in f for poschgtype in poschgtypes) and
                              any(("_fch-" + fitchgtype + "_") in f for fitchgtype in fitchgtypes))]
    elif benchmarkfunction == "mpbnoisy" or \
            benchmarkfunction == "mpbrand" or \
            benchmarkfunction == "mpbcorr":
        selected_exp_files = [f for f in all_experiment_files if (
                              any(("_d-" + str(dim) + "_") in f for dim in dims) and
                              any(("_noise-" + "{:.2f}".format(noise) + "_") in f for noise in noises))]
    return selected_exp_files


def get_full_tl_model_name(path_to_parent_dir, dim):
    '''
    @returns full path to saved pre-trained tl model for desired dimensionality
    '''
    model_file = [f for f in listdir(path_to_parent_dir) if
                  (isfile(join(path_to_parent_dir, f)) and
                   f.endswith('.ckpt.meta') and
                   "dim-" + str(dim) + "_" in f)]
    assert len(model_file) == 1, "false number of files: " + \
        str(len(model_file))

    # clip ".meta" from the end
    model_file = model_file[0].replace(".meta", "")
    full_path = path_to_parent_dir + model_file
    return full_path

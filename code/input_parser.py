'''
Created on May 9, 2018

@author: ameier
'''
import os
import sys

from comparison import PredictorComparator
from utils.utils_print import get_current_day_time


sys.path.append(os.path.abspath(os.pardir))
# from predictor_comparison import get_logs_file_name,\
#   run_experiments_for_sphere_rosenbrock_rastrigin, mpb_evaluation


def define_parser_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    # "dynea" or "dynpso"
    parser.add_argument("-algorithm", type=str)

    parser.add_argument("-repetitions", type=int)
    # "mpb" or "sphere-rastrigin-rosenbrock" (alt)
    # sphere, rosenbrock, rastrigin, mpbnoisy, mpbrandom (neu)
    # defines the benchmark function, must be located in the datasets folder of
    # this project
    parser.add_argument("-benchmarkfunction", type=str)
    # parent directory of the benchmark functions and child directory of the
    # datasets folder of this project
    parser.add_argument("-benchmarkfunctionfolder", type=str)

    # str, mpbnoisy, mpbrand, roslenchg, roslenchggen,strneurons
    # TODO es könnte sein, dass dieselbe Benchmarkfunktion mit unterschiedlichen
    # Parametereinstellungen des Algorithmus ausgeführt wird. Dann dürfen die
    # Ergebnisse nicht im selben Ordner abgelegt werden -> Ausgabeordner
    # also auf andere Weise festlegen?
    # name of output folder
    parser.add_argument("-outputdirectory", type=str)
    # path to output folder
    parser.add_argument("-outputdirectorypath", type=str)

    # for PSO
    parser.add_argument("-c1", type=float)  # sind Zahlen erlaubt?
    parser.add_argument("-c2", type=float)
    parser.add_argument("-c3", type=float)
    parser.add_argument("-insertpred", type=bool)
    parser.add_argument("-adaptivec3", type=bool)
    parser.add_argument("-nparticles", type=int)

    # for EA
    parser.add_argument("-mu", type=int)
    parser.add_argument("-la", type=int)
    parser.add_argument("-ro", type=int)
    parser.add_argument("-mean", type=float)
    parser.add_argument("-sigma", type=float)
    parser.add_argument("-trechenberg", type=int)
    parser.add_argument("-tau", type=float)

    # for predictor
    # no, rnn, autoregressive
    parser.add_argument("-predictor", type=str)
    parser.add_argument("-timesteps", type=int)

    # for ANN predictor
    # fixed20 or dyn1.3
    parser.add_argument("-neuronstype", type=str)
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-batchsize", type=int)
    # machine dependent
    parser.add_argument("-ngpus", type=int)

    return parser


def initialize_comparator(parser):
    if not len(sys.argv) > 1:
        # no input values existing
        # https://stackoverflow.com/questions/10698468/argparse-check-if-any-arguments-have-been-passed/10699527
        # (11.5.18)
        initialize_comparator_manually()
    else:
        initialize_comparator_manually(parser)


def initialize_comparator_manually():
    comparator = PredictorComparator()
    comparator.algorithm = "dynpso"
    comparator.repetitions = 1
    comparator.benchmarkfunction = "sphere"
    comparator.benchmarkfunctionfolder = "GECCO_2018"
    comparator.outputdirectory = "mytests"
    comparator.outputdirectorypath = ""

    if args.algorithm == "dynpso":
        comparator.c1 = args.c1
        comparator.c2 = args.c2
        comparator.c3 = args.c3
        comparator.insertpred = args.insertpred
        comparator.adaptivec3 = args.adaptivec3
        comparator.nparticles = args.nparticles

        print(args.algorithm)
    elif args.algorithm == "dynea":
        comparator.mu = args.mu
        comparator.la = args.la
        comparator.ro = args.ro
        comparator.mean = args.mean
        comparator.sigma = args.sigma
        comparator.trechenberg = args.trechenberg
        comparator.tau = args.tau

    if args.predictor:
        comparator.predictor = args.predictor
        comparator.timesteps = args.timesteps
        if args.predictor == "rnn":
            comparator.neuronstype = args.neuronstype
            comparator.epochs = args.epochs
            comparator.batchsize = args.batchsize
            comparator.ngpus = args.ngpus


def initialize_comparator_with_read_inputs():
    args = parser.parse_args()

    n_current_inputs = len(vars(args))

    if n_current_inputs != 25:
        print("false number of inputs")
        exit(0)

    comparator = PredictorComparator()
    comparator.algorithm = args.algorithm
    comparator.repetitions = args.repetitions
    comparator.benchmarkfunction = args.benchmarkfunction
    comparator.benchmarkfunctionfolder = args.benchmarkfunctionfolder
    comparator.outputdirectory = args.outputdirectory
    comparator.outputdirectorypath = args.outputdirectorypath

    if args.algorithm == "dynpso":
        comparator.c1 = args.c1
        comparator.c2 = args.c2
        comparator.c3 = args.c3
        comparator.insertpred = args.insertpred
        comparator.adaptivec3 = args.adaptivec3
        comparator.nparticles = args.nparticles

        print(args.algorithm)
    elif args.algorithm == "dynea":
        comparator.mu = args.mu
        comparator.la = args.la
        comparator.ro = args.ro
        comparator.mean = args.mean
        comparator.sigma = args.sigma
        comparator.trechenberg = args.trechenberg
        comparator.tau = args.tau

    if args.predictor:
        comparator.predictor = args.predictor
        comparator.timesteps = args.timesteps
        if args.predictor == "rnn":
            comparator.neuronstype = args.neuronstype
            comparator.epochs = args.epochs
            comparator.batchsize = args.batchsize
            comparator.ngpus = args.ngpus


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('fork')

    import copy
    import itertools

    from os import listdir
    from os.path import isfile, join
    import random

    import numpy as np

    from multiprocessing import Pool
    # tf.reset_default_graph()

    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    # (17.8.17)
    #os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(7)
    random.seed(98)

    # =======================================================================
    # define parser arguments
    # =======================================================================
    parser = define_parser_arguments()
    # =======================================================================
    # read parser inputs
    # =======================================================================
    initialize_comparator(parser)
    # read_inputs_and_initialize_comparator(parser)

    # -------------------------------------------------------------------------

    # algorithm = "dynea"  # "dynea" or "dynpso"
    if args.algorithm == "dynea":
        from algorithms.dynea import define_settings_and_run
    elif args.algorithm == "dynpso":
        from algorithms.dynpso import define_settings_and_run

    if len(sys.argv) <= 2:
        if len(sys.argv) <= 1:  # one argument is always in the input
            predictor_name = "no"  # no, rnn, autoregressive
        elif len(sys.argv) == 2:
            predictor_name = sys.argv[1]
        n_gpus = 1
        c1 = 1.496180
        c2 = 1.496180
        c3 = 1.496180
        insert_pred_as_ind = True
        adaptive_c3 = False
        # "mpb" or "sphere-rastrigin-rosenbrock"
        problem = "sphere-rastrigin-rosenbrock"
        # str, mpbnoisy, mpbrand, roslenchg, roslenchggen (folder name!),
        # strneurons
        experiment_folder = "str"
        if predictor_name == "no":  # TODO immer einstellen
            addition = ""  # "3/"  # "2plusInd/"  # 3/, 3plusInd/, 2plusInd/
        else:
            addition = ""
        results_path = os.path.expanduser(
            "~/Documents/Promotion/Ausgaben/Predictorvergleich/2018-03-13/" + experiment_folder + "/" + predictor_name + "/" + addition)
    else:
        assert len(sys.argv) == 26, "false number of input arguments"
        # example call:
        # python3.5 predictor_comparison.py rnn 1.49618 1.49618 1.49618 True False
        # sphere-rastrigin-rosenbrock str ausprobieren/ 3plusInd/ 2
        predictor_name = sys.argv[1]
        c1 = float(sys.argv[2])
        c2 = float(sys.argv[3])
        c3 = float(sys.argv[4])
        insert_pred_as_ind = bool(sys.argv[5])  # True, False
        adaptive_c3 = bool(sys.argv[6])
        problem = sys.argv[7]  # "mpb" or "sphere-rastrigin-rosenbrock"
        experiment_folder = sys.argv[8]  # str, mpbnoisy, mpbrand, roslenchg..
        date = sys.argv[9]
        addition = sys.argv[10]
        n_gpus = int(sys.argv[11])
        results_path = os.path.expanduser(
            "~/Documents/Promotion/Ausgaben/Predictorvergleich/2018-03-13/" + date + experiment_folder + "/" + predictor_name + "/" + addition)
    print("Predictor: ", predictor_name)

    repetitions = 1
    conf_folder = "EvoStar_2018/"  # "GECCO_2018/"

    day, time = get_current_day_time()
    arrays_file_path = results_path + "arrays/"
    metrics_file_path = results_path + "metrics/"
    logs_file_path = results_path + "logs/"

    log_file_name = get_logs_file_name(
        logs_file_path, predictor_name, problem, experiment_folder, day, time)
    print("Write log and errors to file ", log_file_name)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    f = open(
        log_file_name, 'w')
    #sys.stdout = f
    # sys.stderr = f  # TODO wieder einkommentieren

    if problem == "sphere-rastrigin-rosenbrock":
        run_experiments_for_sphere_rosenbrock_rastrigin(conf_folder, repetitions,
                                                        arrays_file_path, metrics_file_path, experiment_folder, predictor_name,
                                                        problem, day, time, c1, c2, c3, insert_pred_as_ind, n_gpus, adaptive_c3,
                                                        n_neurons_type, n_epochs, batch_size, n_time_steps)
    elif problem == "mpb":
        mpb_evaluation(conf_folder, repetitions, arrays_file_path, metrics_file_path,
                       experiment_folder, predictor_name, problem, day, time,
                       c1, c2, c3, insert_pred_as_ind, n_gpus, adaptive_c3,
                       n_neurons_type, n_epochs, batch_size, n_time_steps)

    # reset stdout
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    f.close()
# https://github.com/tensorflow/tensorflow/issues/8220

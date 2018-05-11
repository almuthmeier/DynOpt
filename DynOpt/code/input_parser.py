'''
Created on May 9, 2018

@author: ameier
'''
import os
import sys
import warnings

from comparison import PredictorComparator
from utils.utils_print import get_current_day_time, get_logs_file_name


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


def initialize_comparator(parser, comparator):
    if not len(sys.argv) > 1:
        # no input values existing
        # https://stackoverflow.com/questions/10698468/argparse-check-if-any-arguments-have-been-passed/10699527
        # (11.5.18)
        initialize_comparator_manually(comparator)
    else:
        initialize_comparator_with_read_inputs(parser, comparator)

    # additional data for comparator
    day, time = get_current_day_time()
    comparator.day = day
    comparator.time = time
    comparator.arrays_file_path = comparator.outputdirectorypath + "arrays/"
    comparator.metrics_file_path = comparator.outputdirectorypath + "metrics/"


def initialize_comparator_manually(comparator):
    # path to ".../DynOptimization"
    path_to_dynopt = '/'.join(os.path.abspath(os.pardir).split('/')[:])
    print("pa_to-dynaopt: ", path_to_dynopt)
    comparator.algorithm = "dynpso"
    comparator.repetitions = 1
    comparator.benchmarkfunction = "sphere"
    comparator.benchmarkfunctionfolder = "GECCO_2018"
    comparator.outputdirectorypath = path_to_dynopt + \
        "/DynOpt/output/" + "myexperiments/" + "ff_sphere_1/"
    print("bal: ", comparator.outputdirectorypath)
    if comparator.algorithm == "dynpso":
        comparator.c1 = 1.496180
        comparator.c2 = 1.496180
        comparator.c3 = 1.496180
        comparator.insertpred = False
        comparator.adaptivec3 = False
        comparator.nparticles = 200

    elif comparator.algorithm == "dynea":
        comparator.mu = 5
        comparator.la = 10
        comparator.ro = 2
        comparator.mean = 0.0
        comparator.sigma = 1.0
        comparator.trechenberg = 5
        comparator.tau = 0.5

    comparator.predictor = "no"
    comparator.timesteps = 7
    if comparator.predictor == "rnn":
        comparator.neuronstype = "fixed20"
        comparator.epochs = 30
        comparator.batchsize = 1
        comparator.ngpus = 1


def initialize_comparator_with_read_inputs(parser, comparator):
    args = parser.parse_args()

    n_current_inputs = len(vars(args))

    if n_current_inputs != 24:
        print("false number of inputs")
        exit(0)

    comparator.algorithm = args.algorithm
    comparator.repetitions = args.repetitions
    comparator.benchmarkfunction = args.benchmarkfunction
    comparator.benchmarkfunctionfolder = args.benchmarkfunctionfolder
    comparator.outputdirectorypath = args.outputdirectorypath

    if args.algorithm == "dynpso":
        comparator.c1 = args.c1
        comparator.c2 = args.c2
        comparator.c3 = args.c3
        comparator.insertpred = args.insertpred
        comparator.adaptivec3 = args.adaptivec3
        comparator.nparticles = args.nparticles

    elif args.algorithm == "dynea":
        comparator.mu = args.mu
        comparator.la = args.la
        comparator.ro = args.ro
        comparator.mean = args.mean
        comparator.sigma = args.sigma
        comparator.trechenberg = args.trechenberg
        comparator.tau = args.tau

    comparator.predictor = args.predictor
    comparator.timesteps = args.timesteps
    if args.predictor == "rnn":
        comparator.neuronstype = args.neuronstype
        comparator.epochs = args.epochs
        comparator.batchsize = args.batchsize
        comparator.ngpus = args.ngpus


def run_parser():
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
    comparator = PredictorComparator()
    initialize_comparator(parser, comparator)

    # =======================================================================
    # configure log file
    # =======================================================================
    logs_file_path = comparator.outputdirectorypath + "logs/"
    if not os.path.exists(logs_file_path):
        warnings.warn("Warning: the directory " + logs_file_path +
                      "did not exist and is therefore created")
        os.makedirs(logs_file_path)

    log_file_name = get_logs_file_name(logs_file_path, comparator.predictor,
                                       comparator.benchmarkfunction,
                                       comparator.day, comparator.time)
    print(log_file_name)
    print("Write log and errors to file ", log_file_name, flush=True)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    f = open(
        log_file_name, 'w')
    # sys.stdout = f # TODO in-comment this
    # sys.stderr = f

    # =======================================================================
    # run experiments
    # =======================================================================
    comparator.run_experiments()

    # =======================================================================
    # reset standard output
    # =======================================================================
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    f.close()
# https://github.com/tensorflow/tensorflow/issues/8220


if __name__ == '__main__':
    run_parser()

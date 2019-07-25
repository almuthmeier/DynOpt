'''
Main class to execute all the experiments.

To execute the experiments 
    - either execute this file after having specified all parameters in the
      function initialize_comparator_manually()
    - or specify the parameters in the scripts/run_local.sh file and run that script.

The meaning of the input parameters is explained in the file scipts/run_local.sh

Created on May 9, 2018

@author: ameier
'''
import os
import random
import sys
import warnings

import numpy as np
sys.path.append(os.path.abspath(os.pardir))


def define_parser_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    # benchmark problem
    # "dynea" or "dynpso"
    parser.add_argument("-algorithm", type=str)
    parser.add_argument("-repetitions", type=int)
    parser.add_argument("-chgperiodrepetitions", type=int)
    parser.add_argument("-chgperiods", type=int)  # n_chgs = chgperiods - 1
    parser.add_argument("-lenchgperiod", type=int)
    # true, if changes occur at random time points
    parser.add_argument("-ischgperiodrandom", type=str)
    # "mpb" or "sphere-rastrigin-rosenbrock" (alt)
    # sphere, rosenbrock, rastrigin, mpbnoisy, mpbrandom (neu), mpbcorr
    # defines the benchmark function, must be located in the datasets folder of
    # this project
    parser.add_argument("-benchmarkfunction", type=str)
    # parent directory of the benchmark functions and child directory of the
    # datasets folder of this project
    parser.add_argument("-benchmarkfunctionfolderpath", type=str)

    parser.add_argument("-outputdirectory", type=str)
    # path to output folder
    parser.add_argument("-outputdirectorypath", type=str)
    parser.add_argument("-lbound", type=float)
    parser.add_argument("-ubound", type=float)

    # run only some experiments of all for the benchmark problem
    parser.add_argument("-poschgtypes", type=string_list_type)
    parser.add_argument("-fitchgtypes", type=string_list_type)
    parser.add_argument("-dims", type=int_list_type)  # list of int
    parser.add_argument("-noises", type=float_list_type)

    # for PSO
    parser.add_argument("-c1", type=float)  # sind Zahlen erlaubt?
    parser.add_argument("-c2", type=float)
    parser.add_argument("-c3", type=float)
    parser.add_argument("-insertpred", type=str)
    parser.add_argument("-adaptivec3", type=str)
    parser.add_argument("-nparticles", type=int)

    # for EA
    parser.add_argument("-mu", type=int)
    parser.add_argument("-la", type=int)
    parser.add_argument("-ro", type=int)
    parser.add_argument("-mean", type=float)
    parser.add_argument("-sigma", type=float)
    parser.add_argument("-trechenberg", type=int)
    parser.add_argument("-tau", type=float)
    parser.add_argument("-reinitializationmode", type=str)
    parser.add_argument("-sigmafactors", type=float_list_type)

    # for CMA-ES
    parser.add_argument("-cmavariant", type=str)
    parser.add_argument("-predvariant", type=str)

    # for predictor
    # no, rnn, autoregressive, tfrnn, tftlrnn, tftlrnndense, tcn, kalman
    parser.add_argument("-predictor", type=str)
    parser.add_argument("-trueprednoise", type=float)
    parser.add_argument("-timesteps", type=int)
    parser.add_argument("-addnoisytraindata", type=str)
    parser.add_argument("-traininterval", type=int)
    parser.add_argument("-nrequiredtraindata", type=int)
    parser.add_argument("-useuncs", type=str)
    parser.add_argument("-trainmcruns", type=int)
    parser.add_argument("-testmcruns", type=int)
    parser.add_argument("-traindropout", type=float)
    parser.add_argument("-testdropout", type=float)
    parser.add_argument("-kernelsize", type=int)
    parser.add_argument("-nkernels", type=int)
    parser.add_argument("-lr", type=float)

    # for ANN predictor
    # fixed20 or dyn1.3
    parser.add_argument("-neuronstype", type=str)
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-batchsize", type=int)
    parser.add_argument("-nlayers", type=int)
    # transfer learning
    parser.add_argument("-tlmodelpath", type=str)
    parser.add_argument("-ntllayers", type=int)
    # machine dependent
    parser.add_argument("-ngpus", type=int)

    # runtime
    parser.add_argument("-ncpus", type=int)

    return parser


def create_dir_if_not_existing(directory_path):
    if not os.path.exists(directory_path):
        warnings.warn("Warning: the directory " + directory_path +
                      "did not exist and is therefore created")
        os.makedirs(directory_path)


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

    full_output_path = comparator.outputdirectorypath + comparator.outputdirectory
    arrays_file_path = full_output_path + "arrays/"
    metrics_file_path = full_output_path + "metrics/"
    logs_file_path = full_output_path + "logs/"
    create_dir_if_not_existing(arrays_file_path)
    create_dir_if_not_existing(metrics_file_path)
    create_dir_if_not_existing(logs_file_path)

    comparator.arrays_file_path = arrays_file_path
    comparator.metrics_file_path = metrics_file_path
    comparator.logs_file_path = logs_file_path


def initialize_comparator_manually(comparator):
    '''
    Here boolean values have to be set as booleans (not as strings), because the
    function initialize_comparator_with_read_inputs() is not called afterwards 
    to convert the strings to booleans.
    '''
    # path to parent directory of the project directory DynOpt/
    path_to_dynoptim = '/'.join(os.path.abspath(os.pardir).split('/')[:])

    # benchmark problem
    comparator.algorithm = "dyncma"  # "dyncma"  # "dynea"
    comparator.repetitions = 1
    comparator.chgperiodrepetitions = 1
    comparator.chgperiods = 50
    comparator.lenchgperiod = 10
    comparator.ischgperiodrandom = False
    comparator.benchmarkfunction = "sphere"
    #comparator.benchmarkfunctionfolderpath = path_to_dynoptim + "/DynOpt/datasets/" + "GECCO_2019/"
    comparator.benchmarkfunctionfolderpath = "/home/ameier/Documents/Promotion/Ausgaben/Uncertainty/Ausgaben/data_2019-01-19_final/"
    # attention: naming should be consistent to predictor/other params
    comparator.outputdirectory = "ersterTest/ea_no/"
    comparator.outputdirectorypath = path_to_dynoptim + \
        "/DynOpt/output/" + "EvoStar_2020/" + "sphere/"
    comparator.lbound = 0
    comparator.ubound = 100

    # run only some experiments of all for the benchmark problem
    # ["linear", "sine", "circle", "mixture"])
    comparator.poschgtypes = np.array(["sinefreq"])
    comparator.fitchgtypes = np.array(["none"])
    comparator.dims = np.array([2])
    # TODO must not be a list (otherwise: log-file name is wrong)
    comparator.noises = np.array([0.0])

    # PSO
    if comparator.algorithm == "dynpso":
        comparator.c1 = 1.496180
        comparator.c2 = 1.496180
        comparator.c3 = 1.496180
        comparator.insertpred = False
        comparator.adaptivec3 = False
        comparator.nparticles = 200

    # EA
    elif comparator.algorithm == "dynea":
        comparator.mu = 5
        comparator.la = 10
        comparator.ro = 2
        comparator.mean = 0.0
        comparator.sigma = 1.0
        comparator.trechenberg = 5
        comparator.tau = 0.5
        # "no-RND" "no-VAR" "no-PRE" "pred-RND" "pred-UNC" "pred-DEV" "pred-KAL"
        comparator.reinitializationmode = "no-PRE"  # "no-PRE"
        comparator.sigmafactors = [0.01, 0.1, 1.0, 10.0]
    # CMA
    elif comparator.algorithm == "dyncma":
        # "resetcma" "predcma_internal" "predcma_external"
        comparator.cmavariant = "predcma_internal"
        # "simplest", "a", "b", "c", "d", "g" ,"branke", "f", "ha", "hb", "hd",
        # "hawom", "hbwom", "hdwom"
        comparator.predvariant = "h"

    # for predictor
    # "tcn", "tfrnn", "no", "tftlrnn" "autoregressive" "tftlrnndense" "kalman"
    # "truepred" (true prediction, disturbed with known noise)
    comparator.predictor = "truepred"
    # known prediction noise (standard deviation) of predition "truepred"
    comparator.trueprednoise = 0.0
    comparator.timesteps = 4
    comparator.addnoisytraindata = False  # must be true if addnoisytraindata
    comparator.traininterval = 5
    comparator.nrequiredtraindata = 10
    comparator.useuncs = False
    comparator.trainmcruns = 5 if comparator.useuncs else 0
    comparator.testmcruns = 5 if comparator.useuncs else 0
    comparator.traindropout = 0.1
    comparator.testdropout = 0.1 if comparator.useuncs else 0.0
    comparator.kernelsize = 3
    comparator.nkernels = 16
    comparator.lr = 0.002

    # for ANN predictor
    if comparator.predictor in ["rnn", "tfrnn", "tftlrnn", "tftlrnndense", "tcn"]:
        # (not everything is necessary for every predictor)
        comparator.neuronstype = "fixed20"
        comparator.epochs = 80
        comparator.batchsize = 8
        comparator.n_layers = 1
        # apply transfer learning only for tftlrnn
        comparator.apply_tl = comparator.predictor == "tftlrnn" or comparator.predictor == "tftlrnndense"
        comparator.tl_model_path = "/home/ameier/Documents/Promotion/Ausgaben/TransferLearning/TrainTLNet/Testmodell/"  # + \
        #"tl_nntype-RNN_tllayers-1_dim-5_retseq-True_preddiffs-True_steps-50_repetition-0_epoch-499.ckpt"
        comparator.n_tllayers = 1
        comparator.withdensefirst = comparator.predictor == "tftlrnndense"
        comparator.tl_learn_rate = 0.0001 if comparator.n_layers > 1 else 0.001
        comparator.ngpus = 1

    # runtime
    comparator.ncpus = 2

    # assertions
    if comparator.addnoisytraindata:
        assert comparator.chgperiodrepetitions > 1, "chgperiodrepetitions must be > 1"
    if not comparator.useuncs and comparator.algorithm == "dynea":
        assert (comparator.reinitializationmode !=
                "pred-UNC" and comparator.reinitializationmode != "pred-KAL")
    if (not comparator.predictor == 'kalman' and not comparator.predictor == "truepred"
            and not (comparator.predictor == "tcn" and comparator.useuncs)):
        # if neither Kalman prediction model, truepred, nor AutoTCN is used reinitialization
        # type pred-KAL must not be employed (since no uncertainty estimations
        # available)
        assert comparator.reinitializationmode != "pred-KAL"


def initialize_comparator_with_read_inputs(parser, comparator):
    args = parser.parse_args()

    n_current_inputs = len(vars(args))

    if n_current_inputs != 55:
        print("input_parser.py: false number of inputs: ", n_current_inputs)
        exit(0)

    # benchmark problem
    comparator.algorithm = args.algorithm
    comparator.repetitions = args.repetitions
    comparator.chgperiodrepetitions = args.chgperiodrepetitions
    comparator.chgperiods = args.chgperiods
    comparator.lenchgperiod = args.lenchgperiod
    comparator.ischgperiodrandom = args.ischgperiodrandom == 'True'  # convert str to bool
    comparator.benchmarkfunction = args.benchmarkfunction
    comparator.benchmarkfunctionfolderpath = args.benchmarkfunctionfolderpath
    comparator.outputdirectory = args.outputdirectory
    comparator.outputdirectorypath = args.outputdirectorypath
    comparator.lbound = args.lbound
    comparator.ubound = args.ubound

    # run only some experiments of all for the benchark problem
    comparator.poschgtypes = args.poschgtypes
    comparator.fitchgtypes = args.fitchgtypes
    comparator.dims = args.dims
    comparator.noises = args.noises

    # PSO
    if args.algorithm == "dynpso":
        comparator.c1 = args.c1
        comparator.c2 = args.c2
        comparator.c3 = args.c3
        comparator.insertpred = args.insertpred == 'True'
        comparator.adaptivec3 = args.adaptivec3 == 'True'
        comparator.nparticles = args.nparticles

    # EA
    elif args.algorithm == "dynea":
        comparator.mu = args.mu
        comparator.la = args.la
        comparator.ro = args.ro
        comparator.mean = args.mean
        comparator.sigma = args.sigma
        comparator.trechenberg = args.trechenberg
        comparator.tau = args.tau
        comparator.reinitializationmode = args.reinitializationmode
        comparator.sigmafactors = args.sigmafactors

    # CMA
    elif comparator.algorithm == "dyncma":
        comparator.cmavariant = args.cmavariant
        comparator.predvariant = args.predvariant

    # predictor
    comparator.predictor = args.predictor
    comparator.trueprednoise = args.trueprednoise
    comparator.timesteps = args.timesteps
    comparator.addnoisytraindata = args.addnoisytraindata == 'True'
    comparator.traininterval = args.traininterval
    comparator.nrequiredtraindata = args.nrequiredtraindata
    comparator.useuncs = args.useuncs == 'True'
    comparator.trainmcruns = args.trainmcruns if comparator.useuncs else 0
    comparator.testmcruns = args.testmcruns if comparator.useuncs else 0
    comparator.traindropout = args.traindropout
    comparator.testdropout = args.testdropout if comparator.useuncs else 0.0
    comparator.kernelsize = args.kernelsize
    comparator.nkernels = args.nkernels
    comparator.lr = args.lr

    # for ANN predictor
    if (args.predictor == "rnn" or args.predictor == "tfrnn" or
            args.predictor == "tftlrnn" or args.predictor == "tftlrnndense" or
            args.predictor == "tcn"):
        comparator.neuronstype = args.neuronstype
        comparator.epochs = args.epochs
        comparator.batchsize = args.batchsize
        comparator.n_layers = args.nlayers
        # apply transfer learning only for tftlrnn
        comparator.apply_tl = args.predictor == 'tftlrnn' or args.predictor == "tftlrnndense"
        comparator.tl_model_path = args.tlmodelpath
        comparator.n_tllayers = args.ntllayers
        comparator.withdensefirst = args.predictor == 'tftlrnndense'
        comparator.tl_learn_rate = 0.0001 if comparator.n_layers > 1 else 0.001
        comparator.ngpus = args.ngpus

    # runtime
    comparator.ncpus = args.ncpus

    # assertions
    if comparator.addnoisytraindata:
        assert comparator.chgperiodrepetitions > 1, "chgperiodrepetitions must be > 1"
    if not comparator.useuncs and comparator.algorithm == "dynea":
        assert (comparator.reinitializationmode !=
                "pred-UNC" and comparator.reinitializationmode != "pred-KAL")
    if (not comparator.predictor == 'kalman' and not comparator.predictor == "truepred"
            and not (comparator.predictor == "tcn" and comparator.useuncs)):
        # if neither Kalman prediction model nor AutoTCN is used reinitialization
        # type pred-KAL must not be employed (since no uncertainty estimations
        # available)
        assert comparator.reinitializationmode != "pred-KAL"


def int_list_type(string):
    '''
    Input type for argparse.
    Otherwise it is not possible to read in a list of int.
    https://www.tuxevara.de/2015/01/pythons-argparse-and-lists/ (12.5.18)

    @param string: a list as string, e.g.: '2,5,7,50'
    @return the string as numpy array of integers
    '''
    strings = string.split(',')
    return np.array([int(i) for i in strings])


def float_list_type(string):
    strings = string.split(',')
    return np.array([float(i) for i in strings])


def string_list_type(string):
    strings = string.split(',')
    return np.array([i for i in strings])


def run_parser():

    # tf.reset_default_graph()

    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    # (17.8.17)
    #os.environ['PYTHONHASHSEED'] = '0'

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
    # configure output
    # =======================================================================
    # create output directories

    # set log file
    log_file_name = get_logs_file_name(comparator.logs_file_path,
                                       comparator.predictor,
                                       comparator.benchmarkfunction,
                                       # TODO
                                       comparator.dims[0], comparator.poschgtypes[0],
                                       comparator.day, comparator.time,
                                       # TODO assumes that only one noise value
                                       # is in the array
                                       comparator.noises[0],
                                       comparator.kernelsize, comparator.nkernels,
                                       comparator.lr, comparator.epochs,
                                       comparator.batchsize, comparator.traindropout,
                                       comparator.testdropout)
    print(log_file_name)
    print("Write log and errors to file ", log_file_name, flush=True)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    f = open(
        log_file_name, 'w')
    sys.stdout = f  # TODO(exe) in-comment this
    sys.stderr = f
    #
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

    np.random.seed(7)
    random.seed(98)

    # this import has to be done before imports of own packages
    import multiprocessing as mp
    mp.set_start_method('fork')

    from comparison import PredictorComparator
    from utils.utils_files import get_current_day_time, get_logs_file_name

    run_parser()

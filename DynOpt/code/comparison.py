'''
Is called by the module input_parser.py.

Created on May 9, 2018

@author: ameier
'''
import copy
import math
from os.path import isfile, join
from posix import listdir
import re
import warnings

from algorithms.dynea import DynamicEA
from algorithms.dynpso import DynamicPSO
import numpy as np
from utils.utils_files import get_current_day_time
from utils.utils_prediction import get_n_neurons


class PredictorComparator(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor

        Parameters are set by input_parser.py
        '''
        # TODO testen, wenn Algorithm==dynpso, ob dann auch alle PSO-relevanten
        # Parameter gesetzt sind

        # benchmark problem
        self.algorithm = None  # string
        self.repetitions = None  # int
        self.chgperiods = None  # int
        self.lenchgperiod = None  # int
        self.ischgperiodrandom = None  # bool
        self.benchmarkfunction = None  # string
        self.benchmarkfunctionfolderpath = None  # string
        self.outputdirectorypath = None  # string

        # run only some experiments of all for the benchark problem
        self.poschgtypes = None  # str
        self.fitchgtypes = None  # str
        self.dims = None  # int
        self.noises = None  # float

        # PSO
        self.c1 = None  # float
        self.c2 = None  # float
        self.c3 = None  # float
        self.insertpred = None  # bool
        self.adaptivec3 = None  # bool
        self.nparticles = None  # int

        # EA
        self.mu = None  # int
        self.la = None  # int
        self.ro = None  # int
        self.mean = None  # float
        self.sigma = None  # float
        self.trechenberg = None  # int
        self.tau = None  # float

        # predictor
        self.predictor = None  # string
        self.timesteps = None  # int

        # ANN predictor
        self.neuronstype = None  # string
        self.epochs = None  # int
        self.batchsize = None  # int
        self.ngpus = None  # int

        # runtime
        self.ncpus = None  # int

        # set in this class (in method run_experiment())
        self.exp_file_name = None
        self.experiment_data = None
        self.chgperiods_for_gens = None

        # set in input parser
        self.day = None
        self.time = None
        self.arrays_file_path = None
        self.metrics_file_path = None
        self.logs_file_path = None

    def instantiate_optimization_alg(self):
        # random number generators

        # TODO seeds nicht gleich lassen
        # random generator for the optimization algorithm
        #  (e.g. for creation of population, random immigrants)
        alg_np_rnd_generator = np.random.RandomState(29405601)
        # so?: np.random.RandomState(random.randint(1, 567))
        # for predictor related stuff: random generator for  numpy arrays
        pred_np_rnd_generator = np.random.RandomState(23044820)

        dimensionality = len(self.experiment_data['orig_global_opt_pos'])
        n_generations = self.get_n_generations()
        if self.predictor == "no":
            n_neurons = None
        else:
            n_neurons = get_n_neurons(self.neuronstype, dimensionality)
        if self.algorithm == "dynea":
            alg = DynamicEA(self.benchmarkfunction, dimensionality,
                            n_generations, self.experiment_data, self.predictor,
                            alg_np_rnd_generator, pred_np_rnd_generator,
                            self.mu, self.la, self.ro, self.mean, self.sigma,
                            self.trechenberg, self.tau, self.timesteps,
                            n_neurons, self.epochs, self.batchsize)
        elif self.algorithm == "dynpso":
            alg = DynamicPSO(self.benchmarkfunction, dimensionality,
                             n_generations, self.experiment_data, self.predictor,
                             alg_np_rnd_generator, pred_np_rnd_generator,
                             self.c1, self.c2, self.c3, self.insertpred,
                             self.adaptivec3, self.nparticles, self.timesteps,
                             n_neurons, self.epochs, self.batchsize)
        else:
            warnings.warn("unknown optimization algorithm")
            exit(1)
        return alg

    def save_results(self, repetition_ID, alg):
        '''
        @param alg: the algorithm object that did the optimization and contains
        the data to be stored
        '''
        # generate array file name from the exeriment file name with some
        # replacements:
        # append predictor name at the beginning
        arrays_file_name = self.predictor + "_" + self.exp_file_name
        # replace date and time with start date and time
        arrays_file_name = arrays_file_name.replace(
            "_.*_.*\.npz", "_" + self.day + "_" + self.time + ".npz")
        # append the repetition number at the end before the file ending
        arrays_file_name = arrays_file_name.replace(
            ".npz", "_" + str(repetition_ID) + ".npz")
        # insert the correct number of change periods
        # https://stackoverflow.com/questions/16720541/python-string-replace-regular-expression
        # (15.5.18)
        arrays_file_name = re.sub(
            "_chgperiods-[0-9]+_", "_chgperiods-" + str(self.chgperiods) + "_", arrays_file_name)

        # TODO what if an algorithm doesn't provide one of the variables? (e.g.
        # because it doesn't have change detection?)
        # TODO will perhaps be extended, e.g for computing prediction quality
        np.savez(self.arrays_file_path + arrays_file_name,
                 best_found_fit_per_gen=alg.best_found_fit_per_gen,
                 best_found_pos_per_gen=alg.best_found_pos_per_gen,
                 best_found_fit_per_chgperiod=alg.best_found_fit_per_chgperiod,
                 best_found_pos_per_chgperiod=alg.best_found_pos_per_chgperiod,
                 pred_opt_fit_per_chgperiod=alg.pred_opt_fit_per_chgperiod,
                 pred_opt_pos_per_chgperiod=alg.pred_opt_pos_per_chgperiod,
                 detected_chgperiods_for_gens=alg.detected_chgperiods_for_gens,
                 # information about the real changes (is not in benchmark file
                 # because it is a general file for 10000 change periods)
                 real_chgperiods_for_gens=self.chgperiods_for_gens)

    def instantiate_and_run_algorithm(self, repetition_ID, gpu_ID, ):
        '''
        @param gpu_ID: is None if no GPU is required.
        '''
        print("run: ", repetition_ID, flush=True)
        # =====================================================================
        # instantiate algorithm
        alg = self.instantiate_optimization_alg()

        # =====================================================================
        # run algorithm
        if gpu_ID is None:
            alg.optimize()
        else:
            # make tensorflow deterministic
            import tensorflow as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # prevent using whole GPU
            tf.Session(config=config)
            tf.set_random_seed(1234)
            from keras import backend as K

            # run algorithm on specified GPU
            with tf.device('/gpu:' + str(gpu_ID)):
                alg.optimize()
                K.clear_session()

        # =====================================================================
        # save results
        self.save_results(repetition_ID, alg)

    def run_runs_parallel(self):
        '''
        Starts all runs of the experiment
        '''
        from multiprocessing import Pool
        # do parallel: instantiate and run algorithm (separate instance for
        # each run)

        # distribute runs on different GPUs so that each GPU has the same number of
        # processes
        if self.ngpus is None or self.ngpus is 0 or self.predictor == "no" or self.predictor == "arr":
            # no gpus required
            gpus_for_runs = np.array(self.repetitions * [None])
        else:
            max_runs_per_gpu = math.ceil(self.repetitions / self.ngpus)
            gpus_for_runs = np.floor(
                np.arange(self.repetitions) / max_runs_per_gpu).astype(int)

        # copy arguments for each run
        arguments = [None, None]
        # create list of arguments for the repetitions:
        argument_list = []
        for i in range(self.repetitions):
            argument_list.append(copy.deepcopy(arguments))
            # set repetition IDs  to identify the output files
            argument_list[-1][0] = i
            # set gpu ID
            argument_list[-1][1] = gpus_for_runs[i]
        # execute repetitions of the experiments on different CPUs
        n_kernels = self.ncpus
        with Pool(n_kernels) as pool:
            list(pool.starmap(self.instantiate_and_run_algorithm, argument_list))

    def convert_data_to_per_generation(self):
        '''
        Repeat all entries of experiment_data.
        '''
        n_gens = self.get_n_generations()
        # for all (key-value)-pairs in experiment_data:
        for key, property_per_chg in self.experiment_data.items():
            if not key == "orig_global_opt_pos":
                # pop old data
                self.experiment_data.pop(key)
                # rename key to prevent confusion (if there is something to
                # rename)
                key = key.replace("_per_chgperiod", "_per_gen")
                # convert and store data
                self.experiment_data[key] = property_per_chg[self.chgperiods_for_gens]
                assert n_gens == len(self.experiment_data[key])

    def get_chgperiods_for_gens(self):
        '''
        @param max_n_gens: integer: number of generations after that the EA stops
        @param len_change_period: number of generations per change period (used only
        when the time points of changes are deterministic)
        @param n_changes: number of changes (only used if is_change_time_random is True 
        @param is_change_time_random: true if the time points of changes are random
        @return: 1d numpy array containing for each generation the change
         period number it belongs to 
        '''
        if self.chgperiods == 1:  # no changes
            chgperiods_for_gens = np.zeros(self.get_n_generations(), int)
        elif not self.ischgperiodrandom:  # equidistant changes
            chgperiods_for_gens = np.array(
                [self.lenchgperiod * [i] for i in range(self.chgperiods)]).flatten()
        elif self.ischgperiodrandom:  # random change time points
            max_n_gens = self.get_n_generations()
            unsorted_periods_for_gens = np.random.randint(
                0, self.chgperiods, max_n_gens)
            chgperiods_for_gens = np.sort(unsorted_periods_for_gens)
        else:
            warnings.warn("unhandled case")
        return chgperiods_for_gens

    def extract_data_from_file(self, experiment_file_path):
        exp_file = np.load(experiment_file_path)

        global_opt_fit_per_chgperiod = exp_file['global_opt_fit_per_chgperiod']
        global_opt_pos_per_chgperiod = exp_file['global_opt_pos_per_chgperiod']
        orig_global_opt_pos = exp_file['orig_global_opt_pos']

        experiment_data = {'global_opt_fit_per_chgperiod': global_opt_fit_per_chgperiod,
                           'global_opt_pos_per_chgperiod': global_opt_pos_per_chgperiod,
                           'orig_global_opt_pos': orig_global_opt_pos}

        # additional data for some benchmark functions
        if self.benchmarkfunction == "sphere" or self.benchmarkfunction == \
                "rastrigin" or self.benchmarkfunction == "rosenbrock":
            pass
        if self.benchmarkfunction == "mpbnoisy" or \
                self.benchmarkfunction == "mpbrandom":
            heights = exp_file['heights']
            widths = exp_file['widths']
            positions = exp_file['positions']
            experiment_data['heights'] = heights
            experiment_data['widths'] = widths
            experiment_data['positions'] = positions

        exp_file.close()
        return experiment_data

    def select_experiment_files(self, benchmark_path):
        '''
        Selects some experiment files for a benchmark function to run only these experiments.

        @param all_experiment_files: list of all filenames (absolute paths) 
        being in the benchmark function directory
        '''

        all_experiment_files = [f for f in listdir(benchmark_path) if
                                (isfile(join(benchmark_path, f)) and
                                 f.endswith('.npz') and
                                 self.benchmarkfunction in f)]
        # TODO(dev) add further benchmarks
        selected_exp_files = None
        if self.benchmarkfunction == "sphere" or \
                self.benchmarkfunction == "rastrigin" or \
                self.benchmarkfunction == "rosenbrock":
            selected_exp_files = [f for f in all_experiment_files if (
                                  any(("_d-" + str(dim) + "_") in f for dim in self.dims) and
                                  any(("_pch-" + poschgtype) + "_" in f for poschgtype in self.poschgtypes) and
                                  any(("_fch-" + fitchgtype) + "_" in f for fitchgtype in self.fitchgtypes))]
        elif self.benchmarkfunction == "mpbnoisy" or \
                self.benchmarkfunction == "mpbrandom":
            selected_exp_files = [f for f in all_experiment_files if (
                                  any(("_d-" + str(dim) + "_" in f for dim in self.dims) and
                                      ("_noise-" + str(noise) + "_") in f for noise in self.noises))]
        return selected_exp_files

    def get_n_generations(self):
        return self.lenchgperiod * self.chgperiods

    def run_experiments(self):
        print("run experiments")

        # load the files of the benchmark that correspond to the specified
        # dimensionality, position/fitness change type ...
        benchmark_path = self.benchmarkfunctionfolderpath + self.benchmarkfunction + "/"
        selected_exp_files = self.select_experiment_files(benchmark_path)
        print("selected_exp_files: ", selected_exp_files)
        # only for logging
        n_experiments = len(selected_exp_files)
        exp_counter = 1
        for file_name in selected_exp_files:
            self.exp_file_name = file_name
            exp_file_path = benchmark_path + self.exp_file_name
            # =================================================================
            # logging
            curr_day, curr_time = get_current_day_time()
            print("Started experiment ", exp_counter, " of ", n_experiments, ". ",
                  curr_day, " ", curr_time, flush=True)
            exp_counter += 1

            # =================================================================
            # load data from file
            self.experiment_data = self.extract_data_from_file(exp_file_path)
            # generator change periods for generations
            self.chgperiods_for_gens = self.get_chgperiods_for_gens()
            # convert per CHANGE to per GENERATION
            self.convert_data_to_per_generation()

            # =================================================================
            # instantiate algorithm

            # start optimization
            self.run_runs_parallel()
            # save results
            # (plot results)6

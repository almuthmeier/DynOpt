'''
Created on May 9, 2018

@author: ameier
'''
import copy
import itertools
import math
from os.path import isfile, join
from posix import listdir
import warnings

from algorithms.dynea import DynamicEA
from algorithms.dynpso import DynamicPSO
import numpy as np
from utils.utils_print import get_current_day_time


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

    def instantiate_optimization_alg(self, experiment_data):
        # random number generators

        # TODO seeds nicht gleich lassen
        # random generator for the optimization algorithm
        #  (e.g. for creation of population, random immigrants)
        alg_np_rnd_generator = np.random.RandomState(29405601)
        # for predictor related stuff: random generator for  numpy arrays
        pred_np_rnd_generator = np.random.RandomState(23044820)

        dimensionality = len(experiment_data['orig_global_opt_pos'])
        n_generations = self.get_n_generations()
        if self.predictor == "no":
            n_neurons = None
        else:
            n_neurons = self.get_n_neurons(dimensionality)
        if self.algorithm == "dynea":
            alg = DynamicEA(self.benchmarkfunction, dimensionality,
                            n_generations, experiment_data, self.predictor,
                            alg_np_rnd_generator, pred_np_rnd_generator,
                            self.mu, self.la, self.ro, self.mean, self.sigma,
                            self.trechenberg, self.tau, self.timesteps,
                            n_neurons, self.epochs, self.batchsize)
        elif self.algorithm == "dynpso":
            alg = DynamicPSO(self.benchmarkfunction, dimensionality,
                             n_generations, experiment_data, self.predictor,
                             alg_np_rnd_generator, pred_np_rnd_generator,
                             self.c1, self.c2, self.c3, self.insertpred,
                             self.adaptivec3, self.nparticles, self.timesteps,
                             n_neurons, self.epochs, self.batchsize)
        else:
            warnings.warn("unknown optimization algorithm")
            exit(1)
        return alg

    def instantiate_and_run_algorithm(self, repetition_ID, gpu_ID, experiment_data):
        '''
        @param gpu_ID: is None if no GPU is required.
        '''
        print("run: ", repetition_ID, flush=True)
        # =====================================================================
        # instantiate algorithm
        alg = self.instantiate_optimization_alg(experiment_data)

        # =====================================================================
        # run algorithm
        if gpu_ID is None:
            alg.optimize()
        else:
            # make tensorflow deterministic
            import tensorflow as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # prevent using whole GPU
            session = tf.Session(config=config)
            tf.set_random_seed(1234)
            from keras import backend as K

            # run algorithm on specified GPU
            with tf.device('/gpu:' + str(gpu_ID)):
                alg.optimize()
                K.clear_session()

        # =====================================================================
        # save results

    def run_runs_parallel(self, experiment_data):
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
        arguments = [None, None, experiment_data]
        # create list of arguments for the repetitions:
        argument_list = []
        for i in range(self.repetitions):
            argument_list.append(copy.copy(arguments))
            # set repetition IDs  to identify the output files
            argument_list[-1][0] = i
            # set gpu ID
            argument_list[-1][1] = gpus_for_runs[i]
        # execute experiment with repetitions
        not_parallel = False
        if not_parallel:
            result = list(itertools.starmap(
                self.instantiate_and_run_algorithm, argument_list))
        else:
            n_kernels = self.ncpus
            with Pool(n_kernels) as pool:
                # TODO is that parallel though it is called on same object
                # (self)?
                result = list(pool.starmap(
                    self.instantiate_and_run_algorithm, argument_list))
        # TODO use results

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
        return experiment_data  # TODO or as class variable??

    def convert_data_to_per_generation(self, experiment_data, chgperiods_for_gens):
        '''
        Repeat all entries of experiment_data.
        '''
        n_gens = self.get_n_generations()
        # for all (key-value)-pairs in experiment_data:
        for key, property_per_chg in experiment_data.items():
            if not key == "orig_global_opt_pos":
                # pop old data
                experiment_data.pop(key)
                # rename key to prevent confusion (if there is something to
                # rename)
                key = key.replace("_per_chgperiod", "_per_gen")
                # convert and store data
                experiment_data[key] = property_per_chg[chgperiods_for_gens]
                assert n_gens == len(experiment_data[key])

    def get_n_generations(self):
        return self.lenchgperiod * self.chgperiods

    def get_n_neurons(self, dim):
        '''
        (number of neurons can not directly be specified as input because it is 
        computed in some cases depending on the problem dimensionality)
        '''
        if self.n_neurons_type == "fixed20":
            n_neurons = 20
        elif self.n_neurons_type == "dyn1.3":
            n_neurons = math.ceil(dim * 1.3)
        else:
            msg = "unkwnown type for neuronstype (number of neurons): " + \
                str(self.n_neurons_type)
            warnings.warn(msg)
        return n_neurons

    def get_chgperiods_for_gens(self):
        '''
        @param max_n_gens: integer: number of generations after that the EA stops
        @param len_change_period: number of generations per change period (used only
        when the time points of changes are deterministic)
        @param n_changes: number of changes (only used if is_change_time_random is True 
        @param is_change_time_random: true if the time points of changes are random
        @return: tupel
                    - 1d numpy array containing for each generation the change 
                      period number it belongs to
                    - overall number of changes occurred during all generations 
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

    def save_results(self):
        pass
        # TODO to be stored
        #     - chgperiods_for_gens (because otherwise it is unknown where the
        #       changes happened (especially in random change frequencies)

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
        for exp_file_name in selected_exp_files:
            exp_file_path = benchmark_path + exp_file_name
            # =================================================================
            # logging
            curr_day, curr_time = get_current_day_time()
            print("Started experiment ", exp_counter, " of ", n_experiments, ". ",
                  curr_day, " ", curr_time, flush=True)
            exp_counter += 1

            # =================================================================
            # load data from file
            experiment_data = self.extract_data_from_file(exp_file_path)
            # generator change periods for generations
            chgperiods_for_gens = self.get_chgperiods_for_gens()
            # convert per CHANGE to per GENERATION
            self.convert_data_to_per_generation(
                experiment_data, chgperiods_for_gens)

            # =================================================================
            # instantiate algorithm

            # start optimization
            self.run_runs_parallel(experiment_data)
            # save results
            # (plot results)6

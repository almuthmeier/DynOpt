'''
Created on May 9, 2018

@author: ameier
'''
import math
from os.path import isfile, join
from posix import listdir
import warnings

import numpy as np


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
        self.poschgtype = None  # str
        self.fitchgtype = None  # str
        self.dim = None  # int
        self.noise = None  # float

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

    def convert_data_to_per_generation(self, experiment_file_path, experiment_data):
        pass

    def get_chgperiods_for_gens(self, alg_np_rnd_generator):
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
            chgperiods_for_gens = np.zeros(self.lenchgperiod, int)
        elif not self.ischgperiodrandom:  # equidistant changes
            chgperiods_for_gens = np.array(
                [self.lenchgperiod * [i] for i in range(self.chgperiods)]).flatten()
        elif self.ischgperiodrandom:  # random change time points
            max_n_gens = self.lenchgperiod * self.chgperiods
            unsorted_periods_for_gens = alg_np_rnd_generator.randint(
                0, self.chgperiods, max_n_gens)
            chgperiods_for_gens = np.sort(unsorted_periods_for_gens)
        else:
            warnings.warn("unhandled case")
        return chgperiods_for_gens

    def select_experiment_files(self, all_experiment_files):
        '''
        Selects some experiment files for a benchmark function to run only these experiments.

        @param all_experiment_files: list of all filenames (absolute paths) 
        being in the benchmark function directory
        '''
        # TODO(dev) add further benchmarks
        selected_exp_files = None
        if self.benchmarkfunction == "sphere" or \
                self.benchmarkfunction == "rastrigin" or \
                self.benchmarkfunction == "rosenbrock":
            selected_exp_files = [f for f in all_experiment_files if (
                                  ("_d-" + str(self.dim) + "_") in f and
                                  ("_pch-" + self.poschgtype) in f and
                                  ("_fch-" + self.fitchgtype) in f)]
        elif self.benchmarkfunction == "mpbnoisy" or \
                self.benchmarkfunction == "mpbrandom":
            selected_exp_files = [f for f in all_experiment_files if (
                                  ("_d-" + str(self.dim) + "_") in f and
                                  ("_noise-" + str(self.noise)) in f)]
        return selected_exp_files

    def run_experiments(self):
        print("run experiments")

        # führe alle Dateien im Benchmarkordner aus und speichere Arrays

        # load all files of the benchmark
        # TODO evtl nur die, für die übergebenen Dimensionen/chg-typen/... ???

        benchmark_path = self.benchmarkfunctionfolderpath + self.benchmarkfunction + "/"
        all_experiment_files = [f for f in listdir(benchmark_path) if
                                (isfile(join(benchmark_path, f)) and
                                 f.endswith('.npz') and
                                 self.benchmarkfunction in f)]
        selected_exp_files = self.select_experiment_files(all_experiment_files)

        for exp_file_path in selected_exp_files:
            # load data from file
            experiment_data = self.extract_data_from_file(exp_file_path)
            # convert per CHANGE to per GENERATION
            alg_np_rnd_generator = None  # TODO irgendwo instanziieren
            experiment_data = self.convert_data_to_per_generation(exp_file_path,
                                                                  experiment_data)

            # instantiate algorithm
            # start optimization
            # save results
            # (plot results)6

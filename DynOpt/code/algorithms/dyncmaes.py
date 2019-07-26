'''
Created on May 25, 2018

@author: ameier
'''
# CMA-ES source code can be found here
# path-to-python-environment/lib/python3.5/site-packages/cma
import copy
from math import floor, log, sqrt
import sys

import numpy as np
from utils import utils_dynopt
from utils.utils_cmaes import get_best_fit_and_ind_so_far
from utils.utils_cmaes import get_new_sig, get_mue_best_individuals,\
    get_weighted_avg, get_inverse_sqroot, get_new_p_sig, get_offsprings, \
    get_h_sig, get_new_p_c, visualize_dominant_eigvector, get_C_mu, get_new_C
from utils.utils_dynopt import environment_changed
from utils.utils_prediction import build_predictor
from utils.utils_prediction import prepare_data_train_and_predict


class DynamicCMAES(object):
    '''
    classdocs
    '''

    def __init__(self,
                 benchmarkfunction, dim, lenchgperiod,
                 n_generations, experiment_data, predictor_name,
                 trueprednoise, lbound, ubound,
                 cma_np_rnd_generator, pred_np_rnd_generator,
                 timesteps, n_neurons, epochs, batchsize, n_layers,
                 train_interval, n_required_train_data, use_uncs,
                 train_mc_runs, test_mc_runs, train_dropout, test_dropout,
                 kernel_size, n_kernels, lr, cma_variant, pred_variant):
        '''
        Constructor
        '''
        # TODOs:
        # - alle "Speichervariablen" vom EA nutzen, damit Ausgabedateien identisch sind
        # ---------------------------------------------------------------------
        # for the problem
        # ---------------------------------------------------------------------
        self.benchmarkfunction = benchmarkfunction
        self.dim = dim
        self.lenchgperiod = lenchgperiod
        self.n_generations = n_generations  # TODO rename
        self.experiment_data = experiment_data
        self.predictor_name = predictor_name
        self.trueprednoise = trueprednoise

        self.lbound = lbound  # 100  # assumed that the input data follow this assumption
        self.ubound = ubound  # 200  # TODO(exe) , TODO insert into dynPSO
        # ---------------------------------------------------------------------
        # for the predictor
        # ---------------------------------------------------------------------
        self.n_time_steps = timesteps
        self.n_neurons = n_neurons
        self.n_epochs = epochs
        self.batch_size = batchsize
        self.n_layers = n_layers
        self.train_interval = train_interval

        # predictive uncertainty
        self.train_mc_runs = train_mc_runs
        self.test_mc_runs = test_mc_runs
        self.train_dropout = train_dropout
        self.test_dropout = test_dropout
        self.use_uncs = use_uncs  # True if uncertainties should be trained, predicted and used

        # TCN
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.lr = lr

        # training/testing specifications
        # number train data with that the network at least is trained
        self.n_required_train_data = max(
            n_required_train_data, self.n_time_steps)
        self.predict_diffs = True  # predict position differences, TODO insert into PSO
        self.return_seq = False  # return values for all time steps not only the last one
        # True -> train data are shuffled before training and between epochs
        self.shuffle_train_data = True  # TODO move into script?

        # ---------------------------------------------------------------------
        # for EA/CMA-ES (fixed values)
        # ---------------------------------------------------------------------
        self.cma_np_rnd_generator = cma_np_rnd_generator
        self.pred_np_rnd_generator = pred_np_rnd_generator

        # -------------------------------------------------------------------------
        # fixed parameters
        # -------------------------------------------------------------------------
        self.lambd = 4 + floor(3 * log(self.dim))  # offsprings
        self.mu = floor(self.lambd / 2)  # parents
        # weights (vector of size ń)
        w_divisor = np.sum([(log(self.mu + 0.5) - log(j))
                            for j in range(1, self.mu + 1)])
        self.w = np.array([((log(self.mu + 0.5) - log(i)) / w_divisor)
                           for i in range(1, self.mu + 1)])
        # other
        self.mu_w = 1 / np.sum(np.square(self.w))
        self.c_sig = (self.mu_w + 2) / (self.dim + self.mu_w + 3)
        self.d_sig = 1 + self.c_sig + 2 * \
            max(0, sqrt((self.mu_w - 1) / (self.dim + 1)) - 1)
        self.c_c = 4 / (self.dim + 4)
        self.c_1 = (2 * min(1, self.lambd / 6)) / \
            ((self.dim + 1.3)**2 + self.mu_w)
        self.c_mu = (2 * (self.mu_w - 2 + 1 / self.mu_w)) / \
            ((self.dim + 2)**2 + self.mu_w)
        self.E = sqrt(self.dim) * \
            (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        self.c_o = self.c_c
        self.c_o1 = self.c_1

        # -------------------------------------------------------------------------
        # initialization
        # -------------------------------------------------------------------------
        self.m = self.cma_np_rnd_generator.randint(
            self.lbound, self.ubound, self.dim)
        self.sig = cma_np_rnd_generator.rand()
        self.p_sig = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.C = np.identity(self.dim)
        self.p_sig_pred = np.zeros(self.dim)

        # ---------------------------------------------------------------------
        # options
        # ---------------------------------------------------------------------
        self.cma_variant = cma_variant
        self.pred_variant = pred_variant

        # ---------------------------------------------------------------------
        # values that are not passed as parameters to the constructor
        # ---------------------------------------------------------------------
        self.init_sigma = 1

        # ---------------------------------------------------------------------
        # for EA (variable values)
        # ---------------------------------------------------------------------
        # initialize population (mu candidates) and compute fitness.
        self.population = self.cma_np_rnd_generator.uniform(self.lbound,
                                                            self.ubound, (self.mu, self.dim))
        # 2d numpy array (for each individual one row)
        self.population_fitness = np.array([utils_dynopt.fitness(self.benchmarkfunction,
                                                                 individual, 0,
                                                                 self.experiment_data)
                                            for individual in self.population]).reshape(-1, 1)

        # ---------------------------------------------------------------------
        # for EA (prediction and evaluation)
        # ---------------------------------------------------------------------
        # number of change periods (new training data) since last training
        self.n_new_train_data = 0
        # number of changes detected by the EA
        self.detected_n_changes = 0
        # for each detected change the corresponding generation numbers
        self.detected_chgperiods_for_gens = []
        # best found individual for each generation (2d numpy array)
        self.best_found_pos_per_gen = np.zeros((self.n_generations, self.dim))
        # fitness of best found individual for each generation (1d numpy array)
        self.best_found_fit_per_gen = np.zeros(self.n_generations)
        # position of found optima (one for each change period)
        self.best_found_pos_per_chgperiod = []
        # fitness of found optima (one for each change period)
        self.best_found_fit_per_chgperiod = []
        # position, fitness & epistemic uncertainty of predicted optima (one for
        # each change period)
        self.pred_opt_pos_per_chgperiod = []
        self.pred_opt_fit_per_chgperiod = []
        self.pred_unc_per_chgperiod = []  # predictive variance
        self.aleat_unc_per_chgperiod = []  # average aleatoric uncertainty
        # training error per chgperiod (if prediction was done)
        self.train_error_per_chgperiod = []
        # training error per epoch for each chgperiod (if prediction was done)
        self.train_error_for_epochs_per_chgperiod = []

        # ---------------------------------------------------------------------
        # CMA-ES variables
        # ---------------------------------------------------------------------
        #self.angle_per_gen = []
        #self.sig_per_gen = []
        #self.p_sig_per_gen = []
        #self.h_sig_per_gen = []
        #self.p_c_per_gen = []
        #self.p_sig_pred_per_gen = []
        #self.m_per_gen = []

        # global optimum
        self.glob_opt_per_gen = []

        # ---------------------------------------------------------------------
        # for EA (evaluation of variance) (repetitions of change periods)
        # not used for CMA-ES, only for similar output files like EA
        # ---------------------------------------------------------------------
        # add noisy data (noise equals standard deviation among change period
        # runs TODO could be replaced by "max_n_chgperiod_reps > 1"
        self.add_noisy_train_data = False  # False since unused
        self.n_noisy_series = 20
        # number repetitions of the single change periods (at least 1 -> 1 run)
        # TODO insert into PSO
        self.max_n_chgperiod_reps = 1  # arbitrary value since unused
        # population for last generation of change period (for each run)
        # used for determining the EAs variance for change periods
        # 4d list [runs, chgperiods, parents, dims]
        # TODO insert into PSO
        self.final_pop_per_run_per_chgperiod = [
            [] for _ in range(self.max_n_chgperiod_reps)]
        # fitness of final population per run of each chgperiod
        # 3d list [runs, chgperiods, parents]
        # TODO insert into PSO
        self.final_pop_fitness_per_run_per_changeperiod = [
            [] for _ in range(self.max_n_chgperiod_reps)]
        # best found solution for each run of all change periods
        # format [runs, chgperiods, dims]
        self.best_found_pos_per_run_per_chgp = [
            [] for _ in range(self.max_n_chgperiod_reps)]
        # standard deviation and mean of the position of the best found solution
        # (computed over the change period runs)
        self.stddev_among_runs_per_chgp = [
            [] for _ in range(self.max_n_chgperiod_reps)]
        self.mean_among_runs_per_chgp = [
            [] for _ in range(self.max_n_chgperiod_reps)]

        # ---------------------------------------------------------------------

        assert pred_variant in ["simplest", "a", "b", "c", "d", "g"] and cma_variant == "predcma_external" or \
            (pred_variant in ["branke", "f"] or pred_variant.startswith("h")) and cma_variant == "predcma_internal" or \
            pred_variant == "None" and cma_variant in ["resetcma"]

    # =============================================================================
    # for dynamic CMA-ES

    def get_new_pred_path(self, diff_elem_0, diff_elem_1):
        diff_vector = np.linalg.norm(diff_elem_0 - diff_elem_1) / 2

        new_p_sig = (1 - self.c_sig) * self.p_sig_pred + \
            sqrt(self.c_sig * (2 - self.c_sig)) * sqrt(self.mu_w) * diff_vector
        assert new_p_sig.shape == (self.dim,)
        return new_p_sig

    def update_sig_and_m_after_chage(self, m_old, my_pred_mode):
        if my_pred_mode == "no" and self.predictor_name != "no":
            # there are not yet enough training data -> reset sigma
            self.sig = 1
            return

        # get prediction&uncertainty if available
        try:
            pred = self.pred_opt_pos_per_chgperiod[-1]
        except IndexError:
            pred = None
        try:
            unc = self.pred_unc_per_chgperiod[-1]
        except IndexError:
            unc = None
        #print("pred: ", pred)
        #print("unc: ", unc)

        # set sig and m
        if self.cma_variant == "predcma_external" and len(self.pred_opt_pos_per_chgperiod) > 1:
            if self.pred_variant in ["simplest", "a", "b", "c", "d"]:
                if pred is None:
                    sys.exit("Error: pred_variant " + self.pred_variant +
                             " requires a prediction")
                self.m = pred

            if self.pred_variant == "a":
                if unc is None:
                    sys.exit("Error: pred_variant " + self.pred_variant +
                             " requires an uncertainty estimation")
                self.sig = np.sqrt(unc)
            elif self.pred_variant == "b":
                self.sig = (
                    self.pred_opt_pos_per_chgperiod[-2] - self.best_found_pos_per_chgperiod[-1]) / 2
            elif self.pred_variant == "c":
                self.p_sig_pred = self.get_new_pred_path(
                    self.pred_opt_pos_per_chgperiod[-2], self.best_found_pos_per_chgperiod[-1])
                self.sig = self.p_sig_pred
            elif self.pred_variant in ["simplest", "d"]:
                self.sig = self.init_sigma
            elif self.pred_variant == "g":
                self.sig = np.linalg.norm(
                    self.pred_opt_pos_per_chgperiod[-1] - self.best_found_pos_per_chgperiod[-1]) / 2
            else:
                sys.exit("Error: unknown pred_variant: " + self.pred_variant)

        elif self.cma_variant == "predcma_internal" and len(self.best_found_pos_per_chgperiod) > 1:
            if self.pred_variant == "e":
                pass
            elif self.pred_variant == "f":
                self.p_sig_pred = self.get_new_pred_path(
                    self.best_found_pos_per_chgperiod[-1], self.best_found_pos_per_chgperiod[-2])
                self.sig = self.p_sig_pred
            elif self.pred_variant.startswith("h"):
                # auskommentieren für hc)
                if not self.pred_variant.endswith("wom"):
                    # wom = without mean, i.e., no adaptation of the mean
                    self.m = self.m + \
                        self.best_found_pos_per_chgperiod[-1] - \
                        self.best_found_pos_per_chgperiod[-2]
                if self.pred_variant.startswith("ha"):  # ha)
                    self.sig = 1
                elif self.pred_variant.startswith("hb"):  # hb)  [hc) = hbwom]
                    self.sig = np.linalg.norm(
                        m_old - self.best_found_pos_per_chgperiod[-1]) / 2
                elif self.pred_variant.startswith("hd"):  # hd)
                    self.p_sig_pred = self.get_new_pred_path(
                        m_old, self.best_found_pos_per_chgperiod[-1])
                    self.sig = self.p_sig_pred
                else:
                    sys.exit("Error: unknown pred_variant: " +
                             self.pred_variant)

            elif self.pred_variant == "branke":
                tmp_mu_best_individuals = get_mue_best_individuals(
                    self.dim, self.mu, self.population, self.population_fitness)
                self.m = get_weighted_avg(
                    self.dim, self.w, tmp_mu_best_individuals)

                diff_vals = np.subtract(
                    self.best_found_pos_per_chgperiod[:-1], self.best_found_pos_per_chgperiod[1:])
                norm_vals = np.linalg.norm(diff_vals, axis=1)
                s = np.average(norm_vals)
                self.sig = s / 2
            else:
                sys.exit("Error: unknown pred_variant: " + self.pred_variant)
        else:
            if self.cma_variant in ["resetcma", "predcma_internal", "predcma_external"]:
                # is the case e.g. when not yet enough predictions were made
                self.sig = self.init_sigma
            else:
                sys.exit("Error: unknown cma_variant: " + self.cma_variant)

    # =============================================================================

    def optimize(self):
        # ---------------------------------------------------------------------
        # local variables for predictor
        # ---------------------------------------------------------------------
        predictor = build_predictor(self.predictor_name, self.n_time_steps,
                                    self.dim, self.batch_size, self.n_neurons,
                                    self.return_seq, False, self.n_layers,
                                    self.n_epochs, None, None,
                                    None, None, self.use_uncs,
                                    self.train_mc_runs, self.train_dropout, self.test_dropout,
                                    self.kernel_size, self.n_kernels, self.lr)
        ar_predictor = None
        sess = None  # necessary since is assigned anew after each training
        if self.predictor_name in ["tfrnn", "tftlrnn", "tftlrnndense", "tcn"]:
            import tensorflow as tf
            # if transfer learning then load weights

            # start session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # initialize empty model (otherwise exception)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        # ---------------------------------------------------------------------
        # local variables for CMA-ES
        # ---------------------------------------------------------------------
        # mean after first generation in current change period
        m_begin_chgp = None
        # best individual/fitness in change period so far, is reset after chgp
        so_far_best_fit = sys.float_info.max
        so_far_best_ind = None

        for t in range(self.n_generations):
            glob_opt = self.experiment_data['global_opt_pos_per_gen'][t]
            #print("generation , ", t, " glob opt: ", glob_opt)

            # env_changed = environment_changed(t, self.population, self.population_fitness,
            # self.benchmarkfunction, self.experiment_data,
            # self.cma_np_rnd_generator)
            env_changed = t % self.lenchgperiod == 0 and t > 0  # TODO
            if env_changed and t != 0:
                # print("\nchanged")
                # count change
                self.detected_n_changes += 1
                # count new train data
                self.n_new_train_data += 1
                print("(chg/gen)-(" + str(self.detected_n_changes) +
                      "/" + str(t) + ") ", end="", flush=True)
                # store best found solution during change period as training data
                # for predictor
                self.best_found_pos_per_chgperiod.append(
                    copy.copy(so_far_best_ind))
                self.best_found_fit_per_chgperiod.append(
                    copy.copy(so_far_best_fit))
                # reset values
                so_far_best_fit = sys.float_info.max
                so_far_best_ind = None

                # prepare data and predict optimum
                (my_pred_mode,
                 ar_predictor,
                 self.n_new_train_data) = prepare_data_train_and_predict(sess, t, self.dim, predictor,
                                                                         self.experiment_data, self.n_epochs, self.batch_size,
                                                                         self.return_seq, self.shuffle_train_data, self.n_new_train_data,
                                                                         self.best_found_pos_per_chgperiod, self.train_interval,
                                                                         self.predict_diffs, self.n_time_steps, self.n_required_train_data,
                                                                         self.predictor_name, self.add_noisy_train_data,
                                                                         self.n_noisy_series, self.stddev_among_runs_per_chgp,
                                                                         self.test_mc_runs, self.benchmarkfunction, self.use_uncs,
                                                                         self.pred_unc_per_chgperiod, self.aleat_unc_per_chgperiod,
                                                                         self.pred_opt_pos_per_chgperiod, self.pred_opt_fit_per_chgperiod,
                                                                         self.train_error_per_chgperiod,
                                                                         self.train_error_for_epochs_per_chgperiod,
                                                                         glob_opt, self.trueprednoise, self.pred_np_rnd_generator)

                if not ar_predictor is None:
                    predictor = ar_predictor

                # (re-)set variables
                self.update_sig_and_m_after_chage(m_begin_chgp, my_pred_mode)
                self.p_sig = np.zeros(self.dim)
                self.p_c = np.zeros(self.dim)
                self.C = np.identity(self.dim)

            # ---------------------------------------------------------------------
            # eigenvalue decomposition

            inv_squareroot_C, sqrt_of_eig_vals_C, eig_vals_C, eig_vctrs_C = get_inverse_sqroot(
                self.C)

            #print("m    : ", self.m)
            #print("C: ", self.C)
            #print("sig: ", self.sig)
            # ---------------------------------------------------------------------

            self.population, self.population_fitness = get_offsprings(
                self.dim, self.m, self.sig, self.lambd, sqrt_of_eig_vals_C, eig_vctrs_C, t,
                self.benchmarkfunction, self.experiment_data, self.cma_np_rnd_generator)
            mu_best_individuals = get_mue_best_individuals(
                self.dim, self.mu, self.population, self.population_fitness)

            # parameter update
            m_new = get_weighted_avg(self.dim, self.w, mu_best_individuals)
            p_sig_new = get_new_p_sig(
                self.dim, self.c_sig, self.p_sig, self.mu_w, self.m, m_new, self.sig, inv_squareroot_C)
            h_sig = get_h_sig(p_sig_new, self.c_sig, t, self.dim, self.E)
            p_c_new = get_new_p_c(self.dim, self.c_c, self.p_c, h_sig, self.mu_w,
                                  m_new, self.m, self.sig)
            C_mu = get_C_mu(self.dim, mu_best_individuals,
                            self.m, self.sig, self.w)
            C_new = get_new_C(self.dim, self.c_1, self.c_mu,
                              self.C, p_c_new, C_mu)
            sig_new = get_new_sig(self.sig, self.c_sig,
                                  self.d_sig, p_sig_new, self.E)

            # ---------------------------------------------------------------------
            # store old variables
            # self.glob_opt_per_gen.append(glob_opt)
            # self.angle_per_gen.append(
            # visualize_dominant_eigvector(self.dim, eig_vals_C, eig_vctrs_C))
            # self.sig_per_gen.append(self.sig)
            # self.p_sig_per_gen.append(self.p_sig)
            # self.h_sig_per_gen.append(h_sig)
            # self.p_c_per_gen.append(self.p_c)
            # self.p_sig_pred_per_gen.append(self.p_sig_pred_per_gen)
            # self.m_per_gen.append(self.m)

            # ---------------------------------------------------------------------
            # set variables for next generation
            if env_changed:
                m_begin_chgp = self.m
            self.m = m_new
            self.p_sig = p_sig_new
            self.p_c = p_c_new
            self.C = C_new
            self.sig = sig_new

            # ---------------------------------------------------------------------
            # store further variables

            # chgp number
            self.detected_chgperiods_for_gens.append(self.detected_n_changes)

            # best fitness and individual so far in change period
            so_far_best_fit, so_far_best_ind = get_best_fit_and_ind_so_far(
                so_far_best_fit, so_far_best_ind, self.population, self.population_fitness)

            # best fitness and individual in generation
            min_fitness_index = np.argmin(self.population_fitness)
            self.best_found_fit_per_gen[t] = copy.copy(
                self.population_fitness[min_fitness_index])
            self.best_found_pos_per_gen[t] = copy.copy(
                self.population[min_fitness_index])
            # print(str(t) + ": best: ", self.population_fitness[min_fitness_index],
            #      "[", self.population[min_fitness_index], "]")

        if self.predictor_name in ["tfrnn", "tftlrnn", "tftlrnndense", "tcn"]:
            # TODO why not for "rnn"?
            sess.close()
            tf.reset_default_graph()

        # save results for last change period
        self.best_found_pos_per_chgperiod.append(copy.copy(so_far_best_ind))
        self.best_found_fit_per_chgperiod.append(copy.copy(so_far_best_fit))

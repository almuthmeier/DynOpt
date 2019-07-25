'''
Created on May 25, 2018

@author: ameier
'''
# CMA-ES source code can be found here
# path-to-python-environment/lib/python3.5/site-packages/cma
import copy
from math import floor, log, sqrt
import sys
import warnings

import numpy as np
from utils import utils_dynopt
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
                 benchmarkfunction, dim,
                 n_generations, experiment_data, predictor_name, lbound, ubound,
                 cma_np_rnd_generator, pred_np_rnd_generator,
                 timesteps, n_neurons, epochs, batchsize, n_layers,
                 train_interval, n_required_train_data, use_uncs,
                 train_mc_runs, test_mc_runs, train_dropout, test_dropout,
                 kernel_size, n_kernels, lr, cma_variant, pred_variant):
        '''
        Constructor
        '''
        # TODOs:
        # - n durch dim ersetzen
        #  - überall self nutzen, also keine Zuweisungen auf gleichnamige Variablen
        # - alle "Speichervariablen" vom EA nutzen, damit Ausgabedateien identisch sind
        # ---------------------------------------------------------------------
        # for the problem
        # ---------------------------------------------------------------------
        self.benchmarkfunction = benchmarkfunction
        self.n = dim  # TODO rename
        self.generations = n_generations  # TODO rename
        self.experiment_data = experiment_data
        self.predictor_name = predictor_name

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
        self.lambd = 4 + floor(3 * log(self.n))  # offsprings
        self.mu = floor(self.lambd / 2)  # parents
        # weights (vector of size ń)
        w_divisor = np.sum([(log(self.mu + 0.5) - log(j))
                            for j in range(1, self.mu + 1)])
        self.w = np.array([((log(self.mu + 0.5) - log(i)) / w_divisor)
                           for i in range(1, self.mu + 1)])
        # other
        self.mu_w = 1 / np.sum(np.square(self.w))
        self.c_sig = (self.mu_w + 2) / (self.n + self.mu_w + 3)
        self.d_sig = 1 + self.c_sig + 2 * \
            max(0, sqrt((self.mu_w - 1) / (self.n + 1)) - 1)
        self.c_c = 4 / (self.n + 4)
        self.c_1 = (2 * min(1, self.lambd / 6)) / \
            ((self.n + 1.3)**2 + self.mu_w)
        self.c_mu = (2 * (self.mu_w - 2 + 1 / self.mu_w)) / \
            ((self.n + 2)**2 + self.mu_w)
        self.E = sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n**2))

        self.c_o = self.c_c
        self.c_o1 = self.c_1

        # -------------------------------------------------------------------------
        # initialization
        # -------------------------------------------------------------------------
        self.m = np.random.randint(self.lbound, self.ubound, self.n)
        self.sig = np.random.rand()
        self.p_sig = np.zeros(self.n)
        self.p_c = np.zeros(self.n)
        self.C = np.identity(self.n)
        self.p_sig_pred = np.zeros(self.n)

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
                                                            self.ubound, (self.mu, self.n))
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
        self.best_found_pos_per_gen = np.zeros((self.generations, self.n))
        # fitness of best found individual for each generation (1d numpy array)
        self.best_found_fit_per_gen = np.zeros(self.generations)
        # position of found optima (one for each change period)
        self.best_found_pos_per_chgperiod = []
        # fitness of found optima (one for each change period)
        self.best_found_fit_per_chgperiod = []
        # position, fitness & epistemic uncertainty of predicted optima (one for
        # each change period)
        self.pred_opt_pos_per_chgperiod = []
        self.pred_opt_fit_per_chgperiod = []
        self.epist_unc_per_chgperiod = []  # predictive variance
        self.aleat_unc_per_chgperiod = []  # average aleatoric uncertainty
        # estimated variance by kal. filter
        # TODO use epist_unc_per_chgperiod also for Kalman filter (both have
        # currently same format)
        self.kal_variance_per_chgperiod = []
        # training error per chgperiod (if prediction was done)
        self.train_error_per_chgperiod = []
        # training error per epoch for each chgperiod (if prediction was done)
        self.train_error_for_epochs_per_chgperiod = []

        # CMA-ES variables
        self.angle_per_gen = []
        self.sig_per_gen = []
        self.p_sig_per_gen = []
        self.h_sig_per_gen = []
        self.p_c_per_gen = []
        self.p_sig_pred_per_gen = []
        self.m_per_gen = []

        # global optimum
        self.glob_opt_per_gen = []

        assert pred_variant in ["simplest", "a", "b", "c", "d", "g"] and cma_variant == "predcma_external" or \
            pred_variant in ["branke", "f", "h"] and cma_variant == "predcma_internal" or \
            pred_variant is None and cma_variant not in [
                "predcma_external", "predcma_internal"]

    # =============================================================================
    # for dynamic CMA-ES

    def get_new_pred_path(self, diff_elem_0, diff_elem_1):
        diff_vector = np.linalg.norm(diff_elem_0 - diff_elem_1) / 2

        new_p_sig = (1 - self.c_sig) * self.p_sig_pred + \
            sqrt(self.c_sig * (2 - self.c_sig)) * sqrt(self.mu_w) * diff_vector
        assert new_p_sig.shape == (self.n,)
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
            unc = self.epist_unc_per_chgperiod[-1]
        except IndexError:
            try:
                unc = self.kal_variance_per_chgperiod[-1]
            except IndexError:
                unc = None
        print("pred: ", pred)
        print("unc: ", unc)

        # set sig and m
        if self.cma_variant == "predcma_external" and len(self.pred_opt_pos_per_chgperiod) > 1:
            if self.pred_variant in ["simplest", "a", "b", "c", "d"]:
                if pred is None:
                    warnings.warn(
                        "pred_variant " + self.pred_variant + " requires a prediction")
                    sys.exit()
                self.m = pred

            if self.pred_variant == "a":
                if unc is None:
                    warnings.warn(
                        "pred_variant " + self.pred_variant + " requires an uncertainty estimation")
                    sys.exit()
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
                warnings.warn("unkown pred_variant: " + self.pred_variant)
                sys.exit()

        elif self.cma_variant == "predcma_internal" and len(self.best_found_pos_per_chgperiod) > 1:
            if self.pred_variant == "e":
                pass
            elif self.pred_variant == "f":
                self.p_sig_pred = self.get_new_pred_path(
                    self.best_found_pos_per_chgperiod[-1], self.best_found_pos_per_chgperiod[-2])
                self.sig = self.p_sig_pred
            elif self.pred_variant == "h":
                # auskommentieren für hc)
                self.m = self.m + \
                    self.best_found_pos_per_chgperiod[-1] - \
                    self.best_found_pos_per_chgperiod[-2]
                # ha)
                #sig = 1
                # hb) + hc)
                #sig = np.linalg.norm(m_old - self.best_found_pos_per_chgperiod[-1]) / 2
                # hd)
                self.p_sig_pred = self.get_new_pred_path(
                    m_old, self.best_found_pos_per_chgperiod[-1])
                self.sig = self.p_sig_pred
            elif self.pred_variant == "branke":
                tmp_mu_best_individuals = get_mue_best_individuals(
                    self.n, self.mu, self.population, self.population_fitness)
                self.m = get_weighted_avg(
                    self.n, self.w, tmp_mu_best_individuals)

                diff_vals = np.subtract(
                    self.best_found_pos_per_chgperiod[:-1], self.best_found_pos_per_chgperiod[1:])
                norm_vals = np.linalg.norm(diff_vals, axis=1)
                s = np.average(norm_vals)
                self.sig = s / 2
            else:
                warnings.warn("unkown pred_variant: ", self.pred_variant)
                sys.exit()
        else:
            self.sig = self.init_sigma

    # =============================================================================

    def optimize(self):
        # ---------------------------------------------------------------------
        # local variables for predictor
        # ---------------------------------------------------------------------
        predictor = build_predictor(self.predictor_name, self.n_time_steps,
                                    self.dim, self.batch_size, self.n_neurons,
                                    self.return_seq, False, self.n_layers,
                                    self.n_epochs, self.tl_rnn_type, self.n_tllayers,
                                    self.with_dense_first, self.tl_learn_rate, self.use_uncs,
                                    self.train_mc_runs, self.train_dropout, self.test_dropout,
                                    self.kernel_size, self.n_kernels, self.lr)
        ar_predictor = None
        sess = None  # necessary since is assigned anew after each training
        if self.predictor_name == "tfrnn" or self.predictor_name == "tftlrnn" or \
                self.predictor_name == "tftlrnndense" or self.predictor_name == "tcn":
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

        for t in range(self.generations):
            glob_opt = self.experiment_data['global_opt_pos_per_gen'][t]
            print("generation , ", t, " glob opt: ", glob_opt)

            env_changed = environment_changed(t, self.population, self.population_fitness,
                                              self.benchmarkfunction, self.experiment_data, self.cma_np_rnd_generator)
            if env_changed and t != 0:
                print("\nchanged")
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
                 ar_predictor) = prepare_data_train_and_predict(sess, t, self.dim, predictor,
                                                                self.experiment_data, self.n_epochs, self.batch_size,
                                                                self.return_seq, self.shuffle_train_data, self.n_new_train_data,
                                                                self.best_found_pos_per_chgperiod, self.train_interval,
                                                                self.predict_diffs, self.n_time_steps, self.n_required_train_data,
                                                                self.predictor_name, self.add_noisy_train_data,
                                                                self.n_noisy_series, self.stddev_among_runs_per_chgp,
                                                                self.test_mc_runs, self.benchmarkfunction, self.use_uncs,
                                                                self.epist_unc_per_chgperiod, self.aleat_unc_per_chgperiod,
                                                                self.pred_opt_pos_per_chgperiod, self.pred_opt_fit_per_chgperiod,
                                                                self.kal_variance_per_chgperiod, self.train_error_per_chgperiod,
                                                                self.train_error_for_epochs_per_chgperiod)

                if not ar_predictor is None:
                    predictor = ar_predictor

                # (re-)set variables
                self.update_sig_and_m_after_chage(m_begin_chgp, my_pred_mode)
                self.p_sig = np.zeros(self.n)
                self.p_c = np.zeros(self.n)
                self.C = np.identity(self.n)

            # ---------------------------------------------------------------------
            # eigenvalue decomposition

            inv_squareroot_C, sqrt_of_eig_vals_C, eig_vals_C, eig_vctrs_C = get_inverse_sqroot(
                self.C)

            print("m    : ", self.m)
            print("C: ", self.C)
            print("sig: ", self.sig)
            # ---------------------------------------------------------------------

            self.population, self.population_fitness = get_offsprings(
                self.n, self.m, self.sig, self.lambd, sqrt_of_eig_vals_C, eig_vctrs_C, t)
            mu_best_individuals = get_mue_best_individuals(
                self.n, self.mu, self.population, self.population_fitness)

            # parameter update
            m_new = get_weighted_avg(self.n, self.w, mu_best_individuals)
            p_sig_new = get_new_p_sig(
                self.n, self.c_sig, self.p_sig, self.mu_w, self.m, m_new, self.sig, inv_squareroot_C)
            h_sig = get_h_sig(p_sig_new, self.c_sig, t, self.n, self.E)
            p_c_new = get_new_p_c(self.n, self.c_c, self.p_c, h_sig, self.mu_w,
                                  m_new, self.m, self.sig)
            C_mu = get_C_mu(self.n, mu_best_individuals,
                            self.m, self.sig, self.w)
            C_new = get_new_C(self.n, self.c_1, self.c_mu,
                              self.C, self.p_c_new, C_mu)
            sig_new = get_new_sig(self.sig, self.c_sig,
                                  self.d_sig, self.p_sig_new, self.E)

            # ---------------------------------------------------------------------
            # store old variables
            self.glob_opt_per_gen.append(glob_opt)
            self.angle_per_gen.append(
                visualize_dominant_eigvector(self.n, eig_vals_C, eig_vctrs_C))
            self.sig_per_gen.append(self.sig)
            self.p_sig_per_gen.append(self.p_sig)
            self.h_sig_per_gen.append(h_sig)
            self.p_c_per_gen.append(self.p_c)
            self.p_sig_pred_per_gen.append(self.p_sig_pred_per_gen)
            self.m_per_gen.append(self.m)

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
            so_far_best_fit, so_far_best_ind = self.get_best_fit_and_ind_so_far(
                so_far_best_fit, self.population, self.population_fitness)

            # best fitness and individual in generation
            min_fitness_index = np.argmin(self.population_fitness)
            self.best_found_fit_per_gen[t] = copy.copy(
                self.population_fitness[min_fitness_index])
            self.best_found_pos_per_gen[t] = copy.copy(
                self.population[min_fitness_index])
            print("best: ", self.population_fitness[min_fitness_index],
                  "[", self.population[min_fitness_index], "]")

        if self.predictor_name == "tfrnn" or self.predictor_name == "tftlrnn" or \
                self.predictor_name == "tftlrnndense" or self.predictor_name == "tcn":
            # TODO why not for "rnn"?
            sess.close()
            tf.reset_default_graph()

        # save results for last change period
        self.best_found_pos_per_chgperiod.append(copy.copy(so_far_best_ind))
        self.best_found_fit_per_chgperiod.append(copy.copy(so_far_best_fit))

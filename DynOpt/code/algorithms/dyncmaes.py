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

from sklearn.preprocessing.data import MinMaxScaler

import numpy as np
from utils import utils_dynopt
from utils.utils_cmaes import get_new_sig, get_mue_best_individuals,\
    get_weighted_avg, get_inverse_sqroot, get_new_p_sig, get_offsprings, \
    get_h_sig, get_new_p_c, visualize_dominant_eigvector, get_C_mu, get_new_C
from utils.utils_dynopt import environment_changed
from utils.utils_prediction import build_predictor


class MyMinMaxScaler(MinMaxScaler):
    '''
    Changes the inverse_transform method.
    '''

    def inverse_transform(self, X, only_range=False):
        '''
        @param X: data to be inverse transformed
        @param only_range: False for normal inverse transformation behavior
        (i.e. that of the super class). If True, only the width of the data 
        range is adapted but not the position of that range.
        '''
        if only_range:
            # e.g. for re-scaling aleatoric uncertainty: only the range should
            # be adapted but not the "position" since the values have to be
            # positive
            X = super(MyMinMaxScaler, self).inverse_transform(X)
            # In inverse_transform() are the last two lines:
            #      X -= self.min_
            #      X /= self.scale_
            # In order to only adapt the width of the range but not the position
            # only the last line X /= self.scale_ is needed. Therefore here
            # the second last line X -= self.min_ is un-done.
            X *= self.scale_  # undo
            X += self.min_   # undo
            X /= self.scale_  # redo
            return X
        else:
            return super(MyMinMaxScaler, self).inverse_transform(X)
#------------------------------------------------------------------------------


class DynamicCMAES(object):
    '''
    classdocs
    '''

    def __init__(self,
                 benchmarkfunction, dim,
                 n_generations, experiment_data, predictor_name, lbound, ubound,
                 cma_np_rnd_generator, pred_np_rnd_generator,
                 mean, sigma,
                 reinitialization_mode, sigma_factors,
                 timesteps, n_neurons, epochs, batchsize, n_layers, apply_tl,
                 n_tllayers, tl_model_path, tl_learn_rate, max_n_chperiod_reps,
                 add_noisy_train_data, train_interval, n_required_train_data, use_uncs,
                 train_mc_runs, test_mc_runs, train_dropout, test_dropout,
                 kernel_size, n_kernels, lr, cma_variant, impr_fct, pred_variant):
                #mu_w, w, c_sig, d_sig, c_c, c_1, c_mu, p_sig,
                # p_c, C, E, chg_freq, c_o, c_o1, p_o, C_o):
        '''
        Constructor
        '''
        # TODOs:
        # - n durch dim ersetzen
        # - self-Uebergabeparameter erglassen. Stattdessen in Methoden self. nutzen
        #  - überall self nutzen, also keine Zuweisungen auf gleichnamige Variablen
        # ---------------------------------------------------------------------
        # for the problem
        # ---------------------------------------------------------------------
        self.benchmarkfunction = benchmarkfunction
        self.n = dim  # TODO renames
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
        #self.m = mean
        #self.sig = sigma
        self.reinitialization_mode = reinitialization_mode
        self.sigma_factors = sigma_factors

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
        # d_sig = 0.6  # 0.3
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

        self.p_o = np.zeros(self.n)
        self.C_o = np.identity(self.n)

        self.sig_pred = np.random.rand()
        self.p_sig_pred = np.zeros(self.n)

        # ---------------------------------------------------------------------
        # options
        # ---------------------------------------------------------------------
        self.cma_variant = cma_variant
        self.impr_fct = impr_fct
        self.pred_variant = pred_variant

        # ---------------------------------------------------------------------
        # values that are not passed as parameters to the constructor
        # ---------------------------------------------------------------------
        self.init_sigma = self.sigma

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
        # stores the population of the (beginning of the) last generation
        self.population_of_last_gen = None

        # ---------------------------------------------------------------------
        # for EA (evaluation of variance) (repetitions of change periods)
        # ---------------------------------------------------------------------
        # add noisy data (noise equals standard deviation among change period
        # runs TODO could be replaced by "max_n_chgperiod_reps > 1"
        self.add_noisy_train_data = add_noisy_train_data
        self.n_noisy_series = 20
        # number repetitions of the single change periods (at least 1 -> 1 run)
        # TODO insert into PSO
        self.max_n_chgperiod_reps = max_n_chperiod_reps
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

# =============================================================================

        # -----
        self.fit_per_gen = []
        self.ind_per_gen = []
        self.angle_per_gen = []
        self.sig_per_gen = []
        self.p_sig_per_gen = []
        self.h_sig_per_gen = []
        self.p_c_per_gen = []
        self.frst = []
        self.scnd = []
        self.thrd = []
        self.sig_exp = []
        self.sig_norm = []
        self.sig_inner = []
        self.sig_sub = []
        self.m_per_gen = []
        self.glob_opt_per_gen = []
        self.best_ind_per_chgp = []
        self.p_o_per_gen = []
        self.max_C_frst = []
        self.max_C_scnd = []
        self.max_C_thrd = []
        self.max_C_per_gen = []
        self.cov_ellipses_per_intv = []
        self.pop_per_intv = []
        self.contour_X_per_intv = []
        self.contour_Y_per_intv = []
        self.contour_Z_per_intv = []
        self.c_idx = 0
        self.means_per_chgp = []
        self.m_new_per_intv = []
        self.glob_opt_per_intv = []
        self.m_per_intv = []
        self.stagnation_markers = []
        self.reset_markers = []
        self.max_sampling_cov_per_gen = []
        self.max_C_elem_idx_changed_marker = []
        self.max_C_elem_idx_per_gen = []
        self.predictions = []

        use_p_o_for_sig_setups = {"resetcma": False,
                                  "pathcma_p_o_for_sig": True,
                                  "pathcma_p_o_for_m": False,
                                  "predcma_external": False,
                                  "predcma_internal": False,
                                  "predcma_adapt_sig": False,
                                  "test": False}

        do_pred_setups = {"resetcma": False,
                          "pathcma_p_o_for_sig": False,
                          "pathcma_p_o_for_m": False,
                          "predcma_external": True,
                          "predcma_internal": False,
                          "predcma_adapt_sig": True,
                          "test": False}

        set_sig_manually_setups = {"resetcma": False,
                                   "pathcma_p_o_for_sig": False,  # schlecht
                                   "pathcma_p_o_for_m": False,
                                   "predcma_external": False,
                                   "predcma_internal": False,
                                   # ||(sig = m-pred)||/2
                                   "predcma_adapt_sig": True,
                                   "test": True}
        self.use_p_o_for_sig = use_p_o_for_sig_setups[cma_variant]
        self.do_pred = do_pred_setups[cma_variant]
        self.set_sig_manually = set_sig_manually_setups[cma_variant]

        assert pred_variant in ["simplest", "a", "b", "c", "d", "g"] and cma_variant == "predcma_external" or \
            pred_variant in ["branke", "f", "h"] and cma_variant == "predcma_internal" or \
            pred_variant is None and cma_variant not in [
            "predcma_external", "predcma_internal"]

    # -------------------------------------------------------------------------------
    # for prediction
    def get_prediction(self, pred_type, predictor, train_data, glob_opt, pred_noise,
                       do_training, curr_m):
        '''
        @param pred_noise: is standard deviation
        @return tupel: (predicted optimum, prediction uncertainty)
        - prediction uncertainty is (error co-)variance 
        '''
        if pred_type == "truepred":
            return glob_opt + np.random.normal(0, pred_noise), pred_noise**2
        elif pred_type == "kalman":
            scaler = None
            pred, unc = self.predict_with_kalman(
                train_data, scaler, predictor,  do_training)
            return curr_m + pred, unc

    def fit_scaler(self, data_for_fitting):
        scaler = MyMinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data_for_fitting)
        return scaler

    def predict_with_kalman(self, train_data, scaler, predictor,  do_training):
        '''
        Predicts next optimum position with a Kalman filter.
        @param new_train_data: format [n_data, dims]
        '''

        # scale data (the data are re-scaled directly after the
        # prediction in this iteration)
        scaler = self.fit_scaler(train_data)
        train_data = scaler.transform(copy.copy(train_data))
        # -----------------
        if do_training:
            # "training" of parameters
            predictor.em(train_data)

        # computation of states for past observations
        means, covariances = predictor.filter(train_data)

        # predicting the next step
        new_measurement = None  # not yet known
        next_mean, next_covariance = predictor.filter_update(
            means[-1], covariances[-1], new_measurement)
        # variance per dimension
        next_variance = np.diagonal(next_covariance)

        # invert scaling (1d array would result in DeprecatedWarning -> pass
        # 2d)
        next_mean = next_mean.reshape(1, -1)
        next_variance = next_variance.reshape(1, -1)
        next_mean = scaler.inverse_transform(next_mean, False).flatten()
        next_variance = scaler.inverse_transform(next_variance, True).flatten()
        assert (next_variance >= 0).all()
        return next_mean, next_variance
    # -------------------------------------------------------------------------------
    # for dynamic CMA-ES

    def get_new_p_o(self, mu_w, sig, n, c_o, p_o, o_new, o, use_p_o_for_sig=False):
        if True:
            h_sig = 1  # TODO
            first = (1 - c_o) * p_o
            second = sqrt(c_o * (2 - c_o))
            if use_p_o_for_sig:
                third = (o_new - o) / np.linalg.norm(o_new - o)
            else:
                third = o_new - o
            #third = np.sqrt(abs(o_new - o))

            # Division macht keinen Sinn, weil sigma nichts mit Optimumbewegung zu tun
            # hat; dadurch würde Kovarianz auch viel zu riesig
            new_p_o = first + h_sig * second * sqrt(mu_w) * third  # / sig
            # new_p_o = first + h_sig * second * third  # / sig
            #new_p_o = first + second * sqrt(mu_w) * third
            #new_p_o = np.sqrt(abs(new_p_o))
        else:
            first = 0.5 * p_o
            second = 0.5
            third = (o_new - o)
            new_p_o = first + second * third  # / sig
        assert new_p_o.shape == (n,)
        return new_p_o

    def get_new_pred_path(self, n, c_sig, old_pred_path, mu_w, diff_elem_0, diff_elem_1):
        diff_vector = np.linalg.norm(diff_elem_0 - diff_elem_1) / 2

        new_p_sig = (1 - c_sig) * old_pred_path + \
            sqrt(c_sig * (2 - c_sig)) * sqrt(mu_w) * diff_vector
        assert new_p_sig.shape == (n,)
        return new_p_sig

    def update_sig_and_m_after_chage(self):
        if cma_variant == "pathcma_p_o_for_m" and len(means) > 1:
            sig = np.linalg.norm(p_o) / 2
            #sig = np.linalg.norm(np.abs(p_o)) / 2
        elif cma_variant == "test" and set_sig_manually and len(means) > 1:
            #sig = (means[-1] - means[-2])
            sig = np.linalg.norm(means[-1] - means[-2]) / 2
        elif set_sig_manually and do_pred:
            sig = np.linalg.norm(m - pred) / 2
        elif use_p_o_for_sig:
            sig = get_new_sig(1, c_sig, d_sig, p_o,
                              E, sig_exp, sig_norm, sig_inner, sig_sub)
        elif False and do_pred and len(predictions) > 1 and cma_variant == "predcma_simplest" and len(means) > 1:
            #sig = pred_noise / 2
            pass
        elif cma_variant == "pathcma_p_o_for_m":
            m = m + p_o
            #sig = np.linalg.norm(m - pred) / 2

        if cma_variant == "predcma_external" and len(predictions) > 1:
            if pred_variant in ["simplest", "a", "b", "c", "d"]:
                m = pred

            if pred_variant == "a":
                sig = np.sqrt(unc)
            elif pred_variant == "b":
                sig = (predictions[-2] - best_ind_per_chgp[-1]) / 2
                # sig = np.linalg.norm(
                #    predictions[-2] - train_data[-1])
            elif pred_variant == "c":
                # aus Versehen vorher p_sig statt p_sig_pred
                p_sig_pred = self.get_new_pred_path(
                    n, c_sig, p_sig_pred, mu_w, predictions[-2], best_ind_per_chgp[-1])
                sig = p_sig_pred
            elif pred_variant in ["simplest", "d"]:
                sig = 1
            elif pred_variant == "g":
                sig = np.linalg.norm(
                    predictions[-1] - best_ind_per_chgp[-1]) / 2
            else:
                sig = None
                m = None
                warnings.warn("unkown pred_variant: " + pred_variant)

        elif cma_variant == "predcma_internal" and len(best_ind_per_chgp) > 1:
            if pred_variant == "e":
                pass
            elif pred_variant == "f":
                p_sig_pred = self.get_new_pred_path(
                    n, c_sig, p_sig_pred, mu_w, best_ind_per_chgp[-1], best_ind_per_chgp[-2])
                sig = p_sig_pred
            elif pred_variant == "h":
                # auskommentieren für hc)
                m = m + \
                    best_ind_per_chgp[-1] - best_ind_per_chgp[-2]
                # ha)
                #sig = 1
                # hb) + hc)
                #sig = np.linalg.norm(m_old - best_ind_per_chgp[-1]) / 2
                # hd)
                p_sig_pred = self.get_new_pred_path(
                    n, c_sig, p_sig_pred, mu_w, m_old, best_ind_per_chgp[-1])
                sig = p_sig_pred
            elif pred_variant == "branke":
                tmp_mu_best_individuals = get_mue_best_individuals(
                    n, mu, offspring_population, offspring_fitnesses)
                m = get_weighted_avg(
                    n, w, tmp_mu_best_individuals)

                diff_vals = np.subtract(
                    best_ind_per_chgp[:-1], best_ind_per_chgp[1:])
                norm_vals = np.linalg.norm(diff_vals, axis=1)
                s = np.average(norm_vals)
                sig = s / 2
            else:
                warnings.warn("unkown pred_variant: ", pred_variant)
                return None, None
        else:
            # sig = np.random.rand()  # sonst Error (singuläre Matrix)
            sig = 1

        return sig, m, p_sig_pred
    # -------------------------------------------------------------------------------

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
        m_old = None
        best_fit = sys.float_info.max
        unsucc_count = 0
        not_yet_reset = True  # True if C was not manually reset to I during change period
        max_C_elem_idx = 0.3  # index of the maximum element in C
        for t in range(self.generations):
            glob_opt = self.experiment_data['global_opt_pos_per_gen'][t]
            print("generation , ", t, " glob opt: ", glob_opt)

            env_changed = environment_changed(t, self.population, self.population_fitness,
                                              self.benchmarkfunction, self.experiment_data, self.cma_np_rnd_generator)
            if env_changed:
                print("\nchanged")

                best_fit = sys.float_info.max
                self.means_per_chgp.append(self.m)
                self.best_ind_per_chgp.append(self.ind_per_gen[-1])

                if self.use_p_o_for_sig and len(self.means_per_chgp) > 1:
                    p_o = self.get_new_p_o(self.mu_w, self.sig, self.n, self.c_o, self.p_o,
                                           self.means_per_chgp[-1], self.means_per_chgp[-2], self.use_p_o_for_sig)

                if self.do_pred and len(self.means_per_chgp) > 1:
                    curr_train_data = min(
                        len(self.best_ind_per_chgp), self.n_required_train_data)
                    train_data = self.best_ind_per_chgp[-curr_train_data:]
                    diff_train_data = np.subtract(
                        train_data[1:], train_data[:-1])
                    do_training = t % self.n_required_train_data == 0 or t < 10  # also in first chgp
                    pred, unc = self.get_prediction(self.predictor_name, predictor, diff_train_data,
                                                    glob_opt, None, do_training,
                                                    self.means_per_chgp[-1])
                    self.pred_opt_pos_per_chgperiod.append(pred)

                self.sig, self.m, self.p_sig_pred = self.update_sig_and_m_after_chage(self.cma_variant, self.pred_variant, self.set_sig_manually,
                                                                                      self.do_pred, self.use_p_o_for_sig,
                                                                                      self.p_o, self.mu, self.w, self.p_sig, self.n, self.m, m_old, self.c_sig, self.d_sig,
                                                                                      self.E, pred, unc, self.mu_w, self.sig, self.sig_pred,
                                                                                      self.population, self.population_fitness,
                                                                                      self.sig_exp, self.sig_norm, self.sig_inner, self.sig_sub,
                                                                                      self.means_per_chgp, np.array(
                                                                                          self.best_ind_per_chgp), self.pred_opt_pos_per_chgperiod,
                                                                                      self.p_sig_pred)
                self.p_sig = np.zeros(self.n)
                self.p_c = np.zeros(self.n)
                self.C = np.identity(self.n)

            # ---------------------------------------------------------------------
            # eigenvalue decomposition

            inv_squareroot_C, sqrt_of_eig_vals_C, eig_vals_C, eig_vctrs_C = get_inverse_sqroot(
                self.C)

            # ---------------------------------------------------------------------
            print("pred: ", pred)
            print("unc: ", unc)
            #print("new_p_o: ", p_o)
            #print("new_sig: ", sig)
            print("m    : ", self.m)
            print("C: ", self.C)
            print("sig: ", self.sig)
            # ---------------------------------------------------------------------

            # offsprings
            # if env_changed and len(predictions) > 1:
            #    offspring_population, offspring_fitnesses = get_offsprings(
            #        n, m, sig_pred, lambd, sqrt_of_eig_vals_C, eig_vctrs_C, t, chg_freq)
            # else:
            offspring_population, offspring_fitnesses = get_offsprings(
                self.n, self.m, self.sig, self.lambd, sqrt_of_eig_vals_C, eig_vctrs_C, t)
            mu_best_individuals = get_mue_best_individuals(
                self.n, self.mu, offspring_population, offspring_fitnesses)

            # parameter update
            m_new = get_weighted_avg(self.n, self.w, mu_best_individuals)
            print("m_new: ", m_new)
            p_sig_new = get_new_p_sig(
                self.n, self.c_sig, self.p_sig, self.mu_w, self.m, m_new, self.sig, inv_squareroot_C)
            self.h_sig = get_h_sig(p_sig_new, self.c_sig, t, self.n, self.E)
            p_c_new = get_new_p_c(self.n, self.c_c, self.p_c, self.h_sig, self.mu_w,
                                  m_new, self.m, self.sig, self.frst, self.scnd, self.thrd)
            self.C_mu = get_C_mu(self.n, mu_best_individuals, self.m,
                                 self.sig, self.w, None, pred)
            C_new = get_new_C(self.n, self.c_1, self.c_mu, self.C, self.p_c_new, self.C_mu,
                              self.max_C_frst, self.max_C_scnd, self.max_C_thrd)
            # try:
            sig_new = get_new_sig(self.sig, self.c_sig, self.d_sig, self.p_sig_new,
                                  self.E, self.sig_exp, self.sig_norm, self.sig_inner, self.sig_sub)
            # except:
            # break
            # print("error")
            #sig_new = 1

            # ---------------------------------------------------------------------
            # store old variables
            self.max_sampling_cov_per_gen.append(np.max(self.C))  # for plot
            max_C_elem_idx = np.argmax(self.C)
            self.max_C_elem_idx_per_gen.append(max_C_elem_idx)
            if len(self.max_C_elem_idx_per_gen) > 1 and self.max_C_elem_idx_per_gen[-2] != self.max_C_elem_idx_per_gen[-1]:
                self.max_C_elem_idx_changed_marker.append(t)
            self.max_C_per_gen.append(np.max(self.C))
            self.glob_opt_per_gen.append(glob_opt)

            self.angle_per_gen.append(
                visualize_dominant_eigvector(self.n, eig_vals_C, eig_vctrs_C))
            try:
                self.sig_per_gen.append(log(self.sig))
            except:
                self.sig_per_gen.append(np.log(np.max(self.sig)))
            self.p_sig_per_gen.append(self.p_sig)
            try:
                self.h_sig_per_gen.append(self.h_sig)
            except UnboundLocalError:  # when do_param_update is False
                self.h_sig_per_gen.append(-1)
            self.p_c_per_gen.append(self.p_c)
            self.m_per_gen.append(self.m)
            self.p_o_per_gen.append(self.p_o)

            # ---------------------------------------------------------------------

            # set variables for next generation
            if env_changed:
                m_old = self.m
            self.m = m_new
            self.p_sig = p_sig_new
            self.p_c = p_c_new
            self.C = C_new
            self.sig = sig_new

            curr_best_fit, curr_best_ind = self.print_success(
                t, offspring_population, offspring_fitnesses)
            self.fit_per_gen.append(curr_best_fit)
            self.ind_per_gen.append(curr_best_ind)

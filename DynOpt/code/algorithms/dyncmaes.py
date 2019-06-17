'''
Created on May 25, 2018

@author: ameier
'''
# CMA-ES source code can be found here
# path-to-python-environment/lib/python3.5/site-packages/cma
import copy
from math import floor, log, sqrt, exp
import sys

from numpy import linalg as npla
from scipy import linalg as spla

import numpy as np
from utils import utils_dynopt
from utils.utils_dynopt import environment_changed


class DynamicCMAES(object):
    '''
    classdocs
    '''

    def __init__(self,
                 benchmarkfunction, dim,
                 n_generations, experiment_data, predictor_name, lbound, ubound,
                 ea_np_rnd_generator, pred_np_rnd_generator,
                 mu, la, mean, sigma,
                 reinitialization_mode, sigma_factors,
                 timesteps, n_neurons, epochs, batchsize, n_layers, apply_tl,
                 n_tllayers, tl_model_path, tl_learn_rate, max_n_chperiod_reps,
                 add_noisy_train_data, train_interval, n_required_train_data, use_uncs,
                 train_mc_runs, test_mc_runs, train_dropout, test_dropout,
                 kernel_size, n_kernels, lr, cma_variant, impr_fct):
                #mu_w, w, c_sig, d_sig, c_c, c_1, c_mu, p_sig,
                # p_c, C, E, chg_freq, c_o, c_o1, p_o, C_o):
        '''
        Constructor
        '''

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

        # transfer learning
        self.apply_tl = apply_tl  # True if pre-trained model should be used
        self.tl_model_path = tl_model_path  # path to the trained tl model
        self.tl_rnn_type = "RNN"
        self.n_tllayers = n_tllayers
        self.with_dense_first = None
        self.tl_learn_rate = tl_learn_rate

        # training/testing specifications
        # number train data with that the network at least is trained
        self.n_required_train_data = max(
            n_required_train_data, self.n_time_steps)
        self.predict_diffs = True  # predict position differences, TODO insert into PSO
        self.return_seq = False  # return values for all time steps not only the last one
        # True -> train data are shuffled before training and between epochs
        self.shuffle_train_data = True  # TODO move into script?

        # ---------------------------------------------------------------------
        # for EA (fixed values)
        # ---------------------------------------------------------------------
        self.ea_np_rnd_generator = ea_np_rnd_generator
        self.pred_np_rnd_generator = pred_np_rnd_generator
        self.mu = mu
        self.lambd = la
        self.m = mean
        self.sig = sigma
        self.reinitialization_mode = reinitialization_mode
        self.sigma_factors = sigma_factors

        ################
        # -------------------------------------------------------------------------
        # search space
        # -------------------------------------------------------------------------
        #min_val = -5
        #max_val = 5
        # n = 2  # dimensions

        #generations = 5000
        # chg_freq = 50  # Winkel konvergiert bei 50 ganz gut gegen 0?
        # -------------------------------------------------------------------------
        # fixed parameters
        # -------------------------------------------------------------------------
        lambd = 4 + floor(3 * log(self.n))  # offsprings
        mu = floor(lambd / 2)  # parents
        # weights (vector of size ń)
        w_divisor = np.sum([(log(mu + 0.5) - log(j))
                            for j in range(1, mu + 1)])
        self.w = np.array([((log(mu + 0.5) - log(i)) / w_divisor)
                           for i in range(1, mu + 1)])
        # other
        self.mu_w = 1 / np.sum(np.square(self.w))
        self.c_sig = (self.mu_w + 2) / (self.n + self.mu_w + 3)
        self.d_sig = 1 + self.c_sig + 2 * \
            max(0, sqrt((self.mu_w - 1) / (self.n + 1)) - 1)
        # d_sig = 0.6  # 0.3
        self.c_c = 4 / (self.n + 4)
        self.c_1 = (2 * min(1, lambd / 6)) / ((self.n + 1.3)**2 + self.mu_w)
        self.c_mu = (2 * (self.mu_w - 2 + 1 / self.mu_w)) / \
            ((self.n + 2)**2 + self.mu_w)
        self.E = sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n**2))

        self.c_o = self.c_c
        self.c_o1 = self.c_1

        # prints
        print("mu: ", mu)
        print("lambda: ", lambd)
        print("weight_divisor: ", w_divisor)
        print("weights: ", self.w)
        print("mu_w: ", self.mu_w)
        print("c_sig: ", self.c_sig)
        print("d_sig: ", self.d_sig)
        print("c_c: ", self.c_c)
        print("c_1: ", self.c_1)
        print("c_mu: ", self.c_mu)
        print("E: ", self.E)

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
        ################
        #self.mu_w = mu_w
        #self.w = w
        #self.c_sig = c_sig
        #self.d_sig = d_sig
        #self.c_c = c_c
        #self.c_1 = c_1
        #self.c_mu = c_mu
        #self.p_sig = p_sig
        #self.p_c = p_c
        #self.C = C
        #self.E = E
        #self.c_o = c_o
        #self.c_o1 = c_o1
        #self.p_o = p_o
        #self.C_o = C_o
        self.cma_variant = cma_variant
        self.impr_fct = impr_fct

        # ---------------------------------------------------------------------
        # values that are not passed as parameters to the constructor
        # ---------------------------------------------------------------------
        self.init_sigma = self.sigma

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

        print("Constructor")

        # -----
        self.fit_per_gen = []
        self.angle_per_gen = []
        self.sig_per_gen = []
        self.p_sig_per_gen = []
        self.h_sig_per_gen = []
        self.p_c_per_gen = []
        self.frst = []
        self.scnd = []
        self.self.thrd = []
        self.sig_exp = []
        self.sig_norm = []
        self.sig_inner = []
        self.sig_sub = []
        self.m_per_gen = []
        self.opt_per_gen = []
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
        self.optima = []
        self.m_new_per_intv = []
        self.m_per_intv = []
        self.glob_opt_per_intv = []
        self.stagnation_markers = []
        self.reset_markers = []
        self.max_sampling_cov_per_gen = []
        self.max_C_elem_idx_changed_marker = []
        self.max_C_elem_idx_per_gen = []

        use_pre_pop_setups = {"resetcma": False,
                              "pathcma_simplest": False,
                              "pathcma_prepop_Co-I_sig-random": True,
                              "pathcma_prepop": True,
                              "pathcma_prepop_scaled_Co": True,
                              "pathcma_prepop_wo_Co": True,
                              "predcma_simplest": False,
                              "predcma_sig": True}

        use_C_o_for_sampling_setups = {"resetcma": False,
                                       "pathcma_simplest": True,
                                       "pathcma_prepop_Co-I_sig-random": False,
                                       "pathcma_prepop": True,
                                       "pathcma_prepop_scaled_Co": True,
                                       "pathcma_prepop_wo_Co": False,
                                       "predcma_simplest": False,
                                       "predcma_sig": True}

        use_C_o_for_p_sig_setups = {"resetcma": False,
                                    "pathcma_simplest": True,
                                    "pathcma_prepop_Co-I_sig-random": False,
                                    "pathcma_prepop": True,
                                    "pathcma_prepop_scaled_Co": True,
                                    "pathcma_prepop_wo_Co": False,
                                    "predcma_simplest": False,
                                    "predcma_sig": True}

        use_C_o_for_C_setups = {"resetcma": False,
                                "pathcma_simplest": False,
                                "pathcma_prepop_Co-I_sig-random": False,
                                "pathcma_prepop": False,
                                "pathcma_prepop_scaled_Co": False,
                                "pathcma_prepop_wo_Co": False,
                                "predcma_simplest": False,
                                "predcma_sig": False}

        scale_C_o_setups = {"resetcma": False,
                            "pathcma_simplest": False,
                            "pathcma_prepop_Co-I_sig-random": False,
                            "pathcma_prepop": False,
                            "pathcma_prepop_scaled_Co": True,
                            "pathcma_prepop_wo_Co": False,
                            "predcma_simplest": False,
                            "predcma_sig": True}

        use_pred_for_m_setups = {"resetcma": False,
                                 "pathcma_simplest": False,
                                 "pathcma_prepop_Co-I_sig-random": False,
                                 "pathcma_prepop": False,
                                 "pathcma_prepop_scaled_Co": False,
                                 "pathcma_prepop_wo_Co": False,
                                 "predcma_simplest": True,
                                 "predcma_sig": False}

        scale_sig_setups = {"resetcma": False,
                            "pathcma_simplest": False,
                            "pathcma_prepop_Co-I_sig-random": False,
                            "pathcma_prepop": False,
                            "pathcma_prepop_scaled_Co": False,
                            "pathcma_prepop_wo_Co": False,
                            "predcma_simplest": False,
                            "predcma_sig": True}

        self.use_pre_pop = use_pre_pop_setups[cma_variant]
        self.use_C_o_for_sampling = use_C_o_for_sampling_setups[cma_variant]
        self.use_C_o_for_p_sig = use_C_o_for_p_sig_setups[cma_variant]
        self.use_C_o_for_C = use_C_o_for_C_setups[cma_variant]
        self.scale_C_o = scale_C_o_setups[cma_variant]
        self.use_pred_for_m = use_pred_for_m_setups[cma_variant]
        self.scale_sig = scale_sig_setups[cma_variant]

    def get_new_p_o(self, mu_w, sig, n, c_o, p_o, o_new, o):
        if True:
            h_sig = 1  # TODO
            first = (1 - c_o) * p_o
            second = sqrt(c_o * (2 - c_o))
            third = (o_new - o)
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

    def update_C_after_change(self, n, c_o1, C_old, p_o):
        first = (1 - c_o1) * C_old
        col_vec = p_o[:, np.newaxis]  # format [n,1]
        second = c_o1 * np.matmul(col_vec, col_vec.T)
        new_C = first + second
        assert new_C.shape == (n, n)
        return new_C

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between_vectors(self, v1, v2):
        """
        (14.3.19) 
        https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249

            Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        #angle = np.arccos(np.dot(v1_u, v2_u))
        return np.degrees(angle)

    def signed_angle_between_vectors(self, v1, v2):
        if len(v1) == 2:
            if True:
                v1 = np.concatenate((v1, [0]))
                v2 = np.concatenate((v2, [0]))
            if False:
                x1 = v1[0]
                x2 = v2[0]
                y1 = v1[1]
                y2 = v2[1]
                # dot product between [x1, y1] and [x2, y2]
                dot = x1 * x2 + y1 * y2
                det = x1 * y2 - y1 * x2     # determinant
                angle = np.arctan2(det, dot)     #
                deg = np.degrees(angle)
                #print("2d degree: ", deg)
                return deg
        from numpy import (dot, arccos, clip)
        from numpy.linalg import norm
        # https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
        # (4.4.19)
        # https://web.ma.utexas.edu/users/m408m/Display12-5-4.shtml

        # normal vector of plane that constructs the coordinate system
        vn = np.ones(len(v1))

        norm1 = norm(v1)
        norm2 = norm(v2)
        c = dot(v1, v2) / (norm1 * norm2)  # -> cosine of the angle
        # angle = arccos(clip(c, -1, 1))  # if you really want the angle
        angle = arccos(c)  # if you really want the angle

        cross = np.cross(v1, v2)

        dot_comp = dot(vn, cross)

        if dot_comp < 0:
            angle = -angle
        deg = np.degrees(angle)
        return deg

    def visualize_dominant_eigvector(self, n, eig_vals, eig_vctrs):
        # determine dominant eigenvector
        max_idx = np.argmax(np.absolute(eig_vals))
        dom_vect = eig_vctrs[:, max_idx]
        #print("dom_vect: ", dom_vect)
        #print("length of dom_vect: ", np.linalg.norm(dom_vect))

        # for i in range(n):
        #print("length of vect: ", np.linalg.norm(eig_vctrs[:, i]))

        # compute angle to first axis
        orig_first_axis = np.zeros(n)
        orig_first_axis[0] = 5  # arbitrary point on first axis
        #angle = signed_angle_between_vectors(orig_first_axis, dom_vect)
        #angle = signed_angle_between_vectors(dom_vect, orig_first_axis)
        angle = np.rad2deg(np.arccos(dom_vect[0]))
        #print("anlge: ", angle)
        #print("eig_vec: ", dom_vect)
        return angle

    #------------------------

    def get_inverse_sqroot(self, M):
        '''
        M = TDT^(-1)
            T: eigenvectors
            D: eigenvalues

        Computes inverse square root of M on the eigenvalues of M and re-transforms
        results into original space of M:
        M^(-1/2) = TD^(-1/2)T^(-1) 

        Returns 
            - inverse square root of M
            - square root of eigenvalues of M
            - eigenvectors of M

        Alternative: Use pre-defined packages:
        exp_inv = npla.inv(M)  # inverse
        exp_sqr_inv = spla.sqrtm(exp_inv) # square root
        '''
        # eigenvalues & eigenvectors
        eig_vals, eig_vctrs = npla.eig(M)

        # diagonal matrix of eigenvalues
        eig_val_diag = np.diag(eig_vals)

        # square root
        sqrt_of_eig_vals = spla.sqrtm(eig_val_diag)

        # inverse of square root
        inv_of_sqrt = npla.inv(sqrt_of_eig_vals)

        # re-transform into original space
        new_M = np.matmul(np.matmul(eig_vctrs, inv_of_sqrt),
                          npla.inv(eig_vctrs))
        assert new_M.shape == M.shape
        return new_M, sqrt_of_eig_vals, eig_vals, eig_vctrs

    def get_offsprings(self, n, m, sig, lambd, sqrt_of_eig_vals, eig_vctrs, t):
        offspring_population = np.zeros((lambd, n))
        offspring_fitnesses = np.zeros(lambd)

        # decompose covariance matrix for sampling ("The CMA evolution strategy: a
        # tutorial, Hansen (2016), p. 28+29)
        # A = B*sqrt(M), with C = BMB^T (C=covariance matrix)
        A = np.matmul(eig_vctrs, sqrt_of_eig_vals)

        for k in range(lambd):
            z_k = np.random.normal(size=n)
            y_k = np.matmul(A, z_k)
            x_k = m + sig * y_k
            # x_k is equal to the following line but saves eigendecompositions:
            # m + sig * np.random.multivariate_normal(np.zeros(n), C)

            f_k = utils_dynopt.fitness(self.benchmarkfunction,
                                       x_k, t, self.experiment_data)

            offspring_population[k] = copy.copy(x_k)
            offspring_fitnesses[k] = f_k
        return offspring_population, offspring_fitnesses

    def get_mue_best_individuals(self, n, mu, offspring_population, offspring_fitnesses):
        # sort individuals according to fitness
        sorted_indices = np.argsort(offspring_fitnesses)
        sorted_individuals = offspring_population[sorted_indices]
        # select mu best individuals
        mu_best_individuals = sorted_individuals[:mu, :]
        assert mu_best_individuals.shape == (mu, n)
        return mu_best_individuals  # format [individuals, dimensions]

    def get_weighted_avg(self, n, w, mu_best_individuals):
        # weight individuals
        weighted_indvds = mu_best_individuals * w[:, np.newaxis]
        # sum averaged individuals
        weighted_avg_indvds = np.sum(weighted_indvds, axis=0)
        assert weighted_avg_indvds.shape == (n,)
        return weighted_avg_indvds

    def get_new_p_sig(self, n, c_sig, p_sig, mu_w, m, m_new, sig, inv_squareroot_C):
        succ_mutation_steps = (m_new - m) / sig
        # tmp = np.matmul(succ_mutation_steps, inv_squareroot_C) #geht beides
        tmp = np.matmul(inv_squareroot_C, succ_mutation_steps)

        new_p_sig = (1 - c_sig) * p_sig + \
            sqrt(c_sig * (2 - c_sig)) * sqrt(mu_w) * tmp
        assert new_p_sig.shape == (n,)
        return new_p_sig

    def get_h_sig(self, p_sig_new, c_sig, t, n, E):
        right_side = sqrt(1 - (1 - c_sig)**(2 * (t + 1))) * \
            (1.4 + 2 / (n + 1)) * E
        cond = npla.norm(p_sig_new) < right_side
        return int(cond == True)  # 0 is False, 1 otherwise

    def get_new_p_c(self, n, c_c, p_c, h_sig, mu_w, m_new, m, sig, frst, scnd, thrd):
        first = (1 - c_c) * p_c
        second = sqrt(c_c * (2 - c_c))
        third = (m_new - m)
        frst.append(first)
        scnd.append(second)
        thrd.append(third)
        new_p_c = first + h_sig * second * \
            sqrt(mu_w) * third / sig
        assert new_p_c.shape == (n,)
        return new_p_c

    def get_C_mu(self, n, mu_best_individuals, m, sig, w):
        # subtract mean from each row, scale with sigma
        # format: [individuals, dimensions]
        matrix = (mu_best_individuals - m) / sig

        # multiply each row with respective weight (4.3.19)
        # https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
        weighted_matrix = w[:, np.newaxis] * matrix

        # in the multiplication the matrix is first transposed so that covariance
        # matrix has format [dimension, dimensions] otherwise it would have format
        # [individuals, individuals]
        C_mu = np.matmul(weighted_matrix.T, matrix)
        assert C_mu.shape == (n, n)
        return C_mu

    def get_new_C(self, n, c_1, c_mu, C, p_c_new, C_mu, max_C_frst, max_C_scnd, max_C_thrd):
        if False:  # test other learning rates
            c_1 = 0.2
            c_mu = 0.5
            first = (0.3) * C  # 0.3
        else:
            first = (1 - c_1 - c_mu) * C
        col_vec = p_c_new[:, np.newaxis]  # format [n,1]
        second = c_1 * np.matmul(col_vec, col_vec.T)
        third = c_mu * C_mu
        new_C = first + second + third
        max_C_frst.append(np.max(first))
        max_C_scnd.append(np.max(second))
        max_C_thrd.append(np.max(third))
        # print("max_new_C: ", np.max(new_C), "1. ",
        #      max_C_frst[-1], " 2. ", max_C_scnd[-1], " 3. ", max_C_thrd[-1])
        assert new_C.shape == (n, n)
        return new_C

    def get_new_sig(self, sig, c_sig, d_sig, p_sig_new, E, sig_exp, sig_norm, sig_inner, sig_sub):

        # [ 2.08807787e+09 -2.96451771e-01] davor: [4.27932469e-19 7.26925101e-01]
        #print("p_sig_new: ", p_sig_new)
        norm_val = npla.norm(p_sig_new)
        # print("norm: ", norm_val)  # 2088077871.9406497
        subtr = norm_val / E - 1
        inner = (c_sig / d_sig) * subtr
        # print(c_sig / d_sig) #0.36434206766003985
        # print("subt: ", subtr) # 1664771783.2035427
        # print("inner: ", inner)  # 606546393.6744703
        tmp = exp(inner)
        sig_exp.append(tmp)
        sig_norm.append(norm_val)
        sig_inner.append(inner)
        sig_sub.append(subtr)
        return sig * tmp

    def get_new_sig_quadr(self, sig, c_sig, d_sig, p_sig_new, E, sig_tmp, sig_norm, sig_inner, sig_sub):
        norm_val = npla.norm(p_sig_new)**2
        subtr = norm_val / 2 - 1
        inner = (c_sig / 2 * d_sig) * subtr
        #print("inner: ", inner)
        tmp = exp(inner)
        sig_tmp.append(tmp)
        sig_norm.append(norm_val)
        sig_inner.append(inner)
        sig_sub.append(subtr)
        return sig * tmp
    # -------------------------------------------------------------------------------

    def print_success(self, t, offspring_population, offspring_fitnesses):
        min_idx = np.argmin(offspring_fitnesses)
        print(t, "    ", offspring_fitnesses[min_idx],
              "    ", offspring_population[min_idx])
        # return offspring_fitnesses[min_idx]
        try:
            return log(offspring_fitnesses[min_idx])  # TODO 23.4 wieder log
        except ValueError:  # since fitness is zero
            return log(np.nextafter(0, 1))  # smallest float larger than 0

    # -------------------------------------------------------------------------------

    def optimize(self):
        best_fit = sys.float_info.max
        unsucc_count = 0
        not_yet_reset = True  # True if C was not manually reset to I during change period
        max_C_elem_idx = 0.3  # index of the maximum element in C
        for t in range(self.generations):
            env_changed = environment_changed(t, self.population, self.population_fitness,
                                              self.benchmarkfunction, self.experiment_data, self.cma_np_rnd_generator)
            if env_changed:
                best_fit = sys.float_info.max
                not_yet_reset = True
                t_last_chg = t
                self.detected_n_changes += 1
                # if False:
                print("\nchanged")

                self.optima.append(self.m)  # TODO oder m_t-1???

                if len(self.optima) > 1:
                    self.p_o = self.get_new_p_o(self.mu_w, self.sig, self.n, self.c_o, self.p_o,
                                                self.optima[-1], self.optima[-2])
                    self.C_o = self.update_C_after_change(
                        self.n, self.c_o1, self.C_o, self.p_o)
                inv_squareroot_C_o, sqrt_of_eig_vals_C_o, eig_vals_C_o, eig_vctrs_C_o = self.get_inverse_sqroot(
                    self.C_o)
                if self.scale_C_o:
                    C_o_scaled = self.C_o / np.max(self.C_o)
                    inv_squareroot_C_o_scaled, sqrt_of_eig_vals_C_o_scaled, eig_vals_C_o_scaled, eig_vctrs_C_o_scaled = self.get_inverse_sqroot(
                        C_o_scaled)
                #m = np.random.randint(0, 100, n)
                # m = get_moved_glob_opt(t, n, chg_freq)  # + 0.1
                #print("m: ", m)
                self.sig = np.random.rand()  # ansonsten Error, wegen singulärer Matrix
                #p_sig = np.zeros(n)
                #p_c = np.zeros(n)
                self.C = np.identity(self.n)

            # restart if stagnation occurs
            stagnated = len(
                self.stagnation_markers) > 1 and self.stagnation_markers[-1] == t - 1
            # if True and t >= chg_freq and t == t_last_chg + chg_freq // 4:
            # nicht in erster Generation der change period zuruecksetzen
            if False and stagnated and not_yet_reset and not env_changed:
                # if stagnated:
                print("reset: t: ", t)
                self.sig = np.random.rand()
                self.C = np.identity(self.n)
                not_yet_reset = False
                self.reset_markers.append(t)
            # ---------------------------------------------------------------------
            # eigenvalue decomposition
            inv_squareroot_C, sqrt_of_eig_vals_C, eig_vals_C, eig_vctrs_C = self.get_inverse_sqroot(
                self.C)

            # ---------------------------------------------------------------------
            # set values

            # for C
            if self.use_C_o_for_C and env_changed:
                if self.scale_C_o:
                    C_for_C = C_o_scaled
                else:
                    C_for_C = self.C_o
            else:
                C_for_C = self.C

            # for p_sig
            if self.use_C_o_for_p_sig and env_changed:
                if self.scale_C_o:
                    inv_squareroot_for_p_sig = inv_squareroot_C_o_scaled
                else:
                    inv_squareroot_for_p_sig = inv_squareroot_C_o
            else:
                inv_squareroot_for_p_sig = inv_squareroot_C

            # for sampling
            if self.use_C_o_for_sampling and env_changed:
                if self.scale_C_o:
                    eig_vals_smpl = eig_vals_C_o_scaled
                    sqrt_of_eig_vals_smpl = sqrt_of_eig_vals_C_o_scaled
                    eig_vctrs_smpl = eig_vctrs_C_o_scaled
                    C_for_sampl = C_o_scaled
                else:
                    eig_vals_smpl = eig_vals_C_o
                    sqrt_of_eig_vals_smpl = sqrt_of_eig_vals_C_o
                    eig_vctrs_smpl = eig_vctrs_C_o
                    C_for_sampl = self.C_o
            else:
                eig_vals_smpl = eig_vals_C
                sqrt_of_eig_vals_smpl = sqrt_of_eig_vals_C
                eig_vctrs_smpl = eig_vctrs_C
                C_for_sampl = self.C

            self.max_sampling_cov_per_gen.append(
                np.max(C_for_sampl))  # for plot
            max_C_elem_idx = np.argmax(C_for_sampl)

            self.max_C_elem_idx_per_gen.append(max_C_elem_idx)

            if len(self.max_C_elem_idx_per_gen) > 1 and self.max_C_elem_idx_per_gen[-2] != self.max_C_elem_idx_per_gen[-1]:
                self.max_C_elem_idx_changed_marker.append(t)

            # new m with new evolution path
            if env_changed and self.use_pre_pop:
                pop_for_C_o, fit_for_C_o = self.get_offsprings(
                    self.n, self.m, self.sig, self.lambd, sqrt_of_eig_vals_C_o, eig_vctrs_C_o, t)  # TODO welches sig nehmen (das zurückgesetzte?)
                mu_best_inds_for_C_o = self.get_mue_best_individuals(
                    self.n, self.mu, pop_for_C_o, fit_for_C_o)

                # parameter update
                self.m = self.get_weighted_avg(
                    self.n, self.w, mu_best_inds_for_C_o)

            # ---------------------------------------------------------------------
            glob_opt = self.experiment_data['global_opt_pos_per_chgperiod'][self.detected_n_changes]
            print("glob opt: ", glob_opt)
            #print("C: ", C)
            #print("max_C: ", np.max(C))
            self.max_C_per_gen.append(np.max(self.C))
            self.opt_per_gen.append(glob_opt)

            if env_changed and self.use_pred_for_m:
                self.m = glob_opt + np.random.rand()

            # ---------------------------------------------------------------------
            if env_changed and self.scale_sig:
                self.sig = np.sqrt(np.diagonal(C_for_sampl))
            print("sig: ", self.sig)

            # offsprings
            offspring_population, offspring_fitnesses = self.get_offsprings(
                self.n, self.m, self.sig, self.lambd, sqrt_of_eig_vals_smpl, eig_vctrs_smpl, t)
            mu_best_individuals = self.get_mue_best_individuals(
                self.n, self.mu, offspring_population, offspring_fitnesses)

            # parameter update
            m_new = self.get_weighted_avg(self.n, self.w, mu_best_individuals)
            p_sig_new = self.get_new_p_sig(
                self.n, self.c_sig, self.p_sig, self.mu_w, self.m, m_new, self.sig, inv_squareroot_for_p_sig)
            h_sig = self.get_h_sig(p_sig_new, self.c_sig, t, self.n, self.E)
            p_c_new = self.get_new_p_c(self.n, self.c_c, self.p_c, self.h_sig, self.mu_w,
                                       m_new, self.m, self.sig, self.frst, self.scnd, self.thrd)
            C_mu = self.get_C_mu(self.n, mu_best_individuals,
                                 self.m, self.sig, self.w)
            C_new = self.get_new_C(self.n, self.c_1, self.c_mu, C_for_C, p_c_new, C_mu,
                                   self.max_C_frst, self.max_C_scnd, self.max_C_thrd)
            #print("p_sig_new: ", p_sig_new)
            try:
                sig_new = self.get_new_sig(self.sig, self.c_sig, self.d_sig, p_sig_new,
                                           self.E, self.sig_exp, self.sig_norm, self.sig_inner, self.sig_sub)
            except:
                break
            print("m    : ", self.m)
            print("m_new: ", m_new)

            # ---------------------------------------------------------------------
            # store old variables
            self.angle_per_gen.append(
                self.visualize_dominant_eigvector(self.n, eig_vals_C, eig_vctrs_C))
            try:
                self.sig_per_gen.append(log(self.sig))
            except:
                self.sig_per_gen.append(np.log(np.max(self.sig)))
            self.p_sig_per_gen.append(self.p_sig)
            self.h_sig_per_gen.append(h_sig)
            self.p_c_per_gen.append(self.p_c)
            self.m_per_gen.append(self.m)
            self.p_o_per_gen.append(self.p_o)

            # set variables for next generation
            self.m = m_new
            self.p_sig = p_sig_new
            self.p_c = p_c_new
            self.C = C_new
            self.sig = sig_new

            # ---------------------------------------------------------------------
            # print

            min_fit = self.print_success(
                t, offspring_population, offspring_fitnesses)
            self.fit_per_gen.append(min_fit)

            curr_fit = min_fit  # fitness(m_new, t, chg_freq)

            # detect stagnation
            if t > 5:
                fit_diff_to_5_last = abs(curr_fit - self.fit_per_gen[t - 5])
                fit_diff_to_first = abs(curr_fit - self.fit_per_gen[0])
                enough_improvement = fit_diff_to_5_last / fit_diff_to_first > self.impr_fct
            else:
                enough_improvement = True
            if curr_fit < best_fit and enough_improvement:
                best_fit = curr_fit
                unsucc_count = 0
            else:
                unsucc_count += 1
                if unsucc_count >= 5:
                    self.stagnation_markers.append(t)

            # TODO gibt man das beste jemals gefundene Individuum zurück? oder das
            # aktuell beste?
        #plot_ev_path(p_c_per_gen, p_sig_per_gen)
        return self.fit_per_gen

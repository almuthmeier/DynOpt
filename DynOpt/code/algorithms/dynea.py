'''
Contains the dynamic evolutionary algorithm with prediction model.

Created on Mar 13, 2018

@author: ameier
'''
import copy
import math
import warnings

import numpy as np
from utils import utils_dynopt
from utils.utils_dynopt import environment_changed
from utils.utils_ea import dominant_recombination, gaussian_mutation,\
    mu_plus_lambda_selection, adapt_sigma
from utils.utils_prediction import build_all_predictors
from utils.utils_prediction import prepare_data_train_and_predict
from utils.utils_transferlearning import get_variables_and_names
from utils.utils_values import make_values_feasible_for_square


class DynamicEA():
    def __init__(self, benchmarkfunction, dim,
                 n_generations, experiment_data, predictor_name,
                 trueprednoise, lbound, ubound,
                 ea_np_rnd_generator, pred_np_rnd_generator,
                 mu, la, ro, mean, sigma, trechenberg, tau,
                 reinitialization_mode, sigma_factors,
                 timesteps, n_neurons, epochs, batchsize, n_layers, apply_tl,
                 n_tllayers, tl_model_path, tl_learn_rate, max_n_chperiod_reps,
                 add_noisy_train_data, train_interval, n_required_train_data, use_uncs,
                 train_mc_runs, test_mc_runs, train_dropout, test_dropout,
                 kernel_size, n_kernels, lr):
        '''
        Initialize a DynamicEA object.
        @param benchmarkfunction: (string)
        @param dim: (int) dimensionality of objective function, i.e. number of 
        features for each individual
        @param n_generations: (int) number of generations
        @param experiment_data: (dictionary)
        @param predictor_name: (string)
        @param lbound, ubound: (floats) lower/upper bound of solution space; 
        must fit the data in experiment_data!
        @param ea_np_rnd_generator: numpy random generator for the EA
        @param pred_np_rnd_generator: numpy random generator for the predictor
        @param mu: (int) population size
        @param la: (int) lambda, number of offspring individuals
        @param ro: (int) number parents for recombination
        @param mean: (float) mean for the Gaussian mutation of the EA
        @param sigma: (float) mutation strength for EA 
        @param trechenberg: number of mutations after which sigma is adapted
        @param tau: 0 < tau < 1, factor to adapt sigma (for Rechenberg 1/5 rule)
        @param timesteps: (int) number of time steps the predictions use for the
        prediction
        @param n_neurons: (int) number of neurons within the first layer of the 
        RNN prediction model
        @param epochs: (int) number of epochs to train the RNN predicton model
        @param batch_size: (int) batch size for the RNN predictor
        @param n_layers (int): overall number of RNN layers (incl. pre-trained
        ones), only for "tfrnn", "tftlrnn", or "tftlrnndense"
        @param apply_tl: (bool) True if transfer learning should be applied
        @param n_tllayers (int) number layers in pre-trained RNN 
        only for "tfrnn", "tftlrnn", or "tftlrnndense"
        @param tl_model_path: (string) path to pre-trained RNN
        @param tl_learn_rate (float): learning rate for pre-trained layers
        @param max_n_chperiod_reps: (int): how often each change period should
        be repeated (used for evaluation of variance of EA)
        @param add_noisy_train_data (bool): True if the variance over the 
        repeated change periods should be used as noise to disturbe the time
        series the prediction models learn from
        @param train_interval: (int) number of change periods that must have 
        passed before predictor is trained anew
        @param n_required_train_data: (int) number of training data that is
        used for training
        @param use_uncs: (True) if predictive uncertainty should be estimated; 
        only possible for predictors "kalman", "tcn" and "truepred" 
        @param train_mc_runs: (int) number of Monte Carlo runs during training
        (when predictive uncertainty is estimated)
        @param test_mc_runs: (int) number of Monte Carlo runs during prediciton
        @param train_dropout, test_dropout (float): dropout for training/test
        @param kernel_size (int): filter size for "tcn"
        @param n_kernels: (int) number filters for "tcn"
        @param lr: (float) learning rate for "tcn"
        '''
        # ---------------------------------------------------------------------
        # for the problem
        # ---------------------------------------------------------------------
        self.benchmarkfunction = benchmarkfunction
        self.dim = dim
        self.n_generations = n_generations
        self.experiment_data = experiment_data
        self.predictor_name = predictor_name
        self.trueprednoise = trueprednoise  # TODO unused so far

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
        self.la = la
        self.ro = ro
        self.mean = mean
        self.sigma = sigma
        self.t_rechenberg = trechenberg
        self.tau = tau
        self.reinitialization_mode = reinitialization_mode
        self.sigma_factors = sigma_factors

        # ---------------------------------------------------------------------
        # values that are not passed as parameters to the constructor
        # ---------------------------------------------------------------------
        self.init_sigma = self.sigma

        # ---------------------------------------------------------------------
        # for EA (variable values)
        # ---------------------------------------------------------------------
        # initialize population (mu candidates) and compute fitness.
        self.population = self.ea_np_rnd_generator.uniform(self.lbound,
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
# for (static) EA

    def recombinate(self):
        '''
        @return offspring individual 
        '''
        return dominant_recombination(self.population, self.ro, self.ea_np_rnd_generator)

    def mutate(self, x):
        '''
        @return: mutated individual.
        '''
        return gaussian_mutation(x, self.mean, self.sigma, self.ea_np_rnd_generator)

    def select(self, offsprings, offsprings_fitness):
        '''
        Selects the new individuals for the next generation and updates the
        new population.
        '''
        self.population, self.population_fitness = mu_plus_lambda_selection(
            self.mu, self.population, offsprings, self.population_fitness, offsprings_fitness)

# =============================================================================
# for dynamic EA: for re-initialization strategies

    def compute_noisy_opt_positions(self, z, pred_optimum_position, n_immigrants_per_noise):
        '''
        Generates immigrants around the predected optimum, following the
        current re-initialization strategy.

        @param z: factor for the interval
        @param pred_optimum_position: predicted optimum
        @param n_immigrants_per_noise: (int) number of immigrants per noise level
        '''
        mean = 0.0
        covariance = None  # covariance (matrix) or variance (vector)
        sigma = None  # standard deviation (scalar)
        try:
            n_preds, _ = self.pred_opt_pos_per_chgperiod.shape
        except AttributeError:
            #"AttributeError: 'list' object has no attribute 'shape'"
            # -> only one prediction was made until now
            n_preds = 1

        if self.reinitialization_mode == "pred-RND":
            # -> one sigma for all dimensions
            sigma = 1.0
        elif self.reinitialization_mode == "pred-UNC":
            # predictive variance for re-initialization
            # -> different variance for each dimension
            covariance = self.pred_unc_per_chgperiod[-1]
            assert len(covariance) == self.dim
        elif self.reinitialization_mode == "pred-DEV":
            # -> one variance for all dimensions
            # difference of the last prediction and the last best found position;
            # average over all dimensions

            diff = np.array(
                self.best_found_pos_per_chgperiod[-n_preds:]) - np.array(self.pred_opt_pos_per_chgperiod)
            try:
                avg_squared_diff = np.average(np.square(diff))
                # https://stackoverflow.com/questions/17208567/how-to-find-out-where-a-python-warning-is-from
                warnings.filterwarnings(
                    'error', message='overflow encountered in square')
            except RuntimeWarning:
                print("dynea.compute_noisy_opt_positions: caught warning", flush=True)
                diff = make_values_feasible_for_square(diff)
                # next try for computing square
                avg_squared_diff = np.average(np.square(diff))

            sigma = np.sqrt(avg_squared_diff)  # scalar
        elif self.reinitialization_mode == "pred-KAL":
            assert z == 1.0  # should only have one noise level
            # for explanation of this type see the paper "Tracking moving optima
            # using Kalman-Based predictions"
            c = 0.1  # seemed to be good setting in the paper
            covariance = self.pred_unc_per_chgperiod[-1]
            max_variance = np.max(covariance)
            max_sigma = np.sqrt(max_variance)
            g = c / (1 + max_sigma)
            # number of immigrants produced around the prediction (remaining
            # immigrants are created randomly afterwards)
            n_immigrants_per_noise = math.floor(g * n_immigrants_per_noise)
            assert len(covariance) == self.dim
        else:
            warnings.warn("unknown reinitialization mode: " +
                          self.reinitialization_mode)

        if n_immigrants_per_noise == 0:
            return np.array([])
        noisy_optimum_positions = np.array(
            [gaussian_mutation(pred_optimum_position, mean,
                               sigma, self.pred_np_rnd_generator, z, covariance)
             for _ in range(n_immigrants_per_noise)])
        return noisy_optimum_positions

    def get_parent_population(self):
        '''
        As called in the paper "Prediction-based population re-initialization...."
        Returns for each individual the nearest in the last population.

        For re-initialization types "no_VAR" und "no_PRE".
        '''
        last_pop_copy = copy.copy(self.population_of_last_gen)
        parent_population = []
        for ind in self.population:
            # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
            # (24.1.18)
            squared_diff = np.square(last_pop_copy - ind)
            # sum  within row
            sqrt_diff = np.sqrt(np.sum(squared_diff, axis=1))
            next_neighbor_idx = (sqrt_diff).argmin()
            parent_population.append(self.population[next_neighbor_idx])
            # remove row (the nearest neighbor)
            last_pop_copy = np.delete(last_pop_copy, next_neighbor_idx, axis=0)
        return np.array(parent_population)

    def get_delta(self):
        '''
        As called in the paper "Prediction-based population re-initialization...."

        For the re-initialization types "no_VAR" und "no_PRE".
        '''
        # "parent population" in time as defined in the paper. The first entry
        # is the nearest individual from self.population_of_last_gen
        # to the first entry in self.population, the second one the
        # second next one ...
        parent_population = self.get_parent_population()
        # averages over all dimensions (separately for individuals)
        avg_squared_diffs = 1 / (4 * self.dim) * np.sum(
            np.square(self.population - parent_population), axis=1)
        # one entry for each individual
        sigma_per_ind = np.sqrt(avg_squared_diffs)
        assert sigma_per_ind.shape == (len(self.population),)
        return sigma_per_ind, parent_population

    def adapt_population(self, curr_gen, my_pred_mode):
        '''
        Re-initialize (parts of) the population: insert immigrants that are
        generated by means of different re-initialization strategies.
        Explanations for re-initialization strategies (nRND, nVAR, nPRE, pKAL, 
        pUNC, pDEV, pRND) can be found in the ICANN 2019 paper.
        '''
        mean = 0.0
        n_immigrants = self.mu
        assert n_immigrants == self.mu
        random_immigrants = self.ea_np_rnd_generator.uniform(self.lbound,
                                                             self.ubound, (n_immigrants, self.dim))
        if my_pred_mode == "no" or n_immigrants == 0:
            if self.reinitialization_mode == "no-RND" or self.predictor_name != "no":
                # completely random if no prediction is applied or the predictor
                # is only applied later when enough training data is available
                immigrants = random_immigrants
            elif self.reinitialization_mode == "no-VAR":
                # add to current individuals a noise with standard
                # deviation (x_t - x_t-1)
                sigma_per_ind, _ = self.get_delta()
                immigrants = np.array(
                    [gaussian_mutation(x, mean, float(s), self.pred_np_rnd_generator)
                     for x, s in zip(self.population, sigma_per_ind)])
            elif self.reinitialization_mode == "no-PRE":
                sigma_per_ind, parent_population = self.get_delta()
                predicted_pop = self.population + \
                    (self.population - parent_population)
                immigrants = np.array(
                    [gaussian_mutation(x, mean, float(s), self.pred_np_rnd_generator)
                     for x, s in zip(predicted_pop, sigma_per_ind)])
            else:
                warnings.warn("unknown reinitialization mode: " +
                              self.reinitialization_mode)

        elif my_pred_mode in ["rnn", "autoregressive", "tfrnn", "tftlrnn",
                              "tftlrnndense", "tcn", "kalman", "truepred",
                              "hybrid-autoregressive-rnn"]:
            # last predicted optimum
            pred_optimum_position = self.pred_opt_pos_per_chgperiod[-1]
            # insert predicted optimum into immigrants
            immigrants = np.array([pred_optimum_position])
            n_remaining_immigrants = n_immigrants - len(immigrants)
            # initialize remaining immigrants
            # a) within the neighborhood of the predicted optimum and
            # b) completely randomly
            # ratio of a) and b): 2/3, 1/3
            if n_remaining_immigrants > 1:
                if self.reinitialization_mode == "pred-KAL":
                    self.sigma_factors = [1.0]
                # a)
                # immigrants randomly in the area around the optimum (in case
                # of TCN the area is bound to the predictive variance). There
                # are 4 different realizations of this type.
                two_third = math.ceil((n_remaining_immigrants / 3) * 2)
                n_immigrants_per_noise = two_third // len(self.sigma_factors)

                for z in self.sigma_factors:
                    noisy_optimum_positions = self.compute_noisy_opt_positions(
                        z, pred_optimum_position, n_immigrants_per_noise)
                    if len(noisy_optimum_positions) != 0:
                        immigrants = np.concatenate(
                            (immigrants, noisy_optimum_positions))
                # b) TODO only use respective re-initialization strategy
                # initialize remaining immigrants completely randomly
                n_remaining_immigrants = n_immigrants - len(immigrants)
                immigrants = np.concatenate(
                    (immigrants, random_immigrants[:n_remaining_immigrants]))
            else:
                # take one of the random immigrants
                immigrants = np.concatenate((immigrants, random_immigrants[0]))
        else:
            msg = "unknown prediction mode " + my_pred_mode
            warnings.warn(msg)

        assert len(
            immigrants) == n_immigrants, "false number of immigrants: " + str(len(immigrants))
        # build new population
        self.population = np.concatenate((self.population, immigrants))
        # compute fitness of new population
        self.population_fitness = np.array([utils_dynopt.fitness(self.benchmarkfunction, individual, curr_gen,  self.experiment_data)
                                            for individual in self.population]).reshape(-1, 1)

# =============================================================================
# for dynamic EA

    def reset_parameters(self):
        '''
        Resets sigma after (after a change).
        '''
        self.sigma = self.init_sigma


# =============================================================================

    def optimize(self):
        # ---------------------------------------------------------------------
        # local variables for predictor
        # ---------------------------------------------------------------------
        predictors = build_all_predictors(self.predictor_name, self.n_time_steps,
                                          self.dim, self.batch_size, self.n_neurons,
                                          self.return_seq, self.apply_tl, self.n_layers,
                                          self.n_epochs, self.tl_rnn_type, self.n_tllayers,
                                          self.with_dense_first, self.tl_learn_rate, self.use_uncs,
                                          self.train_mc_runs, self.train_dropout, self.test_dropout,
                                          self.kernel_size, self.n_kernels, self.lr)
        sess = None  # necessary since is assigned anew after each training
        if self.predictor_name in ["rnn", "tfrnn", "tftlrnn", "tftlrnndense", "tcn",
                                   "hybrid-autoregressive-rnn"]:
            import tensorflow as tf
            # if transfer learning then load weights
            if self.apply_tl:
                # instantiate saver to restore pre-trained weights/biases
                tl_variables, _, _, _, _, _ = get_variables_and_names(
                    self.n_tllayers)
                saver = tf.train.Saver(tl_variables)
            # start session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # initialize empty model (otherwise exception)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if self.apply_tl:
                # overwrite initial values with pre-trained weights/biases
                saver.restore(sess, self.tl_model_path)

        # ---------------------------------------------------------------------
        # local variables for EA
        # ---------------------------------------------------------------------
        # number of successful mutations during t_rechenberg generations
        t_s = 0
        # overall number of mutations
        t_all = 0

        # ---------------------------------------------------------------------
        # generations for that the repetitions are executed (is initialized
        # first time: all generations until a change is detected are appended)
        gens_for_rep = []
        # ---------------------------------------------------------------------
        start_pop_for_curr_chgperiod = self.population
        start_pops_fit_for_curr_chgperiod = self.population_fitness
        for i in range(self.n_generations):
            glob_opt = self.experiment_data['global_opt_pos_per_gen'][i]
            #print("generation , ", i, " glob opt: ", glob_opt)
            # store generations that have to be repeated (to compute the
            # variance of the ES)
            gens_for_rep.append(i)
            # test for environment change
            env_changed = environment_changed(i, self.population, self.population_fitness,
                                              self.benchmarkfunction, self.experiment_data, self.ea_np_rnd_generator)

            # test for environment change (but not in first generation)
            if env_changed and i != 0:
                # -------------------------------------------
                # for repetitions of chgperiods:
                # save population for first run of current chgperiod
                self.final_pop_per_run_per_chgperiod[0].append(copy.deepcopy(
                    self.population))
                self.final_pop_fitness_per_run_per_changeperiod[0].append(
                    copy.deepcopy(self.population_fitness))
                self.best_found_pos_per_run_per_chgp[0].append(
                    copy.deepcopy(self.best_found_pos_per_gen[i - 1]))

                # -------------------------------------------
                # only for experiments with repetitions of change periods
                if self.max_n_chgperiod_reps > 1:
                    run_indices = range(1, self.max_n_chgperiod_reps)
                    self.repeat_selected_generations(gens_for_rep, run_indices,
                                                     start_pop_for_curr_chgperiod,
                                                     start_pops_fit_for_curr_chgperiod)
                    gens_for_rep = []

                # in the following the population of the first run is used (for prediction,...)
                # -------------------------------------------
                # reset sigma to initial value
                self.reset_parameters()
                # count change
                self.detected_n_changes += 1
                # count new train data
                self.n_new_train_data += 1
                print("(chg/gen)-(" + str(self.detected_n_changes) +
                      "/" + str(i) + ") ", end="", flush=True)
                # store best found solution during change period as training data for predictor
                # TODO(dev) works only for plus-selection (not for
                # comma-selection)
                self.best_found_pos_per_chgperiod.append(
                    copy.copy(self.best_found_pos_per_gen[i - 1]))
                self.best_found_fit_per_chgperiod.append(
                    copy.copy(self.best_found_fit_per_gen[i - 1]))

                # prepare data and predict optimum
                (my_pred_mode,
                 updated_predictors,
                 self.n_new_train_data) = prepare_data_train_and_predict(sess, i, self.dim, predictors,
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
                predictors = updated_predictors

                # adapt population to environment change
                self.adapt_population(i, my_pred_mode)
                # store population
                start_pop_for_curr_chgperiod = copy.deepcopy(self.population)
                start_pops_fit_for_curr_chgperiod = copy.deepcopy(
                    self.population_fitness)

            self.detected_chgperiods_for_gens.append(self.detected_n_changes)

            # save start population of this generation
            self.population_of_last_gen = copy.copy(self.population)

            # create la offsprings
            offspring_population = np.zeros((self.la, self.dim))
            offspring_pop_fitness = np.zeros((self.la, 1))
            for j in range(self.la):
                # adapt sigma after t_rechenberg mutations
                if t_all % self.t_rechenberg == 0 and t_all != 0:
                    adapt_sigma(
                        self.sigma, self.t_rechenberg, self.tau, t_s)
                    # reset counters
                    t_all = 0
                    t_s = 0
                # recombination
                offspring = self.recombinate()
                # mutation
                mutated_offspring = self.mutate(offspring)
                t_all += 1
                # evaluate offspring
                offspring_fitness = utils_dynopt.fitness(self.benchmarkfunction,
                                                         mutated_offspring, i, self.experiment_data)

                # add offspring to offspring population
                offspring_population[j] = copy.copy(mutated_offspring)
                offspring_pop_fitness[j][0] = offspring_fitness

                # mutation successful?
                if offspring_fitness < utils_dynopt.fitness(self.benchmarkfunction, offspring, i, self.experiment_data):
                    t_s += 1
            # select new population
            self.select(offspring_population, offspring_pop_fitness)
            min_fitness_index = np.argmin(self.population_fitness)
            self.best_found_fit_per_gen[i] = copy.copy(
                self.population_fitness[min_fitness_index])
            self.best_found_pos_per_gen[i] = copy.copy(
                self.population[min_fitness_index])
            # print("best: ", self.population_fitness[min_fitness_index],
            #      "[", self.population[min_fitness_index], "]")
        if self.predictor_name in ["tfrnn", "tftlrnn", "tftlrnndense", "tcn", "rnn",
                                  "hybrid-autoregressive-rnn"]:
            sess.close()
            tf.reset_default_graph()

        # save results for last change period
        self.best_found_pos_per_chgperiod.append(
            copy.copy(self.best_found_pos_per_gen[i - 1]))
        self.best_found_fit_per_chgperiod.append(
            copy.copy(self.best_found_fit_per_gen[i - 1]))

    def repeat_selected_generations(self, generation_idcs, run_idcs, original_pop, original_pops_fit):
        '''
        @param generation_idcs: contains indices of generations that are repeated
        @param run_idcs: contains indices of runs that are repeated (begin with
        1 since one run already is run before)
        @param original_pop: population 
        '''

        # ---------------------------------------------------------------------
        # local variables for EA
        # ---------------------------------------------------------------------
        # number of successful mutations during t_rechenberg generations
        t_s = 0
        # overall number of mutations
        t_all = 0

        # ---------------------------------------------------------------------
        # store old values of class variables
        old_pop = copy.deepcopy(self.population)
        old_pop_fit = copy.deepcopy(self.population_fitness)
        # ---------------------------------------------------------------------
        #print("repeat chgperiod", flush=True)
        for r in run_idcs:
            #print("    repetition: ", r, flush=True)
            # set new values to class variables
            self.population = copy.deepcopy(original_pop)
            self.population_fitness = copy.deepcopy(original_pops_fit)

            for i in generation_idcs:
                # create la offsprings
                offspring_population = np.zeros((self.la, self.dim))
                offspring_pop_fitness = np.zeros((self.la, 1))
                for j in range(self.la):
                    # adapt sigma after t_rechenberg mutations
                    if t_all % self.t_rechenberg == 0 and t_all != 0:
                        adapt_sigma(
                            self.sigma, self.t_rechenberg, self.tau, t_s)
                        # reset counters
                        t_all = 0
                        t_s = 0
                    # recombination
                    offspring = self.recombinate()
                    # mutation
                    mutated_offspring = self.mutate(offspring)
                    t_all += 1
                    # evaluate offspring
                    offspring_fitness = utils_dynopt.fitness(self.benchmarkfunction,
                                                             mutated_offspring, i, self.experiment_data)

                    # add offspring to offspring population
                    offspring_population[j] = copy.copy(mutated_offspring)
                    offspring_pop_fitness[j][0] = offspring_fitness

                    # mutation successful?
                    if offspring_fitness < utils_dynopt.fitness(self.benchmarkfunction, offspring, i, self.experiment_data):
                        t_s += 1
                # select new population
                self.select(offspring_population, offspring_pop_fitness)
            # save final population of this run and its fitness values
            self.final_pop_per_run_per_chgperiod[r].append(copy.deepcopy(
                self.population))
            self.final_pop_fitness_per_run_per_changeperiod[r].append(
                copy.deepcopy(self.population_fitness))
            # determine best found position (with minimum fitness)
            min_fitness_index = np.argmin(self.population_fitness)
            self.best_found_pos_per_run_per_chgp[r].append(
                copy.deepcopy(self.population[min_fitness_index]))

        # ---------------------------------------------------------------------
        # compute standard deviation of best found position among runs
        # ---------------------------------------------------------------------
        # [chgperiods, dims]
        self.stddev_among_runs_per_chgp = np.std(
            self.best_found_pos_per_run_per_chgp, axis=0)
        self.mean_among_runs_per_chgp = np.average(
            self.best_found_pos_per_run_per_chgp, axis=0)

        # ---------------------------------------------------------------------
        # restore old values
        # ---------------------------------------------------------------------
        self.population = old_pop
        self.population_fitness = old_pop_fit

'''
Contains the dynamic evolutionary algorithm with prediction model as 
described in the Evo* 2018 paper.

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
from utils.utils_prediction import build_predictor,\
    predict_next_optimum_position, prepare_scaler
from utils.utils_transferlearning import get_variables_and_names


class DynamicEA():
    def __init__(self, benchmarkfunction, dim,
                 n_generations, experiment_data, predictor_name,
                 ea_np_rnd_generator, pred_np_rnd_generator,
                 mu, la, ro, mean, sigma, trechenberg, tau,
                 timesteps, n_neurons, epochs, batchsize, n_layers, apply_tl,
                 n_tllayers, tl_model_path):
        '''
        Initialize a DynamicEA object.
        @param benchmarkfunction: (string)
        @param dim: (int) dimensionality of objective function, i.e. number of 
        features for each individual
        @param n_generations: (int) number of generations
        @param experiment_data: (dictionary)
        @param predictor_name:
        @param ea_np_rnd_generator: numpy random generator for the EA
        @param pred_np_rnd_generator: numpy random generator for the predictor
        @param mu: (int) population size
        @param la: (int) lambda, number of offspring individuals
        @param ro: (int) number parents for recombination
        @param mean: (float) mean for the Gaussian mutation of the EA
        @param sigma: (float) mutation strength for EA 
        @param trechenberg: number of mutations after which sigma is adapted
        @param tau: 0 < tau < 1, factor to adapt sigma (for Rechenberg 1/5 rule)
        @param time_steps: (int) number of time steps the predictions use for the
        prediction
        @param n_neurons: (int) number of neurons within the first layer of the 
        RNN prediction model
        @param epochs: (int) number of epochs to train the RNN predicton model
        @param batch_size: (int) batch size for the RNN predictor
        '''
        # ---------------------------------------------------------------------
        # for the problem
        # ---------------------------------------------------------------------
        self.benchmarkfunction = benchmarkfunction
        self.dim = dim
        self.n_generations = n_generations
        self.experiment_data = experiment_data
        self.predictor_name = predictor_name

        self.lbound = 0  # assumed that the input data follow this assumption
        self.ubound = 100  # TODO(exe) , TODO insert into dynPSO
        # ---------------------------------------------------------------------
        # for the predictor
        # ---------------------------------------------------------------------
        self.n_time_steps = timesteps
        # number steps used to predict before the number of training data is
        # equal or larger than self.n_time_steps
        # TODO  set at another location? (value depends on n_time_steps)
        self.preliminary_n_steps = 5
        self.n_neurons = n_neurons
        self.n_epochs = epochs
        self.batch_size = batchsize
        self.n_layers = n_layers

        # transfer learning
        self.apply_tl = apply_tl  # True if pre-trained model should be used
        self.tl_model_path = tl_model_path  # path to the trained tl model
        self.tl_rnn_type = "RNN"
        self.n_tllayers = n_tllayers

        # training/testing specifications
        self.use_all_train_data = True  # user all previous data to train with
        self.predict_diffs = True  # predict position differences, TODO insert into PSO
        self.return_seq = True  # return values for all time steps not only the last one
        self.shuffle_train_data = True
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

        # ---------------------------------------------------------------------
        # values that are not passed as parameters to the constructor
        # ---------------------------------------------------------------------
        self.init_sigma = self.sigma

        # ---------------------------------------------------------------------
        # for EA (variable values)
        # ---------------------------------------------------------------------
        # initialize population (mu candidates) and compute fitness.
        # np.random.rand has values in [0, 1). Therefore multiply with 100 for
        # larger values. And make column vector from row vector.
        self.population = self.ea_np_rnd_generator.rand(
            self.mu, self.dim) * 100
        # 2d numpy array (for each individual one row)
        self.population_fitness = np.array([utils_dynopt.fitness(self.benchmarkfunction,
                                                                 individual, 0,
                                                                 self.experiment_data)
                                            for individual in self.population]).reshape(-1, 1)

        # ---------------------------------------------------------------------
        # for EA (prediction and evaluation)
        # ---------------------------------------------------------------------
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
        # position & fitness of predicted optima (one for each change period)
        self.pred_opt_pos_per_chgperiod = []
        self.pred_opt_fit_per_chgperiod = []
        # training error per chgperiod (if prediction was done)
        self.train_error_per_chgperiod = []
        # training error per epoch for each chgperiod (if prediction was done)
        self.train_error_for_epochs_per_chgperiod = []
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
# for dynamic EA

    def reset_parameters(self):
        '''
        Resets sigma after (after a change).
        '''
        self.sigma = self.init_sigma

    def adapt_population(self, curr_gen, my_pred_mode):
        # create new random individual
        n_immigrants = self.mu
        random_immigrants = self.ea_np_rnd_generator.uniform(self.lbound,
                                                             self.ubound, (n_immigrants, self.dim))
        if my_pred_mode == "no" or n_immigrants == 0:
            # randomly
            immigrants = random_immigrants

        elif my_pred_mode == "rnn" or my_pred_mode == "autoregressive" or \
                my_pred_mode == "tfrnn":
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
                # a)
                # immigrants randomly in the area around the optimum:

                two_third = math.ceil((n_remaining_immigrants / 3) * 2)
                # insert new generated noisy neighbors of the predicted
                # optimum (noise has different levels (equal number of
                # immigrants per noise level)
                noise_steps = [1, 10, 100, 1000]
                n_noise_steps = len(noise_steps)
                n_immigrants_per_noise = two_third // n_noise_steps
                mean = 0.0
                for i in noise_steps:
                    if n_immigrants_per_noise > 0:
                        sigma = i * 0.01  # 0.01, 0.1, 1.0, 10.0
                        noisy_optimum_positions = np.array(
                            [gaussian_mutation(pred_optimum_position, mean, sigma, self.pred_np_rnd_generator) for _ in range(n_immigrants_per_noise)])
                        immigrants = np.concatenate(
                            (immigrants, noisy_optimum_positions))

                # b)
                # initialize remaining immigrants completely randomly
                n_remaining_immigrants = n_immigrants - len(immigrants)
                immigrants = np.concatenate(
                    (immigrants, random_immigrants[:n_remaining_immigrants]))
            else:
                # take one of the random immigrants
                immigrants = np.concatenate(
                    (immigrants, random_immigrants[0]))

            assert len(
                immigrants) == n_immigrants, "false number of immigrants: " + str(len(immigrants))
        else:
            msg = "unknown prediction mode " + my_pred_mode
            warnings.warn(msg)

        # build new population
        self.population = np.concatenate((self.population, immigrants))
        # compute fitness of new population
        self.population_fitness = np.array([utils_dynopt.fitness(self.benchmarkfunction, individual, curr_gen,  self.experiment_data)
                                            for individual in self.population]).reshape(-1, 1)

    def prepare_data_train_and_predict(self, sess, gen_idx, trained_first_time, scaler,
                                       n_features, predictor):
        '''
        TODO insert this function into dynpso
        '''

        overall_n_train_data = len(self.best_found_pos_per_chgperiod)
        n_steps_to_use = self.n_time_steps if overall_n_train_data > self.n_time_steps else self.preliminary_n_steps

        # prevent training with too few train data
        if (overall_n_train_data <= n_steps_to_use or self.predictor_name == "no") or\
                (self.predict_diffs and overall_n_train_data <= n_steps_to_use + 1):  # to build differences 1 item more is required
            my_pred_mode = "no"
            train_data = None
            prediction = None

        else:
            my_pred_mode = self.predictor_name

            # transform absolute values to differences
            if self.predict_diffs:
                best_found_vals_per_chgperiod = np.subtract(
                    self.best_found_pos_per_chgperiod[1:], self.best_found_pos_per_chgperiod[:-1])
            else:
                best_found_vals_per_chgperiod = self.best_found_pos_per_chgperiod

            # scale data (the data are re-scaled directly after the
            # prediction in this iteration)
            #scaler = scaler.fit(best_found_vals_per_chgperiod)
            transf_best_found_pos_per_chgperiod = scaler.transform(
                copy.copy(best_found_vals_per_chgperiod))

            # choose training data
            if not trained_first_time or self.use_all_train_data:
                # use all found optimum positions (if it's the first time of
                # training; or if desired)
                trained_first_time = True
                train_data = transf_best_found_pos_per_chgperiod
            else:
                # append the last new train data (one) and in addition
                # n_steps_to_use already evaluated data in order to create a
                # whole time series of n_steps_to_use together with the new
                # data. The oldest data is appended first, then a newer
                # one and so on.
                train_data = []
                for step_idx in range(n_steps_to_use + 1, 0, -1):
                    train_data.append(
                        transf_best_found_pos_per_chgperiod[-step_idx])
                train_data = np.array(train_data)
            # predict next optimum position or difference (and re-scale value)
            prediction, train_error, train_err_per_epoch = predict_next_optimum_position(my_pred_mode, sess, train_data,
                                                                                         self.n_epochs, self.batch_size,
                                                                                         n_steps_to_use, n_features,
                                                                                         scaler, predictor, self.return_seq, self.shuffle_train_data)
            # convert predicted difference into position
            if self.predict_diffs:
                prediction = np.add(
                    self.best_found_pos_per_chgperiod[-1], prediction)

            self.pred_opt_pos_per_chgperiod.append(
                copy.copy(prediction))
            self.pred_opt_fit_per_chgperiod.append(utils_dynopt.fitness(
                self.benchmarkfunction, prediction, gen_idx, self.experiment_data))
            self.train_error_per_chgperiod.append(train_error)
            self.train_error_for_epochs_per_chgperiod.append(
                train_err_per_epoch)
        return my_pred_mode

    def optimize(self):
        #n_overall_layers = 2

        # if apply_tl:
        #    ntllayers = 1  # TODO
        # else:
        #    ntllayers = 0

        # ---------------------------------------------------------------------
        # local variables for predictor
        # ---------------------------------------------------------------------
        predictor = build_predictor(self.predictor_name, self.n_time_steps,
                                    self.dim, self.batch_size, self.n_neurons,
                                    self.return_seq, self.apply_tl, self.n_layers,
                                    self.n_epochs, self.tl_rnn_type, self.n_tllayers)
        sess = None
        if self.predictor_name == "tfrnn":
            import tensorflow as tf
            # if transfer leanring than load weights
            if self.apply_tl:
                # instantiate saver to restore pre-trained weights/biases
                tl_variables, _, _, _, _, _ = get_variables_and_names(
                    self.n_tllayers)
                saver = tf.train.Saver(tl_variables)
            # start session
            sess = tf.Session()
            # initialize empty model (otherwise exception)
            sess.run(tf.global_variables_initializer())
            if self.apply_tl:
                # overwrite initial values with pre-trained weights/biases
                saver.restore(sess, self.tl_model_path)

        # denotes whether the predictor has been trained or not
        trained_first_time = False
        scaler = prepare_scaler(self.lbound, self.ubound, self.dim)
        # ---------------------------------------------------------------------
        # local variables for EA
        # ---------------------------------------------------------------------
        # number of successful mutations during t_rechenberg generations
        t_s = 0
        # overall number of mutations
        t_all = 0

        # ---------------------------------------------------------------------
        for i in range(self.n_generations):
            # test for environment change
            env_changed = environment_changed(i, self.population, self.population_fitness,
                                              self.benchmarkfunction, self.experiment_data, self.ea_np_rnd_generator)
            # test for environment change (but not in first generation)
            if env_changed and i != 0:
                # reset sigma to initial value
                self.reset_parameters()
                # count change
                self.detected_n_changes += 1
                print("detected change ", self.detected_n_changes, flush=True)
                # store best found solution during change period as training data for predictor
                # TODO(dev) works only for plus-selection (not for
                # comma-selection)
                self.best_found_pos_per_chgperiod.append(
                    copy.copy(self.best_found_pos_per_gen[i - 1]))
                self.best_found_fit_per_chgperiod.append(
                    copy.copy(self.best_found_fit_per_gen[i - 1]))

                # prepare data and predict optimum
                my_pred_mode = self.prepare_data_train_and_predict(sess, i, trained_first_time, scaler,
                                                                   self.dim, predictor)

                # adapt population to environment change
                self.adapt_population(i, my_pred_mode)

            self.detected_chgperiods_for_gens.append(self.detected_n_changes)

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
        if self.predictor_name == "tfrnn":
            sess.close()
            tf.reset_default_graph()

'''
Contains the dynamic evolutionary algorithm with prediction model as 
described in the Evo* 2018 paper.

Created on Mar 13, 2018

@author: ameier
'''
import copy
import math
import warnings

from sklearn.preprocessing.data import MinMaxScaler

import numpy as np
from utils import utils_dynopt
from utils.utils_dynopt import environment_changed,\
    get_global_optimum_pos_and_fit_for_all_generations,\
    compute_real_gens_of_chgs
from utils.utils_ea import dominant_recombination, gaussian_mutation,\
    mu_plus_lambda_selection, adapt_sigma
from utils.utils_prediction import build_predictor,\
    predict_next_optimum_position


#from dynamicopt.metrics.metrics_dynea import arr, best_error_before_change
#from dynamicopt.utils import utils_dynopt
# from dynamicopt.utils.utils_dynopt import environment_changed,\
#    get_global_optimum_pos_and_fit_for_all_generations, compute_real_gens_of_chgs
# from dynamicopt.utils.utils_ea import dominant_recombination, gaussian_mutation,\
#    mu_plus_lambda_selection, adapt_sigma
# from dynamicopt.utils.utils_prediction import build_predictor,\
#    predict_next_optimum_position
class DynamicEA():
    def __init__(self, benchmarkfunction, dim,
                 n_generations, experiment_data, predictor_name,
                 ea_np_rnd_generator, pred_np_rnd_generator,
                 mu, la, ro, mean, sigma, trechenberg, tau,
                 timesteps, n_neurons, epochs, batchsize):
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

        # ---------------------------------------------------------------------
        # for the predictor
        # ---------------------------------------------------------------------
        self.n_time_steps = timesteps
        self.n_neurons = n_neurons
        self.n_epochs = epochs
        self.batch_size = batchsize

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
        self.gens_of_detected_chngs = {self.detected_n_changes: []}
        # best found individual for each generation (2d numpy array)
        self.best_individuals = np.zeros((self.n_generations, self.dim))
        # fitness of best found individual for each generation (1d numpy array)
        self.best_fitness_evals = np.zeros(self.n_generations)
        # position of found optima (one for each change period)
        self.prev_optima_pos = []
        # fitness of found optima (one for each change period)
        self.prev_optima_fit = []
        # position & fitness of predicted optima (one for each change period)
        self.pred_optima_pos = []
        self.pred_optima_fit = []

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
        init_range = 200  # TODO
        random_immigrants = self.ea_np_rnd_generator.uniform(-init_range,
                                                             init_range, (n_immigrants, self.dim))
        if my_pred_mode == "no" or n_immigrants == 0:
            # randomly
            immigrants = random_immigrants

        elif my_pred_mode == "rnn" or my_pred_mode == "autoregressive":
            # last predicted optimum
            pred_optimum_position = self.pred_optima_pos[-1]
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

    def dynea(self):
        # ---------------------------------------------------------------------
        # local variables for predictor
        # ---------------------------------------------------------------------
        train_data = []
        n_features = self.dim
        predictor = build_predictor(
            self.predictor_name, self.n_time_steps, n_features, self.batch_size, self.n_neurons)
        # denotes whether the predictor has been trained or not
        trained_first_time = False
        scaler = MinMaxScaler(feature_range=(-1, 1))

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

                # store best found solution during change period as training data for predictor
                # TODO works only for plus-selection (not for comma-selection)
                self.prev_optima_pos.append(
                    copy.copy(self.best_individuals[i - 1]))
                self.prev_optima_fit.append(
                    copy.copy(self.best_fitness_evals[i - 1]))
                overall_n_train_data = len(self.prev_optima_pos)

                # prevent training with too few train data
                if overall_n_train_data <= self.n_time_steps or overall_n_train_data < 50 or self.predictor_name == "no":
                    my_pred_mode = "no"
                    train_data = None
                    prediction = None
                else:
                    my_pred_mode = self.predictor_name

                    # scale data (the data are re-scaled directly after the
                    # prediction in this iteration)
                    scaler = scaler.fit(self.prev_optima_pos)
                    transf_prev_optima_positions = scaler.transform(
                        copy.copy(self.prev_optima_pos))

                    # choose training data
                    if not trained_first_time:
                        # first time, has not been trained before, therefore use all
                        # found optimum positions
                        trained_first_time = True
                        train_data = transf_prev_optima_positions
                    else:
                        # append the last new train data (one) and in addition
                        # n_time_steps already evaluated data in order to create a
                        # whole time series of n_time_steps together with the new
                        # data. The oldest data is appended first, then a newer
                        # one and so on.
                        train_data = []
                        for step_idx in range(self.n_time_steps + 1, 0, -1):
                            train_data.append(
                                transf_prev_optima_positions[-step_idx])
                        train_data = np.array(train_data)
                    # predict next optimum position
                    prediction = predict_next_optimum_position(my_pred_mode, train_data,
                                                               self.n_epochs, self.batch_size,
                                                               self.n_time_steps, n_features,
                                                               scaler, predictor)
                    self.pred_optima_pos.append(copy.copy(prediction))
                    self.pred_optima_fit.append(utils_dynopt.fitness(
                        self.benchmarkfunction, prediction, i, self.experiment_data))

                # adapt population to environment change
                self.adapt_population(i, my_pred_mode)
            try:
                self.gens_of_detected_chngs[self.detected_n_changes].append(i)
            except KeyError:
                # occurs first time after a change occurred
                self.gens_of_detected_chngs[self.detected_n_changes] = []
                self.gens_of_detected_chngs[self.detected_n_changes].append(i)

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
            self.best_fitness_evals[i] = copy.copy(
                self.population_fitness[min_fitness_index])
            self.best_individuals[i] = copy.copy(
                self.population[min_fitness_index])

# =============================================================================
    def save_results(self, arrays_file_name, periods_for_generations, act_n_chngs,
                     global_opt_pos_of_changes,
                     global_opt_fit_of_changes):
        np.savez(arrays_file_name,
                 # data from the EA
                 best_fitness_evals=self.best_fitness_evals,
                 best_individuals=self.best_individuals,
                 prev_optima_fit=self.prev_optima_fit,
                 prev_optima_pos=self.prev_optima_pos,
                 pred_optima_fit=self.pred_optima_fit,
                 pred_optima_pos=self.pred_optima_pos,
                 detected_n_changes=self.detected_n_changes,
                 gens_of_detected_chngs=self.gens_of_detected_chngs,
                 # data about the problem
                 periods_for_generations=periods_for_generations,
                 act_n_chngs=act_n_chngs,
                 global_opt_pos_of_changes=global_opt_pos_of_changes,
                 global_opt_fit_of_changes=global_opt_fit_of_changes)

# =============================================================================
# settings for the EA


def define_settings_and_run(repetition, gpu_ID,
                            predictor_name, problem, experiment_name, dim, generations, len_chg_period, pos_chng_type,
                            fit_chng_type, problem_data, n_peaks, arrays_file_path, day, time, noise,
                            ea_np_rnd_generator, pred_np_rnd_generator, periods_for_generations, act_n_chngs,
                            c1, c2, c3, insert_pred_as_ind, adaptive_c3,
                            n_neurons, n_epochs, batch_size, n_time_steps):
    '''
    repetition has to be the first argument, because predictor_comparison.run_runs_parallel() assumes that.
    '''
    print("Started repetition with ID ", repetition, flush=True)
    #==========================================================================
    # make tensorflow deterministic
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # prevent using whole GPU
    session = tf.Session(config=config)
    tf.set_random_seed(1234)
    from keras import backend as K

    #==========================================================================
    # must be an even number so that half of mu is an integer number
    # (necessary for number of random_immigrants
    # TODO read the following from input
    mu = 5  # TODO hat to be set
    la = 10  # TODO hat to be set
    ro = 2
    mean = 0.0
    sigma = 1.0
    # adapt sigma after 5 mutations
    t_rechenberg = 5
    tau = 0.5  # 0 < tau < 1, for Rechenberg

    # initialize EA object
    ea = DynamicEA(problem, dim, generations, problem_data, predictor_name,
                   mu, la, ro, mean, sigma, t_rechenberg, tau,
                   ea_np_rnd_generator, pred_np_rnd_generator,
                   n_neurons, n_epochs, batch_size, n_time_steps)

    # TODO sind opt. pos/fit per GENERATION (not per change) !!!!!
    # ist daher auch in den gespeicherten array-files verkehrt!!
    global_opt_pos_of_changes, global_opt_fit_of_changes = get_global_optimum_pos_and_fit_for_all_generations(
        problem, problem_data)
    real_gens_of_chgs = compute_real_gens_of_chgs(
        periods_for_generations, act_n_chngs)

    # ==========================================================================
    # run EA on specified GPU
    with tf.device('/gpu:' + str(gpu_ID)):
        ea.dynea()
        K.clear_session()
    # ==========================================================================
    # compute metrics
    arr_value = None
    bebc = None
    # arr_value = arr(real_gens_of_chgs,
    #                global_opt_fit_of_changes, ea.best_fitness_evals)
    # bebc = best_error_before_change(
    #    real_gens_of_chgs, global_opt_fit_of_changes,  ea.best_fitness_evals)

    # ==========================================================================
    # save results
    arrays_file_name = arrays_file_path + predictor_name + "_" + problem + "_" + experiment_name + "_" + \
        str(dim) + "_" + pos_chng_type + "_" + \
        fit_chng_type + "_lenchgperiod-" + str(len_chg_period) + "_noise-" + str(noise) + "_" + day + '_' + \
        time + "_" + str(repetition) + ".npz"
    ea.save_results(arrays_file_name,
                    periods_for_generations, act_n_chngs,
                    global_opt_pos_of_changes,
                    global_opt_fit_of_changes)

    return ea.best_fitness_evals, arr_value, bebc

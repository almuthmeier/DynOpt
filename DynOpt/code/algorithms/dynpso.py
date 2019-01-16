'''
Contains the dynamic PSO variants with prediction models as proposed in the
GECCO 2018 paper.

Created on Dec 13, 2017

@author: ameier
'''

import copy

from code.utils.my_scaler import MyMinMaxScaler
import numpy as np
from utils import utils_dynopt
from utils.utils_dynopt import environment_changed, replace_worst_individuals
from utils.utils_prediction import build_predictor,\
    predict_next_optimum_position


class DynamicPSO():

    def __init__(self, benchmarkfunction, dim,
                 n_generations, experiment_data, predictor_name,
                 pso_np_rnd_generator, pred_np_rnd_generator,
                 c1, c2, c3, insert_pred_as_ind,
                 adaptive_c3, n_particles,
                 timesteps, n_neurons, epochs, batchsize, n_layers, apply_tl,
                 n_tllayers, tl_model_path, tl_learn_rate, max_n_chperiod_reps):
        '''
        Initialize a DynamicPSO object.
        @param benchmarkfunction: (string)
        @param dim: (int) dimensionality of objective function, i.e. number of 
        features for each individual
        @param n_generations: (int) number of generations
        @param experiment_data: (dictionary)
        @param predictor_name: (string)
        @param pso_np_rnd_generator: numpy random generator for the PSO    
        @param pred_np_rnd_generator: numpy random generator for the predictor
        @param c1: (float) influence of particle's best solution
        @param c2: (float) influence of swarm's best solution
        @param c3: (float) influence of prediction
        @param insert_pred_as_ind: (bool) True if the predicted optimum should
        be inserted into the swarm
        @param adaptive_c3: (bool) True if c3 is set with adaptive control
        @param n_particles: (int) swarm size
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
        self.n_layers = n_layers
        self.apply_tl = apply_tl
        self.tl_model_path = tl_model_path
        self.n_tllayers = n_tllayers

        # ---------------------------------------------------------------------
        # for PSO (fixed values)
        # ---------------------------------------------------------------------
        self.pso_np_rnd_generator = pso_np_rnd_generator
        self.pred_np_rnd_generator = pred_np_rnd_generator
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.insert_pred_as_ind = insert_pred_as_ind
        self.adaptive_c3 = adaptive_c3
        self.n_particles = n_particles

        # ---------------------------------------------------------------------
        # values that are not passed as parameters to the constructor
        # ---------------------------------------------------------------------
        self.init_c1 = self.c1
        self.init_c2 = self.c2
        self.init_c3 = self.c3

        # TODO(exe) set following parameters as desired
        self.init_inertia = 0.7298
        self.inertia = self.init_inertia
        self.vmax = 1000.0  # must not be too small
        self.inertia_adapt_factor = 0.5  # like tau for Rechenberg
        self.inertia_min = 0.1
        self.inertia_max = 0.78540

        # ---------------------------------------------------------------------
        # for PSO (variable values)
        # ---------------------------------------------------------------------

        # initialize population and compute fitness.
        # np.random.rand has values in [0, 1). Therefore multiply with 100 for
        # larger values (one row for each particle)
        self.particles = self.pso_np_rnd_generator.rand(
            self.n_particles, self.dim) * 100  # TODO(exe)
        self.v_vals = self.pso_np_rnd_generator.rand(
            self.n_particles, self.dim)
        # 1d numpy array
        self.fitness_vals = np.array(
            [utils_dynopt.fitness(self.benchmarkfunction, p, 0,  self.experiment_data)
             for p in self.particles])
        # best history value (position) for each particle
        # (2d numpy array: 1 row for each particle)
        self.p_best_pos = copy.copy(self.particles)
        self.p_best_fit = copy.copy(self.fitness_vals)
        # best history value (position) of whole swarm (1d numpy array: 1 row)
        self.s_best_pos = copy.copy(
            self.particles[np.argmin(self.fitness_vals)])
        self.s_best_fit = np.min(self.fitness_vals)

        # ---------------------------------------------------------------------
        # for PSO (prediction and evaluation)
        # ---------------------------------------------------------------------

        # number of change period detected by the PSO
        self.detected_n_changes = 0
        # 1d numpy array containing for each generation the number of the
        # detected change period it belongs to
        self.detected_chgperiods_for_gens = []
        # storage for best fitness evaluation of each generation
        self.best_found_pos_per_gen = np.zeros((self.n_generations, self.dim))
        self.best_found_fit_per_gen = np.zeros(self.n_generations)
        # position & fitness of found optima (one for each change period)
        self.best_found_pos_per_chgperiod = []
        self.best_found_fit_per_chgperiod = []
        # position & fitness of predicted optima (one for each change period)
        self.pred_opt_pos_per_chgperiod = []
        self.pred_opt_fit_per_chgperiod = []

    def reset_factors(self):
        '''
        Resets c1, c2, c3 and inertia to their initial values.
        '''
        self.c1 = self.init_c1
        self.c2 = self.init_c2
        self.c3 = self.init_c3
        self.inertia = self.init_inertia

    def adapt_inertia(self, n_succ_particles):
        '''
        Apply Rechenberg 1/5 success rule to inertia weight.

        If for more than 1/5th of the particles the particle update was 
        successful, i.e., the new particle is better than p_best, increase 
        inertia weight.  

        (Stefan Oehmcke's idea, 19.12.17)

        @param n_succ_particles: number of particles for which the particle 
        update resulted in a particle that is better than p_best
        '''
        succes_rate = n_succ_particles / self.n_particles
        if succes_rate > 1 / 5:
            self.inertia /= self.inertia_adapt_factor
        elif succes_rate < 1 / 5:
            self.inertia *= self.inertia_adapt_factor
        else:
            # do not change inertia
            pass
        self.inertia = max(self.inertia_min, self.inertia)
        self.inertia = min(self.inertia, self.inertia_max)

    def adapt_c3(self):
        if len(self.pred_opt_pos_per_chgperiod) > 0 and not self.pred_opt_pos_per_chgperiod[-1] is None:
            if self.pred_opt_fit_per_chgperiod[-1] < 1.05 * self.s_best_fit:
                # prediction has good fitness -> increase its influence
                self.c3 *= 2
                #print("besser", flush=True)
            elif self.pred_opt_fit_per_chgperiod[-1] > 1.05 * self.s_best_fit:
                # prediction has bad fitness -> decrease its influence
                self.c3 /= 2
            else:
                pass

    def optimize(self):
        '''
        @param mu: population size
        @param la: lambda (offspring population size)
        @param dim: dimensionality of objective function, i.e. number of features for each individual
        @param ro: number parents for recombination
        @param sel_strategie: selection strategy: 'plus' or 'comma'
        @param t_rechenberg: number of mutations after which sigma is adapted
        @param tau: # 0 < tau < 1, for Rechenberg
        @return: best fitness occured in any iteration
        '''
        # ---------------------------------------------------------------------
        # local variables for predictor
        # ---------------------------------------------------------------------
        train_data = []
        n_features = self.dim
        predictor = build_predictor(
            self.predictor_name, self.n_time_steps, n_features, self.batch_size, self.n_neurons)
        # denotes whether the predictor has been trained or not
        trained_first_time = False
        scaler = MyMinMaxScaler(feature_range=(-1, 1))

        # ---------------------------------------------------------------------
        # local variables for PSO
        # ---------------------------------------------------------------------
        # number of particles that are better than p_best after particle update
        n_succ_particles = 0

        # ---------------------------------------------------------------------
        for i in range(self.n_generations):
            print("generation: ", i)
            # adapt inertia weight
            self.adapt_inertia(n_succ_particles)
            # reset n_succ_particles after inertia adaptation
            n_succ_particles = 0

            # test for environment change
            env_changed = environment_changed(i, self.particles, self.fitness_vals,
                                              self.benchmarkfunction, self.experiment_data, self.pso_np_rnd_generator)

            # iteration number in which the last change was detected
            last_it_with_chg = 0
            if env_changed and i != 0:
                print("changed")
                # set c1, c2, c3 and inertia to their initial values
                self.reset_factors()

                # count change
                self.detected_n_changes += 1
                last_it_with_chg = i

                # store best found solution during change period as training data for predictor
                # TODO(dev) works only for plus-selection (not for
                # comma-selection)
                self.best_found_pos_per_chgperiod.append(
                    copy.copy(self.best_found_pos_per_gen[i - 1]))
                self.best_found_fit_per_chgperiod.append(
                    copy.copy(self.best_found_fit_per_gen[i - 1]))
                overall_n_train_data = len(self.best_found_pos_per_chgperiod)

                # prevent training with too few train data
                if overall_n_train_data <= self.n_time_steps or overall_n_train_data < 50 or self.predictor_name == "no":
                    my_pred_mode = "no"
                    # train_data is empty list
                    train_data = None
                    prediction = None
                else:
                    my_pred_mode = self.predictor_name

                    # scale data
                    scaler = scaler.fit(self.best_found_pos_per_chgperiod)
                    transf_best_found_pos_per_chgperiod = scaler.transform(
                        copy.copy(self.best_found_pos_per_chgperiod))

                    # choose training data
                    if not trained_first_time:
                        # first time, has not been trained before, therefore use all
                        # found optimum positions
                        trained_first_time = True
                        train_data = transf_best_found_pos_per_chgperiod
                    else:
                        # append the last new train data (one) and in addition
                        # n_time_steps already evaluated data in order to create a
                        # whole time series of n_time_steps together with the new
                        # data
                        train_data = []
                        for step_idx in range(self.n_time_steps + 1, 0, -1):
                            train_data.append(
                                transf_best_found_pos_per_chgperiod[-step_idx])
                        train_data = np.array(train_data)
                    # predict next optimum position
                    prediction = predict_next_optimum_position(my_pred_mode, train_data,
                                                               self.n_epochs, self.batch_size,
                                                               self.n_time_steps, n_features,
                                                               scaler, predictor)
                    self.pred_opt_pos_per_chgperiod.append(
                        copy.copy(prediction))
                    self.pred_opt_fit_per_chgperiod.append(utils_dynopt.fitness(
                        self.benchmarkfunction, prediction, i, self.experiment_data))

                # compute fitness again since fitness function changed
                self.fitness_vals = np.array(
                    [utils_dynopt.fitness(self.benchmarkfunction, p, i, self.experiment_data)
                     for p in self.particles])
                # reset p_best to current position
                self.p_best_pos = copy.copy(self.particles)
                self.p_best_fit = copy.copy(self.fitness_vals)
                # update s_best
                min_p_best_ind = np.argmin(self.p_best_fit)
                self.s_best_pos = copy.copy(self.p_best_pos[min_p_best_ind])
                self.s_best_fit = self.p_best_fit[min_p_best_ind]

                # adapt population (replace worst by random individuals)
                if self.insert_pred_as_ind:
                    pred_to_insert = copy.copy(prediction)
                else:
                    pred_to_insert = None
                self.particles, self.fitness_vals = replace_worst_individuals(self.pso_np_rnd_generator, self.benchmarkfunction,
                                                                              i, self.particles, self.fitness_vals,
                                                                              self.n_particles,  n_features,
                                                                              self.experiment_data, pred_to_insert)

                if prediction is None:
                    self.p_pred_diff_vals = np.zeros(
                        (self.n_particles, self.dim))
                else:
                    self.p_pred_diff_vals = prediction - self.particles
            else:
                self.p_pred_diff_vals = np.zeros((self.n_particles, self.dim))

            # store which change period the current generation belongs to
            self.detected_chgperiods_for_gens.append(self.detected_n_changes)
            # random values
            # TODO new values in every iteration?
            r1_vals = self.pso_np_rnd_generator.rand(
                self.n_particles, self.dim)
            r2_vals = self.pso_np_rnd_generator.rand(
                self.n_particles, self.dim)
            r3_vals = self.pso_np_rnd_generator.rand(
                self.n_particles, self.dim)

            # update velocity
            self.s_best_diff_vals = self.s_best_pos - self.particles
            self.p_best_diff_vals = self.p_best_pos - self.particles

            # decrease importance of prediction with increasing #iterations
            try:
                c3 = self.c3 / (i - last_it_with_chg)
            except ZeroDivisionError:  # if i == last_it_with_chg
                c3 = self.c3
            c3 = self.c3

            tmp1 = self.inertia * self.v_vals
            tmp2 = self.c1 * r1_vals * self.p_best_diff_vals
            tmp3 = self.c2 * r2_vals * self.s_best_diff_vals
            tmp4 = c3 * r3_vals * self.p_pred_diff_vals
            self.v_vals = tmp1 + tmp2 + tmp3 + tmp4

            # velocity should be <= vmax
            too_large = np.abs(self.v_vals) > self.vmax
            self.v_vals[too_large] = (
                np.sign(self.v_vals) * self.vmax)[too_large]

            # update particles
            self.particles = self.particles + self.v_vals

            # compute fitness
            self.fitness_vals = np.array(
                [utils_dynopt.fitness(self.benchmarkfunction, p, i, self.experiment_data) for p in self.particles])

            # update p_best
            particle_better = self.fitness_vals < self.p_best_fit
            n_succ_particles = np.sum(particle_better)
            self.p_best_pos[particle_better] = copy.copy(
                self.particles[particle_better])
            self.p_best_fit[particle_better] = self.fitness_vals[particle_better]
            # update s_best
            self.s_best_pos = copy.copy(
                self.p_best_pos[np.argmin(self.p_best_fit)])
            self.s_best_fit = np.min(self.p_best_fit)

            # determine best particles/fitness (for evaluation/statistic)
            self.best_found_fit_per_gen[i] = copy.copy(self.s_best_fit)
            self.best_found_pos_per_gen[i] = copy.copy(self.s_best_pos)

            # adapt c3
            if self.adaptive_c3:
                self.adapt_c3()
        # store things last time
        self.best_found_pos_per_chgperiod.append(
            copy.copy(self.best_found_pos_per_gen[self.n_generations - 1]))
        self.best_found_fit_per_chgperiod.append(
            copy.copy(self.best_found_fit_per_gen[self.n_generations - 1]))

        # conversion
        self.best_found_pos_per_chgperiod = np.array(
            self.best_found_pos_per_chgperiod)
        self.pred_opt_pos_per_chgperiod = np.array(
            self.pred_opt_pos_per_chgperiod)

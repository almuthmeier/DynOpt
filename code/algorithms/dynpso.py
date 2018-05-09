'''
Contains the dynamic PSO variants with prediction models as proposed in the
GECCO 2018 paper.

Created on Dec 13, 2017

@author: ameier
'''

import copy
import os
import sys

from sklearn.preprocessing.data import MinMaxScaler

import numpy as np
from utils import utils_dynopt
from utils.utils_dynopt import environment_changed, replace_worst_individuals,\
    get_global_optimum_pos_and_fit_for_all_generations,\
    compute_real_gens_of_chgs
from utils.utils_plot import plot_fitness, plot_best_ind_per_gen, plot_points,\
    plot_diff_pred_and_optimum
from utils.utils_prediction import build_predictor,\
    predict_next_optimum_position


#from dynamicopt.metrics.metrics_dynea import arr, best_error_before_change
#from dynamicopt.utils import utils_dynopt
#from dynamicopt.utils.utils_dynopt import replace_worst_individuals, environment_changed, compute_real_gens_of_chgs, get_global_optimum_pos_and_fit_for_all_generations
# from dynamicopt.utils.utils_plot import plot_diff_pred_and_optimum, plot_fitness,\
#    plot_best_ind_per_gen, plot_points
# from dynamicopt.utils.utils_prediction import build_predictor,\
#    predict_next_optimum_position
sys.path.append(os.path.abspath(os.pardir))


class DynamicPSO():

    def __init__(self, problem, dim, iterations, problem_data, predictor_name,
                 n_particles, pso_np_rnd_generator, pred_np_rnd_generator,
                 c1, c2, c3, insert_pred_as_ind, adaptive_c3,
                 n_neurons, n_epochs, batch_size, n_time_steps):
        '''
        Initialize a DynamicPSO object.
        @param problem:
        @param dim: (int) dimensionality of objective function, i.e. number of 
        features for each individual
        @param iterations: (int) number of generations
        @param problem_data:
        @param predictor_name:
        @param n_particles: (int) swarm size
        @param pso_np_rnd_generator: numpy random generator for the PSO
        @param pred_np_rnd_generator: numpy random generator for the predictor
        @param c1: (float) influence of particle's best solution
        @param c2: (float) influence of swarm's best solution
        @param c3: (float) influence of prediction
        @param insert_pred_as_ind: (bool) True if the predicted optimum should
        be inserted into the swarm
        @param adaptive_c3: (bool) True if c3 is set with adaptive control
        @param n_neurons: (int) number of neurons within the first layer of the 
        RNN prediction model
        @param n_epochs: (int) number of epochs to train the RNN predicton model
        @param batch_size: (int) batch size for the RNN predictor
        @param n_time_steps: (int) number of time steps the predictions use for the
        prediction
        '''
        # ---------------------------------------------------------------------
        # for the problem
        # ---------------------------------------------------------------------
        self.problem = problem
        self.dim = dim
        self.iterations = iterations
        self.problem_data = problem_data
        self.pred_mode = predictor_name

        # ---------------------------------------------------------------------
        # for the predictor
        # ---------------------------------------------------------------------
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_time_steps = n_time_steps

        # ---------------------------------------------------------------------
        # for PSO (fixed values)
        # ---------------------------------------------------------------------
        self.n_particles = n_particles
        self.pso_np_rnd_generator = pso_np_rnd_generator
        self.pred_np_rnd_generator = pred_np_rnd_generator

        self.init_c1 = c1
        self.init_c2 = c2
        self.init_c3 = c3
        self.c1 = self.init_c1
        self.c2 = self.init_c2
        self.c3 = self.init_c3
        self.init_inertia = 0.7298
        self.inertia = self.init_inertia
        self.vmax = 1000.0  # TODO must not be too small
        self.inertia_adapt_factor = 0.5  # like tau for Rechenberg
        self.inertia_min = 0.1
        self.inertia_max = 0.78540

        self.insert_pred_as_ind = insert_pred_as_ind
        self.adaptive_c3 = adaptive_c3

        # ---------------------------------------------------------------------
        # for PSO (variable values)
        # ---------------------------------------------------------------------

        # initialize population and compute fitness.
        # np.random.rand has values in [0, 1). Therefore multiply with 100 for
        # larger values (one row for each particle)
        self.particles = self.pso_np_rnd_generator.rand(
            self.n_particles, self.dim) * 100  # TODO
        self.v_vals = self.pso_np_rnd_generator.rand(
            self.n_particles, self.dim)
        # 1d numpy array
        self.fitness_vals = np.array(
            [utils_dynopt.fitness(self.problem, p, 0,  self.problem_data)
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

        # number of changed detected by the PSO
        self.detected_n_changes = 0
        # for each detected change the corresponding generation numbers
        self.gens_of_detected_chngs = {self.detected_n_changes: []}
        # storage for best fitness evaluation of each generation
        self.best_particles = np.zeros((self.iterations, self.dim))
        self.best_fitness_evals = np.zeros(self.iterations)
        # position & fitness of found optima (one for each change period)
        self.prev_optima_pos = []
        self.prev_optima_fit = []
        # position & fitness of predicted optima (one for each change period)
        self.pred_optima_pos = []
        self.pred_optima_fit = []

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
        if len(self.pred_optima_pos) > 0 and not self.pred_optima_pos[-1] is None:
            if self.pred_optima_fit[-1] < 1.05 * self.s_best_fit:
                # prediction has good fitness -> increase its influence
                self.c3 *= 2
                #print("besser", flush=True)
            elif self.pred_optima_fit[-1] > 1.05 * self.s_best_fit:
                # prediction has bad fitness -> decrease its influence
                self.c3 /= 2
            else:
                pass

    def dynpso(self):
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
        # TODO auslagern: dem Optimierer einfach den fertigen Predictor geben?
        # es gibt nur bei RNN ein richtiges Object
        predictor = build_predictor(
            self.pred_mode, self.n_time_steps, n_features, self.batch_size, self.n_neurons)
        # denotes whether the predictor has been trained or not
        trained_first_time = False
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # ---------------------------------------------------------------------
        # local variables for PSO
        # ---------------------------------------------------------------------
        # number of particles that are better than p_best after particle update
        n_succ_particles = 0

        # ---------------------------------------------------------------------
        for i in range(self.iterations):
            # adapt inertia weight
            self.adapt_inertia(n_succ_particles)
            # reset n_succ_particles after inertia adaptation
            n_succ_particles = 0

            # test for environment change
            env_changed = environment_changed(i, self.particles, self.fitness_vals,
                                              self.problem, self.problem_data, self.pso_np_rnd_generator)

            # iteration number in which the last change was detected
            last_it_with_chg = 0
            if env_changed and i != 0:
                # set c1, c2, c3 and inertia to their initial values
                self.reset_factors()

                # count change
                self.detected_n_changes += 1
                last_it_with_chg = i

                # store best found solution during change period as training data for predictor
                # TODO works only for plus-selection (not for comma-selection)
                self.prev_optima_pos.append(
                    copy.copy(self.best_particles[i - 1]))
                self.prev_optima_fit.append(
                    copy.copy(self.best_fitness_evals[i - 1]))
                overall_n_train_data = len(self.prev_optima_pos)

                # prevent training with too few train data
                if overall_n_train_data <= self.n_time_steps or overall_n_train_data < 50 or self.pred_mode == "no":
                    my_pred_mode = "no"
                    # train_data is empty list
                    train_data = None
                    prediction = None
                else:
                    my_pred_mode = self.pred_mode

                    # scale data
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
                        # data
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
                        self.problem, prediction, i, self.problem_data))

                # compute fitness again since fitness function changed
                self.fitness_vals = np.array(
                    [utils_dynopt.fitness(self.problem, p, i, self.problem_data)
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
                self.particles, self.fitness_vals = replace_worst_individuals(self.pso_np_rnd_generator, self.problem,
                                                                              i, self.particles, self.fitness_vals,
                                                                              self.n_particles,  n_features,
                                                                              self.problem_data, pred_to_insert)

                if prediction is None:
                    self.p_pred_diff_vals = np.zeros(
                        (self.n_particles, self.dim))
                else:
                    self.p_pred_diff_vals = prediction - self.particles
            else:
                self.p_pred_diff_vals = np.zeros((self.n_particles, self.dim))

            # end: if_changed
            try:
                self.gens_of_detected_chngs[self.detected_n_changes].append(i)
            except KeyError:
                # occurs first time after a change occurred
                self.gens_of_detected_chngs[self.detected_n_changes] = []
                self.gens_of_detected_chngs[self.detected_n_changes].append(i)

            # random values
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
                [utils_dynopt.fitness(self.problem, p, i, self.problem_data) for p in self.particles])

            # update p_best after saving old
            old_p_best_fit = copy.copy(self.p_best_fit)
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
            self.best_fitness_evals[i] = copy.copy(self.s_best_fit)
            self.best_particles[i] = copy.copy(self.s_best_pos)

            # adapt c3
            if self.adaptive_c3:
                self.adapt_c3()
        # store things last time
        self.prev_optima_pos.append(
            copy.copy(self.best_particles[self.iterations - 1]))
        self.prev_optima_fit.append(
            copy.copy(self.best_fitness_evals[self.iterations - 1]))

        # conversion
        self.prev_optima_pos = np.array(self.prev_optima_pos)
        self.pred_optima_pos = np.array(self.pred_optima_pos)

    def save_results(self, arrays_file_name, periods_for_generations, act_n_chngs,
                     global_opt_pos_of_changes,
                     global_opt_fit_of_changes):
        np.savez(arrays_file_name,
                 # data from the PSO
                 best_fitness_evals=self.best_fitness_evals,
                 best_individuals=self.best_particles,
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


def define_settings_and_run(repetition, gpu_ID,
                            predictor_name, problem, experiment_name, dim, generations, len_chg_period, pos_chng_type,
                            fit_chng_type, problem_data, n_peaks, arrays_file_path, day, time, noise,
                            pso_np_rnd_generator, pred_np_rnd_generator, periods_for_generations, act_n_chngs,
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
    n_particles = 200  # has to be set
    # initialize EA object
    pso = DynamicPSO(problem, dim, generations, problem_data, predictor_name,
                     n_particles, pso_np_rnd_generator, pred_np_rnd_generator,
                     c1, c2, c3, insert_pred_as_ind, adaptive_c3,
                     n_neurons, n_epochs, batch_size, n_time_steps)

    # TODO sind opt. pos/fit per GENERATION (not per change) !!!!!
    # ist daher auch in den gespeicherten array-files verkehrt!!
    global_opt_pos_of_changes, global_opt_fit_of_changes = get_global_optimum_pos_and_fit_for_all_generations(
        problem, problem_data)
    real_gens_of_chgs = compute_real_gens_of_chgs(
        periods_for_generations, act_n_chngs)

    # ==========================================================================
    # run PSO on specified GPU
    with tf.device('/gpu:' + str(gpu_ID)):
        pso.dynpso()
        K.clear_session()
    # ==========================================================================
    # compute metrics
    # arr_value = arr(real_gens_of_chgs,
    #                global_opt_fit_of_changes, pso.best_fitness_evals)
    arr_value = None
    # bebc = best_error_before_change(
    #    real_gens_of_chgs, global_opt_fit_of_changes,  pso.best_fitness_evals)
    bebc = None

    # ==========================================================================
    # save results
    arrays_file_name = arrays_file_path + predictor_name + "_" + problem + "_" + experiment_name + "_" + \
        str(dim) + "_" + pos_chng_type + "_" + \
        fit_chng_type + "_lenchgperiod-" + str(len_chg_period) + "_noise-" + str(noise) + "_" + day + '_' + \
        time + "_" + str(repetition) + ".npz"
    pso.save_results(arrays_file_name,
                     periods_for_generations, act_n_chngs,
                     global_opt_pos_of_changes,
                     global_opt_fit_of_changes)

    # ==========================================================================
    plots_allowed = False
    if plots_allowed == True:
        plot_results(pso, act_n_chngs, arr_value, bebc, global_opt_pos_of_changes,
                     global_opt_fit_of_changes)

    # do not change order of return values!!! (otherwise change it in
    # predictor_comparison.py as well
    return pso.best_fitness_evals, arr_value, bebc


def plot_results(pso, act_n_chngs, arr_value, bebc, global_opt_pos_of_changes,
                 global_opt_fit_of_changes):
    print("detected ", pso.detected_n_changes, " of ", act_n_chngs, " changes")
    print("best fitness: " + str(min(pso.best_fitness_evals)))
    print("Best-fitness-before-change: ", np.average(pso.prev_optima_fit))
    print("ARR: ", arr_value)
    print("Best-error-before-change: ", bebc)

    plot_fitness(pso.best_fitness_evals)
    plot_best_ind_per_gen(pso.best_particles)
    plot_points(pso.pred_optima_pos, 'Predicted optimum position per change')
    plot_points(pso.prev_optima_pos, 'Found optimum position per change')
    # ==========================================================================
    # compute and plot Eucl. difference between predicted and real optimum
    if pso.detected_n_changes == act_n_chngs:
        n_change_periods = act_n_chngs + 1
        plot_points(global_opt_pos_of_changes, 'Real optimum per change')

        n_predicted_opt = len(pso.pred_optima_pos)
        tmp1 = (pso.pred_optima_pos -
                global_opt_pos_of_changes[-n_predicted_opt:])**2
        tmp2 = np.sum(tmp1, axis=1)  # compute row sums
        diff_pred_and_opt = np.sqrt(tmp2)

        # Eucl. norm: real optimum - found optimum
        tmp1 = (pso.prev_optima_pos[-n_predicted_opt:] -
                global_opt_pos_of_changes[-n_predicted_opt:])**2
        tmp2 = np.sum(tmp1, axis=1)  # compute row sums
        diff_found_and_opt = np.sqrt(tmp2)

        plot_diff_pred_and_optimum(
            diff_pred_and_opt,  "position", diff_found_and_opt)

        #----------------------------------
        # plot fitness differences
        opt_array = np.array(
            [global_opt_fit_of_changes[k] for k in range(n_change_periods)])
        fit_diff_found_and_opt = abs(
            pso.prev_optima_fit - opt_array)
        fit_diff_pred_and_opt = abs(
            pso.pred_optima_fit - opt_array[-n_predicted_opt:])
        plot_diff_pred_and_optimum(
            fit_diff_pred_and_opt, "fitness", fit_diff_found_and_opt[-n_predicted_opt:])
    else:
        print("len_found: ", len(pso.pred_optima_pos),
              ", real: ", len(global_opt_pos_of_changes))

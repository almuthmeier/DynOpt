'''
Functionality for dynamic evolution strategies, e.g.:
    - fitness computation
    - change detection
    - population adaption
 
Created on Nov 23, 2017

@author: ameier
'''
import copy
import warnings

from benchmarks import mpb, dynposbenchmark
import numpy as np


def fitness(problem, individual, curr_gen, problem_data):
    '''
    Compute fitness for one individual.
    @param problem: string: mpb, sphere, rastrigin or rosenbrock
    @param individual: individual for that the fitness is computed
    @param curr_gen: number of current generation as fitness is time-dependent
    @param problem_data: containing all the information required to compute the
    fitness (loaded from file)
    '''

    # TODO(dev)
    if problem == "mpb" or problem == "mpbnoisy" or problem == "mpbrandom":
        return mpb.compute_fitness(individual, curr_gen, problem_data['heights'],
                                   problem_data['widths'], problem_data['positions'])
    elif problem == "sphere" or problem == "rastrigin" or problem == "rosenbrock":
        return dynposbenchmark.compute_fitness(individual, curr_gen, problem,
                                               problem_data['global_opt_pos_per_gen'],
                                               problem_data['orig_global_opt_pos'])


def get_global_optimum_pos_and_fit_for_all_generations(problem, problem_data):
    '''
    Get position and fitness of the global optimum for each generation.
    @param problem: string: mpb, sphere, rastrigin or rosenbrock
    @param problem_data: containing all the information required to compute the
    fitness (loaded from file)
    @return tupel: [0]: 2d numpy array: for each generation a row that contains
                        the global optimum position
                   [1]: 1d numpy array: for each generation the global optimum 
                        fitness
    '''
    if problem == "mpb" or problem == "mpbnoisy" or problem == "mpbrandom":  # TODO(dev)
        global_optimum_fit = problem_data['global_opt_fit']
        global_optimum_pos = problem_data['global_opt_pos']
    elif problem == "sphere" or problem == "rastrigin" or problem == "rosenbrock":
        global_optimum_fit = problem_data['global_opt_fit_per_gen']
        global_optimum_pos = problem_data['global_opt_pos_per_gen']
    else:
        msg = "get_global_optimum_pos_and_fit(): unknown problem " + problem
        warnings.warn(msg)
    return global_optimum_pos, global_optimum_fit


def environment_changed(curr_gen, individuals_from_last_gen, fitness_from_last_gen,
                        problem, problem_data, alg_np_rnd_generator):
    '''
    Tests whether the fitness function has changed.

    Re-evaluates half of the passed population and check same as much random
    points.
    @param curr_gen: current generation
    @param individuals_from_last_gen: population (generated in last generation)
    @param fitness_from_last_gen: populations fitness (computed in last generation)
    @param problem: string: mpb, sphere, rastrigin or rosenbrock
    @param problem_data: containing all the information required to compute the
    fitness (loaded from file)
    @param alg_np_rnd_generator: random generator of the optimization algorithm
    (not the random generator of the prediction model) 
    @return: true if the fitness function has changed
    '''
    n_to_reevaluate = len(individuals_from_last_gen)
    dim = len(individuals_from_last_gen[0])

    # re-evaluate half of the individuals created in previous generation
    indices = np.arange(len(individuals_from_last_gen))
    idx_to_reevaluate = alg_np_rnd_generator.choice(
        indices, size=n_to_reevaluate // 2, replace=False)
    # convert list to array, otherwise indexing with array does not work
    individuals_from_last_gen = np.asarray(individuals_from_last_gen)
    fitness_from_last_gen = np.asarray(fitness_from_last_gen)
    # select the individuals w.r.t. the random indices
    indv_to_reevaluate = individuals_from_last_gen[idx_to_reevaluate]
    fit_to_reevaluate = fitness_from_last_gen[idx_to_reevaluate]
    curr_fit = np.array([fitness(problem, ind, curr_gen, problem_data)
                         for ind in indv_to_reevaluate])

    if not np.array_equal(fit_to_reevaluate, curr_fit):
        return True

    # additionally evaluate some random points
    random_individuals = alg_np_rnd_generator.rand(
        n_to_reevaluate // 2, dim) * 1000  # TODO warum fest???
    prev_fit = np.array([fitness(problem, ind, curr_gen - 1, problem_data)
                         for ind in random_individuals])
    curr_fit = np.array([fitness(problem, ind, curr_gen,  problem_data)
                         for ind in random_individuals])

    return not np.array_equal(prev_fit, curr_fit)


def replace_worst_individuals(alg_np_rnd_generator, problem, curr_gen, population,
                              pop_fitness, pop_size,  n_features,  problem_data, prediction=None):
    '''
    Replaces individuals with the worst fitness by random immigrants/individuals.
    1/5th of the population/swarm is replaced by this function.
    @param pop_size: size of the swarm/population
    @param n_features: dimensionality of solution space
    @param prediction: predicted optimum position. If it is not None, it
    becomes part of the immigrants.
    '''
    old_pop_len = len(population)
    n_immigrants = pop_size // 5

    # compute min/max bounds for new individuals
    pop_mean = np.mean(population, axis=0)  # column-wise mean
    pop_dev = np.std(population, axis=0)

    factor = 3
    min_val = pop_mean - factor * pop_dev
    max_val = pop_mean + factor * pop_dev

    # alternative: previously used (but population spreads to wide)
    # init_range = 200
    # min_val = -init_range
    # max_val = init_range

    # create immigrants
    random_immigrants = alg_np_rnd_generator.uniform(min_val,
                                                     max_val, (n_immigrants, n_features))
    if not prediction is None:
        # replace one immigrant by the predicted optimum
        random_immigrants[0] = copy.copy(prediction)

    # compute immigrants' fitness
    imm_fit = np.array([fitness(problem, individual, curr_gen, problem_data)
                        for individual in random_immigrants])

    # sort old population according to fitness
    sorted_indices = np.argsort(pop_fitness)
    pop = population[sorted_indices]
    fit = pop_fitness[sorted_indices]

    # build new population ( replace worst individuals by immigrants
    new_pop = np.concatenate((pop[:-n_immigrants], random_immigrants))
    new_pop_fitness = np.concatenate((fit[:-n_immigrants], imm_fit))

    new_pop_len = len(population)
    assert old_pop_len == new_pop_len, "old_len: " + \
        str(old_pop_len) + " new_len: " + str(new_pop_len)
    return new_pop, new_pop_fitness


def convert_chgperiods_for_gens_to_dictionary(chgperiods_for_gens):
    '''
    @param chgperiods_for_gens: 1d numpy array containing for each generation the 
    change period number it belongs to
    @return dictionary: for each change period (even if the EA did not detect 
    it) a list of the corresponding generations
    '''

    n_gens = len(chgperiods_for_gens)
    gens_of_chgperiods = {0: [0]}

    if n_gens > 1:
        change_nr = 0
        for i in range(1, len(chgperiods_for_gens)):
            if chgperiods_for_gens[i - 1] == chgperiods_for_gens[i]:
                gens_of_chgperiods[change_nr].append(i)
            else:
                change_nr += 1
                gens_of_chgperiods[change_nr] = [i]
    # test whether number of change periods is the same
    assert len(gens_of_chgperiods) == len(np.unique(chgperiods_for_gens))
    return gens_of_chgperiods

'''
Functionality for EAs:
    - Rechenberg 1/5 success rule
    - recombination
    - mutation
    - selection
'''

import numpy as np


def adapt_sigma(sigma, t_rechenberg, tau, t_s):
    '''
    Rechenberg 1/5 success rule for mutation strength

    @param t_s: number of successful mutations during t_rechenberg generations
    @return: the adapted sigma
    '''
    p_s = t_s / t_rechenberg
    if p_s > 1 / 5:
        return sigma / tau
    elif p_s < 1 / 5:
        return sigma * tau
    else:
        return sigma


def dominant_recombination(population, ro, ea_np_rnd_generator):
    '''
    Dominant recombination. 

    Creates one offspring from ro parents in the population.
    @param population: each row is one individual 
    @param ea_np_rnd_generator: numpy random number generator, created with
        np.random.RandomState(<seed>)
    @return: the offspring individual
    '''
    # number individuals in population, number features of individuals
    pop_size, ind_size = population.shape

    # randomly select ro parents, i.e. ro different numbers in [0, pop_size[
    # sampling without replacement
    parents_indices = ea_np_rnd_generator.choice(
        range(0, pop_size), ro)

    # for each feature of the offspring, randomly select one of the ro
    # parents
    offspring = np.zeros(ind_size)
    for i in range(ind_size):
        parent_index = ea_np_rnd_generator.choice(parents_indices)
        offspring[i] = population[parent_index][i]

    return offspring


def gaussian_mutation(x, mean, sigma, ea_np_rnd_generator, z=None, covariance=None):
    '''
    Gaussian mutation.

    Mutates individual x by adding Gaussian Noise.

    @param mean: (scalar)
    @param ea_np_rnd_generator: numpy random number generator, created with
        np.random.RandomState(<seed>)
    @param sigma: (scalar) standard deviation of distribution
    @param z: (scalar) scale of the distribution
    @param covariance: (scalar) (co)variance; use either sigma or covariance
    '''
    # TODO das z extra uebergeben, damit es auch bei multivariate funktioniert

    if sigma is None:
        # multivariate variance: for each dimension different variance
        # convert single float to vector by repeating it multiple times
        mean = np.array([mean] * len(x))
        try:
            # throws exception if covariance is a row vector
            # -> contains only variance (since covariance would require matrix)
            covariance.shape[1]
        except IndexError:  # variance case
            "tuple index out of range"
            # convert variance vector to diagonal covariance matrix
            covariance = np.diag(covariance)
        noise = ea_np_rnd_generator.multivariate_normal(mean, covariance)
    else:  # sigma (standard deviation) is scalar
        noise = ea_np_rnd_generator.normal(mean, sigma, len(x))

    if z is None:
        z = 1
    return x + z * noise


def mu_plus_lambda_selection(mu, parents, offsprings, parents_fitness, offsprings_fitness):
    '''
    Mu+lambda-selection.

    Selects mu individials for the next iteration.

    @return: new population (2d array) and fitness of new population (column vector)
    '''
    # make one large array with more rows
    individuals = np.concatenate((parents, offsprings), axis=0)
    # analogously for fitness arrays
    fitness = np.concatenate((parents_fitness, offsprings_fitness), axis=0)

    # sort array according to fitness column
    # returns indices to sort the individuals according to their fitness (low
    # values (=high fitness) first)
    sorted_indices = np.argsort((fitness).flatten())
    sorted_individuals = individuals[sorted_indices]
    sorted_fitness = fitness[sorted_indices]
    # select mu best individuals
    return (sorted_individuals[:mu, :], sorted_fitness[:mu].reshape(-1, 1))

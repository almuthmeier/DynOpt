'''
Created on Jul 24, 2019

@author: ameier
'''
import copy
from math import sqrt, exp

from numpy import linalg as npla
from scipy import linalg as spla

import numpy as np
from utils import utils_dynopt


def visualize_dominant_eigvector(n, eig_vals, eig_vctrs):
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


def get_inverse_sqroot(M):
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


def get_offsprings(n, m, sig, lambd, sqrt_of_eig_vals, eig_vctrs, t,
                   benchmarkfunction, experiment_data, cma_np_rnd_generator):
    offspring_population = np.zeros((lambd, n))
    offspring_fitnesses = np.zeros(lambd)

    # decompose covariance matrix for sampling ("The CMA evolution strategy: a
    # tutorial, Hansen (2016), p. 28+29)
    # A = B*sqrt(M), with C = BMB^T (C=covariance matrix)
    A = np.matmul(eig_vctrs, sqrt_of_eig_vals)

    for k in range(lambd):
        z_k = cma_np_rnd_generator.normal(size=n)
        y_k = np.matmul(A, z_k)
        x_k = m + sig * y_k
        # x_k is equal to the following line but saves eigendecompositions:
        # m + sig * cma_np_rnd_generatorndom.multivariate_normal(np.zeros(n), C)

        f_k = utils_dynopt.fitness(benchmarkfunction,
                                   x_k, t, experiment_data)

        offspring_population[k] = copy.copy(x_k)
        offspring_fitnesses[k] = f_k
    return offspring_population, offspring_fitnesses


def get_mue_best_individuals(n, mu, offspring_population, offspring_fitnesses):
    # sort individuals according to fitness
    sorted_indices = np.argsort(offspring_fitnesses)
    sorted_individuals = offspring_population[sorted_indices]
    # select mu best individuals
    mu_best_individuals = sorted_individuals[:mu, :]
    assert mu_best_individuals.shape == (mu, n)
    return mu_best_individuals  # format [individuals, dimensions]


def get_weighted_avg(n, w, mu_best_individuals):
    # weight individuals
    weighted_indvds = mu_best_individuals * w[:, np.newaxis]
    # sum averaged individuals
    weighted_avg_indvds = np.sum(weighted_indvds, axis=0)
    assert weighted_avg_indvds.shape == (n,)
    return weighted_avg_indvds


def get_new_p_sig(n, c_sig, p_sig, mu_w, m, m_new, sig, inv_squareroot_C):
    diff_vector = m_new - m
    succ_mutation_steps = diff_vector / sig
    tmp = np.matmul(inv_squareroot_C, succ_mutation_steps)

    new_p_sig = (1 - c_sig) * p_sig + \
        sqrt(c_sig * (2 - c_sig)) * sqrt(mu_w) * tmp
    assert new_p_sig.shape == (n,)
    return new_p_sig


def get_h_sig(p_sig_new, c_sig, t, n, E):
    right_side = sqrt(1 - (1 - c_sig)**(2 * (t + 1))) * (1.4 + 2 / (n + 1)) * E
    cond = npla.norm(p_sig_new) < right_side
    return int(cond == True)  # 0 is False, 1 otherwise


def get_new_p_c(n, c_c, p_c, h_sig, mu_w, m_new, m, sig):
    diff_vector = m_new - m

    first = (1 - c_c) * p_c
    second = sqrt(c_c * (2 - c_c))
    third = diff_vector

    new_p_c = first + h_sig * second * \
        sqrt(mu_w) * third / sig
    assert new_p_c.shape == (n,)
    return new_p_c


def get_C_mu(n, mu_best_individuals, m, sig, w):
    # subtract mean from each row, scale with sigma
    # format: [individuals, dimensions]
    # [:,None] necessary to devide each row by another vector element
    # https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    # (24.6.19)
    #matrix = (mu_best_individuals - m) / scales[:, None]
    matrix = (mu_best_individuals - m) / sig  # different value per dim

    # multiply each row with respective weight (4.3.19)
    # https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
    weighted_matrix = w[:, np.newaxis] * matrix

    # in the multiplication the matrix is first transposed so that covariance
    # matrix has format [dimension, dimensions] otherwise it would have format
    # [individuals, individuals]
    C_mu = np.matmul(weighted_matrix.T, matrix)
    assert C_mu.shape == (n, n)
    return C_mu


def get_new_C(n, c_1, c_mu, C, p_c_new, C_mu):
    first = (1 - c_1 - c_mu) * C
    col_vec = p_c_new[:, np.newaxis]  # format [n,1]
    second = c_1 * np.matmul(col_vec, col_vec.T)
    third = c_mu * C_mu
    new_C = first + second + third
    assert new_C.shape == (n, n)
    return new_C


def get_new_sig(sig, c_sig, d_sig, p_sig_new, E):
    norm_val = npla.norm(p_sig_new)
    subtr = norm_val / E - 1
    inner = (c_sig / d_sig) * subtr
    tmp = exp(inner)
    return sig * tmp


def get_best_fit_and_ind_so_far(curr_best_fit, curr_best_ind, offspring_population, offspring_fitnesses):
    '''
    Checks whether the fitness of the best individual from the current generation
    is better than a given fitness.
    @return the possibly new best fitness and the corresponding individual
    '''
    min_idx = np.argmin(offspring_fitnesses)
    if offspring_fitnesses[min_idx] < curr_best_fit:
        return offspring_fitnesses[min_idx], offspring_population[min_idx]
    else:
        return curr_best_fit, curr_best_ind

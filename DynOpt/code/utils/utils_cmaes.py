'''
Created on Jul 24, 2019

@author: ameier
'''
import copy
from math import sqrt, exp, log

from numpy import linalg as npla
from scipy import linalg as spla

import numpy as np
from utils import utils_dynopt


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


def print_success(self, t, offspring_population, offspring_fitnesses):
    min_idx = np.argmin(offspring_fitnesses)
    print(t, "    ", offspring_fitnesses[min_idx],
          "    ", offspring_population[min_idx])
    # return offspring_fitnesses[min_idx]
    try:
        return log(offspring_fitnesses[min_idx])  # TODO 23.4 wieder log
    except ValueError:  # since fitness is zero
        return log(np.nextafter(0, 1))  # smallest float larger than 0

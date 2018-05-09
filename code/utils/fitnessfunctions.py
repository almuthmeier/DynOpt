'''
Classical fitness functions for static optimization.

All functions expect a one-dimensional numpy array as input (the individual for
that the fitness should be computed) and return the fitness real value.

Created on May 5, 2017

@author: ameier
'''

from math import cos, pi, sqrt

import numpy as np


def sphere(x):
    return sum(x * x)


def rastrigin(x):
    tmp = 2 * pi * x
    cos_array = np.array([cos(xi) for xi in tmp])
    tmp = (x * x) - (10 * cos_array) + 10
    return sum(tmp)


def rosenbrock(x):
    assert len(x) > 1, "Rosenbrock function supports only dimensions > 1"
    x_square = x * x
    tmp = x_square[:(x_square.size - 1)] - x[1:]
    summand_1 = 100 * np.square(tmp)  # has n-1 elements
    summand_2 = np.square(x - 1)  # has n elements

    sum_vector = summand_1 + summand_2[:(summand_2.size - 1)]
    return sum(sum_vector)


def cigar(x):
    assert len(x) > 1, "Cigar function supports only dimensions > 1"
    return np.square(x[0]) + 1e6 * sum(np.square(x[1:]))


def griewank(x):
    sum_part = np.sum(np.square(x) / 4000)
    # sqrt(i+1) since otherwise devision by zero would occur
    product_part = np.prod(
        np.array([cos(x[i] / sqrt(i + 1)) for i in range(len(x))]))
    return sum_part - product_part + 1


def niching(x):
    '''
    0 <= x_i <= 2 ( works without this constraint as well !?!)
    '''
    # has 2^d_ optima
    d_ = 2  # < d
    sum1 = sphere((x[:d_] % 1) - 0.5)
    sum2 = sphere(x[d_:])
    return sum1 + sum2


def double_sum(x):
    # https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/ortizboyer05a-html/node6.html
    s = 0
    for i in range(len(x)):
        # i+1, otherwise false result if array has only one entry
        s = s + np.square(sum(x[:(i + 1)]))
    return s


def get_original_global_opt_pos_and_fit(function, dimensionality):
    '''
    Global optimum position and its fitness for unmoved fitness function.

    TODO: what if function has multiple global optima?
    TODO: extend this function if more fitness functions are desired
    @return: tupel: (position, fitness)
    '''
    function_name = function.__name__
    glob_opt_pos = {'sphere': np.array(dimensionality * [0]),
                    'rosenbrock': np.array(dimensionality * [1]),
                    'rastrigin': np.array(dimensionality * [0])}
    pos = glob_opt_pos[function_name]
    return pos, function(pos)
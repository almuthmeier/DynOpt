'''
Created on Jan 28, 2019

@author: ameier
'''

import numpy as np


def make_values_feasible_for_square(values):
    '''
    Should be applied when np.square(values) caused the following error:
        "RuntimeWarning: overflow encountered in square"

    Corrects entries in values so that afterards np.square(values) can be
    computed.

    @return the corrected values
    '''
    # compute lowest and largest value that can be feed into np.square
    # without throwing an exception
    max_float = np.finfo(np.float).max
    sqrt_max_float = np.sqrt(max_float)
    min_float = np.finfo(np.float).min
    sqrt_min_float = np.sqrt(-min_float)  # make value positive

    # entries that are too large are made smaller
    indices = np.argwhere(values > sqrt_max_float)
    values[indices] = sqrt_max_float
    # entries that are too small are made larger
    indices = np.argwhere(values < -sqrt_min_float)
    values[indices] = sqrt_min_float

    return values

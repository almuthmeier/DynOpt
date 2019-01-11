'''
Created on Jan 10, 2019

@author: ameier
'''

import math

import matplotlib.pyplot as plt
import numpy as np


def first_function(n_data):
    step_size = 0.1
    max_time_point = math.ceil(n_data * 0.1)
    time = np.arange(0, max_time_point, step_size)
    print("max_time_point: ", max_time_point)
    # original (sinusartige Parabel)
    values_d1 = np.sin(time + 2) * 3 * np.sin(2 * time) + \
        0.0001 * np.square(time)
    # Trichter (wird immer größer)
    # values_d1 = np.sin(time + 2) * 3 * np.sin(2 * time) * \
    #    0.0001 * (np.square(time) * 0.3 + 4)
    # ausprobiert
    # values_d1 = np.sin(3 * time + 2) * 3 * np.sin(2 * time) + \
    #    0.001 * np.square(2 * time)
    values_d2 = 5 * np.sin(5 * time) * (np.exp(-0.003 * (time - 50))) + 4

    noise = 0
    noisy_values_d1 = values_d1 + noise * np.random.rand(len(values_d1))
    noisy_values_d2 = values_d2 + noise * np.random.rand(len(values_d1))

    data = np.transpose(np.array([values_d1, values_d2]))
    noisy_data = np.transpose(np.array([noisy_values_d1, noisy_values_d2]))

    return data[:n_data], noisy_data[:n_data]


def create_sinefreq_benchmark(n_data):
    data, noisy_data = first_function(n_data)
    plt.plot(data[:, 0])
    plt.plot(data[:, 1])
    plt.show()
    return data


if __name__ == '__main__':
    create_sinefreq_benchmark(10000)

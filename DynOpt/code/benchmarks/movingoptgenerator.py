'''
Time series generator. Uses artificial functions (sine, polynomial, linear,...) 
and combines them in order to get more complicated time series.

Created on Oct 22, 2018

@author: ameier
'''

import copy
from datetime import datetime
import math
import warnings

from pandas import Series
from pandas import read_csv
from scipy import interpolate
from scipy.interpolate import interp1d
from sklearn.preprocessing.data import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np


def champanger():
    '''
    https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/ (23.10.18)
    '''
    from matplotlib import pyplot
    series = Series.from_csv('monthly-champagne.csv')

    # 'Perrin Freres monthly champagne sales millions ?64-?72'

    print(series.describe())
    series.plot()
    pyplot.show()


def parse(x):
    '''
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ (23.10.18)
    '''
    return datetime.strptime(x, '%Y %m %d %H')


def prepare_real_world_series():
    '''
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ (23.10.18)
    '''
    dataset = read_csv('pollution_original.csv',  parse_dates=[
                       ['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp',
                       'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv('pollution.csv')


def real_word_series(col_name=None):
    '''
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ (23.10.18)
    '''

    from pandas import read_csv
    from matplotlib import pyplot
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    n_data = len(values)
    print("len(dataset): ", n_data, " = ", n_data / 24,
          " days ", " = ", n_data / 24 / 365, " years")
    # specify columns to plotvalues)
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

    if col_name is not None:
        return dataset[col_name].values


def plot_into_subplots(list_of_Xs, list_of_Ys, colors):
    n_plots = len(list_of_Xs)
    n_cols = n_plots
    n_rows = 1
    print("n_plots: ", n_plots)

    fig = plt.figure()
    p_idx = 1
    for X, Y, col in zip(list_of_Xs, list_of_Ys, colors):
        ax = fig.add_subplot(n_rows, n_cols, p_idx)
        ax.plot(X, Y, color=col, linewidth=0.5, antialiased=False)
        p_idx += 1
    plt.draw()  # continue execution
    # plt.show() # break execution until figure is closed
    #fig.savefig('demo.pdf', bbox_inches=0, transparent=True)


def trial():
    n_time_points = 50
    time_points = np.arange(n_time_points)
    ts1 = -2 * time_points + 2 * n_time_points  # linear decreasing function
    ts2 = 0.1 * np.power(time_points, 2) - 25

    list_of_Xs = [time_points, time_points, ts1]
    list_of_Ys = [ts1, ts2, ts2]
    colors = ['blue', 'green', 'red']
    plot_into_subplots(list_of_Xs, list_of_Ys, colors)
    plt.show()

# =============================================================================


def get_series_for_type(type, dims, n_points):
    '''
    @return: format [time_steps, dims]
    '''
    # format [dims, time steps]
    if type == 'linear':
        points_per_dim = [start_linear(n_points) for _ in range(dims)]
    elif type == 'sine':
        points_per_dim = [start_sine(n_points) for _ in range(dims)]
    elif type == 'polynomial':
        points_per_dim = [start_polynomial(n_points) for _ in range(dims)]
    elif type == 'spline':
        points_per_dim = [start_splines(n_points) for _ in range(dims)]
    else:
        warnings.warn("undefined function type: " + type)

    # transpose array so that format is [time steps, dims]
    ts = np.transpose(points_per_dim)
    return ts


def plot_same_axis(x1, x2, y1, y2):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(x1, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('sin', color=color)
    ax2.plot(x2, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def correct_values(values, min_value, max_value):
    '''
    Ensures that values are in given range
    @param values: 1d numpy array
    '''
    # scale
    # do nothing if valid values
    lowest_val = np.min(values)
    largest_val = np.max(values)
    lowest_val_valid = lowest_val >= min_value and lowest_val < max_value
    largest_val_valid = largest_val <= max_value and largest_val > min_value
    #print("allowed: min_val: ", min_value, " max_val: ", max_value)
    #print("current: min_val: ", lowest_val, "max_val: ", largest_val)
    if lowest_val_valid and largest_val_valid:
        pass
    else:
        #print("at least one not valid")
        # +/-1 to prevent AssertionErrors caused by rounding errors
        # -> +/-1 introduces new excpetion: "ValueError: Minimum of desired
        # feature range must be smaller than maximum. Got (84.80001171045868,
        # 84). -> Therefore used without +-1 and adapted assertions.
        min_value_for_scaler = min_value  # + 1
        max_value_for_scaler = max_value  # - 1
        # re-use max/min values in data if valid, otherwise all functions would
        # be in same range
        if lowest_val_valid:
            #print("lowest valid")
            min_value_for_scaler = lowest_val
        if largest_val_valid:
            #print("largest valid")
            max_value_for_scaler = largest_val
        scaler = MinMaxScaler(feature_range=(
            min_value_for_scaler, max_value_for_scaler))
        reshaped_values = values.reshape(-1, 1)  # otherwise DeprecationWarning
        scaler = scaler.fit(reshaped_values)
        values = scaler.transform(reshaped_values)
        values = np.reshape(values, len(values))  # original shape
    # print("afterwards: min_val: ", np.min(
    #    values), " max_val: ", np.max(values))
    min_in_scaled = np.min(values)
    max_in_scaled = np.max(values)
    # test whether min_value <= min_in_scaled
    assert min_value - min_in_scaled <= 0.0000001, "current min: " + \
        str(min_in_scaled) + "but allowed min is: " + str(min_value)
    # test wheter max_in_scaled <= max_value
    assert max_in_scaled - max_value <= 0.000001, "current max: " + str(max_in_scaled) + \
        " but allowed max is: " + str(max_value)
    return values
# =============================================================================
# Spline #
# =============================================================================


def multidim_spline_ts(min_value, max_value, n_points, n_base_points,
                       n_points_between_base_points, n_dims, with_plots=False):
    # format [dims, time steps]
    points_per_dim = [random_spline_ts(min_value, max_value,
                                       n_points, n_base_points, n_points_between_base_points) for _ in range(n_dims)]
    # transpose array so that format is [time steps, dims]
    ts = np.transpose(points_per_dim)

    # plot
    if with_plots:
        time_steps = np.arange(len(ts))
        if n_dims == 1:
            plt.plot(time_steps, ts, '-')
            plt.legend(['dim0'], loc='best')
            plt.show()
        elif n_dims >= 2:
            plots = []
            descriptions = []
            for i in range(n_dims):
                p, = plt.plot(time_steps, ts[:, i])
                plots.append(p)
                descriptions.append('dim' + str(i))
            plt.legend(plots, descriptions, loc='best')
            plt.show()
    return ts


def random_spline_ts(min_value, max_value, desired_n_points, n_base_points,
                     n_points_between_base_points, plots_allowed=False):

    # normal-distributed values
    range_width = max_value - min_value
    mean = min_value + range_width / 2
    # nearly all points are located within [mean-4*std_dev, mean + 4*std_dev]
    dev = range_width / 8
    values = np.random.normal(mean, dev, n_base_points)
    # correct values that are too large/low
    #values = correct_values(values, min_value, max_value)

    # model function
    start = 0
    stop = n_base_points
    base_points = np.arange(start, stop)
    assert len(base_points) == n_base_points
    # kubische Spline-Interpolation
    # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    # (27.10.18)
    f = interp1d(base_points, values, kind='cubic')

    # interpolate new points
    n_new_points = n_base_points * n_points_between_base_points
    xnew = np.linspace(start, np.max(base_points),
                       num=n_new_points, endpoint=True)
    ynew = f(xnew)
    diff = abs(ynew[:-1] - ynew[1:])
    avg_diff = np.average(diff)

    if plots_allowed:
        # visualization
        print("avg_diff: ", avg_diff)
        plt.plot(base_points, values, 'o-', xnew, ynew, '.-')
        plt.legend(['data',  'cubic'], loc='best')
        plt.show()

        # plot differences
        plt.plot(xnew[:-1], diff, '.')
        plt.show()

        # plot in same plot
        plot_same_axis(base_points, xnew[:-1], values, diff)

    # may be larger than desired_n_points
    assert len(ynew) >= desired_n_points
    ynew = correct_values(ynew, min_value, max_value)
    return ynew[:desired_n_points]


def with_spline():
    '''
    Only for trying another function.
    '''
    x = np.arange(0, 2 * np.pi + np.pi / 4, 2 * np.pi / 8)
    y = np.sin(x)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, 2 * np.pi, np.pi / 50)
    ynew = interpolate.splev(xnew, tck, der=0)

    plt.figure()
    plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
    plt.legend(['Linear', 'Cubic Spline', 'True'])
    plt.axis([-0.05, 6.33, -1.05, 1.05])
    plt.title('Cubic-spline interpolation')
    plt.show()


# =============================================================================
# Polynomial #
# =============================================================================
def polynomial():
    '''
    Only for trying polynomials.
    '''
    n_time_points = 50
    x = np.arange(n_time_points)
    start = -6.1
    stop = 3.1
    step = (stop - start) / n_time_points
    x = np.arange(start, stop, step)
    c = 0.03  # -2
    c = 1e-2  # 0.0000000000006
    y = c * (x + 6) * (x + 2) * x * (x - 3)
    y = c * (x + 6) * (x - 1.5) * x * (x + 2) * (x + 4) * (x - 3)

    list_of_Xs = [x]
    list_of_Ys = [y]
    colors = ['blue']
    plot_into_subplots(list_of_Xs, list_of_Ys, colors)
    plt.show()


def random_polynomial(min_value, max_value, n_points, n_base_points, min_order,
                      max_order, plots_allowed=False):

    base_points = np.random.randint(1, n_points - 1, n_base_points - 2)
    base_points = np.append(base_points, [0, n_points - 1])

    # normal-distributed values
    range_width = max_value - min_value
    mean = min_value + range_width / 2
    # nearly all points are located within [mean-4*std_dev, mean + 4*std_dev]
    dev = range_width / 8
    base_values = np.random.normal(mean, dev, n_base_points)
    #base_values = np.random.randint(min_value, max_value, n_base_points)
    # correct values that are too large/low
    #base_values = correct_values(base_values, min_value, max_value)

    # random order of polynomial
    poly_order = np.random.randint(min_order, max_order + 1)

    # fit polynomial (30.10.18)
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.polyfit.html
    z = np.polyfit(base_points, base_values, poly_order)
    p = np.poly1d(z)

    # interpolate
    xnew = np.arange(0, n_points)
    ynew = p(xnew)

    # visualize
    if plots_allowed:
        plt.plot(base_points, base_values, 'o', xnew, ynew, '.')
        plt.show()

    assert len(ynew) == n_points
    ynew = correct_values(ynew, min_value, max_value)
    return ynew

# =============================================================================
# miscellaneous #
# =============================================================================


def linear(min_value, max_value, n_points, plots_allowed=False):
    '''
    f(x) = mx + n
    '''
    # sample two base points
    base_points = np.array([0, n_points - 1])
    base_values = np.random.randint(min_value, max_value, 2)

    poly_order = 1  # linear function
    z = np.polyfit(base_points, base_values, poly_order)
    p = np.poly1d(z)

    # interpolate
    xnew = np.arange(0, n_points)
    ynew = p(xnew)

    # visualize
    if plots_allowed:
        plt.plot(base_points, base_values, 'o', xnew, ynew, '.')
        plt.show()

    assert len(ynew) == n_points
    ynew = correct_values(ynew, min_value, max_value)
    return ynew


def sine(min_value, max_value, n_points, plots_allowed=False):
    '''
    TODO enable increasing amplitude
    https://en.wikipedia.org/wiki/Sine_wave (30.10.18)
    '''
    # amplitude: normal-distributed value within possible range
    # can use only half, otherwise negative values
    range_width = (max_value - min_value) / 2
    mean = min_value + range_width / 2
    # nearly all points are located within [mean-4*std_dev, mean + 4*std_dev]
    dev = range_width / 8
    #amplitude = np.random.normal(mean, dev, 1)
    amplitude = 0.5 * np.random.rand() * range_width + \
        min_value  # amplitude quite large  ('0.5*' to shrink values)
    # frequency in [0,4
    min_freq = 0
    max_freq = 2
    frequency = np.random.rand() * (max_freq - min_freq) + min_freq
    xshift = 0
    yshift = mean  # prevent negative values

    points = np.arange(n_points)
    values = amplitude * \
        np.sin(2 * math.pi * frequency * points + xshift) + yshift

    # visualize
    if plots_allowed:
        print("amplitude: ", amplitude, " frequency: ", frequency)
        plt.plot(points, values,  '-')
        plt.show()

    assert len(values) == n_points

    values = correct_values(values, min_value, max_value)
    return values

# =============================================================================
# mixture #
# =============================================================================


def noise_function(n_points):
    return np.random.normal(0, 1, n_points)


def multidim_section_functions(n_dims, n_points, available_base_fcts, mixture_fcts,
                               min_value, max_value, mix_within_dim, mix_between_dims, with_plots):
    '''
    @return: 2d numpy array: format [time steps, dims]
    '''
    prob_for_mixing = 0.6  # probability for mixing within dimension
    values_per_dim = []
    for _ in range(n_dims):
        if mix_within_dim:
            # mixing within dimension?
            mix_within = np.random.randn() < prob_for_mixing
        else:
            mix_within = False
        fct_min_value = min_value + np.random.randint(0, 20)
        fct_max_value = max_value - np.random.randint(0, 20)
        assert fct_max_value > fct_min_value + (max_value - min_value) / 2
        print("mix: ", mix_within)
        values_per_dim.append(section_functions(n_points, available_base_fcts, mixture_fcts,
                                                fct_min_value, fct_max_value,
                                                mix_within, with_plots))

    if mix_between_dims:
        values_per_dim = mix_dims(
            values_per_dim, min_value, max_value, with_plots)
        # transpose array so that format is [time steps, dims]
    ts = np.transpose(values_per_dim)

    # plot
    if with_plots:
        time_steps = np.arange(len(ts))
        if n_dims == 1:
            plt.plot(time_steps, ts, '-')
            plt.legend(['dim0'], loc='best')
            plt.show()
        elif n_dims >= 2:
            plots = []
            descriptions = []
            for i in range(n_dims):
                p, = plt.plot(time_steps, ts[:, i])
                plots.append(p)
                descriptions.append('dim' + str(i))
            plt.legend(plots, descriptions, loc='best')
            plt.show()

    return ts


def mix_dims(values_per_dim, min_value, max_value, plots_allowed=False):
    '''
    @param values_per_dim: 2d numpy array [dims, time steps]
    '''
    n_dims = len(values_per_dim)
    if n_dims <= 2:
        # no pairs of three dimensions can be constructed
        return values_per_dim
    # =========================================================================
    # choose random pairs of three dimensions that should be mixed
    n_tupels = n_dims // (3 + 1)  # +1 so that at least 3 dims are not tupeled
    tupels_size = 3
    # draw randomly dims that have to be combined
    random_dims = np.random.choice(
        n_dims, n_tupels * tupels_size, replace=False)
    # make tupels with 3 dims each: each row consists of 3 values that are
    # the dimensions that will be combined
    dims_to_combine = random_dims.reshape(-1, 3)

    # =========================================================================
    # determine randomly section length:
    # min: one 20th of whole series
    n_time_steps = len(values_per_dim[0])
    min_length = n_time_steps // 20
    # max: one 10th of whole series
    max_length = n_time_steps // 10
    section_length = np.random.randint(min_length, max_length)

    # =========================================================================
    # mix dimension: add the values of the two first dimensions within the
    # specified section and replace the values of the third dimensions in the
    # following section by this summed values
    for tupel in dims_to_combine:
        # determine randomly section start index: must not be to near at the
        # end
        section_start_idx = np.random.randint(
            0, n_time_steps - section_length - 1)
        section_end_idx = section_start_idx + section_length
        if plots_allowed:
            print("mix between dims: start_idx: ", section_start_idx)
            print("mix between dims: end_idx: ", section_end_idx)
            print("dims that are combined: ", tupel)
        # mix values
        first_vals = values_per_dim[tupel[0]]  # all values of that dimension
        first_vals = first_vals[section_start_idx:section_end_idx]  # section
        second_vals = values_per_dim[tupel[1]]
        second_vals = second_vals[section_start_idx:section_end_idx]
        summed_values = first_vals + second_vals
        # change values in third dimension
        start = section_start_idx + section_length
        end = section_start_idx + 2 * section_length
        values_per_dim[tupel[2]][start:end] = summed_values

    # TODO: remove jumps between the old and the new data

    # =========================================================================
    # correct and return data
    values_per_dim = np.array(values_per_dim)
    for d in range(len(values_per_dim)):
        values_per_dim[d] = correct_values(
            values_per_dim[d], min_value, max_value)
    return values_per_dim


def generate_values_with_fct(fct, min_value, max_value, n_pts):
    if fct == 'linear':
        v = linear(min_value, max_value, n_pts, plots_allowed=False)
    elif fct == 'sine':
        v = sine(min_value, max_value, n_pts, plots_allowed=False)
    elif fct == 'polynomial':
        n_base_points = n_pts // 20
        #poly_order = 18
        min_order = 10
        max_order = 20
        v = random_polynomial(min_value, max_value, n_pts, n_base_points,
                              min_order, max_order, plots_allowed=False)
    elif fct == 'spline':
        n_base_points = n_pts // 10
        n_points_between_base_points = math.ceil(n_pts / n_base_points)
        v = random_spline_ts(min_value, max_value, n_pts, n_base_points,
                             n_points_between_base_points, plots_allowed=False)
    elif fct == 'noise':
        v = noise_function(n_pts)
    else:
        warnings.warn("undefined function: " + fct)
    assert len(v) == n_pts, "require " + str(n_pts) + \
        " points but got " + str(len(v)) + " points"
    return np.array(v)


def section_functions(n_points, available_base_functions, mixture_fcts,
                      min_value, max_value, mix_within_dim, plots_allowed=False):
    '''
    define sections with different functions randomly
    '''
    min_n_points_per_sect = 300  # 300 points per section if equidistant sections
    max_n_points_per_sect = 3000

    # determine number of points per section randomly
    n_points_per_sect = []
    # while n_points - np.sum(n_points_per_sect) >= min_n_points_per_sect:
    while np.sum(n_points_per_sect) <= n_points - min_n_points_per_sect:
        p = np.random.randint(min_n_points_per_sect, max_n_points_per_sect)
        n_points_per_sect.append(p)
    curr_n_pts = np.sum(n_points_per_sect)
    if curr_n_pts > n_points:
        # if already too much points, delete some points in last section
        n_points_per_sect[-1] = n_points_per_sect[-1] - (curr_n_pts - n_points)
    else:
        # fill missing points into last section (missing points cannot be a
        # separate section as it would contain too few points)
        missing = n_points - curr_n_pts
        n_points_per_sect[-1] += missing
    n_points_per_sect = np.array(n_points_per_sect)
    n_sections = len(n_points_per_sect)
    print("n_sections: ", n_sections)
    assert np.sum(n_points_per_sect) == n_points
    assert np.all(n_points_per_sect > min_n_points_per_sect)

    # start indices of sections (0 is in any case a section start index)
    sect_indices = [0]
    for p in n_points_per_sect[:-1]:
        sect_indices.append(sect_indices[-1] + p)
    assert len(sect_indices) == n_sections
    # test whether n_points_per_sect and the section indices are consistent
    assert n_points == sect_indices[-1] + n_points_per_sect[-1]

    # determine base functions for sections
    n_avail_fcts = len(available_base_functions)
    selected_fcts = np.random.randint(0, n_avail_fcts, n_sections)

    # compute values for sections
    values = []
    for s in range(n_sections):
        fct = available_base_functions[selected_fcts[s]]
        # number of points within section
        n_pts = n_points_per_sect[s]
        if plots_allowed:
            print("s: ", s)
            print("fct: ", fct)
            print("n_pts: ", n_pts)

        v = generate_values_with_fct(fct, min_value, max_value, n_pts)
        # not necessary here because done in base functions; for security again
        v = correct_values(v, min_value, max_value)

        # determine with which functions the dim should be mixed
        if mix_within_dim:
            # select at most two other functions
            n_fcts_to_mix = np.random.randint(1, 3)
            selected_mix_fcts = np.random.randint(
                0, len(mixture_fcts), n_fcts_to_mix)
            selected_mix_fcts = mixture_fcts[selected_mix_fcts]
            print("mix function with... ", selected_mix_fcts)
            for f in selected_mix_fcts:
                v += generate_values_with_fct(f, min_value, max_value, n_pts)
            v = correct_values(v, min_value, max_value)

        values = np.concatenate((values, copy.copy(v)))
        # assert min_value <= np.min(values) and np.max(values) <= max_value
        # -> is written with the following line because of rounding errors
        assert min_value - \
            np.min(values) <= 0.000001 and np.max(
                values) - max_value <= 0.000001

    # prevent jumps between sections by inserting small linear sections
    assert np.sum(n_points_per_sect) == n_points
    n_inserted_points = min_n_points_per_sect / 6
    values_without_jumps = []
    for s in range(n_sections):  # for all but the last section
        # end index of current section is summed number points in current in
        # previous sections
        end_idx = np.sum(n_points_per_sect[:s + 1])
        start_idx = np.sum(n_points_per_sect[:s])

        if s == n_sections - 1:  # last section -> insert no values
            values_for_section = values[start_idx:]
            values_without_jumps = np.concatenate(
                (values_without_jumps, values_for_section))
            continue
        values_to_insert = np.linspace(
            values[end_idx], values[end_idx + 1], n_inserted_points)
        # do not use the first and last one, because they are already in the
        # original time series
        values_to_insert = values_to_insert[1:-1]
        # append to time series
        values_for_section = values[start_idx:end_idx]
        values_without_jumps = np.concatenate(
            (values_without_jumps, values_for_section))
        values_without_jumps = np.concatenate(
            (values_without_jumps, values_to_insert))
    # remove the surplus values
    values = values_without_jumps[:n_points]
    # visualize
    if plots_allowed:
        plt.plot(np.arange(n_points), values)
        plt.show()
    return values
    # change base functions randomly:
    #     +/-:
    # + oder * [linear ansteigende Werte]
    #scale = np.arange(n_points)
    #idx = math.floor(n_points / 2)
    #scale[idx:] = np.ones(len(scale[idx:])) * scale[idx]
    #values = values + scale
# =============================================================================
# start functions #
# =============================================================================


def start_splines(n_points=1250):
    # [n_points, n_base_points, distance_base_points}
    # 12500,50,500 -> avg_diff: 0.296
    # 12500,50,1 -> avg_diff: 0.296 ## -> distance_base-points has no effect!
    # 12600,50,n_points_between_base_points -> avg_diff: 0.289
    # -> TODO distance_base_points weglassen
    # -> the more n_base_points, the "longer" the function
    # -> the more n_base_points, less points between two extrema to learn from

    # 12500,500,_ -> avg_diff: 3.18 ## change in n_base_points has effect!
    # 125,50,_ -> avg_diff: 24.559

    # 125,500,_ -> avg_diff: 73.08 ## change in n_points has effect!

    # == changed range: min/max is now 0/100
    # 12500,50,_ -> avg_diff: 0.63

    min_value = 0
    max_value = 100
    #n_points = 1250
    n_base_points = 50  # 0.05
    #distance_base_points = 500
    with_plots = False
    n_points_between_base_points = math.ceil(n_points / n_base_points)

    return random_spline_ts(min_value, max_value, n_points, n_base_points,
                            n_points_between_base_points)
    #n_dims = 3
    # return multidim_spline_ts(min_value, max_value, n_points, n_base_points,
    # n_points_between_base_points, n_dims, with_plots)


def start_polynomial(n_points=1250):

    min_value = 0
    max_value = 100
    #n_points = 1250
    n_base_points = 20  # 5  # 200  # 50
    # poly_order = 18  # 3  # 18  # 8
    min_order = 5
    max_order = 20
    with_plots = False
    return random_polynomial(min_value, max_value, n_points,
                             n_base_points, min_order, max_order, with_plots)


def start_linear(n_points=1250):
    min_value = 0
    max_value = 100
    #n_points = 1250
    with_plots = False
    return linear(min_value, max_value, n_points, with_plots)


def start_sine(n_points=1250):
    min_value = 0
    max_value = 100
    #n_points = 1250
    with_plots = False
    return sine(min_value, max_value, n_points, with_plots)


def start_mixture(dims=3, n_points=12500, seed=None, min_value=100, max_value=200):
    if seed is not None:
        np.random.seed(seed)
    available_base_fcts = ['linear', 'sine', 'polynomial', 'spline']
    # functions that can be mixed to other functions
    mixture_fcts = np.concatenate((available_base_fcts, ['noise']))

    mix_within_dim = True
    mix_between_dims = True
    with_plots = False
    n_dims = dims
    return multidim_section_functions(n_dims, n_points, available_base_fcts, mixture_fcts,
                                      min_value, max_value, mix_within_dim, mix_between_dims, with_plots)


# =============================================================================
# main function #
# =============================================================================
    # what is the Wertebereich (range) of polynomial in given interval
if __name__ == '__main__':
    # np.random.seed(24)  # (3)  # (543)

    # trial()
    # polynomial()
    # with_spline()

    # real-world
    # prepare_real_world_series()
    # output = real_word_series("wnd_spd")

    # champanger()
    # print(len(output))

    # start_splines()
    # start_polynomial()
    # start_linear()
    # start_sine()
    start_mixture(1)

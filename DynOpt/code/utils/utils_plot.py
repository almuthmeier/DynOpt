'''
Functionality to plot the results obtained by dynea or dynpso, e.g. the best 
individuals or their fitness.

Created on Jan 18, 2018

@author: ameier
'''
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness(best_fitness_evals):
    '''
    Plots for each generation the best found fitness value
    @param best_fitness_evals: contains the best fitness of each iteration 
    '''
    plt.figure()
    plt.subplot(111)
    x_values = np.arange(0, best_fitness_evals.size, 1)
    plt.plot(x_values[:1000], best_fitness_evals[:1000])
    plt.plot(x_values[1000:], best_fitness_evals[1000:], color='green')

    plt.title("Best fitness value of each generation")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


def plot_best_ind_per_gen(best_individuals):
    '''
    Plots for each generation the best individual found (for 2-dimensional
    individuals).
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = range(len(best_individuals))

    # colors visualizes order:
    # https://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
    ttt = ax.scatter(best_individuals[:, 0], best_individuals[:, 1],
                     marker='x', c=color)
    #==========
    # other color schema
    # https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
    #color = range(len(best_individuals))
    # ttt = ax.scatter(best_individuals[:, 0], best_individuals[:, 1],
    # marker='x', c=color, cmap=plt.cm.get_cmap('RdYlBu'))  # cm.coolwarm)
    #===========
    # same scaling for both axis (by this means, sine transformed individuals are showed correctly
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_aspect.html
    ax.set_aspect('equal')
    #============
    # https://stackoverflow.com/questions/26034777/matplotlib-same-height-for-colorbar-as-for-plot
    #  divider = make_axes_locatable(ax)
    # cax1 = divider.append_axes("right", size="100%", pad=5.0)

    cb = plt.colorbar(ttt)
    cb.set_label('generation number')
    plt.title("Best individuals per generation")
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.show()


def plot_points(points, title):
    '''
    Plots the passed 2-dimensional points.
    @param points: contains solutions in the solution space; one for each change
     (could be best individual per change, real optimum position per change...) 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = range(len(points))
    ttt = ax.scatter(points[:, 0], points[:, 1],
                     marker='x', c=color)
    cb = plt.colorbar(ttt)
    # same scaling for both axis (by this means, sine transformed individuals are showed correctly
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_aspect.html
    ax.set_aspect('equal')
    cb.set_label('change number')
    plt.title(title)
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.show()


def plot_diff_pred_and_optimum(diff_pred_and_opt, title_addition, diff_found_and_opt=None):
    '''
    Difference between predicted and real optimum is plotted per change.
    If desired, the difference between the found and the real optimum is 
    plotted as well. 
    @param diff_pred_and_opt: Euclidean difference between predicted and real optimum
    @param title_addition: string, makes the title more accurate
    @param diff_found_and_opt: Euclidean difference between found and real optimum
    '''
    changes = np.arange(len(diff_pred_and_opt))
    plt.figure()
    plt.subplot(111)
    plt.plot(changes, diff_pred_and_opt, color='green')
    if not diff_found_and_opt is None:
        plt.plot(changes, diff_found_and_opt, color='blue')

    title = "Difference between predicted (green) or found (blue) and real optimum " + \
        title_addition
    plt.title(title)
    plt.xlabel('Change')
    plt.ylabel('Euclidean difference')
    plt.show()

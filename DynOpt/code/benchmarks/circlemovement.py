'''
Implements the optimum movement along a circle. (only for 2 dimensions) 

Is called from dynposbenchmark.py

Created on Aug 4, 2018

@author: ameier
'''
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np


def plot_movement(points):
    '''
    Visualization of all movement points.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = range(len(points))

    # colors visualizes order:
    # https://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
    ttt = ax.scatter(points[:, 0], points[:, 1],
                     marker='x', c=color)
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


def rotate(degree, point, rotation_center):
    '''
    Rotate one point (vector) with desired degree around another point 
    (rotation_center).
    '''
    # convert from degree to radians
    degree = np.deg2rad(degree)
    # rotation matrix
    rot_matrix = [[np.cos(degree), -np.sin(degree)],
                  [np.sin(degree),  np.cos(degree)]]
    # transform, rotate, re-transform
    transformed_point = point - rotation_center
    rotated_point = np.dot(transformed_point, rot_matrix)
    retransformed_rotated_point = rotated_point + rotation_center
    return retransformed_rotated_point


def create_circle_movement_points(distance, n_points, orig_glob_opt_position=None):
    '''
    Creates first two points and the following ones by rotation of the second 
    last point around the last point in order to get a circle as movement
    pattern.
    @param distance: Euclidean distance between two optimum positions
    @param n_points: number of optima on one circle
    @param orig_glob_opt_position: first optimium point
    @return: list of optimum positions
    '''
    # =========================================================================
    # inner angle at each point of the polygon
    degree = ((n_points - 2) * 180) / n_points
    # =========================================================================
    # for printing
    points = []
    # =========================================================================
    # first point
    if orig_glob_opt_position is None:
        first = np.array([2, 4])
    else:
        first = orig_glob_opt_position
    points.append(first)
    # =========================================================================
    # second point

    # point to get a arbitrary direction
    direction_giving_point = first + [2, 2]
    # vector between this and the first point + normalization
    v = first - direction_giving_point
    v_normalized = preprocessing.normalize(
        v.reshape(1, -1), norm='l2')[0]  # only one element
    # second point with desired distance
    second = first + distance * v_normalized
    points.append(second)
    # =========================================================================
    # remaining points
    for i in range(2, n_points):
        # rotate second last point with last point as rotation center
        to_rotate = points[i - 2]
        rotation_center = points[i - 1]
        new_point = rotate(degree, to_rotate, rotation_center)
        points.append(new_point)

    return points


if __name__ == '__main__':
    # parameters
    distance = 0.5
    n_points = 100
    orig_glob_opt_position = np.array([2, 4])

    points = create_circle_movement_points(
        distance, n_points, orig_glob_opt_position)
    plot_movement(np.array(points))

'''
Created on Nov 1, 2017

@author: ameier
'''


import os
import unittest

from matplotlib import cm

from benchmarks.mpb import compute_fitness, __create_and_save_mpb_problem__
from mpl_toolkits.mplot3d import Axes3D  # necessary for "projection='3d'"
import numpy as np
from utils.utils_plot import plot_points


class Test(unittest.TestCase):

    def setUp(self):
        self.path_test_problems = os.path.abspath(
            os.pardir) + "/tests/test_datasets/"

        '''
        Create two instances of MPB-Function with same parameters.
        '''
        self.n_chg_periods = 12
        self.n_dims = 2
        self.n_peaks = 3
        self.len_chg_period = 4
        self.len_movement_vector = 0.6
        self.f1_name = self.path_test_problems + "f1.npz"
        self.f2_name = self.path_test_problems + "f2.npz"
        self.f3_name = self.path_test_problems + "f3.npz"

        # create first time
        self.np_random_generator = np.random.RandomState(3)
        self.np_peaks_random_generator = np.random.RandomState(55)
        __create_and_save_mpb_problem__(self.n_chg_periods, self.n_dims, self.n_peaks, self.len_movement_vector,
                                        self.np_random_generator, self.np_peaks_random_generator, self.f1_name, noise=None)
        # create second time
        self.np_random_generator = np.random.RandomState(3)
        self.np_peaks_random_generator = np.random.RandomState(55)
        __create_and_save_mpb_problem__(self.n_chg_periods, self.n_dims, self.n_peaks, self.len_movement_vector,
                                        self.np_random_generator, self.np_peaks_random_generator, self.f2_name, noise=None)
        # create third time (with noise)
        self.np_random_generator = np.random.RandomState(3)
        self.np_peaks_random_generator = np.random.RandomState(55)
        __create_and_save_mpb_problem__(self.n_chg_periods, self.n_dims, self.n_peaks, self.len_movement_vector,
                                        self.np_random_generator, self.np_peaks_random_generator, self.f3_name, noise=0.1)

        self.f1 = np.load(self.f1_name)
        self.f2 = np.load(self.f2_name)
        self.f3 = np.load(self.f3_name)

        self.heights1 = self.f1['heights']
        self.heights2 = self.f2['heights']
        self.widths1 = self.f1['widths']
        self.widths2 = self.f2['widths']
        self.positions1 = self.f1['positions']
        self.positions2 = self.f2['positions']
        self.global_opt_fit1 = self.f1['global_opt_fit_per_chgperiod']
        self.global_opt_fit2 = self.f2['global_opt_fit_per_chgperiod']
        self.global_opt_fit3 = self.f3['global_opt_fit_per_chgperiod']
        self.global_opt_pos1 = self.f1['global_opt_pos_per_chgperiod']
        self.global_opt_pos2 = self.f2['global_opt_pos_per_chgperiod']
        self.global_opt_pos3 = self.f3['global_opt_pos_per_chgperiod']

    def tearDown(self):
        '''
        Delete the test instances of MPB-Function.
        '''
        os.remove(self.f1_name)
        os.remove(self.f2_name)
        os.remove(self.f3_name)

    def testReproduciblity(self):
        '''
        Tests whether two test instances have same values, if they are created 
        with same parameters.
        '''
        # exactly same problem values
        np.testing.assert_array_equal(self.heights1, self.heights2)
        np.testing.assert_array_equal(self.widths1, self.widths2)
        np.testing.assert_array_equal(self.positions1, self.positions2)
        np.testing.assert_array_equal(
            self.global_opt_fit1, self.global_opt_fit2)
        np.testing.assert_array_equal(
            self.global_opt_pos1, self.global_opt_pos2)

        # changes correctly?
        self.assertEqual(len(self.global_opt_fit1), self.n_chg_periods)
        self.assertEqual(len(np.unique(self.global_opt_fit1)),
                         self.n_chg_periods)

        # correct number peaks?
        self.assertEqual(len(self.positions1), self.n_peaks)

        # correct dimensionality?
        self.assertEqual(len(self.global_opt_pos1[0]), self.n_dims)

    def test_compute_fitness(self):
        '''
        Tests whether previously computed fitness of global optima is same as
        computed by the normal fitness function
        '''
        for g in range(self.n_chg_periods):
            x = self.global_opt_pos1[g]
            act = compute_fitness(x, g, self.heights1,
                                  self.widths1, self.positions1)
            exp = self.global_opt_fit1[g]
            self.assertEqual(act, exp)

    def test_noise(self):
        '''
        Tests whether noisy variant of the problems really consists different 
        values for the optimum positions while optimum fitness is equal.
        '''
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                self.global_opt_pos1, self.global_opt_pos3)
        np.testing.assert_array_equal(
            self.global_opt_fit1, self.global_opt_fit3)

    def test_global_opt(self):
        '''
        Tests whether the stored global optimum (fitness & position) really is
        the global optimum peak.
        '''
        f1 = np.load(self.f2_name)
        heights1 = f1['heights']
        widths1 = f1['widths']
        positions1 = f1['positions']
        global_opt_fit1 = f1['global_opt_fit_per_chgperiod']
        global_opt_pos1 = f1['global_opt_pos_per_chgperiod']

        # format: n_peaks, n_generations
        _, n_generations = heights1.shape

        # compute max height for each column (for each peak one row, while the
        # columns represent generations)
        max_peak_index_per_gen = np.argmax(heights1, axis=0)
        for gen in range(n_generations):
            highest_peak_hight = heights1[max_peak_index_per_gen[gen]][gen]
            highest_peak_position = positions1[max_peak_index_per_gen[gen]][gen]

            self.assertEqual(-highest_peak_hight, global_opt_fit1[gen])
            np.testing.assert_array_equal(
                highest_peak_position, global_opt_pos1[gen])

            # compute fitness
            act_fit = compute_fitness(highest_peak_position, gen,
                                      heights1, widths1, positions1)
            self.assertEqual(act_fit, global_opt_fit1[gen])

        min_range = -6
        max_range = 6
        step = 2
        X = np.arange(min_range, max_range, step)
        Y = np.arange(min_range, max_range, step)
        X, Y = np.meshgrid(X, Y)

        # test whether the fitness of the global optimum really is better than
        # some other solutions
        my_gen = 5  # test for one arbitrary generation
        rows, cols = X.shape
        for i in range(rows):
            for j in range(cols):
                individual = np.array([X[i][j], Y[i][j]])
                fit = compute_fitness(individual, my_gen,
                                      heights1, widths1, positions1)
                if fit <= global_opt_fit1[my_gen]:
                    print("smaller")
                    opt_pos = global_opt_pos1[my_gen]
                    np.testing.assert_array_equal(individual, opt_pos)

    def visual_optimum_movement_test(self):
        '''
        Plots the global optima to visually check the correctness.
        '''
        f1 = np.load(self.f3_name)
        global_opt_pos1 = f1['global_opt_pos_per_chgperiod']
        plot_points(global_opt_pos1, 'Real optimum per change')

    def plot_mpb_test(self):
        '''
        Plots MPB fitness landscape at a choosen generation (loads MPB data set)
        '''
        #==========================
        # MPB parameters
        gen = 10  # arbitrarily chosen
        f1 = np.load(self.f3_name)
        heights = f1['heights']
        widths = f1['widths']
        positions = f1['positions']
        #==========================
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # https://stackoverflow.com/questions/35346480/matplotlib-save-plot-lines-without-background-and-borders-and-transparency-8-b
        # "setting the position of the axes at the bottom left corner of the figure and filling the entire figure"
        ax.set_position([0, 0, 1, 1])
        # make background transparent for figure and axes
        fig.patch.set_alpha(0.)
        ax.patch.set_alpha(0.)

        # Coordinates
        X = np.arange(-25, 125, 1.0)
        Y = np.arange(-25, 125, 1.0)
        # jeweils ein Eintrag aus beiden ergibt eine Koordinate
        X, Y = np.meshgrid(X, Y)

        # calculate fitness for coordinates
        rows, cols = X.shape
        fitness = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                x = np.array([X[i][j], Y[i][j]])
                fitness[i][j] = compute_fitness(
                    x, gen, heights, widths, positions)

        # Plot the surface.
        ax.plot_surface(X, Y, fitness, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'fitness')
        fig.set_size_inches(10.5, 5.5)

        # Anzeigen und Speichern
        plt.show()
        self.assertTrue(True)

        #fig.savefig('mpb.pdf', bbox_inches=0, transparent=True)
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

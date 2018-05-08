'''
Files for this test have been created by dynposbenchmark.create_str_problems().

Created on Jan 18, 2018

@author: ameier
'''
import os
import unittest

from code.benchmarks.dynposbenchmark import compute_fitness
from code.utils.fitnessfunctions import rosenbrock
import matplotlib.pyplot as plt
import numpy as np


class Test(unittest.TestCase):
    def setUp(self):
        self.path_test_problems = os.path.abspath(
            os.pardir) + "/tests/test_datasets/"

    def test_create_str_problems(self):
        '''
        Only a visual test (for arbitrary data set).

        Plots for each change the first two dimensions of the global optimum.
        '''

        file_name = "rosenbrock_d-2_chgs-10000_pch-sine_fch-none_2018-05-07_15:53.npz"
        file_path = self.path_test_problems + "EvoStar_2018/rosenbrock/" + file_name
        prob_file = np.load(file_path)
        global_opt_pos_per_chg = prob_file['global_opt_pos_per_chg']
        prob_file.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = range(len(global_opt_pos_per_chg))
        ax.scatter(global_opt_pos_per_chg[:, 0], global_opt_pos_per_chg[:, 1],
                   marker='x', c=color)
        plt.title('Optimum position during time')
        plt.xlabel('1st dimension')
        plt.ylabel('2nd dimension')
        plt.show()

    def test_same_sine_movement(self):
        '''
        Tests whether the movement for different functions is the same.
        It should be the same since dynbposbenchmark.create_str_problems() uses 
        the same seed for each function.
        '''
        # load data
        file_name1 = "sphere_d-50_chgs-10000_pch-sine_fch-none_2018-05-07_15:53.npz"
        file_name2 = "rosenbrock_d-50_chgs-10000_pch-sine_fch-none_2018-05-07_15:53.npz"
        dim = 50

        f1 = np.load(self.path_test_problems +
                     "EvoStar_2018/sphere/" + file_name1)
        f2 = np.load(self.path_test_problems +
                     "EvoStar_2018/rosenbrock/" + file_name2)
        global_opt_fit1 = f1['global_opt_fit_per_chg']
        global_opt_pos1 = f1['global_opt_pos_per_chg']
        orig_opt_pos1 = f1['orig_global_opt_pos']
        global_opt_fit2 = f2['global_opt_fit_per_chg']
        global_opt_pos2 = f2['global_opt_pos_per_chg']
        orig_opt_pos2 = f2['orig_global_opt_pos']
        f1.close()
        f2.close()

        # test equality of optimum movement (for that subtraction of original
        # optimum position necessary)
        tolerance = 0.01
        np.testing.assert_array_almost_equal(
            global_opt_pos1 - orig_opt_pos1, global_opt_pos2 - orig_opt_pos2, tolerance)
        np.testing.assert_array_equal(global_opt_fit1, global_opt_fit2)
        np.testing.assert_array_equal(orig_opt_pos1, np.array(dim * [0]))
        np.testing.assert_array_equal(orig_opt_pos2, np.array(dim * [1]))

    def test_linear_movement(self):
        '''
        Tests whether the linear movement is created by adding the value 5 to
        the old optimum.
        For EvoStar 2018 equality with value 2 has to be tested.
        '''
        # load data
        file_name1 = "sphere_d-50_chgs-10000_pch-linear_fch-none_2018-05-07_15:53.npz"
        file_name2 = "rosenbrock_d-50_chgs-10000_pch-linear_fch-none_2018-05-07_15:53.npz"
        f1 = np.load(self.path_test_problems +
                     "GECCO_2018/sphere/" + file_name1)
        f2 = np.load(self.path_test_problems +
                     "GECCO_2018/rosenbrock/" + file_name2)
        global_opt_pos1 = f1['global_opt_pos_per_chg']
        orig_opt_pos1 = f1['orig_global_opt_pos']
        global_opt_pos2 = f2['global_opt_pos_per_chg']
        orig_opt_pos2 = f2['orig_global_opt_pos']
        f1.close()
        f2.close()

        # test equality
        self.assertTrue(np.sum((global_opt_pos1 - orig_opt_pos1) % 5) == 0)
        self.assertTrue(np.sum((global_opt_pos2 - orig_opt_pos2) % 5) == 0)

    def test_compute_fitness(self):
        '''
        Tests whether the fitness is computed correctly
        '''
        ff = rosenbrock
        f_name = "rosenbrock"
        file_name1 = "rosenbrock_d-2_chgs-10000_pch-sine_fch-none_2018-05-07_15:53.npz"
        f1 = np.load(self.path_test_problems +
                     "EvoStar_2018/rosenbrock/" + file_name1)
        global_opt_pos1 = f1['global_opt_pos_per_chg']
        orig_opt_pos1 = f1['orig_global_opt_pos']
        f1.close()

        # assume following data
        ind = np.array([4.2, 56789])
        dim = 2
        gen = 34
        current_change_period = 3  # beginning with 0, every 10th generation a change

        # mock the following array
        global_opt_pos_per_gen = np.zeros((50, dim))
        global_opt_pos_per_gen[gen] = global_opt_pos1[current_change_period]

        # compute fitness manually
        moved_ind = ind - (global_opt_pos_per_gen[gen] - orig_opt_pos1)
        exp_fit = ff(moved_ind)

        # compare with function
        act_fit = compute_fitness(ind, gen, f_name,
                                  global_opt_pos_per_gen, orig_opt_pos1)
        self.assertEqual(exp_fit, act_fit)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_create_str_problems']
    unittest.main()

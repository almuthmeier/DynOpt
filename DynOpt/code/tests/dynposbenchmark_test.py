'''
TODO use also circle movement in tests?
Files for this test have been created by dynposbenchmark.create_problems().

Created on Jan 18, 2018

@author: ameier
'''


import os
import sys
import unittest

sys.path.append(os.path.abspath(os.pardir))

from benchmarks.dynposbenchmark import compute_fitness
import matplotlib.pyplot as plt
import numpy as np
from utils.fitnessfunctions import rosenbrock, sphere, rastrigin





class Test(unittest.TestCase):
    def setUp(self):
        self.path_test_problems = os.path.abspath(
            os.pardir) + "/tests/test_datasets/"

        # path to DynOpt
        path_to_dynopt = '/'.join(os.path.abspath(os.pardir).split('/')[:-1])
        self.path_test_problems = path_to_dynopt + "/datasets/"

    def test_create_problems(self):
        '''
        Only a visual test (for arbitrary data set).

        Plots for each change the first two dimensions of the global optimum.
        '''

        file_name = "rosenbrock_d-2_chgperiods-10000_pch-sine_fch-none_2018-05-09_11:13.npz"
        file_path = self.path_test_problems + "EvoStar_2018/rosenbrock/" + file_name
        prob_file = np.load(file_path)
        global_opt_pos_per_chgperiod = prob_file['global_opt_pos_per_chgperiod']
        prob_file.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = range(len(global_opt_pos_per_chgperiod))
        ax.scatter(global_opt_pos_per_chgperiod[:, 0], global_opt_pos_per_chgperiod[:, 1],
                   marker='x', c=color)
        plt.title('Optimum position during time')
        plt.xlabel('1st dimension')
        plt.ylabel('2nd dimension')
        plt.show()

    def test_same_sine_movement(self):
        '''
        Tests whether the movement for different functions is the same.
        It should be the same since dynbposbenchmark.create_problems() uses 
        the same seed for each function.

        Not only for sine movement type, also for mixture ...
        '''
        # load data
        #file_name1 = "sphere_d-50_chgs-10000_pch-sine_fch-none_2018-05-07_15:53.npz"
        #file_name2 = "rosenbrock_d-50_chgs-10000_pch-sine_fch-none_2018-05-07_15:53.npz"
        file_name1 = "sphere_d-20_chgperiods-100_pch-sine_fch-none_2018-06-15_21:27.npz"
        file_name2 = "rosenbrock_d-20_chgperiods-100_pch-sine_fch-none_2018-06-15_21:27.npz"

        file_name1 = "sphere_d-2_chgperiods-10000_pch-mixture_fch-none_2018-11-17_10:15.npz"
        file_name2 = "rosenbrock_d-2_chgperiods-10000_pch-mixture_fch-none_2018-11-17_10:15.npz"
        dim = 20

        f1 = np.load(self.path_test_problems +
                     "MyTest/sphere/" + file_name1)
        f2 = np.load(self.path_test_problems +
                     "MyTest/rosenbrock/" + file_name2)
        global_opt_fit1 = f1['global_opt_fit_per_chgperiod']
        global_opt_pos1 = f1['global_opt_pos_per_chgperiod']
        orig_opt_pos1 = f1['orig_global_opt_pos']
        global_opt_fit2 = f2['global_opt_fit_per_chgperiod']
        global_opt_pos2 = f2['global_opt_pos_per_chgperiod']
        orig_opt_pos2 = f2['orig_global_opt_pos']
        f1.close()
        f2.close()

        # test equality of optimum movement (for that subtraction of original
        # optimum position necessary)
        tolerance = 0.01
        np.testing.assert_array_almost_equal(
            global_opt_pos1 - orig_opt_pos1, global_opt_pos2 - orig_opt_pos2, tolerance)
        np.testing.assert_array_equal(global_opt_fit1, global_opt_fit2)
        global_opt_pos1
        np.testing.assert_array_equal(orig_opt_pos1, global_opt_pos1[0])
        np.testing.assert_array_equal(orig_opt_pos2, global_opt_pos2[0])
        #np.testing.assert_array_equal(orig_opt_pos1, np.array(dim * [0]))
        #np.testing.assert_array_equal(orig_opt_pos2, np.array(dim * [1]))

    def test_linear_movement(self):
        '''
        Tests whether the linear movement is created by adding the value 5 to
        the old optimum.
        For EvoStar 2018 equality with value 2 has to be tested.
        '''
        # load data
        file_name1 = "sphere_d-20_chgperiods-100_pch-linear_fch-none_2018-06-15_21:27.npz"
        file_name2 = "rosenbrock_d-20_chgperiods-100_pch-linear_fch-none_2018-06-15_21:27.npz"
        f1 = np.load(self.path_test_problems +
                     "MyTest/sphere/" + file_name1)
        f2 = np.load(self.path_test_problems +
                     "MyTest/rosenbrock/" + file_name2)
        global_opt_pos1 = f1['global_opt_pos_per_chgperiod']
        orig_opt_pos1 = f1['orig_global_opt_pos']
        global_opt_pos2 = f2['global_opt_pos_per_chgperiod']
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
        # ---------------------------------------------------------------------
        # prepare data
        # ---------------------------------------------------------------------

        ff = rastrigin
        f_name = "rastrigin"
        file_name = "rastrigin_d-10_chgperiods-10000_pch-mixture_fch-none_2018-12-05_09:31.npz"
        file_path = "/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/code/tests/test_datasets/"
        f1 = np.load(file_path + file_name)
        global_opt_pos1 = f1['global_opt_pos_per_chgperiod']
        orig_opt_pos1 = f1['orig_global_opt_pos']
        global_opt_fit_per_chgperiod = f1['global_opt_fit_per_chgperiod']
        f1.close()

        # assume following data
        dim = 10
        gen = 34
        current_change_period = 0  # beginning with 0, every 10th generation a change

        # mock the following array
        n_gens = 50
        global_opt_pos_per_gen = np.zeros((n_gens, dim))
        global_opt_pos_per_gen[gen] = global_opt_pos1[current_change_period]

        # ---------------------------------------------------------------------
        # check whether fitness of global best solution always is 0 (the global
        # best fitness)
        # ---------------------------------------------------------------------

        # first change period (fitness)
        ind = global_opt_pos1[current_change_period]
        exp_fit = global_opt_fit_per_chgperiod[current_change_period]
        act_fit = compute_fitness(ind, gen, f_name,
                                  global_opt_pos_per_gen, orig_opt_pos1)
        self.assertEqual(exp_fit, act_fit)

        # ---------------------------------------------------------------------
        # compute fitness manually
        ind = np.array([4.2, 56789, 23, 12, 6, 4.4, 5, 3, 2, 10])
        moved_ind = ind - (global_opt_pos_per_gen[gen] - orig_opt_pos1)
        exp_fit = ff(moved_ind)

        # compare with function
        act_fit = compute_fitness(ind, gen, f_name,
                                  global_opt_pos_per_gen, orig_opt_pos1)
        self.assertEqual(exp_fit, act_fit)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_create_problems']
    unittest.main()

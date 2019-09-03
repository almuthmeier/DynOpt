'''
Created on Sep 3, 2019

@author: ameier
'''
import os
import unittest


from benchmarks.sine_generator import generate_sine_fcts_for_multiple_dimensions,\
    compute_vals_for_fct
import numpy as np


class Test(unittest.TestCase):

    def setUp(self):
        # path to DynOpt
        path_to_dynopt = '/'.join(os.path.abspath(os.pardir).split('/')[:-1])
        path_test_problems = path_to_dynopt + "/datasets/test_datasets/"

        self.n_chg_periods = 200
        self.dims = 2
        seed = 45
        n_base_time_points = 100
        l_bound = 0
        u_bound = 100
        desired_curv = 10
        desired_med_vel = 0.5
        max_n_functions = 4

        self.data, self.params, self.step_size = generate_sine_fcts_for_multiple_dimensions(self.dims, self.n_chg_periods, seed, n_base_time_points,
                                                                                            l_bound, u_bound, desired_curv,
                                                                                            desired_med_vel, max_n_functions)

        self.ds_file_name = path_test_problems + "dsb_2019-09-03.npz"
        global_opt_fit = np.array(self.n_chg_periods * [0])
        orig_global_opt_position = self.data[0]
        np.savez(self.ds_file_name, global_opt_fit_per_chgperiod=global_opt_fit,
                 global_opt_pos_per_chgperiod=self.data, orig_global_opt_pos=orig_global_opt_position,
                 fcts_params_per_dim=self.params,
                 step_size=self.step_size)

    def testParamsReproducible(self):
        '''
        Tests whether the stored optimum positions can be reproduced by the 
        stored parameterization values. 
        '''
        file = np.load(self.ds_file_name)
        global_opt_fit_per_chgperiod = file['global_opt_fit_per_chgperiod']
        global_opt_pos_per_chgperiod = file['global_opt_pos_per_chgperiod']
        fcts_params_per_dim = file['fcts_params_per_dim']
        s_size = file['step_size']
        file.close()

        time_steps = np.array([i * s_size for i in range(self.n_chg_periods)])
        pos_per_dim = []
        for d in range(self.dims):
            pos_per_dim.append(compute_vals_for_fct(
                fcts_params_per_dim[d], time_steps))
        reproduced_positions = np.transpose(pos_per_dim)
        np.testing.assert_array_almost_equal(
            reproduced_positions, global_opt_pos_per_chgperiod)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testParamsReproducible']
    unittest.main()

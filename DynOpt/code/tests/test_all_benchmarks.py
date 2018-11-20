'''
Tests properties that all data set files have in common.

Created on May 8, 2018

@author: ameier
'''
import os
import unittest

import numpy as np


class Test(unittest.TestCase):

    def setUp(self):
        # path to DynOpt
        path_to_dynopt = '/'.join(os.path.abspath(os.pardir).split('/')[:-1])
        self.path_test_problems = path_to_dynopt + "/datasets/"

    def tearDown(self):
        pass

    def test_dataset_properites(self):
        '''
        Tests whether the data set files certain arrays, the arrays have the 
        same length and whether the values are stored per change or per generation 
        '''
        # for all sub-folders in self.pyth_test_problems and its sub-folders do
        # the following tests
        for ppath, _, files in os.walk(self.path_test_problems):  # ppath, subdirs
            for name in files:
                abs_file_path = os.path.join(ppath, name)
                file = np.load(abs_file_path)

                try:
                    # =========================================================
                    # test existence of certain variables in the file
                    # this part throws exceptions if the variables do not exist
                    msg = "global_opt_fit_per_chgperiod"
                    global_opt_fit = file['global_opt_fit_per_chgperiod']

                    msg = "global_opt_pos_per_chgperiod"
                    global_opt_pos = file['global_opt_pos_per_chgperiod']

                    msg = "orig_global_opt_pos"
                    orig_pos = file['orig_global_opt_pos']

                    # test whether the first global optimum is the same as in
                    # the array storing the global optimum for each change
                    np.testing.assert_array_almost_equal(
                        orig_pos, global_opt_pos[0], 2)  # accuracy 2

                    # =========================================================
                    # test array lengths
                    self.assertEqual(len(global_opt_fit), len(global_opt_pos))

                    # =========================================================
                    # test whether entries are per change or per generation
                    # by testing whether same positions appear successively

                    if "sphere" in abs_file_path or "rosenbrock" in abs_file_path \
                            or "rastrigin" in abs_file_path or "griewank" in abs_file_path:
                        for i in range(1, len(global_opt_pos)):
                            are_not_equal = np.any(np.not_equal(
                                global_opt_pos[i - 1], global_opt_pos[i]))
                            np.testing.assert_equal(are_not_equal, True)
                    elif ("mpbrand" in abs_file_path or "mpbnoisy" in abs_file_path or
                          "mpbcorr" in abs_file_path):  # TODO(dev)
                        # For MPB the optimum may remain the same in successive
                        # change periods and change suddenly after some changes.
                        # Therefore this benchmark is not testable
                        pass
                except AssertionError:
                    print("false " + msg + " field in file " + abs_file_path)
                    raise
                except KeyError:
                    print("no " + msg + " field in file " + abs_file_path)
                    raise
                file.close()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

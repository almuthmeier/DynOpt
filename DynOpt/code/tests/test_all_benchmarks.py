'''
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

    def testName(self):
        # zu testen
        # - Vorhandensein von Variablen mit entsprechenden Namen
        # - nicht gleiche Werte hintereinander

        # for all subfolders in self.pyth_test_problems and its subfolders do
        # the following tests

        for ppath, subdirs, files in os.walk(self.path_test_problems):
            for name in files:
                abs_file_path = os.path.join(ppath, name)
                # if abs_file_path.endswith('.npz'):
                #    pass
                # if "mpb" in abs_file_path:
                #    print("mpb")

                print("hello")
                file = np.load(abs_file_path)
                #msg = "global_opt_fit_per_chgperiod"
                # file['global_opt_fit_per_chgperiod']
                try:
                    msg = "global_opt_fit_per_chgperiod"
                    file['global_opt_fit_per_chgperiod']

                    msg = "global_opt_pos_per_chgperiod"
                    all_positions = file['global_opt_pos_per_chgperiod']

                    msg = "orig_global_opt_pos"
                    orig_pos = file['orig_global_opt_pos']

                    np.testing.assert_array_almost_equal(
                        orig_pos, all_positions[0], 4)
                except AssertionError:
                    print("false " + msg + " field in file " + abs_file_path)
                    # print(ae)
                    raise
                except KeyError:
                    print("no " + msg + " field in file " + abs_file_path)
                file.close()

            #self.assertEqual(1, 6)
        # pop_file_name = [f for f in listdir(mpb_problems_folder) if (isfile(join(
        #    mpb_problems_folder, f)) and f.endswith('.npz') and experiment_name in f
        # and ("_d-" + str(dim) + "_") in f and ("noise-" + str(noise)) in f)]


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

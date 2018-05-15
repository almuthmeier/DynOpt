'''
Created on Oct 5, 2017

@author: ameier
'''
import unittest


from comparison import PredictorComparator
import numpy as np


class Test(unittest.TestCase):

    def setUp(self):
        # mock a comparator object
        self.comparator = PredictorComparator()
        np.random.seed(4)

    def test_get_chgperiods_for_gens(self):
        # =====================================================================
        # deterministic changes
        self.comparator.chgperiods = 5
        self.comparator.ischgperiodrandom = False
        self.comparator.lenchgperiod = 6
        exp_periods_for_gens = np.array([0, 0, 0, 0, 0, 0,
                                         1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2,
                                         3, 3, 3, 3, 3, 3,
                                         4, 4, 4, 4, 4, 4])

        periods_for_gens = self.comparator.get_chgperiods_for_gens()
        print(periods_for_gens)
        np.testing.assert_array_equal(periods_for_gens, exp_periods_for_gens)
        act_n_periods = len(np.unique(periods_for_gens))
        self.assertEqual(act_n_periods, self.comparator.chgperiods)

        # =====================================================================
        # no changes
        self.comparator.chgperiods = 1
        self.comparator.ischgperiodrandom = False
        self.comparator.lenchgperiod = 6
        exp_periods_for_gens = np.array([0, 0, 0, 0, 0, 0])

        periods_for_gens = self.comparator.get_chgperiods_for_gens()
        print(periods_for_gens)
        np.testing.assert_array_equal(periods_for_gens, exp_periods_for_gens)
        act_n_periods = len(np.unique(periods_for_gens))
        self.assertEqual(act_n_periods, self.comparator.chgperiods)

        # =====================================================================
        # random changes
        self.comparator.chgperiods = 4
        self.comparator.ischgperiodrandom = True
        self.comparator.lenchgperiod = 10

        periods_for_gens = self.comparator.get_chgperiods_for_gens()
        print(periods_for_gens)
        self.assertTrue(np.any(periods_for_gens == 0))
        self.assertTrue(np.any(periods_for_gens == 1))
        self.assertTrue(np.any(periods_for_gens == 2))
        self.assertTrue(np.any(periods_for_gens == 3))
        act_n_periods = len(np.unique(periods_for_gens))
        self.assertEqual(act_n_periods, self.comparator.chgperiods)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

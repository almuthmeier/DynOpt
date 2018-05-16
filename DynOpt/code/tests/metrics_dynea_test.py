'''
Created on Oct 6, 2017

@author: ameier
'''
from _collections import OrderedDict
import copy
import unittest

from matplotlib import cm
from matplotlib import colors

import matplotlib.pyplot as plt
from metrics.metrics_dynea import arr, best_error_before_change,\
    avg_best_of_generation, conv_speed, normalized_bog, __convergence_speed__
import numpy as np


# from dynamicopt.metrics.metrics_dynea import arr, normalized_bog, best_error_before_change,\
#    avg_best_of_generation, __convergence_speed__, conv_speed
class Test(unittest.TestCase):

    def test_arr(self):
        #======================================================================
        # test case 0a) fitness becomes worse in period 2
        generations_of_changes = {0: [0, 1, 2, 3]}
        optima_of_changes = {0: 3}

        best_fitness_evals = np.array([5, 9, 4, 4])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        print("case 0a): act_arr ", act_arr)
        # compute ARR manually
        # change 0: (0+4-1-1) / 4*(3-5) = 2/-8 (for ARR without math.abs)
        # change 0: (0+4+1+1) / 4*(2) = 6/8 (for ARR with math.abs)
        exp_arr = (2 / -8) / 1
        exp_arr = 6 / 8
        self.assertEqual(act_arr, exp_arr)  # TODO evtl. auskommentieren
        print("case 0: ", exp_arr)

        #======================================================================
        # test case 0b)  Fitness increases continuously
        generations_of_changes = {0: [0, 1, 2, 3]}
        optima_of_changes = {0: 3}

        best_fitness_evals = np.array([8, 5, 4, 4])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        exp_arr = (0 - 3 - 4 - 4) / (-20)
        self.assertEqual(act_arr, exp_arr)
        print("case 0b: ", exp_arr)
        #======================================================================
        # test case 1
        generations_of_changes = {0: [0, 1]}
        optima_of_changes = {0: 5}

        best_fitness_evals = np.array([8, 6])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        # change 1: (0-2) / 2*(5-8) = -2/-6
        exp_arr = (-2 / -6) / 1
        self.assertEqual(act_arr, exp_arr)
        print("case 1: ", exp_arr)

        #======================================================================
        # test case 2
        generations_of_changes = {0: [0, 1, 2]}
        optima_of_changes = {0: 7}

        best_fitness_evals = np.array([7, 7, 7])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        # change 2: (0+0+0) / 3*(7-7) = 0/0 (math error) -> 1
        exp_arr = 1
        self.assertEqual(act_arr, exp_arr)
        print("case 2: ", exp_arr)

        #======================================================================
        # test case 3
        generations_of_changes = {0: [0]}
        optima_of_changes = {0: 2}

        best_fitness_evals = np.array([6])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        # change 3: (0) / 1*(2-6) = 0
        exp_arr = 0
        self.assertEqual(act_arr, exp_arr)
        print("case 3: ", exp_arr)

        #======================================================================
        # test case 4
        generations_of_changes = {0: [0, 1, 2]}
        optima_of_changes = {0: 5}

        best_fitness_evals = np.array([7, 7, 7])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        # change 4: (0+0+0) / 3*(2-7) = 0
        exp_arr = 0
        self.assertEqual(act_arr, exp_arr)
        print("case 4: ", exp_arr)

        #======================================================================
        # test case 5
        generations_of_changes = {0: [0]}
        optima_of_changes = {0: 66}

        best_fitness_evals = np.array([66])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        # change 5: 0 / 1*(0) = 0/0 (math error) -> 1
        exp_arr = 1
        self.assertEqual(act_arr, exp_arr)
        print("case 5: ", exp_arr)

        #======================================================================
        # test case 6
        generations_of_changes = {
            0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7, 8], 3: [9], 4: [10, 11, 12], 5: [13]}
        # dictionary containing the position of the global optimum for all
        # changes
        optima_of_changes = {0: 3, 1: 5, 2: 7, 3: 2, 4: 5, 5: 66}

        best_fitness_evals = np.array([5, 9, 4, 4,
                                       8, 6,
                                       7, 7, 7,  # always optimum
                                       6,
                                       7, 7, 7,  # always non-optimum
                                       66])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        #--- ARR without math.abs
        # compute ARR manually
        # change 0: (0+4-1-1) 4*(3-5) = -2/-8
        # change 1: (0-2) / 2*(5-8) = -2/-6
        # change 2: (0+0+0) / 3*(7-7) = 0/0 (math error) -> 1
        # change 3: (0) / 1*(2-6) = 0
        # change 4: (0+0+0) / 3*(2-7) = 0
        # change 5: 0 / 1*(0) = 0/0 (math error) -> 1
        exp_arr = (2 / -8 + 2 / 6 + 1 + 0 + 0 + 1) / \
            6  # (ARR without math.abs)
        #----
        exp_arr = (6 / 8 + 2 / 6 + 1 + 0 + 0 + 1) / 6  # (ARR with math.abs)
        self.assertEqual(act_arr, exp_arr)  # TODO evtl. auskommentieren
        print("case 6: ", exp_arr)

        #======================================================================
        # # Comparison of different situations
        #======================================================================
        # test case 7a (should be worse than 7b, because slower convergence)
        generations_of_changes = {0: [0, 1, 2, 3]}
        optima_of_changes = {0: 3}

        best_fitness_evals = np.array([9, 6, 4, 3])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        exp_arr = (0 - 3 - 5 - 6) / (4 * (3 - 9))  # = -14/-24
        self.assertEqual(act_arr, exp_arr)
        print("case 7a: ", exp_arr)

        #================
        # test case 7b (should be better than 7a, because faster convergence)
        generations_of_changes = {0: [0, 1, 2, 3]}
        optima_of_changes = {0: 3}

        best_fitness_evals = np.array([9, 6, 3, 3])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        exp_arr = (0 - 3 - 6 - 6) / (4 * (3 - 9))  # = -15/-24
        self.assertEqual(act_arr, exp_arr)
        print("case 7b: ", exp_arr)

        #======================================================================
        # test case 8a (starting from high fitness and do not reach optimum)
        generations_of_changes = {0: [0, 1, 2, 3, 4, 5]}
        optima_of_changes = {0: 3}

        best_fitness_evals = np.array([34, 27, 20, 13, 10, 8])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        exp_arr = (0 - 7 - 14 - 21 - 24 - 26) / (6 * (3 - 34))
        self.assertEqual(act_arr, exp_arr)
        print("case 8a: ", exp_arr)
        #======================================================================
        # test case 8b (starting from lower and ending at lower fitness )
        # should be better than 8a
        generations_of_changes = {0: [0, 1, 2, 3, 4, 5]}
        optima_of_changes = {0: 3}

        best_fitness_evals = np.array([16, 13, 10, 7, 4, 3])
        act_arr = arr(generations_of_changes,
                      optima_of_changes, best_fitness_evals)

        # compute ARR manually
        exp_arr = (0 - 3 - 6 - 9 - 12 - 13) / (6 * (3 - 16))
        self.assertEqual(act_arr, exp_arr)
        print("case 8b: ", exp_arr)

    def examine_arr(self):
        '''
        Examines which data have a good or a poor ARR.
        '''
        gens_of_chgs = {0: [0, 1, 2, 3, 4, 5, 6]}
        optima_of_chgs = {0: 1}
        best_fit_evals_per_alg = OrderedDict()
        # immediately optimum fitness
        name = "immediately optimum fitness"
        best_fit_evals_1 = np.array([1, 1, 1, 1, 1, 1, 1])
        best_fit_evals_per_alg[name] = best_fit_evals_1
        act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_1)
        bebc = best_error_before_change(
            gens_of_chgs, optima_of_chgs,  best_fit_evals_1)
        avg_bog, _ = avg_best_of_generation(np.array([best_fit_evals_1]))
        speed = conv_speed(
            gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_1})
        print(name, ": \n    ARR ", act_arr, "\n    Speed ", speed,
              "\n    BEBC ", bebc, "\n    avg BOG ", avg_bog)

        # linear decreasing fitness
        name = "linear decreasing fitness"
        best_fit_evals_2 = np.array([13, 11, 9, 7, 5, 3, 1])
        best_fit_evals_per_alg[name] = best_fit_evals_2
        act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_2)
        bebc = best_error_before_change(
            gens_of_chgs, optima_of_chgs,  best_fit_evals_2)
        avg_bog, _ = avg_best_of_generation(np.array([best_fit_evals_2]))
        speed = conv_speed(
            gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_2})
        print(name, ": \n    ARR ", act_arr, "\n    Speed ", speed,
              "\n    BEBC ", bebc, "\n    avg BOG ", avg_bog)

        # in last step optimal fitness
        name = "in last step optimal fitness (worse fitness level at beginning)"
        best_fit_evals_3 = np.array([13, 13, 13, 13, 13, 13, 1])  # -12/(7*-12)
        best_fit_evals_per_alg[name] = best_fit_evals_3
        act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_3)
        bebc = best_error_before_change(
            gens_of_chgs, optima_of_chgs,  best_fit_evals_3)
        avg_bog, _ = avg_best_of_generation(np.array([best_fit_evals_3]))
        speed = conv_speed(
            gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_3})
        print(name, ": \n    ARR ", act_arr, "\n    Speed ", speed,
              "\n    BEBC ", bebc, "\n    avg BOG ", avg_bog)

        # in last step optimal fitness
        name = "in last step optimal fitness (better fitness level at beginning)"
        best_fit_evals_4 = np.array([7, 7, 7, 7, 7, 7, 1])  # -6/(7*-6) = 1/7
        best_fit_evals_per_alg[name] = best_fit_evals_4
        act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_4)
        bebc = best_error_before_change(
            gens_of_chgs, optima_of_chgs,  best_fit_evals_4)
        avg_bog, _ = avg_best_of_generation(np.array([best_fit_evals_4]))
        speed = conv_speed(
            gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_4})
        print(name, ": \n    ARR", act_arr, "\n    Speed ", speed,
              "\n    BEBC ", bebc, "\n    avg BOG ", avg_bog)

        # first constant then convergence
        name = "first constant then convergence"
        best_fit_evals_5 = np.array([13, 13, 13, 13, 3, 2, 1])
        best_fit_evals_per_alg[name] = best_fit_evals_5
        act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_5)
        bebc = best_error_before_change(
            gens_of_chgs, optima_of_chgs,  best_fit_evals_5)
        avg_bog, _ = avg_best_of_generation(np.array([best_fit_evals_5]))
        speed = conv_speed(
            gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_5})
        print(name, ": \n    ARR ", act_arr, "\n    Speed ", speed,
              "\n    BEBC ", bebc, "\n    avg BOG ", avg_bog)

        # first improvement, then convergence
        name = "first improvement, then convergence"
        best_fit_evals_6 = np.array([7, 7, 7, 7, 3, 2, 1])
        best_fit_evals_per_alg[name] = best_fit_evals_6
        act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_6)
        bebc = best_error_before_change(
            gens_of_chgs, optima_of_chgs,  best_fit_evals_6)
        avg_bog, _ = avg_best_of_generation(np.array([best_fit_evals_6]))
        speed = conv_speed(
            gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_6})
        print(name, ": \n    ARR ", act_arr, "\n    Speed ", speed,
              "\n    BEBC ", bebc, "\n    avg BOG ", avg_bog)

        print("AND again")
        print()
        print()
        # overall convergence speed
        overall_speed = conv_speed(
            gens_of_chgs, optima_of_chgs, best_fit_evals_per_alg)
        for alg, speed in overall_speed.items():
            print(alg, "\n    ", speed)

    def test_normalized_bog(self):
        #======================================================================
        # test case 0 (1 problem, 2 algorithms)
        avg_bog_per_alg_and_problem = {
            'alg_1': {          # best algorithm
                'p1': 1},
            'alg_2': {          # worst algorithm
                'p1': 6}
        }
        exp_norm_1 = 1  # best
        exp_norm_2 = 0  # worst

        act_norms = normalized_bog(avg_bog_per_alg_and_problem)
        act_norm_1 = act_norms['alg_1']
        act_norm_2 = act_norms['alg_2']

        self.assertEqual(act_norm_1, exp_norm_1)
        self.assertEqual(act_norm_2, exp_norm_2)

        #======================================================================
        # test case 1 (1 problem, 3 algorithms)
        avg_bog_per_alg_and_problem = {
            'alg_1': {          # best algorithm
                'p1': 1},
            'alg_2': {          # middle algorithm
                'p1': 2},
            'alg_3': {          # worst algorithm
                'p1': 9}
        }
        exp_norm_1 = 1  # best
        exp_norm_2 = 7 / 8  # middle
        exp_norm_3 = 0  # worst

        act_norms = normalized_bog(avg_bog_per_alg_and_problem)
        act_norm_1 = act_norms['alg_1']
        act_norm_2 = act_norms['alg_2']
        act_norm_3 = act_norms['alg_3']

        self.assertEqual(act_norm_1, exp_norm_1)
        self.assertEqual(act_norm_2, exp_norm_2)
        self.assertEqual(act_norm_3, exp_norm_3)

        #======================================================================
        # test case 2 (2 problem, 2 algorithms)
        avg_bog_per_alg_and_problem = {
            'alg_1': {          # one time best, one time worst
                'p1': 1, 'p2': 6},
            'alg_2': {          # same quality as alg_1
                'p1': 6, 'p2': 1}
        }
        exp_norm_1 = 0.5  # best
        exp_norm_2 = 0.5  # worst

        act_norms = normalized_bog(avg_bog_per_alg_and_problem)
        act_norm_1 = act_norms['alg_1']
        act_norm_2 = act_norms['alg_2']

        self.assertEqual(act_norm_1, exp_norm_1)
        self.assertEqual(act_norm_2, exp_norm_2)

    def test_best_error_before_change(self):
        #======================================================================
        # test case 0 (one change)
        generations_of_changes = {0: [0, 1, 2, 3]}
        optima_of_changes = {0: 3}
        best_fitness_evals = np.array([5, 9, 4, 4])

        act = best_error_before_change(
            generations_of_changes, optima_of_changes, best_fitness_evals)
        exp = 1
        self.assertEqual(act, exp)
        #======================================================================
        # test case 1 (3 changes)
        generations_of_changes = {0: [0, 1, 2, 3], 1: [4], 2: [5, 6, 7]}
        optima_of_changes = {0: 3, 1: 1, 2: 0}
        best_fitness_evals = np.array([5, 9, 4, 4,
                                       3,
                                       9, 0, 9])

        act = best_error_before_change(
            generations_of_changes, optima_of_changes, best_fitness_evals)
        exp = (1 + 2 + 0) / 3
        self.assertEqual(act, exp)
        #======================================================================
        # test case 2 (3 changes, negative fitness values)
        generations_of_changes = {0: [0, 1, 2, 3], 1: [4], 2: [5, 6, 7]}
        optima_of_changes = {0: -3, 1: 1, 2: 0}
        best_fitness_evals = np.array([5, 9, -1, -2,
                                       3,
                                       9, 0, 9])

        act = best_error_before_change(
            generations_of_changes, optima_of_changes, best_fitness_evals)
        exp = (1 + 2 + 0) / 3
        self.assertEqual(act, exp)

    def test_convergence_speed(self):
        print()
        print("test_convergence_speed")
        #======================================================================
        # test case 1) no convergence at all (fitness positive)
        gens_of_chgs = {0: [0, 1, 2, 3]}
        optima_of_chgs = {0: 3}

        best_fit_evals = np.array([8, 8, 8, 8])
        worst_fit_per_chg = {0: 8}
        act = __convergence_speed__(
            gens_of_chgs, optima_of_chgs, best_fit_evals, worst_fit_per_chg)
        exp = 1
        self.assertEqual(act, exp)

        #======================================================================
        # test case 2) no convergence at all (fitness negative)
        gens_of_chgs = {0: [0, 1, 2, 3]}
        optima_of_chgs = {0: -12}

        best_fit_evals = np.array([-2, -2, -2, -2])
        worst_fit_per_chg = {0: -2}
        act = __convergence_speed__(
            gens_of_chgs, optima_of_chgs, best_fit_evals, worst_fit_per_chg)
        exp = 1
        self.assertEqual(act, exp)
        #======================================================================
        # test case 2) positive and negative fitness
        gens_of_chgs = {0: [0, 1, 2, 3]}
        optima_of_chgs = {0: -12}

        best_fit_evals = np.array([8, 4, -1, -3])
        worst_fit_per_chg = {0: 8}
        act = __convergence_speed__(
            gens_of_chgs, optima_of_chgs, best_fit_evals, worst_fit_per_chg)
        exp = (1 * 20 + 2 * 16 + 3 * 11 + 4 * 9) / \
            (1 * 20 + 2 * 20 + 3 * 20 + 4 * 20)  # 0.605
        self.assertEqual(act, exp)

        #======================================================================
        # test case 3) direct convergence
        gens_of_chgs = {0: [0, 1, 2, 3]}
        optima_of_chgs = {0: -12}

        best_fit_evals = np.array([-12, -12, -12, -12])
        worst_fit_per_chg = {0: -12}
        act = __convergence_speed__(
            gens_of_chgs, optima_of_chgs, best_fit_evals, worst_fit_per_chg)
        exp = 0
        self.assertEqual(act, exp)

        #======================================================================

        # one generations
        gens_of_chgs = {0: [0]}
        optima_of_chgs = {0: -12}
        best_fit_evals = np.array([3])
        worst_fit_per_chg = {0: 3}
        act = __convergence_speed__(
            gens_of_chgs, optima_of_chgs, best_fit_evals, worst_fit_per_chg)
        exp = 1
        self.assertEqual(act, exp)
        #======================================================================
        # multiple changes
        gens_of_chgs = {0: [0, 1]}
        optima_of_chgs = {0: -4}

        best_fit_evals = np.array([3, -1])
        worst_fit_per_chg = {0: 3}
        act = __convergence_speed__(
            gens_of_chgs, optima_of_chgs, best_fit_evals, worst_fit_per_chg)
        exp = (13 / 21)
        self.assertEqual(act, exp)
        #======================================================================
        # multiple changes
        gens_of_chgs = {0: [0, 1, 2, 3], 1: [4], 2: [5, 6]}
        optima_of_chgs = {0: -12, 1: 5, 2: -4}

        best_fit_evals = np.array([-12, -12, -12, -12, 9, 3, -1])
        worst_fit_per_chg = {0: -12, 1: 9, 2: 3}
        act = __convergence_speed__(
            gens_of_chgs, optima_of_chgs, best_fit_evals, worst_fit_per_chg)
        exp = (0 + (4 / 4) + (13 / 21)) / 3
        self.assertEqual(act, exp)

    def test_conv_speed(self):
        # test case 1: worst and best case (one change)
        gens_of_chgs = {0: [0, 1, 2, 3]}
        optima_of_gens = [-12, -12, -12, -12]
        best_fit_evals_a = np.array([-12, -12, -12, -12])
        best_fit_evals_b = np.array([8, 8, 8, 8])
        best_fit_evals_per_alg = {'a': best_fit_evals_a, 'b': best_fit_evals_b}
        act = conv_speed(gens_of_chgs, optima_of_gens, best_fit_evals_per_alg)
        exp = {'a': 0, 'b': 1}
        self.assertEqual(act, exp)

        # test case 2: worst and best case, but now changed (two changes)
        gens_of_chgs = {0: [0, 1, 2, 3], 1: [4, 5, 6]}
        optima_of_gens = [-12, -12, -12, -12,
                          3, 3, 3]
        best_fit_evals_a = np.array([-12, -12, -12, -12,
                                     8, 8, 8])
        best_fit_evals_b = np.array([8, 8, 8, 8,
                                     3, 3, 3])
        best_fit_evals_per_alg = {'a': best_fit_evals_a, 'b': best_fit_evals_b}
        # TODO this test case fails
        act = conv_speed(gens_of_chgs, optima_of_gens, best_fit_evals_per_alg)
        exp = {'a': 0.5, 'b': 0.5}
        self.assertEqual(act, exp)

        # test case 2: "normal" fitness developments (one always better)
        gens_of_chgs = {0: [0, 1, 2, 3]}
        optima_of_gens = [4, 4, 4, 4]
        best_fit_evals_a = np.array([8, 6, 4, 4])
        best_fit_evals_b = np.array([9, 7, 5, 4])
        best_fit_evals_per_alg = {'a': best_fit_evals_a, 'b': best_fit_evals_b}
        act = conv_speed(gens_of_chgs, optima_of_gens, best_fit_evals_per_alg)
        exp_a = (1 * 4 + 2 * 2 + 0 + 0) / (5 * (1 + 2 + 3 + 4))
        exp_b = (1 * 5 + 2 * 3 + 3 * 1 + 0) / (5 * (1 + 2 + 3 + 4))
        exp = {'a': exp_a, 'b': exp_b}
        self.assertEqual(act, exp)  # -> {'a': 0.16, 'b': 0.28}

        # test case 2: "normal" fitness developments
        gens_of_chgs = {0: [0, 1, 2, 3]}
        optima_of_gens = [4, 4, 4, 4]
        best_fit_evals_a = np.array([8, 6, 4, 4])
        best_fit_evals_b = np.array([9, 5, 5, 4])
        best_fit_evals_per_alg = {'a': best_fit_evals_a, 'b': best_fit_evals_b}
        act = conv_speed(gens_of_chgs, optima_of_gens, best_fit_evals_per_alg)
        exp_a = (1 * 4 + 2 * 2 + 0 + 0) / (5 * (1 + 2 + 3 + 4))
        exp_b = (1 * 5 + 2 * 1 + 3 * 1 + 0) / (5 * (1 + 2 + 3 + 4))
        exp = {'a': exp_a, 'b': exp_b}
        self.assertEqual(act, exp)  # -> {'a': 0.16, 'b': 0.2}

    def test_published_results_GECCO(self):
        # file from GECCO experiments
        # Evo* files have other data structure
        array_path = "/home/ameier/Documents/Promotion/Ausgaben/Predictorvergleich/PSO/Pruefung_GECCO-Ausgabe/"
        array_file_a_name = "no_sphere_str_5_sine_pos_ch_none_fit_ch_lenchgperiod-20_noise-None_2018-01-25_13:48_9.npz"
        #array_file_a_name = "no_sphere_str_5_sine_pos_ch_none_fit_ch_2017-11-01_13:32_9.npz"
        array_file_a = np.load(array_path + array_file_a_name)

        array_file_b_name = "rnn_sphere_str_5_sine_pos_ch_none_fit_ch_lenchgperiod-20_noise-None_2018-01-25_13:53_9.npz"
        array_file_b = np.load(array_path + array_file_b_name)

        periods_for_generations = array_file_a['periods_for_generations']
        gens_of_chgs = {}  # {0: [0, 1, 2, 3]}
        change = 0
        gens_of_chgs[change] = [change]
        for i in range(1, len(periods_for_generations)):
            if periods_for_generations[i] != periods_for_generations[i - 1]:
                change += 1
                gens_of_chgs[change] = []
            gens_of_chgs[change].append(i)

        # {0: -12}
        # TODO: confusing name: actually this array contains optimum fitness values
        # per generation
        optima_of_chgs = array_file_a['global_opt_fit_of_changes']

        best_fit_evals_a = array_file_a['best_fitness_evals']
        best_fit_evals_b = array_file_b['best_fitness_evals']
        best_fit_evals_per_alg = {'a': best_fit_evals_a, 'b': best_fit_evals_b}

        array_file_a.close()
        array_file_b.close()

        act = conv_speed(gens_of_chgs, optima_of_chgs, best_fit_evals_per_alg)
        # "no" faster than "rnn3" (according to results in paper)
        self.assertTrue(act['a'] < act['b'])

    def examine_convergence_speed_II(self):
        '''
        Plots all possible fitness developments (for one change period) with
        specified n_gens and max_fit. 
        Color of the grahps corresponds to their convergence speed measure.
        '''
        n_gens = 2
        max_fit = 10
        gens_of_chgs = {0: [i for i in range(n_gens)]}
        optima_of_chgs = {0: 0}

        # all permutations of posssible fitness values
        graphs = graph_interface(max_fit, n_gens)
        best_fit_evals_per_alg = {}
        graph_nr = 0
        for graph_nr in range(len(graphs)):
            best_fit_evals_per_alg[graph_nr] = graphs[graph_nr]

        # compute convergence speed
        speeds = conv_speed(
            gens_of_chgs, optima_of_chgs, best_fit_evals_per_alg)

        # plot all graphs
        plt.figure()
        plt.subplot(111)
        norm = colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)  # "gray")
        my_colors = {}
        for alg, s in speeds.items():
            # make color
            my_colors[alg] = mapper.to_rgba(s)
            # plot graphs
            plt.plot(gens_of_chgs[0],
                     best_fit_evals_per_alg[alg], c=my_colors[alg], label=s)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    def examine_convergence_speed(self):
        '''
        Plots three possible fitness developments (for one change period). 
        Color of the grahps corresponds to their convergence speed measure.
        '''
        # one change
        gens_of_chgs = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        optima_of_chgs = {0: 0}
        best_fit_evals_per_alg = {'a': [10, 4, 4, 4, 4, 2, 2, 2, 2, 2],
                                  'b': [8, 8, 8, 6, 6, 6, 3, 3, 3, 3],
                                  'c': [7, 7, 7, 7, 7, 7, 1, 1, 1, 1]}
        # immediately optimum fitness
        speed = conv_speed(
            gens_of_chgs, optima_of_chgs, best_fit_evals_per_alg)

        # ===========
        norm = colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)  # ='gray'

        my_colors = {'a': mapper.to_rgba(speed['a']),
                     'b': mapper.to_rgba(speed['b']),
                     'c': mapper.to_rgba(speed['c'])}
        plt.figure()
        plt.subplot(111)
        plt.plot(gens_of_chgs[0],
                 best_fit_evals_per_alg['a'], c=my_colors['a'], label=speed['a'])
        plt.plot(gens_of_chgs[0],
                 best_fit_evals_per_alg['b'], c=my_colors['b'], label=speed['b'])
        plt.plot(gens_of_chgs[0],
                 best_fit_evals_per_alg['c'], c=my_colors['c'], label=speed['c'])
        plt.legend()
        plt.title("Best fitness value of each generation")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()


def graph_interface(max_fit, gens):
    '''
    Creates all possible fitness developments within gens generations.
    '''
    graphs_for_all_prefixes = None
    for fit in range(max_fit):
        graphs = __make_graphs__([fit], gens)
        if graphs_for_all_prefixes is None:
            graphs_for_all_prefixes = copy.copy(graphs)
        else:
            graphs_for_all_prefixes = np.append(
                graphs_for_all_prefixes, graphs, axis=0)
    return graphs_for_all_prefixes


def __make_graphs__(list_prefix, max_len):
    '''
    Is called by graph_interface()
    '''
    curr_ending_value = list_prefix[-1]
    graphs = None
    for i in range(0, curr_ending_value + 1):  # not larger than current last value
        list_new = copy.copy(list_prefix)
        list_new = np.append(list_new, i)

        if max_len == len(list_new):
            # append 1d list
            #graphs = np.append(graphs, np.array([list_new]), axis=0)
            if graphs is None:
                graphs = [list_new]
            else:
                # graphs.append(list_new)
                graphs = np.append(graphs, np.array([list_new]), axis=0)
        else:
            # 2d lists
            continued_lists = __make_graphs__(list_new, max_len)
            if graphs is None:
                graphs = copy.copy(continued_lists)
            else:
                graphs = np.append(graphs, continued_lists, axis=0)
            # for l in continued_lists:
            #    graphs.append(l)
    return graphs


def other_simple_test():
    gens_of_chgs = {0: [0, 1, 2, 3, ]}
    optima_of_chgs = {0: 0}
    best_fit_evals_per_alg = OrderedDict()
    # immediately optimum fitness
    name = "begin worse"
    best_fit_evals_1 = np.array([9, 6, 6, 0])
    best_fit_evals_per_alg[name] = best_fit_evals_1

    act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_1)
    speed = conv_speed(
        gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_1})

    print(name, ": \n    ARR ", act_arr, "\n    Speed ", speed)

    name = "begin good"
    best_fit_evals_2 = np.array([4, 3, 3, 0])
    best_fit_evals_per_alg[name] = best_fit_evals_2

    act_arr = arr(gens_of_chgs, optima_of_chgs, best_fit_evals_2)
    speed = conv_speed(
        gens_of_chgs, optima_of_chgs, {'a': best_fit_evals_2})

    print(name, ": \n    ARR ", act_arr, "\n    Speed ", speed)

    # overall convergence speed
    overall_speed = conv_speed(
        gens_of_chgs, optima_of_chgs, best_fit_evals_per_alg)
    for alg, speed in overall_speed.items():
        print(alg, "\n    ", speed)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_arr']
    t = Test()
    t.examine_arr()
    # t.examine___convergence_speed__()
    t.examine_convergence_speed_II()
    # unittest.main()
    other_simple_test()

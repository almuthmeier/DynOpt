'''
Created on Jan 16, 2019

@author: ameier
'''


import copy
import unittest

from code.utils.my_scaler import MyMinMaxScaler
import numpy as np


class Test(unittest.TestCase):

    def testMyScaler(self):
        # , [0, 8, 3, 0]])
        data_to_fit = np.array([[1, -2, 3, -4], [-2, 7, -5, 1]])

        scaler = MyMinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data_to_fit)

        transformed_data = scaler.transform(data_to_fit)

        retransformed_data = scaler.inverse_transform(
            copy.copy(transformed_data), False)
        range_orig_data = np.abs(retransformed_data[1] - retransformed_data[0])
        print("range_orig_data: ", range_orig_data)

        # =======================================================
        # noise

        noise_to_retrans = np.array([[0, 1, 8, 3], [6, 1, 0.5, 34]])

        noise_retrans_whole_range = scaler.inverse_transform(
            copy.copy(noise_to_retrans), False)
        noise_retrans_positive_range = scaler.inverse_transform(
            copy.copy(noise_to_retrans), True)
        range_of_whole_range = np.abs(
            noise_retrans_whole_range[1] - noise_retrans_whole_range[0])
        range_of_pos_range = np.abs(
            noise_retrans_positive_range[1] - noise_retrans_positive_range[0])

        print("\nnoise_retrans_whole_range:\n", noise_retrans_whole_range)
        print("noise_retrans_positive_range:\n",  noise_retrans_positive_range)

        print("\n")
        # assert equal ranges
        self.assertTrue(
            (np.abs(range_of_whole_range - range_of_pos_range) < 0.001).all())
        # assert no negative values
        self.assertTrue((noise_retrans_positive_range >= 0).all())

        print("\n")
        print("scaler.data_min_: ", scaler.data_min_)
        print("scaler.data_max_: ", scaler.data_max_)
        print("scaler.data_range_: ", scaler.data_range_)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testMyScaler']
    unittest.main()

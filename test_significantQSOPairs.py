from unittest import TestCase

import numpy as np

import significant_qso_pairs

__author__ = 'yishay'


class TestSignificantQSOPairs(TestCase):
    def test_add_if_larger(self):
        s = significant_qso_pairs.SignificantQSOPairs(5, dtype=int, initial_value=-1)
        qso1_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        qso2_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        value_list = [11, 15, 16, 13, 12, 10, 17, 20, 13, 14, 15]
        for qso1, qso2, value in zip(qso1_list, qso2_list, value_list):
            s.add_if_larger(qso1, qso2, value)

        ar_values = np.array(value_list)
        ar_5_largest_values = ar_values[np.argpartition(ar_values, -5)[-5:]]
        self.assertSetEqual(set(s.ar_values), set(ar_5_largest_values))

        print s.ar_qso1
        print s.ar_qso2
        print s.ar_values

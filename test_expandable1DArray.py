from unittest import TestCase

import numpy as np

from bins_2d import Expandable1DArray


__author__ = 'yishay'


class TestExpandable1DArray(TestCase):
    def test_get_array_view(self):
        ar1 = Expandable1DArray(np.arange(0, 5))
        if not np.array_equal(ar1.get_array_view(), np.arange(0, 5)):
            self.fail()

    def test_add_array(self):
        ar1 = Expandable1DArray([0])
        print ar1.size, ar1.ar.size, ar1.ar, ar1.get_array_view()
        ar1.add_array(np.array([1, 2, 3]))
        print ar1.size, ar1.ar.size, ar1.ar, ar1.get_array_view()
        ar1.add_array(np.array([4, 5, 6]))
        print ar1.size, ar1.ar.size, ar1.ar, ar1.get_array_view()
        ar1.add_array(np.array([7, 8, 9]))
        print ar1.size, ar1.ar.size, ar1.ar, ar1.get_array_view()
        ar1.add_array(np.array([10, 11, 12]))
        print ar1.size, ar1.ar.size, ar1.ar, ar1.get_array_view()
        ar1.add_array(np.array([13, 14, 15]))
        print ar1.size, ar1.ar.size, ar1.ar, ar1.get_array_view()
        ar1.add_array(np.array([16, 17, 18]))
        print ar1.size, ar1.ar.size, ar1.ar, ar1.get_array_view()
        if not np.array_equal(ar1.get_array_view(), np.arange(0, 19)):
            self.fail()

    def test__new_size(self):
        ar1 = Expandable1DArray([0])
        new_size = ar1._new_size(16385)
        self.assertEqual(new_size, 32768, "got {0} instead of {1}".format(new_size, 32768))


    def test_get_next_power_of_2(self):
        ar1 = Expandable1DArray([0])
        self.assertEqual(ar1.get_next_power_of_2(16777215), 16777216)

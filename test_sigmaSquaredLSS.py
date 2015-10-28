from unittest import TestCase

import numpy as np

from physics_functions import pixel_weight_coefficients


class TestSigmaSquaredLSS(TestCase):
    def test_evaluate(self):
        sigma = pixel_weight_coefficients.SigmaSquaredLSS(1.9, 3.6, 0.01)
        res = sigma.eval(np.array([1.95, 2.5, 3.55, 1.901, 3.599]))
        print(res)
        self.assertGreater(res[1], res[0])
        self.assertGreater(res[2], res[1])
        self.assertAlmostEqual(res[3], 0.047)
        self.assertAlmostEqual(res[4], 0.2487)

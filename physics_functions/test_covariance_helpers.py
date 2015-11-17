from unittest import TestCase
import numpy as np
import covariance_helpers

class TestJackknife_2d(TestCase):
    f = np.array([[[1.,2],[1,2],[1,4]],[[2,2],[5,6],[1,4]]])
    w = np.ones_like(f)

    def test_jackknife_2d_weighted(self):
        f = self.f
        w = self.w
        a = covariance_helpers.jackknife_2d_weighted(f,w)
        b = covariance_helpers.jackknife_2d(f,w)
        print(a)
        self.failUnless(np.allclose(a,b))

    def test_jackknife_2d(self):
        print(covariance_helpers.jackknife_2d(self.f,self.w))

    def test_subsample_2d_weighted(self):
        print(covariance_helpers.subsample_2d_weighted(self.f, self.w))

    def test_subsample_2d(self):
        print(covariance_helpers.subsample_2d(self.f,self.w))

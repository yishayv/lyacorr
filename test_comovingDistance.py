from unittest import TestCase

import numpy as np
from astropy.cosmology import Planck13
import astropy.units as u

from physics_functions import comoving_distance


class TestComovingDistance(TestCase):
    def test_fast_comoving_distance(self):
        z_params = {'z_start': 1.8, 'z_end': 3.7, 'z_step': 0.001}
        cd = comoving_distance.ComovingDistance(**z_params)
        ar_z = np.arange(1.952, 3.6, 0.132)
        ar_dist = cd.fast_comoving_distance(ar_z)
        ar_dist_reference = Planck13.comoving_transverse_distance(ar_z) / u.Mpc
        print ar_dist - ar_dist_reference
        self.assertTrue(np.allclose(ar_dist, ar_dist_reference))

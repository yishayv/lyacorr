from astropy.cosmology import Planck13
import numpy as np
from astropy import units as u


def fast_linear_interpolate(f, x):
    x = np.asarray(x)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1

    return (x1 - x) * f[x0] + (x - x0) * f[x1]


class ComovingDistance:
    def __init__(self, z_start, z_end, z_step):
        self._comoving_z_table = np.arange(z_start, z_end, z_step)
        self._comoving_distance_table = Planck13.comoving_distance(self._comoving_z_table).to(u.Mpc).value
        self.z_start = z_start
        self.z_end = z_end
        self.z_step = z_step

    def fast_comoving_distance(self, ar_z):
        assert np.all(ar_z < self.z_end) & np.all(ar_z > self.z_start), "redshift value out of range"
        ar_index = self._comoving_distance_table.size * (ar_z - self.z_start) / (self.z_end - self.z_start)
        return fast_linear_interpolate(self._comoving_distance_table, ar_index)


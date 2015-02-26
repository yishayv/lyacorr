from astropy.cosmology import Planck13
from astropy import units as u

import lookup_table


class ComovingDistance:
    def __init__(self, z_start, z_end, z_step):
        self.lookup_table = lookup_table.LinearInterpTable(self._distance_function, z_start, z_end, z_step)

    @staticmethod
    def _distance_function(ar_z):
        return Planck13.comoving_distance(ar_z).to(u.Mpc).value

    def fast_comoving_distance(self, ar_z):
        return self.lookup_table.evaluate(ar_z)


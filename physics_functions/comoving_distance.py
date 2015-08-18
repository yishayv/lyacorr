from astropy.cosmology import Planck13, WMAP5, WMAP7, WMAP9
from astropy import units as u
import common_settings

settings = common_settings.Settings()

import lookup_table


class ComovingDistance:
    def __init__(self, z_start, z_end, z_step, cosmology='from_config_file'):
        if cosmology == 'from_config_file':
            cosmology = settings.get_cosmology()
        self.cosmology = \
            {'Planck13': Planck13,
             'WMAP5': WMAP5,
             'WMAP7': WMAP7,
             'WMAP9': WMAP9}[cosmology]
        self.lookup_table = lookup_table.LinearInterpTable(
            lambda x: self._distance_function(x, self.cosmology), z_start, z_end, z_step)
        self.H0 = self.cosmology.H0

    @staticmethod
    def _distance_function(ar_z, cosmology):
        return cosmology.comoving_distance(ar_z).to(u.Mpc).value

    def comoving_distance(self, ar_z):
        return self._distance_function(ar_z, self.cosmology)

    def fast_comoving_distance(self, ar_z):
        """
        :type ar_z: np.multiarray.ndarray
        :rtype: np.multiarray.ndarray
        """
        """
        :param ar_z:
        :return:
        """
        return self.lookup_table.eval(ar_z)

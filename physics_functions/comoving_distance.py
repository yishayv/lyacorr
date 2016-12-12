"""
This module provides a helper class for obtaining fast comoving distance values, using a lookup table.
"""

from astropy import units as u
from astropy.cosmology import Planck13, WMAP5, WMAP7, WMAP9, FlatLambdaCDM

import common_settings
import lookup_table

settings = common_settings.Settings()  # type: common_settings.Settings

# Cosmological constants used in the fiducial model by Delubac 2015.
h = 0.7
Oc = 0.1090 / h ** 2
Ob = 0.0227 / h ** 2
Onu = 0.0006 / h ** 2
N_nu = 3.
# magic number for m_nu to obtain Onu*h^2=0.0006
m_nu = 0.018563030530909796 * u.eV
Om = Oc + Ob + Onu
fiducial_delubac = FlatLambdaCDM(
    H0=h * 100 * u.km / u.s / u.Mpc, Om0=Om, Ob0=Ob, m_nu=m_nu)


class ComovingDistance:
    def __init__(self, z_start=0, z_end=3.6, z_step=0.001, cosmology='from_config_file'):
        if cosmology == 'from_config_file':
            cosmology = settings.get_cosmology()
        self.cosmology = \
            {'Planck13': Planck13,
             'WMAP5': WMAP5,
             'WMAP7': WMAP7,
             'WMAP9': WMAP9,
             'Fiducial': fiducial_delubac}[cosmology]
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

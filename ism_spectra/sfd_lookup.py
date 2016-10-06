import astropy.io.fits as fits
import numpy as np


class SFDLookUp(object):
    """
    Get extinction values from the SFD maps[1].
    [1]: https://arxiv.org/abs/astro-ph/9809230
    """

    def __init__(self, ngp_filename, sgp_filename):
        """
        :param ngp_filename: NGP filename and path (e.g. 'SFD_dust_4096_ngp.fits')
        :type ngp_filename: str
        :param sgp_filename: SGP filename and path (e.g. 'SFD_dust_4096_sgp.fits')
        :type sgp_filename: str
        """
        ngp_list = fits.open(ngp_filename)
        sgp_list = fits.open(sgp_filename)

        ngp_data = ngp_list[0].data
        sgp_data = sgp_list[0].data

        data_res_x, data_res_y = ngp_data.shape

        if ngp_data.shape != sgp_data.shape:
            raise Exception("NGP and SGP maps must have the same resolution")

        self.half_data_res_x = data_res_x // 2
        self.half_data_res_y = data_res_y // 2

        self.sfd_map = np.dstack((ngp_data, sgp_data))

    def lookup(self, l, b):
        """
        get extinction values in the specified galactic coordinates
        :param l: l-parameter (1D array)
        :type l: np.ndarray
        :param b: b-parameter (1D array)
        :type b: np.ndarray
        :return: 1D array containing extinction values
        :rtype: np.ndarray
        """
        # SFD 98, equations C1 and C2:
        n_mask = b < 0
        n = np.ones_like(b)
        n[n_mask] = -1
        x = self.half_data_res_x * np.sqrt(1 - n * np.sin(b)) * np.cos(l) + (
            self.half_data_res_x - 0.5)  # type: np.ndarray
        y = -self.half_data_res_y * n * np.sqrt(1 - n * np.sin(b)) * np.sin(l) + (
            self.half_data_res_y - 0.5)  # type: np.ndarray

        return self.sfd_map[y.astype(np.int), x.astype(np.int), n_mask.astype(np.int)]

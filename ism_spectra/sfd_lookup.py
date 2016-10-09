import astropy.io.fits as fits
import numpy as np

x_offset = 0.5
y_offset = 0.5


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

        self.data_res_x, self.data_res_y = ngp_data.shape

        if ngp_data.shape != sgp_data.shape:
            raise Exception("NGP and SGP maps must have the same resolution")

        self.half_data_res_x = self.data_res_x // 2
        self.half_data_res_y = self.data_res_y // 2

        self.sfd_map = np.dstack((ngp_data, sgp_data))

    def get_xy(self, l, b):
        """
        get Lambert projection x and y values for the specified galactic coordinates
        :param l: l-parameter (1D array)
        :type l: np.ndarray
        :param b: b-parameter (1D array)
        :type b: np.ndarray
        :return: 1D arrays containing x, y values and the NGP/SGP mask.
        :rtype: np.ndarray
        """
        # SFD 98, equations C1 and C2:
        n_mask = b < 0
        n = np.ones_like(b)
        n[n_mask] = -1
        x = self.half_data_res_x * np.sqrt(1 - n * np.sin(b)) * np.cos(l) + (
            self.half_data_res_x - x_offset)  # type: np.ndarray
        y = -self.half_data_res_y * n * np.sqrt(1 - n * np.sin(b)) * np.sin(l) + (
            self.half_data_res_y - x_offset)  # type: np.ndarray

        return x, y, n_mask

    def lookup_nearest(self, l, b):
        """
        get extinction values in the specified galactic coordinates
        :param l: l-parameter (1D array)
        :type l: np.ndarray
        :param b: b-parameter (1D array)
        :type b: np.ndarray
        :return: 1D array containing extinction values
        :rtype: np.ndarray
        """
        x, y, n_mask = self.get_xy(l, b)
        return self.sfd_map[y.astype(np.int), x.astype(np.int), n_mask.astype(np.int)]

    def lookup_bilinear(self, l, b):
        """
        get extinction values in the specified galactic coordinates, using bilinear interpolation
        :param l: l-parameter (1D array)
        :type l: np.ndarray
        :param b: b-parameter (1D array)
        :type b: np.ndarray
        :return: 1D array containing extinction values
        :rtype: np.ndarray
        """
        x, y, n_mask = self.get_xy(l, b)
        x_fraction, x_floor = np.modf(x)
        y_fraction, y_floor = np.modf(y)

        weights = np.empty(shape=(2, 2,) + x.shape)
        weights[0, 0] = (1 - x_fraction) * (1 - y_fraction)
        weights[0, 1] = (1 - x_fraction) * y_fraction
        weights[1, 0] = x_fraction * (1 - y_fraction)
        weights[1, 1] = x_fraction * y_fraction

        values = np.empty(shape=(2, 2,) + x.shape)
        x_floor_int = x_floor.astype(np.int)
        y_floor_int = y_floor.astype(np.int)
        x_ceil_int = np.clip(x_floor_int + 1, 0, self.data_res_x - 1)
        y_ceil_int = np.clip(y_floor_int + 1, 0, self.data_res_y - 1)
        n_mask_int = n_mask.astype(np.int)

        values[0, 0] = self.sfd_map[y_floor_int, x_floor_int, n_mask_int]
        values[0, 1] = self.sfd_map[y_ceil_int, x_floor_int, n_mask_int]
        values[1, 0] = self.sfd_map[y_floor_int, x_ceil_int, n_mask_int]
        values[1, 1] = self.sfd_map[y_ceil_int, x_ceil_int, n_mask_int]

        return np.sum(values * weights, axis=(0, 1))

import numpy as np

import common_settings
from data_access.numpy_spectrum_container import NpSpectrumContainer
import bin_pixel_pairs
import significant_qso_pairs

NUM_BINS_X = 50
NUM_BINS_Y = 50
MAX_Z_RESOLUTION = 1000

settings = common_settings.Settings()


class CovarianceMatrix:
    def __init__(self, cd, radius, accumulator_type):
        """
        initialize persistent objects
        :type cd: comoving_distance.ComovingDistance
        :type radius: float
        """
        self.cd = cd
        self.radius = radius
        self.significant_qso_pairs = significant_qso_pairs.SignificantQSOPairs()
        self.accumulator_type = accumulator_type
        self.covariance = np.zeros((50, 50, 50, 50, 3))

    def add_quad(self, qso_angle12, qso_angle34, max_range_parallel, max_range_transverse,
                 spec1_index, spec2_index, spec3_index, spec4_index, delta_t_file):
        """
        Find all pixel pairs in QSO1,QSO2 that are closer than radius r
        :type qso_angle: float64
        :type spec1_index: int
        :type spec2_index: int
        :type spec3_index: int
        :type spec4_index: int
        :type delta_t_file: NpSpectrumContainer
        :return:
        """

        # Note: not using pre_alloc_matrices.zero()

        # the maximum distance that can be stored in the accumulator
        # r = float(accumulator.get_max_range())
        # range_parallel = np.float64(accumulator.get_x_range())
        # range_transverse = np.float64(accumulator.get_y_range())
        r = np.sqrt(np.square(max_range_parallel, np.square(max_range_transverse)))

        spec1_z = delta_t_file.get_wavelength(spec1_index)
        spec2_z = delta_t_file.get_wavelength(spec2_index)
        spec3_z = delta_t_file.get_wavelength(spec3_index)
        spec4_z = delta_t_file.get_wavelength(spec4_index)
        if not (spec1_z.size and spec2_z.size and spec3_z.size and spec4_z.size):
            return

        assert spec1_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec1_z.min(), spec1_index)
        assert spec2_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec2_z.min(), spec2_index)
        assert spec3_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec3_z.min(), spec3_index)
        assert spec4_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec4_z.min(), spec4_index)

        # Note: throughout this method, "flux" means delta_f
        spec1_flux = delta_t_file.get_flux(spec1_index)
        spec1_distances = self.cd.fast_comoving_distance(spec1_z)

        spec2_flux = delta_t_file.get_flux(spec2_index)
        spec2_distances = self.cd.fast_comoving_distance(spec2_z)

        spec3_flux = delta_t_file.get_flux(spec3_index)
        spec3_distances = self.cd.fast_comoving_distance(spec3_z)

        spec4_flux = delta_t_file.get_flux(spec4_index)
        spec4_distances = self.cd.fast_comoving_distance(spec4_z)

        # get pre-calculated weights for each QSO
        qso1_weights = delta_t_file.get_ivar(spec1_index)
        qso2_weights = delta_t_file.get_ivar(spec2_index)
        qso3_weights = delta_t_file.get_ivar(spec3_index)
        qso4_weights = delta_t_file.get_ivar(spec4_index)

        # if the parallel distance between forests is too large, they will not form pairs.
        if spec1_distances[0] > r + spec2_distances[-1] or spec2_distances[0] > r + spec1_distances[-1]:
            return

        # if the parallel distance between forests is too large, they will not form pairs.
        if spec3_distances[0] > r + spec4_distances[-1] or spec4_distances[0] > r + spec3_distances[-1]:
            return

        ar = bin_pixel_pairs.bin_pixel_quads(ar_dist1=spec1_distances, ar_dist2=spec2_distances,
                                             ar_flux1=spec1_flux, ar_flux2=spec2_flux,
                                             ar_weights1=qso1_weights, ar_weights2=qso2_weights,
                                             ar_dist3=spec3_distances, ar_dist4=spec4_distances,
                                             ar_flux3=spec3_flux, ar_flux4=spec4_flux,
                                             ar_weights3=qso3_weights, ar_weights4=qso4_weights,
                                             qso_angle12=qso_angle12,
                                             ar_est=self.ar_est,
                                             out=self.covariance,
                                             x_bin_size=50,
                                             y_bin_size=50,
                                             x_bin_count=50,
                                             y_bin_count=50)

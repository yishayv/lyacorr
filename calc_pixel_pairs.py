import numpy as np

import common_settings
import bins_2d
from flux_accumulator import AccumulatorBase
from data_access.numpy_spectrum_container import NpSpectrumContainer
import bin_pixel_pairs

NUM_BINS_X = 50
NUM_BINS_Y = 50
MAX_Z_RESOLUTION = 1000

settings = common_settings.Settings()


class PreAllocMatrices:
    def __init__(self, z_res):
        self.z_res = z_res
        self.m1 = np.zeros([z_res, z_res], dtype='float32')
        self.m2 = np.zeros([z_res, z_res], dtype='float32')
        self.m4 = np.zeros([z_res, z_res], dtype='float32')
        self.m5 = np.zeros([z_res, z_res], dtype='float32')
        self.m6 = np.zeros([z_res, z_res], dtype='float32')
        self.m7 = np.zeros([z_res, z_res], dtype='float32')
        self.v1 = np.zeros(z_res, dtype='float32')
        self.v2 = np.zeros(z_res, dtype='float32')
        self.v3 = np.zeros(z_res, dtype='float32')
        self.v4 = np.zeros(z_res, dtype='float32')
        self.v5 = np.zeros(z_res, dtype='float32')
        self.v6 = np.zeros(z_res, dtype='float32')
        self.mask1 = np.zeros([z_res, z_res], dtype=bool)
        self.mask2 = np.zeros([z_res, z_res], dtype=bool)

    def zero(self):
        self.m1.fill(0)
        self.m2.fill(0)
        self.mask1.fill(0)
        self.mask2.fill(0)
        self.m4.fill(0)
        self.m5.fill(0)
        self.m6.fill(0)
        self.v1.fill(0)
        self.v2.fill(0)
        self.v3.fill(0)
        self.v4.fill(0)
        self.v5.fill(0)
        self.v6.fill(0)


class PixelPairs:
    def __init__(self, cd, radius):
        """
        initialize persistent objects
        :type cd: comoving_distance.ComovingDistance
        :type radius: float
        """
        self.cd = cd
        self.radius = radius
        self.pre_alloc_matrices = PreAllocMatrices(MAX_Z_RESOLUTION)

    def find_nearby_pixels2(self, accumulator, qso_angle,
                            spec1_index, spec2_index, delta_t_file):
        """
        Find all pixel pairs in QSO1,QSO2 that are closer than radius r
        :type accumulator: AccumulatorBase
        :type qso_angle: float64
        :type spec1_index: int
        :type spec2_index: int
        :type delta_t_file: NpSpectrumContainer
        :return:
        """

        # Note: not using pre_alloc_matrices.zero()

        # the maximum distance that can be stored in the accumulator
        r = np.float32(accumulator.get_max_range())
        range_parallel = np.float32(accumulator.get_x_range())
        range_transverse = np.float32(accumulator.get_y_range())

        spec1_z = delta_t_file.get_wavelength(spec1_index)
        spec2_z = delta_t_file.get_wavelength(spec2_index)
        if not (spec1_z.size and spec2_z.size):
            return

        assert spec1_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec1_z.min(), spec1_index)
        assert spec2_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec2_z.min(), spec2_index)

        # Note: throughout this method, "flux" means delta_f
        spec1_flux = delta_t_file.get_flux(spec1_index)
        spec1_distances = self.cd.fast_comoving_distance(spec1_z)

        spec2_flux = delta_t_file.get_flux(spec2_index)
        # print spec2_flux
        spec2_distances = self.cd.fast_comoving_distance(spec2_z)

        # get pre-calculated weights for each QSO
        qso1_weights = delta_t_file.get_ivar(spec1_index)
        qso2_weights = delta_t_file.get_ivar(spec2_index)

        # if the parallel distance between forests is too large, they will not form pairs.
        if spec1_distances[0] > r + spec2_distances[-1] or spec2_distances[0] > r + spec1_distances[-1]:
            return

        # create matrices with first dimension of spec1 data points,
        # second dimension of spec2 data points
        y = spec1_distances.size
        x = spec2_distances.size

        # assign variables to pre-allocated memory
        flux_products = self.pre_alloc_matrices.m2[:y, :x]
        mask_matrix_parallel = self.pre_alloc_matrices.mask1[:y, :x]
        mask_matrix_final = self.pre_alloc_matrices.mask2[:y, :x]
        r_parallel = self.pre_alloc_matrices.m4[:y, :x]
        r_transverse = self.pre_alloc_matrices.m5[:y, :x]
        spec1_distances_sq = self.pre_alloc_matrices.v1[:y]
        spec2_distances_sq = self.pre_alloc_matrices.v2[:x]
        z_weights = self.pre_alloc_matrices.m6[:y, :x]

        np.square(spec1_distances, out=spec1_distances_sq)
        np.square(spec2_distances, out=spec2_distances_sq)

        # a matrix of flux products
        # TODO: add weights for a proper calculation of "xi(i,j)"
        np.outer(spec1_flux, spec2_flux, out=flux_products)

        # r|| = abs(r1 - r2)
        np.subtract(spec1_distances[:, None], spec2_distances, out=r_parallel)
        np.abs(r_parallel, out=r_parallel)
        np.multiply(r_parallel, 1. / accumulator.get_x_bin_size(), out=r_parallel)

        # r_ =  (r1 + r2)/2 * qso_angle
        np.add(spec1_distances[:, None], spec2_distances, out=r_transverse)
        np.multiply(r_transverse, qso_angle / 2. / accumulator.get_y_bin_size(), out=r_transverse)

        # mask all elements that too far apart
        np.less(r_parallel, range_parallel, out=mask_matrix_parallel)
        np.less(r_transverse, range_transverse, out=mask_matrix_final)
        np.logical_and(mask_matrix_parallel, mask_matrix_final, mask_matrix_final)

        np.outer(qso1_weights, qso2_weights, out=z_weights)

        assert not np.isnan(flux_products).any()
        assert not np.isnan(z_weights).any()

        return accumulator.add_array_with_mask(flux_products,
                                               r_parallel,
                                               r_transverse,
                                               mask_matrix_final,
                                               z_weights)

    def find_nearby_pixels(self, accumulator, qso_angle,
                           spec1_index, spec2_index, delta_t_file):
        """
        Find all pixel pairs in QSO1,QSO2 that are closer than radius r
        :type accumulator: AccumulatorBase
        :type qso_angle: float64
        :type spec1_index: int
        :type spec2_index: int
        :type delta_t_file: NpSpectrumContainer
        :return:
        """

        # Note: not using pre_alloc_matrices.zero()

        # the maximum distance that can be stored in the accumulator
        r = np.float64(accumulator.get_max_range())
        range_parallel = np.float64(accumulator.get_x_range())
        range_transverse = np.float64(accumulator.get_y_range())

        spec1_z = delta_t_file.get_wavelength(spec1_index)
        spec2_z = delta_t_file.get_wavelength(spec2_index)
        if not (spec1_z.size and spec2_z.size):
            return

        assert spec1_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec1_z.min(), spec1_index)
        assert spec2_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec2_z.min(), spec2_index)

        # Note: throughout this method, "flux" means delta_f
        spec1_flux = delta_t_file.get_flux(spec1_index)
        spec1_distances = self.cd.fast_comoving_distance(spec1_z)

        spec2_flux = delta_t_file.get_flux(spec2_index)
        # print spec2_flux
        spec2_distances = self.cd.fast_comoving_distance(spec2_z)

        # get pre-calculated weights for each QSO
        qso1_weights = delta_t_file.get_ivar(spec1_index)
        qso2_weights = delta_t_file.get_ivar(spec2_index)

        # if the parallel distance between forests is too large, they will not form pairs.
        if spec1_distances[0] > r + spec2_distances[-1] or spec2_distances[0] > r + spec1_distances[-1]:
            return

        ar = bin_pixel_pairs.bin_pixel_pairs(ar_z1=spec1_z, ar_z2=spec2_z,
                                             ar_dist1=spec1_distances, ar_dist2=spec2_distances,
                                             ar_flux1=spec1_flux, ar_flux2=spec2_flux,
                                             ar_weights1=qso1_weights, ar_weights2=qso2_weights,
                                             qso_angle=qso_angle,
                                             x_bin_size=accumulator.get_x_bin_size(),
                                             y_bin_size=accumulator.get_y_bin_size(),
                                             x_bin_count=accumulator.get_x_count(),
                                             y_bin_count=accumulator.get_y_count())

        # print ar[:,:,0].max()
        local_bins = bins_2d.Bins2D.from_np_arrays(ar[:, :, 1], ar[:, :, 0], ar[:, :, 2],
                                                   accumulator.get_x_range(), accumulator.get_y_range())
        accumulator += local_bins
        # print accumulator.ar_count.max()

    def apply_to_flux_pairs(self, pairs, pairs_angles, delta_t_file, accumulator):
        """

        :type pairs: np.array
        :type pairs_angles: np.array
        :type delta_t_file: NpSpectrumContainer
        :type accumulator
        :rtype: AccumulatorBase
        """

        n = 0
        for i, j, k in pairs:
            qso_angle = pairs_angles[k]
            # r_parallel = abs(ar_distance[i] - ar_distance[j])
            # mean_distance = (ar_distance[i] + ar_distance[j]) / 2
            # r_transverse = mean_distance * qso_angle
            # print 'QSO pair with r_parallel %f, r_transverse %f' % (r_parallel, r_transverse)
            spec1_index = i
            spec2_index = j

            self.find_nearby_pixels(accumulator, qso_angle,
                                    spec1_index, spec2_index, delta_t_file)
            n += 1
        return accumulator

    def add_qso_pairs_to_bins(self, pairs, pairs_angles, delta_t_file):
        """

        :type pairs: np.array
        :type pairs_angles: np.array
        :type delta_t_file: NpSpectrumContainer
        :rtype: bins_2d.Bins2D
        """
        pair_separation_bins = bins_2d.Bins2D(NUM_BINS_X, NUM_BINS_Y, x_range=self.radius, y_range=self.radius)
        pair_separation_bins.set_filename(settings.get_estimator_bins())
        self.apply_to_flux_pairs(pairs, pairs_angles, delta_t_file, pair_separation_bins)
        return pair_separation_bins

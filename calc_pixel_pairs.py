"""
    Accumulate flux products from pixel pairs to bins.
    Uses a Python/C API methods for the actual calculation.
    There are 2 modes of operation:
        - mean: sum of weighted flux products, sum of weights
        - median: a weighted histogram of flux products.
    Also contains a slower reference implementation in pure Python (mean only)
"""
import lyacorr_cython_helper
from collections import namedtuple

import numpy as np

import bins_3d
import bins_3d_with_group_id
import common_settings
import flux_histogram_bins
import significant_qso_pairs
from data_access.numpy_spectrum_container import NpSpectrumContainer
from flux_accumulator import AccumulatorBase

settings = common_settings.Settings()

NUM_BINS_X = 50
NUM_BINS_Y = 50
NUM_BINS_Z = settings.get_num_distance_slices()
MAX_Z_RESOLUTION = 1000

accumulator_types = namedtuple('accumulator_type', ['mean', 'mean_subsample', 'histogram'])


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
    def __init__(self, cd, radius, accumulator_type):
        """
        initialize persistent objects
        :type cd: comoving_distance.ComovingDistance
        :type radius: float
        """
        self.cd = cd
        self.radius = radius
        self.pre_alloc_matrices = PreAllocMatrices(MAX_Z_RESOLUTION)
        self.significant_qso_pairs = significant_qso_pairs.SignificantQSOPairs()
        self.accumulator_type = accumulator_type
        self.min_distance = cd.comoving_distance(settings.get_min_forest_redshift())
        self.max_distance = cd.comoving_distance(settings.get_max_forest_redshift())

    def find_nearby_pixels2(self, accumulator, qso_angle,
                            spec1_index, spec2_index, delta_t_file):
        """
        Find all pixel pairs in QSO1,QSO2 that are closer than radius r.
        This is a reference implementation in pure Python+Numpy.
        :type accumulator: AccumulatorBase
        :type qso_angle: float64
        :type spec1_index: int
        :type spec2_index: int
        :type delta_t_file: NpSpectrumContainer
        :return:
        """

        # Note: not using pre_alloc_matrices.zero()

        # the maximum distance that can be stored in the accumulator
        r = float(accumulator.get_max_range())
        range_parallel = np.float32(accumulator.get_ranges()[1, 0])
        range_transverse = np.float32(accumulator.get_ranges()[1, 1])

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
        # print(spec2_flux)
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
        mean_distance = self.pre_alloc_matrices.m7[:y, :x]

        np.square(spec1_distances, out=spec1_distances_sq)
        np.square(spec2_distances, out=spec2_distances_sq)

        # a matrix of flux products
        np.outer(spec1_flux, spec2_flux, out=flux_products)

        # r|| = abs(r1 - r2)
        np.subtract(spec1_distances[:, None], spec2_distances, out=r_parallel)
        np.abs(r_parallel, out=r_parallel)

        # r_ =  (r1 + r2)/2 * qso_angle
        np.add(spec1_distances[:, None], spec2_distances, out=mean_distance)
        np.multiply(mean_distance, 1. / 2, out=mean_distance)
        np.multiply(mean_distance, qso_angle, out=r_transverse)

        # mask all elements that are too far apart
        np.less(r_parallel, range_parallel, out=mask_matrix_parallel)
        np.less(r_transverse, range_transverse, out=mask_matrix_final)
        np.logical_and(mask_matrix_parallel, mask_matrix_final, mask_matrix_final)

        np.outer(qso1_weights, qso2_weights, out=z_weights)
        # multiply fluxes by their respective weights
        np.multiply(flux_products, z_weights, out=flux_products)

        assert not np.isnan(flux_products).any()
        assert not np.isnan(z_weights).any()

        return accumulator.add_array_with_mask(flux_products,
                                               r_parallel,
                                               r_transverse,
                                               mean_distance,
                                               mask_matrix_final,
                                               z_weights)

    def find_nearby_pixels(self, accumulator, qso_angle,
                           spec1_index, spec2_index, delta_t_file,
                           group_id=0):
        """
        Find all pixel pairs in QSO1,QSO2 that are closer than radius r.
        This is a faster implementation that uses a Python/C API module.
        :type accumulator: AccumulatorBase
        :type qso_angle: float64
        :type spec1_index: int64
        :type spec2_index: int64
        :type delta_t_file: NpSpectrumContainer
        :type group_id: int64
        :return:
        """

        # Note: not using pre_alloc_matrices.zero()

        # the maximum distance that can be stored in the accumulator
        r = float(accumulator.get_max_range())

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
        # print(spec2_flux)
        spec2_distances = self.cd.fast_comoving_distance(spec2_z)

        # get pre-calculated weights for each QSO
        qso1_weights = delta_t_file.get_ivar(spec1_index)
        qso2_weights = delta_t_file.get_ivar(spec2_index)

        # if the parallel distance between forests is too large, they will not form pairs.
        if spec1_distances[0] > r + spec2_distances[-1] or spec2_distances[0] > r + spec1_distances[-1]:
            return

        if self.accumulator_type == accumulator_types.mean:
            ar = lyacorr_cython_helper.bin_pixel_pairs(spec1_distances, spec2_distances,
                                                       spec1_flux, spec2_flux,
                                                       qso1_weights, qso2_weights,
                                                       qso_angle,
                                                       accumulator.get_dims(),
                                                       accumulator.get_ranges())
            local_bins = bins_3d.Bins3D(accumulator.get_dims(),
                                        accumulator.get_ranges(), ar_existing_data=ar)

            flux_contribution = np.nanmax(np.abs(local_bins.ar_flux))
            self.significant_qso_pairs.add_if_larger(spec1_index, spec2_index, flux_contribution)

            accumulator += local_bins
        elif self.accumulator_type == accumulator_types.mean_subsample:
            # local_bins = bins_3d_with_group_id.Bins3DWithGroupID(
            #     accumulator.get_dims(), accumulator.get_ranges())  # type: bins_3d_with_group_id.Bins3DWithGroupID
            # retrieve or create storage for this array
            assert isinstance(accumulator, bins_3d_with_group_id.Bins3DWithGroupID)
            ar_view = accumulator.get_group_view(group_id).ar_data
            lyacorr_cython_helper.bin_pixel_pairs(spec1_distances, spec2_distances,
                                                  spec1_flux, spec2_flux,
                                                  qso1_weights, qso2_weights,
                                                  qso_angle,
                                                  accumulator.get_dims(),
                                                  accumulator.get_ranges(),
                                                  ar_view)

            # disabled for performance reasons
            # flux_contribution = np.nanmax(np.abs(local_bins.dict_bins_3d_data[group_id].ar_flux))
            # self.significant_qso_pairs.add_if_larger(spec1_index, spec2_index, flux_contribution)

            # accumulator += local_bins
        elif self.accumulator_type == accumulator_types.histogram:
            assert isinstance(accumulator, flux_histogram_bins.FluxHistogramBins)
            # TODO: try to avoid using implementation details of the accumulator interface
            accumulator.pair_count += lyacorr_cython_helper.bin_pixel_pairs_histogram(
                spec1_distances, spec2_distances,
                spec1_flux, spec2_flux, qso1_weights, qso2_weights,
                qso_angle,
                accumulator.get_dims(),
                accumulator.get_ranges(),
                accumulator.ar_flux)
            pass

    def apply_to_flux_pairs(self, pairs, pairs_angles, delta_t_file, accumulator):
        """

        :type pairs: np.array
        :type pairs_angles: np.array
        :type delta_t_file: NpSpectrumContainer
        :type accumulator
        :rtype: AccumulatorBase
        """

        n = 0
        for spec1_index, spec2_index, group_id, angle_index in pairs:
            qso_angle = pairs_angles[angle_index]

            self.find_nearby_pixels(accumulator, qso_angle,
                                    spec1_index, spec2_index, delta_t_file,
                                    group_id=group_id)
            n += 1
        return accumulator

    def add_qso_pairs_to_bins(self, pairs, pairs_angles, delta_t_file):
        """

        :type pairs: np.multiarray.ndarray
        :type pairs_angles: np.multiarray.ndarray
        :type delta_t_file: NpSpectrumContainer
        :rtype: AccumulatorBase
        """

        pair_separation_bins = None
        if self.accumulator_type == accumulator_types.mean:
            pair_separation_bins = bins_3d.Bins3D(
                dims=np.array([NUM_BINS_X, NUM_BINS_Y, NUM_BINS_Z]),
                ranges=np.array([[0, 0, self.min_distance], [self.radius, self.radius, self.max_distance]]))
            pair_separation_bins.set_filename(settings.get_mean_estimator_bins())
        elif self.accumulator_type == accumulator_types.mean_subsample:
            pair_separation_bins = bins_3d_with_group_id.Bins3DWithGroupID(
                dims=np.array([NUM_BINS_X, NUM_BINS_Y, NUM_BINS_Z]),
                ranges=np.array([[0, 0, self.min_distance], [self.radius, self.radius, self.max_distance]]))
            pair_separation_bins.set_filename(settings.get_correlation_estimator_subsamples_npz())
        elif self.accumulator_type == accumulator_types.histogram:
            pair_separation_bins = flux_histogram_bins.FluxHistogramBins(
                dims=np.array([NUM_BINS_X, NUM_BINS_Y, 1000]),
                ranges=np.array([[0, 0, -2e3], [self.radius, self.radius, 2e3]]))
            pair_separation_bins.set_filename(settings.get_median_estimator_bins())

        assert pair_separation_bins

        self.apply_to_flux_pairs(pairs, pairs_angles, delta_t_file, pair_separation_bins)

        return pair_separation_bins

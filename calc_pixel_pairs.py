import numpy as np

import common_settings
import bins_2d


# bin size in Mpc/h
BIN_SIZE = 4
NUM_BINS_X = 50
NUM_BINS_Y = 50
MAX_Z_RESOLUTION = 1000

settings = common_settings.Settings()


class PreAllocMatrices:
    def __init__(self, z_res):
        self.z_res = z_res
        self.m1 = np.zeros([z_res, z_res])
        self.m2 = np.zeros([z_res, z_res])
        self.m4 = np.zeros([z_res, z_res])
        self.m5 = np.zeros([z_res, z_res])
        self.m6 = np.zeros([z_res, z_res])
        self.m7 = np.zeros([z_res, z_res])
        self.v1 = np.zeros(z_res)
        self.v2 = np.zeros(z_res)
        self.v3 = np.zeros(z_res)
        self.v4 = np.zeros(z_res)
        self.v5 = np.zeros(z_res)
        self.v6 = np.zeros(z_res)
        self.mask1 = np.zeros([z_res, z_res], dtype=bool)

    def zero(self):
        self.m1.fill(0)
        self.m2.fill(0)
        self.mask1.fill(0)
        self.m4.fill(0)
        self.m5.fill(0)
        self.m6.fill(0)
        self.v1.fill(0)
        self.v2.fill(0)
        self.v3.fill(0)
        self.v4.fill(0)
        self.v5.fill(0)
        self.v6.fill(0)


def find_nearby_pixels(cd, pre_alloc_matrices, pair_separation_bins, qso_angle,
                       spec1_index, spec2_index, delta_t_file, r):
    """
    Find all pixel pairs in QSO1,QSO2 that are closer than radius r
    :type cd: comoving_distance.ComovingDistance
    :type pre_alloc_matrices: PreAllocMatrices
    :type pair_separation_bins: bins_2d.Bins2D
    :type qso_angle: float64
    :type spec1_index: int
    :type spec2_index: int
    :type delta_t_file: NpSpectrumContainer
    :type r: float64
    :return:
    """

    # Note: not using pre_alloc_matrices.zero()

    # use law of cosines to find the distance between pairs of pixels
    qso_angle_cosine = np.cos(qso_angle)
    r_sq = np.square(r)

    spec1_z = delta_t_file.get_wavelength(spec1_index)
    spec2_z = delta_t_file.get_wavelength(spec2_index)
    if not (spec1_z.size and spec2_z.size):
        return

    assert spec1_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec1_z.min(), spec1_index)
    assert spec2_z.min() > 0, "z out of range: {0}, spec index {1}".format(spec2_z.min(), spec2_index)

    # Note: throughout this method, "flux" means delta_f
    spec1_flux = delta_t_file.get_flux(spec1_index)
    spec1_distances = cd.fast_comoving_distance(spec1_z)

    spec2_flux = delta_t_file.get_flux(spec2_index)
    # print spec2_flux
    spec2_distances = cd.fast_comoving_distance(spec2_z)

    # if the parallel distance between forests is too large, they will not form pairs.
    if spec1_distances[0] > r + spec2_distances[-1] or spec2_distances[0] > r + spec1_distances[-1]:
        return

    # create matrices with first dimension of spec1 data points,
    # second dimension of spec2 data points
    y = spec1_distances.size
    x = spec2_distances.size

    # assign variables to pre-allocated memory
    m1 = pre_alloc_matrices.m1[:y, :x]
    flux_products = pre_alloc_matrices.m2[:y, :x]
    mask_matrix = pre_alloc_matrices.mask1[:y, :x]
    r_parallel = pre_alloc_matrices.m4[:y, :x]
    r_transverse = pre_alloc_matrices.m5[:y, :x]
    spec1_distances_sq = pre_alloc_matrices.v1[:y]
    spec2_distances_sq = pre_alloc_matrices.v2[:x]
    z_plus_1_1 = pre_alloc_matrices.v3[:y]
    z_plus_1_2 = pre_alloc_matrices.v4[:x]
    z_plus_1_power_1 = pre_alloc_matrices.v5[:y]
    z_plus_1_power_2 = pre_alloc_matrices.v6[:x]
    z_weights = pre_alloc_matrices.m6[:y, :x]
    weighted_flux_products = pre_alloc_matrices.m7[:y, :x]

    np.square(spec1_distances, out=spec1_distances_sq)
    np.square(spec2_distances, out=spec2_distances_sq)

    # calculate all mutual distances
    # d^2 = r1^2 + r2^2 - 2*r1*r2*cos(a)
    np.outer(spec1_distances, spec2_distances, out=m1)
    np.multiply(m1, - 2 * qso_angle_cosine, out=m1)
    np.add(m1, spec1_distances_sq[:, None], out=m1)
    np.add(m1, spec2_distances_sq[None, :], out=m1)

    spec1_spec2_dist_sq = m1

    # a matrix of flux products
    # TODO: add weights for a proper calculation of "xi(i,j)"
    np.outer(spec1_flux, spec2_flux, out=flux_products)

    # mask all elements that are close enough
    np.less(spec1_spec2_dist_sq, r_sq, out=mask_matrix)

    # r|| = abs(r1 - r2)
    np.subtract(spec1_distances[:, None], spec2_distances, out=r_parallel)
    np.abs(r_parallel, out=r_parallel)
    np.multiply(r_parallel, 1. / BIN_SIZE, out=r_parallel)

    # r_ =  (r1 + r2)/2 * qso_angle
    np.add(spec1_distances[:, None], spec2_distances, out=r_transverse)
    np.multiply(r_transverse, qso_angle / (2 * BIN_SIZE), out=r_transverse)

    # calculate z-based weights
    gamma = 3.8
    np.add(spec1_z, 1, out=z_plus_1_1)
    np.add(spec2_z, 1, out=z_plus_1_2)
    np.power(z_plus_1_1, gamma, out=z_plus_1_power_1)
    np.power(z_plus_1_2, gamma, out=z_plus_1_power_2)
    np.outer(z_plus_1_power_1, z_plus_1_power_2, out=z_weights)

    # np.multiply(flux_products, z_weights, weighted_flux_products)
    assert not np.isnan(flux_products.sum())
    assert not np.isnan(z_weights.sum())

    return pair_separation_bins.add_array_with_mask(flux_products,
                                                    r_parallel,
                                                    r_transverse,
                                                    mask_matrix,
                                                    z_weights)


def apply_to_flux_pairs(cd, pairs, pairs_angles, delta_t_file, accumulator):
    """

    :type cd: comoving_distance.ComovingDistance
    :type pairs: np.array
    :type pairs_angles: np.array
    :type delta_t_file: NpSpectrumContainer
    :type accumulator
    :rtype: accumulator
    """

    pre_alloc_matrices = PreAllocMatrices(MAX_Z_RESOLUTION)
    n = 0
    for i, j, k in pairs:
        qso_angle = pairs_angles[k]
        # r_parallel = abs(ar_distance[i] - ar_distance[j])
        # mean_distance = (ar_distance[i] + ar_distance[j]) / 2
        # r_transverse = mean_distance * qso_angle
        # print 'QSO pair with r_parallel %f, r_transverse %f' % (r_parallel, r_transverse)
        spec1_index = i
        spec2_index = j
        # TODO: read the default 200Mpc value from elsewhere
        find_nearby_pixels(cd, pre_alloc_matrices, accumulator, qso_angle,
                           spec1_index, spec2_index, delta_t_file,
                           r=200 * np.sqrt(2))
        if n % 1000 == 0:
            print 'intermediate number of pixel pairs in bins (qso pair count = %d) :%d' % (
                n, accumulator.ar_count.sum().astype(int))
            accumulator.flush()
        n += 1
    return accumulator


def add_qso_pairs_to_bins(cd, pairs, pairs_angles, delta_t_file):
    """

    :type cd: comoving_distance.ComovingDistance
    :type pairs: np.array
    :type pairs_angles: np.array
    :type delta_t_file: NpSpectrumContainer
    :rtype: bins_2d.Bins2D
    """
    pair_separation_bins = bins_2d.Bins2D(NUM_BINS_X, NUM_BINS_Y)
    pair_separation_bins.set_filename(settings.get_estimator_bins())
    apply_to_flux_pairs(cd, pairs, pairs_angles, delta_t_file, pair_separation_bins)
    return pair_separation_bins

__author__ = 'yishay'

import cProfile
import kernprof

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.cosmology import Planck13
from astropy.coordinates import matching as matching
from astropy.coordinates import Angle
from astropy import table

import read_spectrum_hdf5
import common_settings
import bins_2d
from read_spectrum_fits import QSORecord
import comoving_distance


settings = common_settings.Settings()

# bin size in Mpc/h
BIN_SIZE = 4


# def find_nearby_pixels(qso_angle, spec2, spec1_pixel, r):
# """
# Find the pixels in QSO2 within radius r of an nth-pixel QSO1
# :param qso1:
# :param qso2:
# """
# coord1 = coord.sky

class PreAllocMatrices:
    def __init__(self):
        self.spec1_spec2_dist_sq = np.zeros([5000, 5000])
        self.spec1_times_spec2_dist = np.zeros([5000, 5000])
        self.m1 = np.zeros([5000, 5000])
        self.m2 = np.zeros([5000, 5000])
        self.m4 = np.zeros([5000, 5000])
        self.m5 = np.zeros([5000, 5000])
        self.v1 = np.zeros(5000)
        self.v2 = np.zeros(5000)
        self.mask1 = np.zeros([5000, 5000], dtype=bool)

    def zero(self):
        self.spec1_spec2_dist_sq.fill(0)
        self.spec1_times_spec2_dist.fill(0)
        self.m1.fill(0)
        self.m2.fill(0)
        self.mask1.fill(0)
        self.v1.fill(0)
        self.v2.fill(0)
        self.m4.fill(0)
        self.m5.fill(0)


def find_nearby_pixels(pre_alloc_matrices, pair_separation_bins, qso_angle, spec1, spec2, r):
    """
    Find all pixel pairs in QSO1,QSO2 that are closer than radius r
    :param pre_alloc_matrices: PreAllocMatrices
    :param pair_separation_bins: bins_2d.Bins2D
    :param qso_angle: float64
    :param spec1: [np.array, np.array, QSORecord]
    :param spec2: [np.array, np.array, QSORecord]
    :param r:
    :return:
    """

    # Note: not using pre_alloc_matrices.zero()

    # use law of cosines to find the distance between pairs of pixels
    qso_angle_cosine = np.cos(qso_angle)
    r_sq = np.square(r)

    # Note: throughout this method, "flux" means delta_f
    spec1_distances = spec1[0]
    spec1_flux = spec1[1]

    spec2_distances = spec2[0]
    spec2_flux = spec2[1]

    # create matrices with first dimension of spec1 data points,
    # second dimension of spec2 data points
    y = spec1_distances.size
    x = spec2_distances.size

    m1 = pre_alloc_matrices.m1[:y, :x]
    flux_products = pre_alloc_matrices.m2[:y, :x]
    mask_matrix = pre_alloc_matrices.mask1[:y, :x]
    r_parallel = pre_alloc_matrices.m4[:y, :x]
    r_transverse = pre_alloc_matrices.m5[:y, :x]
    spec1_distances_sq = pre_alloc_matrices.v1[:y]
    spec2_distances_sq = pre_alloc_matrices.v2[:x]

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
    np.multiply(r_parallel, 1 / BIN_SIZE)

    # r_ =  (r1 + r2)/2 * qso_angle
    np.add(spec1_distances[:, None], spec2_distances, out=r_transverse)
    np.multiply(r_transverse, qso_angle / (2 * BIN_SIZE), out=r_transverse)

    # add flux products for all nearby pairs, and bin by r_parallel, r_transverse
    pair_separation_bins.add_array_with_mask(flux_products,
                                             r_parallel,
                                             r_transverse,
                                             mask_matrix)


z_start = 2.1
z_end = 3.5
z_step = 0.001


def add_qso_pairs_to_bins(ar_distance, pairs, pairs_angles, spectra_with_metadata):
    pair_separation_bins = bins_2d.Bins2D(50, 50)
    pre_alloc_matrices = PreAllocMatrices()
    for i, j, k in pairs:
        # find distance between QSOs
        # qso1 = coord_set[i]
        # qso2 = coord_set[j]
        qso_angle = pairs_angles[k]
        r_parallel = abs(ar_distance[i] - ar_distance[j])
        mean_distance = (ar_distance[i] + ar_distance[j]) / 2
        r_transverse = mean_distance * qso_angle
        # print 'QSO pair with r_parallel %s, r_transverse %s' % (r_parallel, r_transverse)
        spec1 = spectra_with_metadata.return_spectrum(i)
        spec2 = spectra_with_metadata.return_spectrum(j)
        # TODO: read the default 200Mpc value from elsewhere
        find_nearby_pixels(pre_alloc_matrices, pair_separation_bins, qso_angle, spec1, spec2, 200)
        print 'intermediate number of pixel pairs in bins:', pair_separation_bins.ar_count.sum().astype(int)
    return pair_separation_bins


def profile_main():
    cd = comoving_distance.ComovingDistance(z_start, z_end, z_step)

    # x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')
    # min_distance = cd.comoving_distance_transverse(2.1, **fidcosmo)
    # print 'minimum distance', min_distance, 'Mpc/rad'

    # initialize data sources
    qso_record_table = table.Table(np.load('../../data/QSO_table.npy'))
    spectra_with_metadata = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table)

    # prepare data for quicker access
    qso_record_list = [QSORecord.from_row(i) for i in qso_record_table]
    ar_ra = np.array([i.ra for i in qso_record_list])
    ar_dec = np.array([i.dec for i in qso_record_list])
    ar_z = np.array([i.z for i in qso_record_list])
    ar_distance = cd.fast_comoving_distance(ar_z)
    print 'QSO table size:', len(ar_distance)

    # set maximum QSO angular separation to 200Mpc/h (in co-moving coordinates)
    # TODO: does the article assume h=100km/s/mpc?
    max_angular_separation = 200 * u.Mpc / (Planck13.comoving_transverse_distance(2.1) / u.radian)
    print 'maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree)

    # print ar_list
    coord_set = coord.SkyCoord(ra=ar_ra * u.degree, dec=ar_dec * u.degree,
                               distance=ar_distance * u.Mpc)
    # print coord_set

    # find all QSO pairs
    # for now, limit to a small set of the pairs, for a reasonable runtime
    x = matching.search_around_sky(coord_set[:2], coord_set[:50], max_angular_separation)

    pairs_with_unity = np.vstack((x[0], x[1], np.arange(x[0].size)))
    pairs = pairs_with_unity.T[pairs_with_unity[1] != pairs_with_unity[0]]
    pairs_angles = x[2].to(u.rad).value
    print 'number of QSO pairs:', pairs.size

    pair_separation_bins = add_qso_pairs_to_bins(ar_distance, pairs, pairs_angles, spectra_with_metadata)
    print 'total number of pixel pairs in bins:', pair_separation_bins.ar_count.sum().astype(int)


if settings.get_profile():
    cProfile.run('profile_main()', filename='generate_pair_list.prof', sort=2)
else:
    profile_main()

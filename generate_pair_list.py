__author__ = 'yishay'

import cProfile

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


def find_nearby_pixels(pair_separation_bins, qso_angle, spec1, spec2, r):
    """
    Find all pixel pairs in QSO1,QSO2 that are closer than radius r
    :param pair_separation_bins: bins_2d.Bins2D
    :param qso_angle: float64
    :param spec1: [np.array, np.array, QSORecord]
    :param spec2: [np.array, np.array, QSORecord]
    :param r:
    :return:
    """
    # for i in spec1:
    # find_nearby_pixels(qso_angle, spec2, i, r)


    # use law of cosines to find the distance between pairs of pixels
    qso_angle_cosine = np.cos(qso_angle)
    r_sq = np.square(r)

    # note: through this method, "flux" means delta_f
    spec1_distances = spec1[0]
    spec1_flux = spec1[1]

    spec2_distances = spec2[0]
    spec2_flux = spec2[1]

    spec1_distances_sq = np.square(spec1_distances)
    spec2_distances_sq = np.square(spec2_distances)

    # create matrices with first dimension of spec1 data points,
    # second dimension of spec2 data points

    spec1_times_spec2_dist = np.outer(spec1_distances, spec2_distances)

    spec1_spec2_dist_sq = np.add(- 2 * spec1_times_spec2_dist * qso_angle_cosine,
                                 spec1_distances_sq[:, None])
    spec1_spec2_dist_sq = np.add(spec1_spec2_dist_sq,
                                 spec2_distances_sq[None, :])

    # a matrix of flux products
    # TODO: add weights for a proper calculation of "xi(i,j)"
    flux_products = np.outer(spec1_flux, spec2_flux)

    # mask all elements that are close enough
    mask_matrix = spec1_spec2_dist_sq > r_sq

    r_parallel = np.abs(spec1_distances[:, None] - spec2_distances)
    r_transverse = qso_angle * (spec1_distances[:, None] + spec2_distances) / 2

    pair_separation_bins.add_array(flux_products[mask_matrix],
                             r_parallel[mask_matrix]/BIN_SIZE,
                             r_transverse[mask_matrix]/BIN_SIZE)


z_start = 2.1
z_end = 3.5
z_step = 0.001


def fast_linear_interpolate(f, x):
    x = np.asarray(x)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1

    return (x1 - x) * f[x0] + (x - x0) * f[x1]


def fast_comoving_distance(ar_z, _comoving_table_distance):
    ar_index = (ar_z - z_start) / (z_end - z_start)
    return fast_linear_interpolate(_comoving_table_distance, ar_index)


def add_qso_pairs_to_bins(ar_distance, pairs, pairs_angles, spectra_with_metadata):
    pair_separation_bins = bins_2d.Bins2D(50, 50)
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
        find_nearby_pixels(pair_separation_bins, qso_angle, spec1, spec2, 200)
    return pair_separation_bins


def profile_main():
    comoving_z_table = np.arange(z_start, z_end, z_step)
    comoving_distance_table = Planck13.comoving_distance(comoving_z_table).to(u.Mpc).value

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
    ar_distance = fast_comoving_distance(ar_z, comoving_distance_table)
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
    # for now, limit to up to 10th of the pairs, for a reasonable runtime
    x = matching.search_around_sky(coord_set[:1], coord_set[:50], max_angular_separation)

    pairs_with_unity = np.vstack((x[0], x[1], np.arange(x[0].size)))
    pairs = pairs_with_unity.T[pairs_with_unity[1] != pairs_with_unity[0]]
    pairs_angles = x[2].to(u.rad).value
    print 'number of QSO pairs:', pairs.size

    add_qso_pairs_to_bins(ar_distance, pairs, pairs_angles, spectra_with_metadata)


if settings.get_profile():
    cProfile.run('profile_main()', filename='generate_pair_list.prof', sort=2)
else:
    profile_main()

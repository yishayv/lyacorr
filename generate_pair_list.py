__author__ = 'yishay'

import cProfile

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.cosmology import Planck13
from astropy.coordinates import matching as matching
from astropy.coordinates import Angle
from astropy import table

import bins_2d
from read_spectrum_fits import QSORecord


def find_nearby_pixels(qso1, n, qso2, r):
    """
    Find the pixels in QSO2 within radius r of the nth-pixel QSO1
    :param qso1:
    :param qso2:
    """
    coord1 = coord.sky


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


def add_qso_pairs_to_bins(ar_distance, pairs, pairs_angles):
    PairSeparationBins = bins_2d.Bins2D(50, 50)
    for i, j, k in pairs:
        # find distance between QSOs
        # qso1 = coord_set[i]
        # qso2 = coord_set[j]
        qso_angle = pairs_angles[k]
        r_parallel = abs(ar_distance[i] - ar_distance[j])
        mean_distance = (ar_distance[i] + ar_distance[j]) / 2
        r_transverse = mean_distance * qso_angle
        # print 'QSO pair with r_parallel %s, r_transverse %s' % (r_parallel, r_transverse)
        # iterate over spectra and then call:

        PairSeparationBins.add(0, r_parallel, r_transverse)


def profile_main():
    comoving_table_z = np.arange(z_start, z_end, z_step)
    comoving_table_distance = Planck13.comoving_distance(comoving_table_z).to(u.Mpc).value

    # x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')
    # min_distance = cd.comoving_distance_transverse(2.1, **fidcosmo)
    # print 'minimum distance', min_distance, 'Mpc/rad'

    qso_record_table = table.Table(np.load('../../data/QSO_table.npy'))
    qso_record_list = [QSORecord.from_row(i) for i in qso_record_table]
    ar_ra = np.array([i.ra for i in qso_record_list])
    ar_dec = np.array([i.dec for i in qso_record_list])
    ar_z = np.array([i.z for i in qso_record_list])
    ar_distance = fast_comoving_distance(ar_z, comoving_table_distance)
    print 'QSO table size:', len(ar_distance)

    max_angular_separation = 200 * u.Mpc / (Planck13.comoving_transverse_distance(2.1) / u.radian)
    print 'maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree)

    # print ar_list
    coord_set = coord.SkyCoord(ra=ar_ra * u.degree, dec=ar_dec * u.degree,
                               distance=ar_distance * u.Mpc)
    # print coord_set

    # find all QSO pairs
    # for now, limit to up to 10th of the pairs, for a reasonable runtime
    x = matching.search_around_sky(coord_set[:17000], coord_set, max_angular_separation)

    pairs_with_unity = np.vstack((x[0], x[1], np.arange(x[0].size)))
    pairs = pairs_with_unity.T[pairs_with_unity[1] != pairs_with_unity[0]]
    pairs_angles = x[2].to(u.rad).value
    print 'number of QSO pairs:', pairs.size

    add_qso_pairs_to_bins(ar_distance, pairs, pairs_angles)


cProfile.run('profile_main()', sort=2)

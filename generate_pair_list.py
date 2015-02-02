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


def profile_main():
    # x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')
    # min_distance = cd.comoving_distance_transverse(2.1, **fidcosmo)
    # print 'minimum distance', min_distance, 'Mpc/rad'

    qso_record_table = table.Table(np.load('../../data/QSO_table.npy'))
    qso_record_list = [QSORecord.from_row(i) for i in qso_record_table]
    ra = [i.ra for i in qso_record_list]
    dec = [i.dec for i in qso_record_list]
    distance = [Planck13.comoving_distance(i.z) for i in qso_record_list]
    print 'QSO table size:', len(distance)

    max_angular_separation = 200 * u.Mpc / (Planck13.comoving_transverse_distance(2.1) / u.radian)
    print 'maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree)

    # print ar_list
    coord_set = coord.SkyCoord(ra=ra * u.degree, dec=dec * u.degree,
                               distance=distance)
    # print coord_set

    x = matching.search_around_sky(coord_set, coord_set, max_angular_separation)

    pairs_with_unity = np.vstack((x[0], x[1], np.arange(x[0].size)))
    pairs = pairs_with_unity.T[pairs_with_unity[1] != pairs_with_unity[0]]
    print 'number of QSO pairs:', pairs.size

    PairSeparationBins = bins_2d.Bins2D(50, 50)
    for i, j, k in pairs[:1000]:
        # find distance between QSOs
        # qso1 = coord_set[i]
        # qso2 = coord_set[j]
        qso_angle = x[2][k].to(u.rad).value
        r_parallel = abs(distance[i] - distance[j])
        mean_distance = (distance[i] + distance[j])/2
        r_transverse = mean_distance * qso_angle
        # print 'QSO pair with r_parallel %s, r_transverse %s' % (r_parallel, r_transverse)
        # iterate over spectra and then call:

        PairSeparationBins.add(0, r_parallel.value, r_transverse.value)

cProfile.run('profile_main()', sort=2)

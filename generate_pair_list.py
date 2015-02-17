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
from read_spectrum_fits import QSORecord
import comoving_distance
import calc_pixel_pairs
from numpy_spectrum_container import NpSpectrumContainer


settings = common_settings.Settings()

z_start = 1.8
z_end = 3.6
z_step = 0.001

cd = comoving_distance.ComovingDistance(z_start, z_end, z_step)


def profile_main():
    # x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')
    # min_distance = cd.comoving_distance_transverse(2.1, **fidcosmo)
    # print 'minimum distance', min_distance, 'Mpc/rad'

    # initialize data sources
    qso_record_table = table.Table(np.load('../../data/QSO_table.npy'))
    spectra_with_metadata = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table)
    delta_t_file = NpSpectrumContainer(True, len(qso_record_table), settings.get_delta_t_npy(),
                                       max_wavelength_count=1000)

    # prepare data for quicker access
    qso_record_list = [QSORecord.from_row(i) for i in qso_record_table]
    ar_ra = np.array([i.ra for i in qso_record_list])
    ar_dec = np.array([i.dec for i in qso_record_list])
    ar_z = np.array([i.z for i in qso_record_list])
    ar_distance = cd.fast_comoving_distance(ar_z)
    print 'QSO table size:', len(ar_distance)

    # set maximum QSO angular separation to 200Mpc/h (in co-moving coordinates)
    # TODO: does the article assume h=100km/s/mpc?
    # TODO: find a more precise value instead of z=1.9
    max_angular_separation = 200 * u.Mpc / (Planck13.comoving_transverse_distance(1.9) / u.radian)
    print 'maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree)

    # print ar_list
    coord_set = coord.SkyCoord(ra=ar_ra * u.degree, dec=ar_dec * u.degree,
                               distance=ar_distance * u.Mpc)
    # print coord_set

    # find all QSO pairs
    # for now, limit to a small set of the pairs, for a reasonable runtime
    x = matching.search_around_sky(coord_set[:800], coord_set[:800], max_angular_separation)

    pairs_with_unity = np.vstack((x[0], x[1], np.arange(x[0].size)))
    pairs = pairs_with_unity.T[pairs_with_unity[1] != pairs_with_unity[0]]
    pairs_angles = x[2].to(u.rad).value
    print 'number of QSO pairs:', pairs.size

    pair_separation_bins = calc_pixel_pairs.add_qso_pairs_to_bins(cd, ar_distance, pairs, pairs_angles,
                                                                  spectra_with_metadata, delta_t_file)
    print 'total number of pixel pairs in bins:', pair_separation_bins.ar_count.sum().astype(int)
    pair_separation_bins.save(settings.get_estimator_bins())


if settings.get_profile():
    cProfile.run('profile_main()', filename='generate_pair_list.prof', sort=2)
else:
    profile_main()

__author__ = 'yishay'

import cProfile
import itertools

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.cosmology import Planck13
from astropy.coordinates import matching as matching
from astropy.coordinates import Angle
from astropy import table
from mpi4py import MPI

import read_spectrum_hdf5
import common_settings
from read_spectrum_fits import QSORecord
import comoving_distance
import calc_pixel_pairs
from numpy_spectrum_container import NpSpectrumContainer
import bins_2d


settings = common_settings.Settings()

z_start = 1.8
z_end = 3.6
z_step = 0.001

cd = comoving_distance.ComovingDistance(z_start, z_end, z_step)

comm = MPI.COMM_WORLD


def r_print(*args):
    if comm.rank == 0:
        print 'ROOT:',
        for i in args:
            print i,
        print


def l_print(*args):
    print comm.rank, ':',
    for i in args:
        print i,
    print


def get_chunks(num_items, comm_size):
    chunk_sizes = np.zeros(comm_size, dtype=int)
    chunk_offsets = np.zeros(comm_size, dtype=int)

    chunk_sizes[:] = num_items // comm_size
    chunk_sizes[:num_items % comm_size] += 1
    chunk_offsets = np.roll(np.cumsum(chunk_sizes), 1)
    chunk_offsets[0] = 0
    return chunk_sizes, chunk_offsets


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
    r_print('QSO table size:', len(ar_distance))

    # set maximum QSO angular separation to 200Mpc/h (in co-moving coordinates)
    # TODO: does the article assume h=100km/s/mpc?
    # TODO: find a more precise value instead of z=1.9
    max_angular_separation = 200 * u.Mpc / (Planck13.comoving_transverse_distance(1.9) / u.radian)
    r_print('maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree))

    # print ar_list
    coord_set = coord.SkyCoord(ra=ar_ra * u.degree, dec=ar_dec * u.degree,
                               distance=ar_distance * u.Mpc)
    # print coord_set

    # find all QSO pairs
    # for now, limit to a small set of the pairs, for a reasonable runtime

    chunk_sizes, chunk_offsets = get_chunks(len(coord_set), comm.size)

    local_start_index = chunk_offsets[comm.rank]
    local_end_index = local_start_index + chunk_sizes[comm.rank]
    l_print('matching objects in range:', local_start_index, 'to', local_end_index)
    # each node matches a range of objects against the full list.
    count = matching.search_around_sky(coord_set[local_start_index:local_end_index],
                                       coord_set,
                                       max_angular_separation)

    # search around sky returns indices in the input lists.
    # each node should add its offset to get the QSO index in the original list (only for x[0]).
    # qso2 which contains the unmodified index to the full list of QSOs.
    # the third vector is a count so we can keep a reference to the angles vector.
    local_pairs_with_unity = np.vstack((count[0] + local_start_index,
                                        count[1],
                                        np.arange(count[0].size)))

    local_pair_angles = count[2].to(u.rad).value
    l_print('number of QSO pairs (including identity pairs):', count[0].size)
    l_print('angle vector size:', local_pair_angles.size)

    # remove pairs of the same QSO.
    pairs = local_pairs_with_unity.T[local_pairs_with_unity[1] != local_pairs_with_unity[0]]
    # l_print(pairs)
    l_print('number of QSO pairs:', pairs.shape[0])
    # l_print('angle vector:', x[2])
    local_pair_separation_bins = \
        calc_pixel_pairs.add_qso_pairs_to_bins(cd, pairs, local_pair_angles, delta_t_file)
    # l_print(local_qso1 + local_start_index)
    l_print('local pair count:', local_pair_separation_bins.ar_count.sum())

    pair_separation_bins_count = np.zeros(shape=(comm.size, calc_pixel_pairs.NUM_BINS_X, calc_pixel_pairs.NUM_BINS_Y))
    pair_separation_bins_flux = np.zeros(shape=(comm.size, calc_pixel_pairs.NUM_BINS_X, calc_pixel_pairs.NUM_BINS_Y))
    comm.Gatherv(local_pair_separation_bins.ar_count, pair_separation_bins_count)
    comm.Gatherv(local_pair_separation_bins.ar_flux, pair_separation_bins_flux)

    if comm.rank == 0:
        # TODO: rewrite!
        list_pair_separation_bins = [bins_2d.Bins2D.from_np_arrays(count, flux) for count, flux in
                                     itertools.izip(pair_separation_bins_count, pair_separation_bins_flux)]
        if list_pair_separation_bins:
            pair_separation_bins = reduce(lambda x, y: x + y, list_pair_separation_bins,
                                          bins_2d.Bins2D.init_as(list_pair_separation_bins[0]))

            r_print('total number of pixel pairs in bins:', pair_separation_bins.ar_count.sum().astype(int))
            pair_separation_bins.save(settings.get_estimator_bins())
        else:
            print('no results received.')


if settings.get_profile():
    cProfile.run('profile_main()', filename='generate_pair_list.prof', sort=2)
else:
    profile_main()

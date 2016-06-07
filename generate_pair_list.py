"""
    Computes the Lyman-alpha forest auto-correlation estimator.
    The work is split between MPI nodes based on the first QSO in each possible pair.
    Partial data is gathered and the correlation estimator file is saved after processing each sub-chunk.
"""
import cProfile
import itertools

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import matching as matching
from astropy.coordinates import Angle
from astropy import table
from mpi4py import MPI

import common_settings
from data_access.read_spectrum_fits import QSORecord
from physics_functions import comoving_distance
import calc_pixel_pairs
from data_access.numpy_spectrum_container import NpSpectrumContainer
import mpi_helper
from physics_functions.spherical_math import SkyGroups, find_spherical_mean_deg

settings = common_settings.Settings()

z_start = 1.8
z_end = 3.6
z_step = 0.001

cd = comoving_distance.ComovingDistance(z_start, z_end, z_step)

comm = MPI.COMM_WORLD


class SubChunkHelper:
    def __init__(self):
        self.pair_separation_bins = None

    def add_pairs_in_sub_chunk(self, delta_t_file, local_pair_angles, pairs, pixel_pairs):
        local_pair_separation_bins = \
            pixel_pairs.add_qso_pairs_to_bins(pairs, local_pair_angles, delta_t_file)

        mpi_helper.l_print('local pair count:', local_pair_separation_bins.get_pair_count())
        local_pair_separation_bins_array = local_pair_separation_bins.get_data_as_array()
        local_pair_separation_bins_metadata = local_pair_separation_bins.get_metadata()
        local_array_shape = local_pair_separation_bins_array.shape
        array_block_size = np.prod(local_array_shape[1:])

        comm.Barrier()
        mpi_helper.r_print("BEGIN GATHER")
        mpi_helper.l_print_no_barrier('local array shape:', local_array_shape)
        array_counts = comm.allgather(local_array_shape[0])

        pair_separation_bins_array = None
        array_endings = np.cumsum(array_counts)
        array_displacements = array_endings - np.array(array_counts)
        if comm.rank == 0:
            mpi_helper.r_print('array count:', array_counts)
            root_array_shape = (np.sum(array_counts),) + local_array_shape[1:]
            mpi_helper.r_print('root array shape:', root_array_shape)
            pair_separation_bins_array = np.ones(shape=root_array_shape, dtype=np.float64)

        send_buf = [local_pair_separation_bins_array,
                    local_array_shape[0] * array_block_size]
        receive_buf = [pair_separation_bins_array, np.multiply(array_counts, array_block_size),
                       np.multiply(array_displacements, array_block_size), MPI.DOUBLE]

        # mpi_helper.l_print(send_buf)

        comm.Gatherv(sendbuf=send_buf, recvbuf=receive_buf)
        list_pair_separation_bins_metadata = comm.gather(local_pair_separation_bins_metadata)
        comm.Barrier()
        mpi_helper.r_print("END_GATHER")

        if comm.rank == 0:
            # mpi_helper.r_print(receive_buf[0][0][0:10])
            list_pair_separation_bins = [
                type(local_pair_separation_bins).load_from(
                    pair_separation_bins_array[array_displacements[rank]:array_endings[rank]], metadata)
                for rank, metadata in enumerate(list_pair_separation_bins_metadata)]

            # initialize bins only if this is the first time we get here
            # for now use a function level static variable
            if not self.pair_separation_bins:
                self.pair_separation_bins = local_pair_separation_bins.init_as(local_pair_separation_bins)

            # add new results to existing bins
            if list_pair_separation_bins:
                for i in list_pair_separation_bins:
                    for g in i.dict_bins_2d_data.keys():
                        mpi_helper.l_print_no_barrier(np.sum(i.dict_bins_2d_data[g].ar_count))
                self.pair_separation_bins = reduce(lambda x, y: x + y, list_pair_separation_bins,
                                                   self.pair_separation_bins)

                mpi_helper.r_print('total number of pixel pairs in bins:',
                                   self.pair_separation_bins.get_pair_count())
                self.pair_separation_bins.flush()
                pixel_pairs.significant_qso_pairs.save(settings.get_significant_qso_pairs_npy())
            else:
                print('no results received.')


def profile_main():
    # x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')
    # min_distance = cd.comoving_distance_transverse(2.1, **fidcosmo)
    # print('minimum distance', min_distance, 'Mpc/rad')

    # initialize data sources
    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    delta_t_file = NpSpectrumContainer(True, num_spectra=len(qso_record_table), filename=settings.get_delta_t_npy(),
                                       max_wavelength_count=1000)

    # prepare data for quicker access
    qso_record_list = [QSORecord.from_row(i) for i in qso_record_table]
    ar_ra = np.array([i.ra for i in qso_record_list])
    ar_dec = np.array([i.dec for i in qso_record_list])
    ar_z = np.array([i.z for i in qso_record_list])
    ar_distance = cd.fast_comoving_distance(ar_z)
    mpi_helper.r_print('QSO table size:', len(ar_distance))

    # TODO: find a more precise value instead of z=1.9
    # set maximum QSO angular separation to 200Mpc/h (in co-moving coordinates)
    # the article assumes h is measured in units of 100km/s/mpc
    radius = (200. * (100. * u.km / (u.Mpc * u.s)) / cd.H0).value
    max_angular_separation = radius / (cd.comoving_distance(1.9) / u.radian)
    mpi_helper.r_print('maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree))

    # print(ar_list)
    coord_set = coord.SkyCoord(ra=ar_ra * u.degree, dec=ar_dec * u.degree,
                               distance=ar_distance * u.Mpc)
    # print(coord_set)

    # find all QSO pairs
    chunk_sizes, chunk_offsets = mpi_helper.get_chunks(len(coord_set), comm.size)

    local_start_index = chunk_offsets[comm.rank]
    local_end_index = local_start_index + chunk_sizes[comm.rank]
    mpi_helper.l_print('matching objects in range:', local_start_index, 'to', local_end_index)
    # each node matches a range of objects against the full list.
    count = matching.search_around_sky(coord_set[local_start_index:local_end_index],
                                       coord_set,
                                       max_angular_separation)

    # search around sky returns indices in the input lists.
    # each node should add its offset to get the QSO index in the original list (only for x[0]).
    # qso2 which contains the unmodified index to the full list of QSOs.
    # the third vector is a count so we can keep a reference to the angles vector.
    local_qso_index_1 = count[0] + local_start_index
    local_qso_index_2 = count[1]

    # find the mean ra,dec for each pair
    local_qso_ra_pairs = np.vstack((ar_ra[local_qso_index_1], ar_ra[local_qso_index_2]))
    local_qso_dec_pairs = np.vstack((ar_dec[local_qso_index_1], ar_dec[local_qso_index_2]))
    # we can safely assume that separations is small enough so we don't have catastrophic cancellation of the mean,
    # so checking the unit radius value is not required
    local_pair_means_ra, local_pair_means_dec, _ = find_spherical_mean_deg(local_qso_ra_pairs, local_qso_dec_pairs,
                                                                           axis=0)

    sky_groups = SkyGroups(nside=settings.get_healpix_nside())
    group_id = sky_groups.get_group_ids(local_pair_means_ra, local_pair_means_dec)

    local_qso_pairs_with_unity = np.vstack((local_qso_index_1,
                                            local_qso_index_2,
                                            group_id,
                                            np.arange(count[0].size)))

    local_qso_pair_angles = count[2].to(u.rad).value
    mpi_helper.l_print('number of QSO pairs (including identity pairs):', count[0].size)
    mpi_helper.l_print('angle vector size:', local_qso_pair_angles.size)

    # remove pairs of the same QSO.
    # local_qso_pairs = local_qso_pairs_with_unity.T[local_qso_pairs_with_unity[1] != local_qso_pairs_with_unity[0]]

    # remove pairs of the same QSO, which have different [plate,mjd,fiber]
    # assume that QSOs within roughly 10 arc-second (5e-5 rads) are the same object.
    local_qso_pairs = local_qso_pairs_with_unity.T[local_qso_pair_angles > 5e-5]

    mpi_helper.l_print('total number of redundant objects removed:', local_qso_pairs_with_unity.shape[1] -
                       local_qso_pairs.shape[0] - chunk_sizes[comm.rank])

    # l_print(pairs)
    mpi_helper.l_print('number of QSO pairs:', local_qso_pairs.shape[0])
    # l_print('angle vector:', x[2])

    if settings.get_enable_weighted_median_estimator():
        accumulator_type = calc_pixel_pairs.accumulator_types.histogram
        assert not settings.get_enable_weighted_mean_estimator(), "Median and mean estimators are mutually exclusive."
        assert not settings.get_enable_estimator_subsamples(), "Subsamples not supported for histogram."
    elif settings.get_enable_weighted_mean_estimator():
        if settings.get_enable_estimator_subsamples():
            accumulator_type = calc_pixel_pairs.accumulator_types.mean_subsample
        else:
            accumulator_type = calc_pixel_pairs.accumulator_types.mean
    else:
        assert False, "Either median or mean estimators must be specified."

    pixel_pairs_object = calc_pixel_pairs.PixelPairs(cd, radius, accumulator_type=accumulator_type)
    # divide the work into sub chunks
    # Warning: the number of sub chunks must be identical for all nodes because gather is called after each sub chunk.
    # divide by comm.size to make sub chunk size independent of number of nodes.
    num_sub_chunks_per_node = settings.get_mpi_num_sub_chunks() // comm.size
    pixel_pair_sub_chunks = mpi_helper.get_chunks(local_qso_pairs.shape[0], num_sub_chunks_per_node)
    sub_chunk_helper = SubChunkHelper()
    for i, j, k in itertools.izip(pixel_pair_sub_chunks[0], pixel_pair_sub_chunks[1], itertools.count()):
        sub_chunk_start = j
        sub_chunk_end = j + i
        mpi_helper.l_print("sub_chunk: size", i, ", starting at", j, ",", k, "out of", len(pixel_pair_sub_chunks[0]))
        sub_chunk_helper.add_pairs_in_sub_chunk(delta_t_file, local_qso_pair_angles,
                                                local_qso_pairs[sub_chunk_start:sub_chunk_end],
                                                pixel_pairs_object)


if settings.get_profile():
    cProfile.run('profile_main()', filename='generate_pair_list.prof', sort=2)
else:
    profile_main()

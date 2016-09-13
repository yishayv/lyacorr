"""
    Computes the Lyman-alpha forest auto-correlation estimator.
    The work is split between MPI nodes based on the first QSO in each possible pair.
    Partial data is gathered and the correlation estimator file is saved after processing each sub-chunk.
"""
import cProfile
import itertools

import numpy as np
from astropy import coordinates as coord
from astropy import table
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import matching as matching
from mpi4py import MPI

import calc_pixel_pairs
import common_settings
import mpi_helper
from data_access.numpy_spectrum_container import NpSpectrumContainer
from data_access.read_spectrum_fits import QSORecord
from physics_functions import comoving_distance
from physics_functions.spherical_math import SkyGroups, find_spherical_mean_deg
from python_compat import reduce, zip, range

settings = common_settings.Settings()

z_start = 1.8
z_end = 3.6
z_step = 0.001

cd = comoving_distance.ComovingDistance(z_start, z_end, z_step)

comm = MPI.COMM_WORLD


def get_bundles(start, end, size):
    """
    split a range into bundles.
    each bundle is a tuple with an offset and size.
    :type start: int
    :type end: int
    :type size: int
    :rtype: tuple(int, int)
    """
    offsets = range(start, end, size)
    sizes = [size] * len(offsets)
    sizes[-1] = end - offsets[-1]
    return zip(offsets, sizes)


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
                    for g in i.dict_bins_3d_data.keys():
                        mpi_helper.l_print_no_barrier(np.sum(i.dict_bins_3d_data[g].ar_count))
                self.pair_separation_bins = reduce(lambda x, y: x + y, list_pair_separation_bins,
                                                   self.pair_separation_bins)

                mpi_helper.r_print('total number of pixel pairs in bins:',
                                   self.pair_separation_bins.get_pair_count())
                self.pair_separation_bins.flush()
                pixel_pairs.significant_qso_pairs.save(settings.get_significant_qso_pairs_npy())
            else:
                print('no results received.')


def generate_pairs(ar_dec, ar_ra, coord_permutation, coord_set, local_end_index, local_start_index,
                   max_angular_separation):
    """
    Generate QSO pairs, in bundles.
    Each time, a bundle of QSOs is matched against the full list
    :param ar_dec: declination array
    :param ar_ra: right ascension array
    :param coord_permutation: pseudo-random permutation of qso indices, for counting each pair only once
    :param coord_set: coordinate set of all QSOs
    :param local_end_index: last QSO index for this MPI node
    :param local_start_index: first QSO index for this MPI node
    :param max_angular_separation: maximum angular separation for sky search
    :return:
    """
    # each node matches a range of objects against the full list.

    qso_bundle_size = settings.get_qso_bundle_size()

    bundles = list(get_bundles(local_start_index, local_end_index, qso_bundle_size))
    num_bundles = len(bundles)

    # if the number of bundles is not the same across all MPI nodes (which should be rare),
    # we are not allowed to use the synchronized version of print.
    num_bundles = comm.allgather(num_bundles)
    mpi_helper.r_print("number of QSO bundles per node:", num_bundles)
    is_num_bundles_equal = np.all(np.array(num_bundles) == num_bundles[0])
    print_func = mpi_helper.l_print if is_num_bundles_equal else mpi_helper.l_print_no_barrier
    mpi_helper.r_print('using print function:', print_func.__name__)

    for bundle_start, bundle_size in bundles:
        print_func('matching ', bundle_size, ' objects, starting at :', bundle_start)
        print_func('node progress:{:.1f}%'.format(
            100. * (bundle_start - local_start_index) / (local_end_index - local_start_index)))
        count = matching.search_around_sky(coord_set[bundle_start:bundle_start + bundle_size],
                                           coord_set,
                                           max_angular_separation)
        # search around sky returns indices in the input lists.
        # each node should add its offset to get the QSO index in the original list (only for x[0]).
        # qso2 contains the unmodified index to the full list of QSOs.
        # the third vector is a count so we can keep a reference to the angles vector.
        qso_index_1 = count[0] + bundle_start
        qso_index_2 = count[1]
        # find the mean ra,dec for each pair
        qso_ra_pairs = np.vstack((ar_ra[qso_index_1], ar_ra[qso_index_2]))
        qso_dec_pairs = np.vstack((ar_dec[qso_index_1], ar_dec[qso_index_2]))
        # we can safely assume that separations are small enough so we don't have catastrophic cancellation of the mean,
        # so checking the unit radius value is not required
        pair_means_ra, local_pair_means_dec, _ = find_spherical_mean_deg(qso_ra_pairs, qso_dec_pairs,
                                                                         axis=0)
        sky_groups = SkyGroups(nside=settings.get_healpix_nside())
        group_id = sky_groups.get_group_ids(pair_means_ra, local_pair_means_dec)
        qso_pairs_with_unity = np.vstack((qso_index_1,
                                          qso_index_2,
                                          group_id,
                                          np.arange(count[0].size)))
        qso_pair_angles = count[2].to(u.rad).value
        print_func('number of QSO pairs (including identity pairs):', count[0].size)
        print_func('angle vector size:', qso_pair_angles.size)
        # remove pairs of the same QSO, which have different [plate,mjd,fiber]
        # assume that QSOs within roughly 10 arc-second (5e-5 rads) are the same object.
        # also keep only 1 instance of each pair (keep only: qso1_index_hash < qso2_index_hash)
        qso_pairs = qso_pairs_with_unity.T[np.logical_and(qso_pair_angles > 5e-5,
                                                          coord_permutation[qso_pairs_with_unity[0]] <
                                                          coord_permutation[qso_pairs_with_unity[1]])]
        print_func('total number of redundant objects removed:',
                   qso_pairs_with_unity.shape[1] - qso_pairs.shape[0])
        print_func('number of QSO pairs:', qso_pairs.shape[0])
        yield qso_pair_angles, qso_pairs


def profile_main():
    # x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')
    # min_distance = cd.comoving_distance_transverse(2.1, **fidcosmo)
    # print('minimum distance', min_distance, 'Mpc/rad')

    # initialize data sources
    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    if settings.get_ism_only_mode():
        delta_t_filename = settings.get_forest_ism_npy()
    else:
        delta_t_filename = settings.get_delta_t_npy()

    delta_t_file = NpSpectrumContainer(True, num_spectra=len(qso_record_table), filename=delta_t_filename,
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
    radius_quantity = (200. * (100. * u.km / (u.Mpc * u.s)) / cd.H0)  # type: u.Quantity
    radius = radius_quantity.value
    max_angular_separation = radius / (cd.comoving_distance(1.9) / u.radian)
    mpi_helper.r_print('maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree))

    # print(ar_list)
    coord_set = coord.SkyCoord(ra=ar_ra * u.degree, dec=ar_dec * u.degree,
                               distance=ar_distance * u.Mpc)
    # create a random permutation of the coordinate set
    # (this is done to balance the load on the nodes)
    coord_permutation = None
    if comm.rank == 0:
        coord_permutation = np.random.permutation(len(coord_set))
    coord_permutation = comm.bcast(coord_permutation)

    # find all QSO pairs
    chunk_sizes, chunk_offsets = mpi_helper.get_chunks(len(coord_set), comm.size)

    local_start_index = chunk_offsets[comm.rank]
    local_end_index = local_start_index + chunk_sizes[comm.rank]

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
    # NOTE: we no longer divide by comm.size to make sub chunk size independent of number of nodes,
    #       because pairs are generated in bundles, instead of once at the beginning.
    num_sub_chunks_per_node = settings.get_mpi_num_sub_chunks()

    sub_chunk_helper = SubChunkHelper()
    for local_qso_pair_angles, local_qso_pairs in generate_pairs(
            ar_dec, ar_ra, coord_permutation, coord_set, local_end_index, local_start_index,
            max_angular_separation):

        pixel_pair_sub_chunks = mpi_helper.get_chunks(local_qso_pairs.shape[0], num_sub_chunks_per_node)
        for i, j, k in zip(pixel_pair_sub_chunks[0], pixel_pair_sub_chunks[1], itertools.count()):
            sub_chunk_start = j
            sub_chunk_end = j + i
            mpi_helper.l_print("sub_chunk: size", i, ", starting at", j, ",", k, "out of",
                               len(pixel_pair_sub_chunks[0]))
            sub_chunk_helper.add_pairs_in_sub_chunk(delta_t_file, local_qso_pair_angles,
                                                    local_qso_pairs[sub_chunk_start:sub_chunk_end],
                                                    pixel_pairs_object)


if settings.get_profile():
    cProfile.run('profile_main()', filename='generate_pair_list.prof', sort=2)
else:
    profile_main()

"""
    Computes the Lyman-alpha forest auto-correlation estimator.
    The work is split between MPI nodes based on the first QSO in each possible pair.
    Partial data is gathered and the correlation estimator file is saved after processing each sub-chunk.
"""
import cProfile
import collections

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
from data_access.numpy_spectrum_container import NpSpectrumContainer
import mpi_helper
import calc_covariance

settings = common_settings.Settings()

z_start = 1.8
z_end = 3.6
z_step = 0.001

cd = comoving_distance.ComovingDistance(z_start, z_end, z_step)

comm = MPI.COMM_WORLD


def gather_concatenate_big_array(local_array, sum_axis=0, max_nbytes=2 ** 31 - 1):
    global_array = None
    global_nbytes = comm.allgather(local_array.nbytes)
    if local_array.shape[0] > 0:
        assert np.take(local_array, [0],
                       axis=sum_axis).nbytes <= max_nbytes, "array elements must not be larger than max_nbytes"
    mpi_helper.r_print("global_nbytes", global_nbytes)
    if np.array(global_nbytes).sum() > max_nbytes:  # 2 ** 31):
        # split the array along the summation axis
        axis_end = local_array.shape[sum_axis]
        axis_split_point = axis_end // 2
        local_array_first_part = np.take(local_array, np.arange(0, axis_split_point), axis=sum_axis)
        local_array_last_part = np.take(local_array, np.arange(axis_split_point, axis_end), axis=sum_axis)
        global_array_first_part = gather_concatenate_big_array(local_array_first_part, sum_axis, max_nbytes)
        global_array_last_part = gather_concatenate_big_array(local_array_last_part, sum_axis, max_nbytes)
        if comm.rank == 0:
            global_array = np.concatenate((global_array_first_part, global_array_last_part), axis=sum_axis)
    else:
        global_array_list = comm.gather(local_array)
        if comm.rank == 0:
            # mpi_helper.r_print(global_array_list)
            global_array = np.concatenate(global_array_list)
    return global_array


def profile_main():
    local_array = np.array([[1., 2], [3, 4], [5, 6], [7, 8], [9., 10]]) if comm.rank == 0 else np.array(
        [[20, 21], [22, 23]])
    temp1 = gather_concatenate_big_array(local_array, sum_axis=0,
                                         max_nbytes=48)
    mpi_helper.r_print(temp1)
    # x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')
    # min_distance = cd.comoving_distance_transverse(2.1, **fidcosmo)
    # print 'minimum distance', min_distance, 'Mpc/rad'

    # initialize data sources
    mpi_helper.l_print("Loading QSO record table")
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
    # the article assumes h=100km/s/mpc
    radius = (200. * (100. * u.km / (u.Mpc * u.s)) / cd.H0).value
    max_angular_separation = radius / (cd.comoving_distance(1.9) / u.radian)
    mpi_helper.r_print('maximum separation of QSOs:', Angle(max_angular_separation).to_string(unit=u.degree))

    # create a CovarianceMatrix instance:
    cov = calc_covariance.CovarianceMatrix(cd=cd, radius=radius)

    # print ar_list
    coord_set = coord.SkyCoord(ra=ar_ra * u.degree, dec=ar_dec * u.degree,
                               distance=ar_distance * u.Mpc)
    # print coord_set

    # find all QSO pairs
    chunk_sizes, chunk_offsets = mpi_helper.get_chunks(len(coord_set), comm.size)

    local_start_index = chunk_offsets[comm.rank]
    local_end_index = local_start_index + chunk_sizes[comm.rank]
    mpi_helper.l_print('Matching objects in range:', local_start_index, 'to', local_end_index)
    # each node matches a range of objects against the full list.
    count = matching.search_around_sky(coord_set[local_start_index:local_end_index],
                                       coord_set,
                                       max_angular_separation)

    # search around sky returns indices for the input lists.
    # each node should add its offset to get the QSO index in the original list (only for x[0]),
    # so that each item contains the unmodified index to the full list of QSOs.
    # the third vector element is the angle between each pair.
    local_qso_pairs_with_unity = np.vstack((count[0] + local_start_index,
                                            count[1],
                                            count[2].to(u.rad).value))

    mpi_helper.l_print('Number of QSO pairs (including identity pairs):', count[0].size)

    # remove pairs of the same QSO, which have different [plate,mjd,fiber]
    # assume that QSOs within roughly 10 arc-second (5e-5 rads) are the same object.
    local_qso_pairs = local_qso_pairs_with_unity.T[local_qso_pairs_with_unity[2] > 5e-5]

    mpi_helper.l_print('Total number of redundant objects removed:', local_qso_pairs_with_unity.shape[1] -
                       local_qso_pairs.shape[0] - chunk_sizes[comm.rank])

    # l_print(pairs)
    mpi_helper.l_print('Number of QSO pairs:', local_qso_pairs.shape[0])
    # l_print('angle vector:', x[2])

    assert settings.get_enable_weighted_mean_estimator(), "covariance requires mean correlation estimator"
    assert not settings.get_enable_weighted_median_estimator(), "covariance requires mean correlation estimator"

    # gather all the qso pairs to rank 0
    mpi_helper.r_print("Gathering QSO pairs")
    mpi_helper.l_print("qso pairs shape:", local_qso_pairs.shape)
    global_qso_pairs = gather_concatenate_big_array(local_qso_pairs, sum_axis=0)

    # initialize variable for non-zero ranks
    global_random_sample = None
    # define count of quads for each chunk
    sample_chunk_size = 40000
    local_random_sample = np.zeros((sample_chunk_size, 10))
    global_pair_dict = None

    if comm.rank == 0:
        mpi_helper.r_print(
            "Gathered QSO pairs, count={0}".format(global_qso_pairs.shape[0]))
        assert isinstance(global_qso_pairs, np.ndarray)

        # build a 2-level dictionary of all QSO pairs.
        global_pair_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: np.inf))
        for qso1, qso2, angle in global_qso_pairs:
            global_pair_dict[qso1][qso2] = angle

    iteration_number = 0
    while True:
        # set chunk size
        if comm.rank == 0:
            mpi_helper.r_print(
                "Iteration {0}".format(iteration_number))
            global_random_sample = create_random_sample(global_qso_pairs, global_pair_dict, sample_chunk_size)
            mpi_helper.r_print("random sample shape:", global_random_sample.shape)

        comm.Scatter(global_random_sample, local_random_sample)

        # reshape the local array to form pairs of pairs
        mpi_helper.l_print(local_random_sample.shape)
        for quad in local_random_sample:
            cov.add_quad(qso_angle12=quad[4], qso_angle34=quad[9],
                         max_range_parallel=radius, max_range_transverse=radius,
                         spec1_index=quad[0], spec2_index=quad[1],
                         spec3_index=quad[2], spec4_index=quad[3],
                         delta_t_file=delta_t_file)

        global_ar_covariance = np.zeros((50, 50, 50, 50, 3))
        comm.Reduce(
            [cov.ar_covariance, MPI.DOUBLE],
            [global_ar_covariance, MPI.DOUBLE],
            op=MPI.SUM, root=0)
        if comm.rank == 0:
            mpi_helper.r_print("Partial Covariance Stats:", global_ar_covariance.sum(axis=(0, 1, 2, 3)))
            np.save(settings.get_correlation_estimator_covariance_npy(), global_ar_covariance)

        iteration_number += 1


# @profile
def create_random_sample(global_qso_pairs, global_pair_dict, sample_chunk_size):
    # create a random sample of pairs
    """

    :type global_qso_pairs: np.multiarray.ndarray
    :type global_pair_dict: collections.defaultdict(dict)
    :type sample_chunk_size: int
    :rtype: np.multiarray.ndarray
    """
    ar_probabilities = np.reciprocal(global_qso_pairs[:, 2])
    # normalize probability vector
    ar_probabilities /= ar_probabilities.sum()
    # draw a random pair as qso1, qso2
    random_sample_indices = np.random.choice(np.arange(global_qso_pairs.shape[0]),
                                             sample_chunk_size * comm.size, p=ar_probabilities)
    random_sample_first_qso_pair = global_qso_pairs[random_sample_indices]

    random_sample_quad = np.zeros((sample_chunk_size, 10))

    for i in np.arange(random_sample_first_qso_pair.shape[0]):
        pair12 = random_sample_first_qso_pair[i]
        # find two more QSOs around the first pair.
        # without loss of generality, we can search around qso1
        qso1 = pair12[0]
        qso2 = pair12[1]
        angle12 = pair12[2]

        qso3_candidates_dict = global_pair_dict[qso1]
        ar_qso3_candidates = np.fromiter(qso3_candidates_dict.iterkeys(), dtype=float)
        ar_qso3_candidates_angles = np.fromiter(qso3_candidates_dict.itervalues(), dtype=float)
        ar_qso3_candidates_probabilities = 1 / ar_qso3_candidates_angles
        ar_qso3_candidates_probabilities /= ar_qso3_candidates_probabilities.sum()

        all_angles_in_range = False
        while not all_angles_in_range:
            # draw qso3 randomly from QSOs around qso1
            qso3 = np.random.choice(ar_qso3_candidates, 1, p=ar_qso3_candidates_probabilities)[0]

            # draw qso4 randomly from QSOs around qso3
            qso4_candidates_dict = global_pair_dict[qso3]
            ar_qso4_candidates = np.fromiter(qso4_candidates_dict.iterkeys(), dtype=float)
            ar_qso4_candidates_angles = np.fromiter(qso4_candidates_dict.itervalues(), dtype=float)
            ar_qso4_candidates_probabilities = 1 / ar_qso4_candidates_angles
            ar_qso4_candidates_probabilities /= ar_qso4_candidates_probabilities.sum()
            qso4 = np.random.choice(ar_qso4_candidates, 1, p=ar_qso4_candidates_probabilities)[0]

            # find the remaining angles: between 1-3, 2-4, 1-4, 2-3, 3-4
            angle13 = global_pair_dict[qso1][qso3]
            angle24 = global_pair_dict[qso2][qso4]
            angle14 = global_pair_dict[qso1][qso4]
            angle23 = global_pair_dict[qso2][qso3]
            angle34 = global_pair_dict[qso3][qso4]

            random_sample_quad[i] = np.array(
                [qso1, qso2, qso3, qso4, angle12, angle13, angle14, angle23, angle24, angle34])
            if not np.any(np.isinf(random_sample_quad[i])):
                all_angles_in_range = True

    print "end of quad sample."
    return random_sample_quad


if settings.get_profile():
    cProfile.run('profile_main()', filename='generate_covariance.prof', sort=2)
else:
    profile_main()

import astropy.units as u
import healpy as hp
import numpy as np
import numpy.random as random
import numpy.random.mtrand
from astropy.coordinates import SkyCoord, Longitude, Latitude
from mpi4py import MPI

from mpi_helper import r_print

comm = MPI.COMM_WORLD
enable_intersection = True
ar_map_nside = 2048

# use a different random seed for each MPI rank, but still use predictable results for easier debugging.
_ = np.random.mtrand.RandomState(comm.rank)


def ra_dec2ang(ra, dec):
    return (90. - dec) * np.pi / 180., ra / 180. * np.pi


def main_loop(ar_map, max_bin_angle, outer_disc_mean, outer_disc, max_angular_separation):
    ar_product = np.zeros(shape=num_bins)
    ar_weights = np.zeros(shape=num_bins)
    chosen_indices = np.random.choice(outer_disc, size=100, replace=False)
    for index in chosen_indices:
        vec_a = hp.pix2vec(ar_map_nside, index)
        inner_disc = hp.query_disc(ar_map_nside, vec=vec_a, radius=max_angular_separation.to(u.rad).value)
        if enable_intersection:
            inner_disc = np.intersect1d(inner_disc, outer_disc, assume_unique=True)
        vec_b = hp.pix2vec(ar_map_nside, inner_disc)
        ar_ang_dist_with_zero = hp.rotator.angdist(vec_b, vec_a)
        a = index
        b = inner_disc[ar_ang_dist_with_zero > 0]
        ar_ang_dist = ar_ang_dist_with_zero[ar_ang_dist_with_zero > 0]
        ar_bins_float = ar_ang_dist / max_bin_angle * num_bins  # type: np.ndarray
        ar_bins = ar_bins_float.astype(int)
        # TODO: filter NaNs earlier so that they don't decrease the correlation
        pair_product = np.nan_to_num((ar_map[a] - outer_disc_mean) * (ar_map[b] - outer_disc_mean))
        ar_product += np.bincount(ar_bins, weights=pair_product, minlength=num_bins)
        ar_weights += np.bincount(ar_bins, minlength=num_bins)

    return ar_product, ar_weights


# load Planck map on the root node
ar_map_shape = None
ar_map_0 = None
ar_map_0_log = None
if comm.rank == 0:
    ar_map_0 = hp.fitsfunc.read_map("../../data/COM_CompMap_Dust-DL07-AvMaps_2048_R2.00.fits", field=0)
    # ar_map_0_log = np.log(ar_map_0)

    # optionally add a mock signal to the map
    mock = False
    if mock:
        ar_mock = ar_map_0
        nside_signal = 32
        radius = hp.nside2resol(nside_signal) / 2 / np.sqrt(2)

        for i in range(hp.nside2npix(nside_signal)):
            vec1 = hp.pix2vec(nside_signal, i)
            mask = hp.query_disc(ar_map_nside, vec=vec1, radius=radius)
            ar_mock[mask] *= 100

        ar_mock /= np.sqrt(100)

# send the map to all other nodes
ar_map_local = comm.bcast(ar_map_0)

# initialize correlation bins
num_bins = 100
ar_product_total = np.zeros(shape=(10, num_bins))
ar_weights_total = np.zeros(shape=(10, num_bins))

# sky scan parameters
num_directions = 4
stripe_step_deg = 10

for current_direction_index in np.arange(num_directions):
    # start with a location within the BOSS field
    center_ra = 180.
    center_dec = 30.
    center_coord = SkyCoord(ra=center_ra * u.degree, dec=center_dec * u.degree)
    center_galactic = center_coord.galactic
    # scan the sky in galactic coordinates:
    galactic_l = Longitude(center_galactic.l + 0 * u.degree)
    # currently we just change the latitude
    galactic_b = Latitude(
        center_galactic.b - (current_direction_index - num_directions * 0.0) * stripe_step_deg * u.degree)
    r_print("galactic l-value:", galactic_l.value)
    r_print("galactic b-value:", galactic_b.value)
    # convert galactic coordinates to healpy theta/phi and create a unit vector:
    center_theta, center_phi = ra_dec2ang(ra=galactic_l.value, dec=galactic_b.value)
    vec = hp.ang2vec(theta=center_theta, phi=center_phi)
    r_print("unit vector:", vec)
    # get all pixels in the current disc
    current_disc_radius = 10 / 180. * np.pi
    disc = hp.query_disc(ar_map_nside, vec=vec, radius=current_disc_radius)
    r_print("disc has ", disc.shape[0], " pixels")

    max_angle_fixed = 5. / 180. * np.pi
    disc_mean = np.nanmean(ar_map_local[disc])
    ar_dec, ar_ra = hp.pix2ang(ar_map_nside, disc)
    global_max_angular_separation = 5. * u.degree  # type: u.Quantity

    # work in iterations so that we don't use too much memory
    for i in np.arange(1):
        # initialize bins for reduce operation
        ar_product_reduce = np.zeros(shape=num_bins)
        ar_weights_reduce = np.zeros(shape=num_bins)

        ar_product_local, ar_weights_local = main_loop(
            ar_map=ar_map_local, max_bin_angle=max_angle_fixed,
            outer_disc_mean=disc_mean, outer_disc=disc,
            max_angular_separation=global_max_angular_separation
        )

        # sum up bins from all nodes
        comm.Reduce(
            [ar_product_local, MPI.DOUBLE],
            [ar_product_reduce, MPI.DOUBLE],
            op=MPI.SUM, root=0)
        comm.Reduce(
            [ar_weights_local, MPI.DOUBLE],
            [ar_weights_reduce, MPI.DOUBLE],
            op=MPI.SUM, root=0)

        if comm.rank == 0:
            r_print("Finished direction ", current_direction_index, ", Iteration ", i)
            ar_product_total[current_direction_index] += ar_weights_reduce
            ar_weights_total[current_direction_index] += ar_weights_reduce
            r_print("total weight: ", ar_weights_total[current_direction_index].sum())

            angular_separation_bins = np.arange(num_bins, dtype=float) / num_bins * max_angle_fixed * 180. / np.pi
            np.savez(
                '../../data/planck_dust_correlation.npz', angular_separation_bins=angular_separation_bins,
                ar_product_total=ar_product_total, ar_weights_total=ar_weights_total)

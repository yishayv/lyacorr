import astropy.units as u
import healpy as hp
import numpy as np
import numpy.random as random
import numpy.random.mtrand
from astropy.coordinates import SkyCoord, Longitude, Latitude
from mpi4py import MPI

from mpi_helper import r_print

comm = MPI.COMM_WORLD


def angular_distance(ar1, ar2):
    theta1, phi1 = hp.pix2ang(2048, ar1)
    theta2, phi2 = hp.pix2ang(2048, ar2)
    ra1 = phi1 * 180. / np.pi
    dec1 = 90. - (theta1 * 180. / np.pi)
    ra2 = phi2 * 180. / np.pi
    dec2 = 90. - (theta2 * 180. / np.pi)
    coord1 = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree)
    coord2 = SkyCoord(ra=ra2 * u.degree, dec=dec2 * u.degree)
    return coord1.separation(coord2).to(u.rad).value


_ = np.random.mtrand.RandomState(comm.rank)

ar_map_shape = None
ar_map_0 = None
ar_map_0_log = None
if comm.rank == 0:
    ar_map_0 = hp.fitsfunc.read_map("/Users/yishay/Downloads/COM_CompMap_Dust-DL07-AvMaps_2048_R2.00.fits", field=0)
    # ar_map_0_log = np.log(ar_map_0)

    mock = False
    if mock:
        ar_mock = ar_map_0
        nside_signal = 32
        radius = hp.nside2resol(nside_signal) / 2 / np.sqrt(2)

        for i in range(hp.nside2npix(nside_signal)):
            vec1 = hp.pix2vec(nside_signal, i)
            mask = hp.query_disc(2048, vec=vec1, radius=radius)
            ar_mock[mask] *= 100

        ar_mock /= np.sqrt(100)

ar_map = comm.bcast(ar_map_0)

num_bins = 100
ar_product_total = np.zeros(shape=(10, num_bins))
ar_weights_total = np.zeros(shape=(10, num_bins))


def ra_dec2ang(ra, dec):
    return (90. - dec) * np.pi / 180., ra / 180. * np.pi


def main_loop(max_angle, disc_part_mean, disc_part, disc_part_pixel_coords, max_angular_separation):
    ar_product = np.zeros(shape=num_bins)
    ar_weights = np.zeros(shape=num_bins)
    ar_product_reduce = np.zeros(shape=num_bins)
    ar_weights_reduce = np.zeros(shape=num_bins)
    chosen_indices = np.random.choice(np.arange(disc_part_pixel_coords.shape[0]), size=100, replace=False)
    for index in chosen_indices:
        vec_a = hp.pix2vec(2048, index)
        disc2 = hp.query_disc(2048, vec=vec_a, radius=max_angular_separation.to(u.rad).value)
        vec_b = hp.pix2vec(2048, disc2)
        ar_ang_dist_with_zero = hp.rotator.angdist(vec_a, vec_b)
        a = index
        b = disc2[ar_ang_dist_with_zero > 0]
        ar_ang_dist_with_zero = ar_ang_dist_with_zero[ar_ang_dist_with_zero > 0]
        ar_bins_float = ar_ang_dist_with_zero / max_angle * num_bins  # type: np.ndarray
        ar_bins = ar_bins_float.astype(int)
        pair_product = np.nan_to_num((ar_map[a] - disc_part_mean) * (ar_map[b] - disc_part_mean))
        ar_product += np.bincount(ar_bins, weights=pair_product, minlength=num_bins)
        ar_weights += np.bincount(ar_bins, minlength=num_bins)

    comm.Reduce(
        [ar_product, MPI.DOUBLE],
        [ar_product_reduce, MPI.DOUBLE],
        op=MPI.SUM, root=0)
    comm.Reduce(
        [ar_weights, MPI.DOUBLE],
        [ar_weights_reduce, MPI.DOUBLE],
        op=MPI.SUM, root=0)
    return ar_product_reduce, ar_weights_reduce


num_directions = 4
stripe_step_deg = 10

for current_direction_index in np.arange(num_directions):
    center_ra = 180.
    center_dec = 30.
    center_coord = SkyCoord(ra=center_ra * u.degree, dec=center_dec * u.degree)
    center_galactic = center_coord.galactic
    galactic_l = Longitude(center_galactic.l + 0 * u.degree)
    galactic_b = Latitude(
        center_galactic.b - (current_direction_index - num_directions * 0.0) * stripe_step_deg * u.degree)
    r_print("galactic l-value:", galactic_l.value)
    r_print("galactic l-value:", galactic_b.value)
    center_theta, center_phi = ra_dec2ang(ra=galactic_l.value, dec=galactic_b.value)
    vec = hp.ang2vec(theta=center_theta, phi=center_phi)
    r_print("unit vector:", vec)
    disc = hp.query_disc(2048, vec=vec, radius=10 / 180. * np.pi)
    r_print("disc has ", disc.shape[0], " pixels")

    max_angle_fixed = 5. / 180. * np.pi
    disc_mean = np.nanmean(ar_map[disc])
    ar_dec, ar_ra = hp.pix2ang(2048, disc)
    pixel_coords = SkyCoord(ra=ar_ra * u.rad, dec=ar_dec * u.rad)
    global_max_angular_separation = 5. * u.degree

    # build initial kd-tree
    # __ = matching.search_around_sky(pixel_coords[0:1],
    #                                 pixel_coords,
    #                                 global_max_angular_separation)

    for i in np.arange(1):

        ar_product_iter, ar_weights_iter = main_loop(
            max_angle=max_angle_fixed, disc_part_mean=disc_mean, disc_part=disc, disc_part_pixel_coords=pixel_coords,
            max_angular_separation=global_max_angular_separation
        )

        if comm.rank == 0:
            r_print("Finished direction ", current_direction_index, ", Iteration ", i)
            ar_product_total[current_direction_index] += ar_product_iter
            ar_weights_total[current_direction_index] += ar_weights_iter
            r_print("total weight: ", ar_weights_total.sum())

            angular_separation_bins = np.arange(num_bins, dtype=float) / num_bins * max_angle_fixed * 180. / np.pi
            np.savez(
                '../../data/planck_dust_correlation.npz', angular_separation_bins=angular_separation_bins,
                ar_product_total=ar_product_total, ar_weights_total=ar_weights_total)
            # plt.plot(angular_separation_bins, ar_corr)
            # plt.show()
            # plt.plot(angular_separation_bins, ar_weights_total)
            # plt.show()

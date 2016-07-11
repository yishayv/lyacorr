import astropy.units as u
import healpy as hp
import numpy as np
import numpy.random as random
import numpy.random.mtrand
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
from astropy.coordinates import matching
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
    pixel_coords_small_sample = disc_part_pixel_coords[chosen_indices]
    count = matching.search_around_sky(pixel_coords_small_sample,
                                       disc_part_pixel_coords,
                                       max_angular_separation)
    a_with_unity = disc_part[chosen_indices[count[0]]]
    b_with_unity = disc_part[count[1]]
    ar_dist_with_unity = count[2].to(u.rad).value
    a = a_with_unity[a_with_unity != b_with_unity]
    b = b_with_unity[a_with_unity != b_with_unity]
    ar_dist = ar_dist_with_unity[a_with_unity != b_with_unity]
    a_part = a[ar_dist < max_angle]
    b_part = b[ar_dist < max_angle]
    ar_dist_part = ar_dist[ar_dist < max_angle]
    ar_bins_float = ar_dist_part / max_angle * num_bins  # type: np.ndarray
    ar_bins = ar_bins_float.astype(int)
    pair_product = np.nan_to_num((ar_map[a_part] - disc_part_mean) * (ar_map[b_part] - disc_part_mean))
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


for j in np.arange(10):
    center_ra = 180.
    center_dec = 30.
    center_coord = SkyCoord(ra=center_ra * u.degree, dec=center_dec * u.degree)
    center_galactic = center_coord.galactic
    galactic_l = Longitude(center_galactic.l + 0 * u.degree)
    galactic_b = Latitude(center_galactic.b + (j - 5) * 10 * u.degree)
    r_print("galactic l-value:", galactic_l.value)
    r_print("galactic l-value:", galactic_b.value)
    center_theta, center_phi = ra_dec2ang(ra=galactic_l.value, dec=galactic_b.value)
    vec = hp.ang2vec(theta=center_theta, phi=center_phi)
    r_print("unit vector:", vec)
    disc = hp.query_disc(2048, vec=vec, radius=10 / 180. * np.pi)

    max_angle_fixed = 5. / 180. * np.pi
    disc_mean = np.nanmean(ar_map[disc])
    ar_dec, ar_ra = hp.pix2ang(2048, disc)
    pixel_coords = SkyCoord(ra=ar_ra * u.rad, dec=ar_dec * u.rad)
    global_max_angular_separation = 5. * u.degree

    for i in np.arange(2):
        # build initial kd-tree
        __ = matching.search_around_sky(pixel_coords[0:1],
                                        pixel_coords,
                                        global_max_angular_separation)

        ar_product_iter, ar_weights_iter = main_loop(
            max_angle=max_angle_fixed, disc_part_mean=disc_mean, disc_part=disc, disc_part_pixel_coords=pixel_coords,
            max_angular_separation=global_max_angular_separation
        )

        if comm.rank == 0:
            r_print(i)
            ar_product_total[j] += ar_product_iter
            ar_weights_total[j] += ar_weights_iter
            r_print("total weight: ", ar_weights_total.sum())

            angular_separation_bins = np.arange(num_bins, dtype=float) / num_bins * max_angle_fixed * 180. / np.pi
            np.savez(
                '../../data/planck_dust_correlation.npz', angular_separation_bins=angular_separation_bins,
                ar_product_total=ar_product_total, ar_weights_total=ar_weights_total)
            # plt.plot(angular_separation_bins, ar_corr)
            # plt.show()
            # plt.plot(angular_separation_bins, ar_weights_total)
            # plt.show()

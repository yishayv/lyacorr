from unittest import TestCase

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

import calc_pixel_pairs
import physics_functions.comoving_distance
import bins_2d
from data_access.numpy_spectrum_container import NpSpectrumContainer

NUM_BINS_X = 50
NUM_BINS_Y = 50

__author__ = 'yishay'

cd = physics_functions.comoving_distance.ComovingDistance(1.9, 3.6, 0.001)


class TestPixelPairs(TestCase):
    def test_find_nearby_pixels(self):
        radius_quantity = (200. * (100. * u.km / (u.Mpc * u.s)) / cd.H0)  # type: u.Quantity
        radius = radius_quantity.value

        pair_separation_bins_1 = bins_2d.Bins2D(NUM_BINS_X, NUM_BINS_Y, x_range=radius, y_range=radius)
        pair_separation_bins_2 = bins_2d.Bins2D(NUM_BINS_X, NUM_BINS_Y, x_range=radius, y_range=radius)
        delta_t_file = NpSpectrumContainer(readonly=False, create_new=True, num_spectra=2, filename=None)

        ar_z0 = np.arange(1.95, 3.56, 0.002)
        delta_t_file.set_wavelength(0, ar_z0)
        delta_t_file.set_flux(0, np.sin(ar_z0 * 50))
        delta_t_file.set_ivar(0, ar_z0)

        ar_z1 = np.arange(1.94, 3.4, 0.002)
        delta_t_file.set_wavelength(1, ar_z1)
        delta_t_file.set_flux(1, np.sin(ar_z1 * 50))
        delta_t_file.set_ivar(1, ar_z1)

        pixel_pairs = calc_pixel_pairs.PixelPairs(cd, radius, 'mean')
        qso_angle = 0.04

        pixel_pairs.find_nearby_pixels(accumulator=pair_separation_bins_1, qso_angle=qso_angle, spec1_index=0,
                                       spec2_index=1,
                                       delta_t_file=delta_t_file)

        pixel_pairs.find_nearby_pixels2(accumulator=pair_separation_bins_2, qso_angle=qso_angle, spec1_index=0,
                                        spec2_index=1,
                                        delta_t_file=delta_t_file)

        print(pair_separation_bins_1.ar_flux.sum(), pair_separation_bins_2.ar_flux.sum())
        print(pair_separation_bins_1.ar_count.sum(), pair_separation_bins_2.ar_count.sum())

        self.assertAlmostEqual((pair_separation_bins_1.ar_flux - pair_separation_bins_2.ar_flux).sum(), 0, 6)
        self.assertAlmostEqual((pair_separation_bins_1.ar_count - pair_separation_bins_2.ar_count).sum(), 0, 6)
        self.assertAlmostEqual((pair_separation_bins_1.ar_weights - pair_separation_bins_2.ar_weights).sum(), 0, 6)

        plot = True
        if plot:
            # plt.set_cmap('gray')
            with np.errstate(divide='ignore', invalid='ignore'):
                ar_est = pair_separation_bins_1.ar_flux / pair_separation_bins_1.ar_weights
            plt.imshow(ar_est, interpolation='nearest')
            plt.show()

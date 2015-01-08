__author__ = 'yishay'

import numpy as np
import scipy.ndimage as ndimage


class ContinuumFitPCA:
    BLUE_START = 1020
    RED_END = 1600
    LY_A_PEAK_BINNED = 1216
    LY_A_PEAK_INDEX = (LY_A_PEAK_BINNED - BLUE_START) / 0.5
    NUM_BINS = (RED_END - LY_A_PEAK_BINNED) * 2 + 1

    def __init__(self, red_pc_text_file, full_pc_text_file, projection_matrix_file):
        self.red_pc_table = np.genfromtxt(red_pc_text_file, skip_header=23)
        self.full_pc_table = np.genfromtxt(full_pc_text_file, skip_header=23)
        self.projection_matrix = np.genfromtxt(projection_matrix_file, delimiter=',')
        self.red_pc = self.red_pc_table[:, 3:13]
        self.full_pc = self.full_pc_table[:, 3:13]
        self.red_mean = self.red_pc_table[:, 1]
        self.full_mean = self.full_pc_table[:, 1]

    def red_to_full(self, red_pc_coefficients):
        return np.dot(self.projection_matrix.T, red_pc_coefficients)

    def project_red_spectrum(self, red_spectrum):
        return np.dot(red_spectrum - self.red_mean, self.red_pc)

    def full_spectrum(self, full_pc_coefficients):
        return np.dot(self.full_pc, full_pc_coefficients) + self.full_mean

    def fit(self, ar_wavelength_rest, ar_flux, normalized):
        red_spectrum = ar_flux[(self.LY_A_PEAK_BINNED <= ar_wavelength_rest) & (ar_wavelength_rest <= self.RED_END)]
        red_spectrum_rebinned = ndimage.zoom(red_spectrum, self.NUM_BINS / float(red_spectrum.size))

        # Suzuki 2004 normalizes flux according to 21 pixels around 1216
        normalization_factor = \
            red_spectrum_rebinned[self.LY_A_PEAK_INDEX - 10:self.LY_A_PEAK_INDEX + 11].mean()
        red_spectrum_rebinned_normalized = red_spectrum_rebinned / float(normalization_factor)
        red_spectrum_coefficients = self.project_red_spectrum(red_spectrum_rebinned_normalized)
        full_spectrum_coefficients = self.red_to_full(red_spectrum_coefficients)
        full_spectrum = self.full_spectrum(full_spectrum_coefficients)
        ar_wavelength_rest_binned = np.arange(self.BLUE_START, self.RED_END + .1, 0.5)
        if ~normalized:
            full_spectrum = full_spectrum * normalization_factor
        return full_spectrum, ar_wavelength_rest_binned
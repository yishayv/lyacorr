__author__ = 'yishay'

import numpy as np
import scipy.linalg
import scipy.interpolate


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

    def least_squares_red_spectrum(self, ar_red_flux, ar_red_ivar):
        ar_red_flux_diff = ar_red_flux - self.red_mean
        ar_sqrt_weights = np.sqrt(ar_red_ivar)
        x = self.red_pc * ar_sqrt_weights[:, None]
        y = ar_red_flux_diff * ar_sqrt_weights
        coefficients = scipy.linalg.lstsq(x, y)
        return coefficients[0]


    def full_spectrum(self, full_pc_coefficients):
        return np.dot(self.full_pc, full_pc_coefficients) + self.full_mean

    def fit_rebin(self, ar_wavelength_rest, ar_flux, ar_ivar, normalized):
        # mask ROUGHLY at the useful spectrum range.
        # include some extra data from the edges for the nearest neighbor interpolation.
        red_spectrum_mask = [(self.LY_A_PEAK_BINNED - 1 <= ar_wavelength_rest) &
                             (ar_wavelength_rest <= self.RED_END + 1)]
        ar_red_wavelength_rest = ar_wavelength_rest[red_spectrum_mask]
        ar_red_flux = ar_flux[red_spectrum_mask]
        ar_red_ivar = ar_ivar[red_spectrum_mask]

        # create wavelength bins (consider moving this elsewhere)
        ar_wavelength_bins = np.arange(self.LY_A_PEAK_BINNED, self.RED_END + .1, 0.5)
        # ar_wavelength_bins = np.arange(self.NUM_BINS) / float(self.NUM_BINS) * \
        # (self.RED_END - self.LY_A_PEAK_BINNED) + self.LY_A_PEAK_BINNED

        # interpolate red spectrum into predefined bins:
        # (use nearest neighbor to avoid leaking bad data)
        f_flux = scipy.interpolate.interp1d(ar_red_wavelength_rest, ar_red_flux,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_red_flux_rebinned = f_flux(ar_wavelength_bins)
        f_ivar = scipy.interpolate.interp1d(ar_red_wavelength_rest, ar_red_ivar,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_red_ivar_rebinned = f_ivar(ar_wavelength_bins)

        # Suzuki 2004 normalizes flux according to 21 pixels around 1216
        normalization_factor = \
            ar_red_flux_rebinned[self.LY_A_PEAK_INDEX - 10:self.LY_A_PEAK_INDEX + 11].mean()
        ar_red_flux_rebinned_normalized = ar_red_flux_rebinned / float(normalization_factor)

        # find the PCA coefficients for the red part of the spectrum.
        red_spectrum_coefficients = self.least_squares_red_spectrum(ar_red_flux_rebinned_normalized,
                                                                    ar_red_ivar_rebinned)

        # map red PCs to full spectrum PCs
        full_spectrum_coefficients = self.red_to_full(red_spectrum_coefficients)

        # convert from PCs to an actual spectrum
        ar_full_spectrum = self.full_spectrum(full_spectrum_coefficients)
        ar_full_wavelength_rest_binned = np.arange(self.BLUE_START, self.RED_END + .1, 0.5)
        if ~normalized:
            ar_full_spectrum = ar_full_spectrum * normalization_factor
        return ar_full_spectrum, ar_full_wavelength_rest_binned, normalization_factor

    def fit(self, ar_wavelength_rest, ar_flux, ar_ivar, normalized, boundary_value=None):
        binned_spectrum, ar_wavelength_rest_binned, normalization_factor = \
            self.fit_rebin(ar_wavelength_rest, ar_flux, ar_ivar, normalized)

        spectrum = np.interp(ar_wavelength_rest, ar_wavelength_rest_binned, binned_spectrum,
                             boundary_value, boundary_value)
        return spectrum, normalization_factor

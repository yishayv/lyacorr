from collections import namedtuple

import lmfit
import numpy as np
import scipy.interpolate
import scipy.linalg
from scipy import signal

import common_settings
import qso_pca_loader

settings = common_settings.Settings()

# by default, use both Paris 2011 and Suzuki 2005 templates
fit_pca_files = settings.get_pca_continuum_tables()
list_pca_loader_default = [qso_pca_loader.PCALoaderSuzuki(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2]),
                           qso_pca_loader.PCALoaderParis(fit_pca_files[3], fit_pca_files[4], fit_pca_files[5])]


# based on [Suzuki 2005] and [Lee, Suzuki, & Spergel 2012]
class ContinuumFitPCA:
    def __init__(self, list_pca_container=list_pca_loader_default,
                 fit_function_name=None, num_components=8):
        assert 0 < num_components <= 10
        if not fit_function_name:
            fit_function_name = settings.get_continuum_fit_method()
        self.fit_function = {'dot_product': self.project_red_spectrum,
                             'weighted_ls': self.fit_least_squares_red_spectrum,
                             'lee_2012': self.fit_red_spectrum}[fit_function_name]

        self.list_pca_container = list_pca_container

    @staticmethod
    def red_to_full(pca, red_pc_coefficients):
        return np.dot(pca.projection_matrix.T, red_pc_coefficients)

    def project_red_spectrum(self, pca, ar_red_flux, ar_red_ivar):
        del ar_red_ivar  # suppress unused variable
        red_spectrum_coefficients = np.dot(ar_red_flux - pca.red_mean, pca.red_pc)
        # map red PCs to full spectrum PCs
        full_spectrum_coefficients = self.red_to_full(pca, red_spectrum_coefficients)
        # convert from PCs to an actual spectrum
        return self.full_spectrum(pca, full_spectrum_coefficients)

    @staticmethod
    def least_squares_red_spectrum_(pca, ar_red_flux, ar_red_ivar):
        ar_red_flux_diff = ar_red_flux - pca.red_mean
        ar_sqrt_weights = np.sqrt(ar_red_ivar)
        x = pca.red_pc * ar_sqrt_weights[:, None]
        y = ar_red_flux_diff * ar_sqrt_weights
        result = scipy.linalg.lstsq(x, y)
        red_spectrum_coefficients = result[0]
        return red_spectrum_coefficients

    def fit_least_squares_red_spectrum(self, pca, ar_red_flux, ar_red_ivar):
        red_spectrum_coefficients = self.least_squares_red_spectrum_(pca, ar_red_flux, ar_red_ivar)
        # map red PCs to full spectrum PCs
        full_spectrum_coefficients = self.red_to_full(pca, red_spectrum_coefficients)
        # convert from PCs to an actual spectrum
        return self.full_spectrum(pca, full_spectrum_coefficients)

    def fit_red_spectrum(self, pca, ar_red_flux, ar_red_ivar):
        params = lmfit.Parameters()
        max_c_z = 1.01
        max_alpha_lambda = 3
        max_f_1280 = 10
        params.add('f_1280', value=1, min=1. / max_f_1280, max=max_f_1280)
        params.add('c_z', value=1, min=1. / max_c_z, max=max_c_z)
        params.add('alpha_lambda', value=1, min=-max_alpha_lambda, max=+max_alpha_lambda)
        result = lmfit.minimize(fcn=self.red_spectrum_residual,
                                params=params, args=(pca, ar_red_flux, ar_red_ivar))
        # get the coefficients of the fitted spectrum:
        red_spectrum_coefficients = self.red_spectrum_fit_coefficients(result.params, pca, ar_red_flux, ar_red_ivar)
        # map red PCs to full spectrum PCs
        full_spectrum_coefficients = self.red_to_full(pca, red_spectrum_coefficients)
        # convert from PCs to an actual spectrum
        full_spectrum_fit = self.full_spectrum(pca, full_spectrum_coefficients)
        # make an adjustment opposite to the one we made during fit.
        return self.inverse_full_spectrum_adjustment(result.params, pca, full_spectrum_fit)

    @staticmethod
    def inverse_full_spectrum_adjustment(params, pca, ar_full_flux):
        ar_full_flux = ar_full_flux / np.power(pca.ar_wavelength_bins / pca.LY_A_PEAK_BINNED,
                                               params['alpha_lambda'].value)
        ar_full_flux = np.interp(pca.ar_wavelength_bins,
                                 pca.ar_wavelength_bins * params['c_z'].value, ar_full_flux)
        ar_full_flux /= params['f_1280'].value
        return ar_full_flux

    def red_spectrum_fit_coefficients(self, params, pca, ar_red_flux, ar_red_ivar):
        # modify qso spectrum according to free parameters:
        # this is not really evaluated at 1280 because we already performed a rough normalization.
        # instead, just multiply everything by this factor
        ar_red_flux = ar_red_flux * params['f_1280'].value
        # red shift correction:
        ar_red_flux = np.interp(pca.ar_red_wavelength_bins * params['c_z'].value,
                                pca.ar_red_wavelength_bins, ar_red_flux)
        ar_red_flux *= np.power(pca.ar_red_wavelength_bins / pca.LY_A_PEAK_BINNED,
                                params['alpha_lambda'].value)
        coefficients = self.least_squares_red_spectrum_(pca, ar_red_flux, ar_red_ivar)
        return coefficients

    def red_spectrum_residual(self, params, pca, ar_red_flux, ar_red_ivar):
        coefficients = self.red_spectrum_fit_coefficients(params, pca, ar_red_flux, ar_red_ivar)
        ar_red_fit = np.dot(pca.red_pc, coefficients) + pca.red_mean
        residual = ar_red_fit - ar_red_flux
        return residual

    @staticmethod
    def full_spectrum(pca, full_pc_coefficients):
        return np.dot(pca.full_pc, full_pc_coefficients) + pca.full_mean

    @staticmethod
    def rebin_red_spectrum(pca, ar_flux, ar_ivar, ar_wavelength_rest):
        # mask ROUGHLY at the useful spectrum range.
        # include some extra data from the edges for the nearest neighbor interpolation.
        red_spectrum_mask = [(pca.LY_A_PEAK_BINNED - 1 <= ar_wavelength_rest) &
                             (ar_wavelength_rest <= pca.RED_END + 1)]
        ar_red_wavelength_rest = ar_wavelength_rest[red_spectrum_mask]
        ar_red_flux = ar_flux[red_spectrum_mask]
        ar_red_ivar = ar_ivar[red_spectrum_mask]
        # interpolate red spectrum into predefined bins:
        # use nearest neighbor to avoid leaking bad data between adjacent pixels.
        f_flux = scipy.interpolate.interp1d(ar_red_wavelength_rest, ar_red_flux,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_red_flux_rebinned = f_flux(pca.ar_red_wavelength_bins)
        f_ivar = scipy.interpolate.interp1d(ar_red_wavelength_rest, ar_red_ivar,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_red_ivar_rebinned = f_ivar(pca.ar_red_wavelength_bins)
        return ar_red_flux_rebinned, ar_red_ivar_rebinned

    @staticmethod
    def rebin_full_spectrum(pca, ar_flux, ar_ivar, ar_wavelength_rest):
        # interpolate spectrum into predefined bins:
        # use nearest neighbor to avoid leaking bad data between adjacent pixels.
        f_flux = scipy.interpolate.interp1d(ar_wavelength_rest, ar_flux,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_flux_rebinned = f_flux(pca.ar_wavelength_bins)
        f_ivar = scipy.interpolate.interp1d(ar_wavelength_rest, ar_ivar,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_ivar_rebinned = f_ivar(pca.ar_wavelength_bins)
        return ar_flux_rebinned, ar_ivar_rebinned

    def fit_binned(self, pca, ar_flux_rebinned, ar_ivar_rebinned,
                   ar_mean_flux_constraint, qso_redshift):

        is_good_fit = True

        ar_red_flux_rebinned = ar_flux_rebinned[pca.LY_A_PEAK_INDEX:]
        ar_red_ivar_rebinned = ar_ivar_rebinned[pca.LY_A_PEAK_INDEX:]

        # Suzuki 2004 normalizes flux according to 21 pixels around 1280
        normalization_factor = \
            ar_red_flux_rebinned[pca.LY_A_NORMALIZATION_INDEX - 10:pca.LY_A_NORMALIZATION_INDEX + 11].mean()
        ar_red_flux_rebinned_normalized = ar_red_flux_rebinned / float(normalization_factor)

        ar_full_fit = None
        for _ in np.arange(3):
            # predict the full spectrum from the red part of the spectrum.
            ar_full_fit = self.fit_function(pca, ar_red_flux_rebinned_normalized,
                                            ar_red_ivar_rebinned)

            # restore the original flux scale
            ar_full_fit = ar_full_fit * normalization_factor
            ar_red_fit = ar_full_fit[pca.LY_A_PEAK_INDEX:]
            # mask 2.5 sigma absorption
            # suppress error when dividing by 0, because 0 ivar is already masked, so the code has no effect anyway.
            with np.errstate(divide='ignore'):
                ar_absorption_mask = ar_red_flux_rebinned - ar_red_fit < - 2.5 * (ar_red_ivar_rebinned ** -0.5)
            # print "masked ", float(ar_absorption_mask.sum())/ar_absorption_mask.size, " of pixels in iteration ", i
            ar_red_ivar_rebinned[ar_absorption_mask] = 0

        ar_blue_fit = ar_full_fit[:pca.LY_A_PEAK_INDEX]
        ar_blue_flux_rebinned = ar_flux_rebinned[:pca.LY_A_PEAK_INDEX]
        ar_blue_ivar_rebinned = ar_ivar_rebinned[:pca.LY_A_PEAK_INDEX]
        ar_blue_fit_mean_flux_rebinned = ar_mean_flux_constraint[:pca.LY_A_PEAK_INDEX] * ar_blue_fit
        # ignore pixels with 0 ivar
        ar_blue_data_mask = np.logical_and(np.isfinite(ar_blue_flux_rebinned), ar_blue_ivar_rebinned)

        if np.array(ar_blue_data_mask).sum() > 50:
            # find the optimal mean flux regulation:
            params = lmfit.Parameters()
            params.add('a_mf', value=0, min=-300, max=300)
            if qso_redshift > 2.4:
                # there are enough forest pixels for a 2nd order fit:
                params.add('b_mf', value=0, min=-300, max=300)
                result = lmfit.minimize(fcn=self.regulate_mean_flux_2nd_order_residual,
                                        params=params, args=(pca,
                                                             ar_blue_flux_rebinned,
                                                             ar_blue_fit_mean_flux_rebinned,
                                                             ar_blue_data_mask))
                # apply the 2nd order mean flux regulation to the continuum fit:
                ar_regulated_blue_flux = self.mean_flux_2nd_order_correction(
                    result.params, ar_blue_fit, pca.delta_wavelength, pca.delta_wavelength_sq)
            else:
                # low redshift makes most of the forest inaccessible,
                # use a 1st order fit to avoid over-fitting.
                result = lmfit.minimize(fcn=self.regulate_mean_flux_1st_order_residual,
                                        params=params, args=(pca,
                                                             ar_blue_flux_rebinned,
                                                             ar_blue_fit_mean_flux_rebinned,
                                                             ar_blue_data_mask))

                # apply the 1st order mean flux regulation to the continuum fit:
                ar_regulated_blue_flux = self.mean_flux_1st_order_correction(
                    result.params, ar_blue_fit, pca.delta_wavelength)

            # overwrite the original blue fit with the regulated fit.
            ar_full_fit[:pca.LY_A_PEAK_INDEX] = ar_regulated_blue_flux
        else:
            is_good_fit = False

        goodness_of_fit = self.get_goodness_of_fit(pca, ar_flux_rebinned, ar_full_fit) if is_good_fit else np.inf
        snr = self.get_simple_snr(ar_flux_rebinned[pca.LY_A_PEAK_INDEX:pca.RED_END_GOODNESS_OF_FIT_INDEX],
                                  ar_ivar_rebinned[pca.LY_A_PEAK_INDEX:pca.RED_END_GOODNESS_OF_FIT_INDEX])

        return ar_full_fit, pca.ar_wavelength_bins, normalization_factor, goodness_of_fit, snr

    def fit(self, ar_wavelength_rest, ar_flux, ar_ivar, qso_redshift, boundary_value=None,
            mean_flux_constraint_func=None):

        # loop over available PCA templates and return the best result
        FitResult = namedtuple('FitResult',
                               ['spectrum', 'normalization_factor', 'is_good_fit', 'goodness_of_fit', 'snr'])
        result_list = []
        for i, pca in enumerate(self.list_pca_container):
            ar_flux_rebinned, ar_ivar_rebinned = self.rebin_full_spectrum(pca, ar_flux, ar_ivar, ar_wavelength_rest)

            # find the theoretical mean flux
            ar_z_rebinned = pca.ar_wavelength_bins * (1 + qso_redshift) / pca.LY_A_PEAK_BINNED - 1

            # use standard mean transmission flux, unless specified otherwise.
            if not mean_flux_constraint_func:
                ar_mean_flux_constraint = self.mean_flux_constraint(ar_z_rebinned)
            else:
                ar_mean_flux_constraint = mean_flux_constraint_func(ar_z_rebinned)

            binned_spectrum, ar_wavelength_rest_binned, normalization_factor, goodness_of_fit, snr = \
                self.fit_binned(pca, ar_flux_rebinned, ar_ivar_rebinned, ar_mean_flux_constraint, qso_redshift)

            spectrum = np.interp(ar_wavelength_rest, ar_wavelength_rest_binned, binned_spectrum,
                                 boundary_value, boundary_value)

            is_good_fit = self._is_good_fit(pca, ar_flux_rebinned, ar_ivar_rebinned, binned_spectrum)

            result_item = FitResult(spectrum, normalization_factor, is_good_fit, goodness_of_fit, snr)

            # append result
            result_list += [result_item]

        # find the best result, that is, the one with smallest goodness_of_fit
        best_index = np.argmin([j.goodness_of_fit for j in result_list])
        best_result = result_list[best_index]

        return best_result

    @classmethod
    def get_goodness_of_fit(cls, pca, ar_flux, ar_flux_fit):
        """
        :type pca:
        :type ar_flux: np.multiarray.ndarray
        :type ar_flux_fit: np.multiarray.ndarray
        """
        # note: we assume standard bins (self.ar_wavelength_bins)
        # get the red part of the spectrum
        ar_red_flux = ar_flux[pca.LY_A_PEAK_INDEX:pca.RED_END_GOODNESS_OF_FIT_INDEX]
        ar_red_flux_fit = ar_flux_fit[pca.LY_A_PEAK_INDEX:pca.RED_END_GOODNESS_OF_FIT_INDEX]
        # smooth the observed flux
        box_size = 15
        boxcar15 = signal.boxcar(box_size)
        # convolve and divide by box_size to keep the same scale
        ar_red_flux_smoothed = signal.convolve(ar_red_flux, boxcar15, mode='same') / box_size
        ar_diff = np.abs((ar_red_flux_fit - ar_red_flux_smoothed) / ar_red_flux_smoothed)
        # since delta wavelength is known, (eq 4) in the 2012 paper simplifies to:
        delta_f = ar_diff.sum() / pca.NUM_RED_BINS
        return delta_f

    def _is_good_fit(self, pca, ar_flux, ar_ivar, ar_flux_fit):
        # threshold is based on signal to noise.
        snr = self.get_simple_snr(ar_flux[pca.LY_A_PEAK_INDEX:pca.RED_END_GOODNESS_OF_FIT_INDEX],
                                  ar_ivar[pca.LY_A_PEAK_INDEX:pca.RED_END_GOODNESS_OF_FIT_INDEX])
        delta_f = self.get_goodness_of_fit(pca, ar_flux, ar_flux_fit)

        return self.is_good_fit(snr, delta_f)

    @staticmethod
    def is_good_fit(snr, goodness_of_fit):
        """
        :type snr: float
        :type goodness_of_fit: np.multiarray.ndarray
        :rtype bool
        """
        # threshold is based on signal to noise.
        # max_delta_f = self.max_delta_f_per_snr(snr) if snr > 0 else 0

        # in addition to a max_delta_f value, avoid suspicious over-fitting and very low SNR values
        # ignore high SNR as well.
        max_delta_f = 1.
        min_delta_f = 0.02
        min_snr = np.exp(0.1)
        max_snr = np.exp(3)

        is_good_fit_result = min_delta_f < goodness_of_fit < max_delta_f and min_snr < snr < max_snr

        return is_good_fit_result

    def regulate_mean_flux_2nd_order_residual(self, params, pca, ar_flux, ar_fit, ar_data_mask):
        ar_regulated_fit = self.mean_flux_2nd_order_correction(params, ar_fit[ar_data_mask],
                                                               pca.delta_wavelength[ar_data_mask],
                                                               pca.delta_wavelength_sq[ar_data_mask])
        residual = ar_regulated_fit - ar_flux[ar_data_mask]
        # note: it seems better to ignore weights for this fit.
        # this is because we are trying to improve the far blue side, which also has larger uncertainty.
        # the fitted parameters have smaller effect as we get near 1280.
        return residual

    def regulate_mean_flux_1st_order_residual(self, params, pca, ar_flux, ar_fit, ar_data_mask):
        ar_regulated_fit = self.mean_flux_1st_order_correction(params, ar_fit[ar_data_mask],
                                                               pca.delta_wavelength[ar_data_mask])
        residual = ar_regulated_fit - ar_flux[ar_data_mask]
        # note: it seems better to ignore weights for this fit.
        # this is because we are trying to improve the far blue side, which also has larger uncertainty.
        # the fitted parameters have smaller effect as we get near 1280.
        return residual

    @staticmethod
    def mean_flux_2nd_order_correction(params, ar_flux, ar_delta_wavelength, ar_delta_wavelength_sq):
        a_mf = params['a_mf'].value
        b_mf = params['b_mf'].value
        return ar_flux * (1 + a_mf * ar_delta_wavelength + b_mf * ar_delta_wavelength_sq)

    @staticmethod
    def mean_flux_1st_order_correction(params, ar_flux, ar_delta_wavelength):
        a_mf = params['a_mf'].value
        return ar_flux * (1 + a_mf * ar_delta_wavelength)

    @staticmethod
    def mean_flux_constraint(z):
        return np.exp(-0.001845 * (1 + z) ** 3.924)

    @staticmethod
    def get_simple_snr(ar_flux, ar_ivar):
        """
        Compute the SNR of a spectrum using median flux and median ivar.
        :type ar_flux: np.ndarray
        :type ar_ivar: np.ndarray
        :rtype: float
        """
        # no need to square ar_flux because the median stays the same
        # no longer using absolute value, because a negative flux should not have high significance.
        return np.nanmedian(ar_flux) * np.sqrt(np.nanmedian(ar_ivar))

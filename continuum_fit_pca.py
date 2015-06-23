__author__ = 'yishay'

import numpy as np
import scipy.linalg
import scipy.interpolate
from scipy import signal
import lmfit
import astropy.table as table

import common_settings
from data_access.numpy_spectrum_container import NpSpectrumContainer


settings = common_settings.Settings()


# based on [Suzuki 2005] and [Lee, Suzuki, & Spergel 2012]
class ContinuumFitPCA:
    BLUE_START = 1020
    RED_END = 1600
    LY_A_PEAK_BINNED = 1216
    LY_A_PEAK_INDEX = (LY_A_PEAK_BINNED - BLUE_START) / 0.5
    NUM_RED_BINS = (RED_END - LY_A_PEAK_BINNED) * 2 + 1

    def __init__(self, red_pc_text_file, full_pc_text_file, projection_matrix_file,
                 fit_function_name=None, num_components=8):
        assert 0 < num_components <= 10
        self.red_pc_table = np.genfromtxt(red_pc_text_file, skip_header=23)
        self.full_pc_table = np.genfromtxt(full_pc_text_file, skip_header=23)
        self.projection_matrix = np.genfromtxt(projection_matrix_file, delimiter=',')[:num_components, :num_components]
        self.num_components = num_components
        self.red_pc = self.red_pc_table[:, 3:3 + num_components]
        self.full_pc = self.full_pc_table[:, 3:3 + num_components]
        self.red_mean = self.red_pc_table[:, 1]
        self.full_mean = self.full_pc_table[:, 1]
        if not fit_function_name:
            fit_function_name = settings.get_continuum_fit_method()
        self.fit_function = {'dot_product': self.project_red_spectrum,
                             'weighted_ls': self.fit_least_squares_red_spectrum,
                             'lee_2012': self.fit_red_spectrum}[fit_function_name]
        # create wavelength bins
        self.ar_wavelength_bins = np.arange(self.BLUE_START, self.RED_END + .1, 0.5)
        self.ar_red_wavelength_bins = np.arange(self.LY_A_PEAK_BINNED, self.RED_END + .1, 0.5)
        self.ar_blue_wavelength_bins = np.arange(self.BLUE_START, self.LY_A_PEAK_BINNED, 0.5)

        # pre-calculated values for mean flux regulation
        self.pivot_wavelength = 1280
        self.delta_wavelength = self.ar_blue_wavelength_bins / self.pivot_wavelength - 1
        self.delta_wavelength_sq = np.square(self.delta_wavelength)
        self.snr_stats = np.zeros(shape=(3, 50, 50))

    def red_to_full(self, red_pc_coefficients):
        return np.dot(self.projection_matrix.T, red_pc_coefficients)

    def project_red_spectrum(self, ar_red_flux, ar_red_ivar):
        del ar_red_ivar  # suppress unused variable
        red_spectrum_coefficients = np.dot(ar_red_flux - self.red_mean, self.red_pc)
        # map red PCs to full spectrum PCs
        full_spectrum_coefficients = self.red_to_full(red_spectrum_coefficients)
        # convert from PCs to an actual spectrum
        return self.full_spectrum(full_spectrum_coefficients)

    def least_squares_red_spectrum_(self, ar_red_flux, ar_red_ivar):
        ar_red_flux_diff = ar_red_flux - self.red_mean
        ar_sqrt_weights = np.sqrt(ar_red_ivar)
        x = self.red_pc * ar_sqrt_weights[:, None]
        y = ar_red_flux_diff * ar_sqrt_weights
        result = scipy.linalg.lstsq(x, y)
        red_spectrum_coefficients = result[0]
        return red_spectrum_coefficients

    def fit_least_squares_red_spectrum(self, ar_red_flux, ar_red_ivar):
        red_spectrum_coefficients = self.least_squares_red_spectrum_(ar_red_flux, ar_red_ivar)
        # map red PCs to full spectrum PCs
        full_spectrum_coefficients = self.red_to_full(red_spectrum_coefficients)
        # convert from PCs to an actual spectrum
        return self.full_spectrum(full_spectrum_coefficients)

    def fit_red_spectrum(self, ar_red_flux, ar_red_ivar):
        params = lmfit.Parameters()
        max_c_z = 1.01
        max_alpha_lambda = 3
        max_f_1280 = 10
        params.add('f_1280', value=1, min=1. / max_f_1280, max=max_f_1280)
        params.add('c_z', value=1, min=1. / max_c_z, max=max_c_z)
        params.add('alpha_lambda', value=1, min=-max_alpha_lambda, max=+max_alpha_lambda)
        result = lmfit.minimize(fcn=self.red_spectrum_residual,
                                params=params, args=(ar_red_flux, ar_red_ivar))
        # get the coefficients of the fitted spectrum:
        red_spectrum_coefficients = self.red_spectrum_fit_coefficients(params, ar_red_flux, ar_red_ivar)
        # map red PCs to full spectrum PCs
        full_spectrum_coefficients = self.red_to_full(red_spectrum_coefficients)
        # convert from PCs to an actual spectrum
        full_spectrum_fit = self.full_spectrum(full_spectrum_coefficients)
        # make an adjustment opposite to the one we made during fit.
        return self.inverse_full_spectrum_adjustment(result.params, full_spectrum_fit)

    def inverse_full_spectrum_adjustment(self, params, ar_full_flux):
        ar_full_flux = ar_full_flux / np.power(self.ar_wavelength_bins / self.LY_A_PEAK_BINNED,
                                               params['alpha_lambda'].value)
        ar_full_flux = np.interp(self.ar_wavelength_bins,
                                 self.ar_wavelength_bins * params['c_z'].value, ar_full_flux)
        ar_full_flux /= params['f_1280'].value
        return ar_full_flux

    def red_spectrum_fit_coefficients(self, params, ar_red_flux, ar_red_ivar):
        # modify qso spectrum according to free parameters:
        # this is not really evaluated at 1280 because we already performed a rough normalization.
        # instead, just multiply everything by this factor
        ar_red_flux = ar_red_flux * params['f_1280'].value
        # red shift correction:
        ar_red_flux = np.interp(self.ar_red_wavelength_bins * params['c_z'].value,
                                self.ar_red_wavelength_bins, ar_red_flux)
        ar_red_flux *= np.power(self.ar_red_wavelength_bins / self.LY_A_PEAK_BINNED,
                                params['alpha_lambda'].value)
        coefficients = self.least_squares_red_spectrum_(ar_red_flux, ar_red_ivar)
        return coefficients

    def red_spectrum_residual(self, params, ar_red_flux, ar_red_ivar):
        coefficients = self.red_spectrum_fit_coefficients(params, ar_red_flux, ar_red_ivar)
        ar_red_fit = np.dot(self.red_pc, coefficients) + self.red_mean
        residual = ar_red_fit - ar_red_flux
        return residual

    def full_spectrum(self, full_pc_coefficients):
        return np.dot(self.full_pc, full_pc_coefficients) + self.full_mean

    def rebin_red_spectrum(self, ar_flux, ar_ivar, ar_wavelength_rest):
        # mask ROUGHLY at the useful spectrum range.
        # include some extra data from the edges for the nearest neighbor interpolation.
        red_spectrum_mask = [(self.LY_A_PEAK_BINNED - 1 <= ar_wavelength_rest) &
                             (ar_wavelength_rest <= self.RED_END + 1)]
        ar_red_wavelength_rest = ar_wavelength_rest[red_spectrum_mask]
        ar_red_flux = ar_flux[red_spectrum_mask]
        ar_red_ivar = ar_ivar[red_spectrum_mask]
        # interpolate red spectrum into predefined bins:
        # (use nearest neighbor to avoid leaking bad data)
        f_flux = scipy.interpolate.interp1d(ar_red_wavelength_rest, ar_red_flux,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_red_flux_rebinned = f_flux(self.ar_red_wavelength_bins)
        f_ivar = scipy.interpolate.interp1d(ar_red_wavelength_rest, ar_red_ivar,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_red_ivar_rebinned = f_ivar(self.ar_red_wavelength_bins)
        return ar_red_flux_rebinned, ar_red_ivar_rebinned

    def rebin_full_spectrum(self, ar_flux, ar_ivar, ar_wavelength_rest):
        # interpolate spectrum into predefined bins:
        # (use nearest neighbor to avoid leaking bad data)
        f_flux = scipy.interpolate.interp1d(ar_wavelength_rest, ar_flux,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_flux_rebinned = f_flux(self.ar_wavelength_bins)
        f_ivar = scipy.interpolate.interp1d(ar_wavelength_rest, ar_ivar,
                                            kind='nearest', bounds_error=False, assume_sorted=True)
        ar_ivar_rebinned = f_ivar(self.ar_wavelength_bins)
        return ar_flux_rebinned, ar_ivar_rebinned

    def fit_binned(self, ar_flux_rebinned, ar_ivar_rebinned,
                   ar_mean_flux_constraint, qso_redshift):

        is_good_fit = True

        ar_red_flux_rebinned = ar_flux_rebinned[self.LY_A_PEAK_INDEX:]
        ar_red_ivar_rebinned = ar_ivar_rebinned[self.LY_A_PEAK_INDEX:]

        # Suzuki 2004 normalizes flux according to 21 pixels around 1216
        normalization_factor = \
            ar_red_flux_rebinned[self.LY_A_PEAK_INDEX - 10:self.LY_A_PEAK_INDEX + 11].mean()
        ar_red_flux_rebinned_normalized = ar_red_flux_rebinned / float(normalization_factor)

        # predict the full spectrum from the red part of the spectrum.
        ar_full_fit = self.fit_function(ar_red_flux_rebinned_normalized,
                                        ar_red_ivar_rebinned)

        # restore the original flux scale
        ar_full_fit = ar_full_fit * normalization_factor

        ar_blue_fit = ar_full_fit[:self.LY_A_PEAK_INDEX]
        ar_blue_flux_rebinned = ar_flux_rebinned[:self.LY_A_PEAK_INDEX]
        ar_blue_fit_mean_flux_rebinned = ar_mean_flux_constraint[:self.LY_A_PEAK_INDEX] * ar_blue_fit
        ar_blue_data_mask = [np.isfinite(ar_blue_flux_rebinned)]

        if np.array(ar_blue_data_mask).sum() > 50:
            # find the optimal mean flux regulation:
            params = lmfit.Parameters()
            params.add('a_mf', value=0, min=-30, max=30)
            if qso_redshift > 2.4:
                # there are enough forest pixels for a 2nd order fit:
                params.add('b_mf', value=0, min=-30, max=30)
                result = lmfit.minimize(fcn=self.regulate_mean_flux_2nd_order_residual,
                                        params=params, args=(ar_blue_flux_rebinned,
                                                             ar_blue_fit_mean_flux_rebinned,
                                                             ar_blue_data_mask))
                # apply the 2nd order mean flux regulation to the continuum fit:
                ar_regulated_blue_flux = self.mean_flux_2nd_order_correction(
                    result.params, ar_blue_fit, self.delta_wavelength, self.delta_wavelength_sq)
            else:
                # low redshift makes most of the forest inaccessible,
                # use a 1st order fit to avoid over-fitting.
                result = lmfit.minimize(fcn=self.regulate_mean_flux_1st_order_residual,
                                        params=params, args=(ar_blue_flux_rebinned,
                                                             ar_blue_fit_mean_flux_rebinned,
                                                             ar_blue_data_mask))

                # apply the 1st order mean flux regulation to the continuum fit:
                ar_regulated_blue_flux = self.mean_flux_1st_order_correction(
                    result.params, ar_blue_fit, self.delta_wavelength)

            # overwrite the original blue fit with the regulated fit.
            ar_full_fit[:self.LY_A_PEAK_INDEX] = ar_regulated_blue_flux
        else:
            is_good_fit = False

        is_good_fit = is_good_fit and self.is_good_fit(ar_flux_rebinned, ar_ivar_rebinned, ar_full_fit)

        return ar_full_fit, self.ar_wavelength_bins, normalization_factor, is_good_fit

    def fit(self, ar_wavelength_rest, ar_flux, ar_ivar, qso_redshift, boundary_value=None,
            mean_flux_constraint_func=None):
        ar_flux_rebinned, ar_ivar_rebinned = self.rebin_full_spectrum(ar_flux, ar_ivar, ar_wavelength_rest)

        # find the theoretical mean flux
        ar_z_rebinned = self.ar_wavelength_bins * (1 + qso_redshift) / self.LY_A_PEAK_BINNED - 1

        # use standard mean transmission flux, unless specified otherwise.
        if not mean_flux_constraint_func:
            ar_mean_flux_constraint = self.mean_flux_constraint(ar_z_rebinned)
        else:
            ar_mean_flux_constraint = mean_flux_constraint_func(ar_z_rebinned)

        binned_spectrum, ar_wavelength_rest_binned, normalization_factor, is_good_fit = \
            self.fit_binned(ar_flux_rebinned, ar_ivar_rebinned, ar_mean_flux_constraint, qso_redshift)

        spectrum = np.interp(ar_wavelength_rest, ar_wavelength_rest_binned, binned_spectrum,
                             boundary_value, boundary_value)

        return spectrum, normalization_factor, is_good_fit

    @classmethod
    def get_goodness_of_fit(cls, ar_flux, ar_flux_fit):
        """
        :type ar_flux: np.multiarray.ndarray
        :type ar_flux_fit: np.multiarray.ndarray
        """
        # note: we assume standard bins (self.ar_wavelength_bins)
        # get the red part of the spectrum
        ar_red_flux = ar_flux[cls.LY_A_PEAK_INDEX:]
        ar_red_flux_fit = ar_flux_fit[cls.LY_A_PEAK_INDEX:]
        # smooth the observed flux
        box_size = 15
        boxcar15 = signal.boxcar(box_size)
        # convolve and divide by box_size to keep the same scale
        ar_red_flux_smoothed = signal.convolve(ar_red_flux, boxcar15, mode='same') / box_size
        ar_diff = np.abs(ar_red_flux_fit - ar_red_flux_smoothed) / ar_red_flux_smoothed
        # since delta wavelength is known, (eq 4) in the 2012 paper simplifies to:
        delta_f = ar_diff.sum() / cls.NUM_RED_BINS
        return delta_f

    def is_good_fit(self, ar_flux, ar_ivar, ar_flux_fit):
        # threshold is based on signal to noise.
        snr = self.get_simple_snr(ar_flux, ar_ivar)
        delta_f = self.get_goodness_of_fit(ar_flux, ar_flux_fit)

        return self._is_good_fit(snr, delta_f)

    def _is_good_fit(self, snr, goodness_of_fit):
        """
        :type snr: float
        :type goodness_of_fit: np.multiarray.ndarray
        """
        # threshold is based on signal to noise.
        max_delta_f = self.max_delta_f_per_snr(snr) if snr != 0 else 0

        delta_f = goodness_of_fit

        # in addition to a max_delta_f value, avoid suspicious over-fitting and very low SNR values
        is_good_fit_result = 0.02 < delta_f < max_delta_f and snr > 0.5

        bin_x = np.clip(snr * 50 / 30, 0, 49)
        bin_y = np.clip(delta_f * 50 / 1., 0, 49)
        self.snr_stats[1 if is_good_fit_result else 0, bin_x, bin_y] += 1
        return is_good_fit_result


    def regulate_mean_flux_2nd_order_residual(self, params, ar_flux, ar_fit, ar_data_mask):
        ar_regulated_fit = self.mean_flux_2nd_order_correction(params, ar_fit[ar_data_mask],
                                                               self.delta_wavelength[ar_data_mask],
                                                               self.delta_wavelength_sq[ar_data_mask])
        residual = ar_regulated_fit - ar_flux[ar_data_mask]
        # note: it seems better to ignore weights for this fit.
        # this is because we are trying to improve the far blue side, which also has larger uncertainty.
        # the fitted parameters have smaller effect as we get near 1280.
        return residual

    def regulate_mean_flux_1st_order_residual(self, params, ar_flux, ar_fit, ar_data_mask):
        ar_regulated_fit = self.mean_flux_1st_order_correction(params, ar_fit[ar_data_mask],
                                                               self.delta_wavelength[ar_data_mask])
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
        # no need to square ar_flux because median
        # TODO: is it correct to use the absolute value even though it has no physical justification?
        return np.percentile(np.abs(ar_flux), 50) * np.sqrt(np.percentile(ar_ivar, 50))

    @staticmethod
    def max_delta_f_per_snr(snr):
	# approximate a fixed quantile of spectra as a function of SNR.
        return ((snr/1.5) ** (-1.5) / 3.6 + 0.05)*1.5


class ContinuumFitContainer(object):
    def __init__(self, num_spectra=-1):
        self.num_spectra = num_spectra
        self.np_spectrum = NpSpectrumContainer(readonly=False, num_spectra=num_spectra)
        self.continuum_fit_metadata = table.Table()
        self.continuum_fit_metadata.add_columns(
            [table.Column(name='index', dtype='i8', unit=None, length=num_spectra),
             table.Column(name='is_good_fit', dtype='b', unit=None, length=num_spectra),
             table.Column(name='goodness_of_fit', dtype='f8', unit=None, length=num_spectra),
             table.Column(name='snr', dtype='f8', unit=None, length=num_spectra)])

        # initialize array
        self.np_spectrum.zero()

    def get_wavelength(self, n):
        return self.np_spectrum.get_wavelength(n)

    def get_flux(self, n):
        return self.np_spectrum.get_flux(n)

    def set_wavelength(self, n, data):
        self.np_spectrum.set_wavelength(n, data)

    def set_flux(self, n, data):
        self.np_spectrum.set_flux(n, data)

    def set_metadata(self, n, is_good_fit, goodness_of_fit, snr):
        self.continuum_fit_metadata[n] = [n, is_good_fit, goodness_of_fit, snr]

    def copy_metadata(self, n, metadata):
        self.continuum_fit_metadata[n] = metadata

    def get_metadata(self, n):
        return self.continuum_fit_metadata[n]

    def get_is_good_fit(self, n):
        return self.get_metadata(n)['is_good_fit']

    def get_goodness_of_fit(self, n):
        return self.get_metadata(n)['goodness_of_fit']

    def get_snr(self, n):
        return self.get_metadata(n)['snr']

    @classmethod
    def from_np_array_and_object(cls, np_array, obj):
        # TODO: consider refactoring.
        np_spectrum = NpSpectrumContainer.from_np_array(np_array, readonly=True)
        new_instance = cls(num_spectra=np_spectrum.num_spectra)
        # replace spectrum container with existing data
        new_instance.np_spectrum = np_spectrum
        # replace metadata with existing metadata object
        assert type(new_instance.continuum_fit_metadata) == type(obj)
        new_instance.continuum_fit_metadata = obj
        return new_instance

    def as_object(self):
        return self.continuum_fit_metadata

    def as_np_array(self):
        return self.np_spectrum.as_np_array()


class ContinuumFitContainerFiles(ContinuumFitContainer):
    # noinspection PyMissingConstructor
    def __init__(self, create_new=False, num_spectra=-1):
        # note: do NOT call super(ContinuumFitContainer, self)
        # we don't want to initialize a very large object in memory.

        if create_new:
            self.num_spectra = num_spectra
            self.np_spectrum = NpSpectrumContainer(readonly=False, num_spectra=num_spectra,
                                                   filename=settings.get_continuum_fit_npy())
            self.continuum_fit_metadata = table.Table()
            self.continuum_fit_metadata.add_columns(
                [table.Column(name='index', dtype='i8', unit=None, length=num_spectra),
                 table.Column(name='is_good_fit', dtype='b', unit=None, length=num_spectra),
                 table.Column(name='goodness_of_fit', dtype='f8', unit=None, length=num_spectra),
                 table.Column(name='snr', dtype='f8', unit=None, length=num_spectra)])

            # initialize file
            self.np_spectrum.zero()
        else:
            self.np_spectrum = NpSpectrumContainer(readonly=True, filename=settings.get_continuum_fit_npy())
            self.continuum_fit_metadata = np.load(settings.get_continuum_fit_metadata_npy())
            self.num_spectra = self.np_spectrum.num_spectra

    def save(self):
        np.save(settings.get_continuum_fit_metadata_npy(), self.continuum_fit_metadata)


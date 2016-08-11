import os.path
import sys
from collections import Counter

import astropy.table as table
import matplotlib.pyplot as plt
import numpy as np
import weighted
from scipy import signal

import calc_mean_transmittance
import common_settings
import continuum_fit_pca
import mean_transmittance
import qso_line_mask
import spectrum
from data_access import read_spectrum_fits
from data_access.qso_data import QSOData
from physics_functions.deredden_func import DereddenSpectrum
from physics_functions.pre_process_spectrum import PreProcessSpectrum
from physics_functions.spectrum_calibration import SpectrumCalibration

# from astropy.convolution import convolve as ap_convolve, Gaussian1DKernel

i = 232

# TODO: replace with a more accurate number
lya_center = 1215.67


def redshift(wavelength, z):
    return (1 + z) * wavelength


def redshift_to_lya_center(z):
    return redshift(lya_center, z)


def lya_center_to_redshift(wavelength):
    return (wavelength / lya_center) - 1


def plot_v_mark(wavelength):
    plt.axvspan(wavelength, wavelength, alpha=0.3, edgecolor='red')


settings = common_settings.Settings()
qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
deredden_spectrum = DereddenSpectrum()
spectrum_calibration = SpectrumCalibration(settings.get_tp_correction_hdf5())
pre_process_spectrum = PreProcessSpectrum()


def rolling_weighted_median(ar_data, ar_weights, box_size):
    ar_flux_smoothed = np.zeros_like(ar_data)
    box_size_lower = - (box_size // 2)
    box_size_upper = box_size // 2 + (box_size & 1)
    for j in xrange(ar_data.size):
        start = max(j + box_size_lower, 0)
        end = min(j + box_size_upper, ar_data.size)
        ar_flux_smoothed[j] = weighted.median(ar_data[start:end], ar_weights[start:end])
    return ar_flux_smoothed


class PlotSpectrum:
    def __init__(self, qso_data_):
        """

        :type qso_data_: QSOData
        """
        self.flux_range = None
        self.wavelength_range = None
        self.qso_data_ = None
        self.qso_data_ = qso_data_
        qso_rec = qso_data_.qso_rec
        qso_z = qso_rec.z
        self.qso_z = qso_z
        print("Plate, FiberID, MJD:", qso_rec.plate, qso_rec.fiberID, qso_rec.mjd)
        print("Z:", self.qso_z)

        fit_pca = continuum_fit_pca.ContinuumFitPCA()

        # create the wavelength series for the measurements
        self.ar_wavelength = np.array(qso_data_.ar_wavelength)
        self.ar_flux = np.array(qso_data_.ar_flux)
        self.ar_ivar = np.array(qso_data_.ar_ivar)

        # correct for flux mis-calibration, milky-way lines, and extinction
        # (enabled/disabled by config options)
        pre_processed_qso_data, result_string = pre_process_spectrum.apply(qso_data_)

        if result_string != 'processed':
            # error during pre-processing. log statistics of error causes.
            print("pre-processing error:", result_string)

        self.ar_flux_correct = pre_processed_qso_data.ar_flux
        self.ar_ivar_correct = pre_processed_qso_data.ar_ivar

        # begin PCA fit:
        ar_wavelength_rest = self.ar_wavelength / (1 + qso_z)
        fit_result = fit_pca.fit(ar_wavelength_rest, self.ar_flux_correct, self.ar_ivar_correct, qso_z,
                                 boundary_value=np.nan)
        self.fit_spectrum = fit_result.spectrum
        is_good_fit = fit_result.is_good_fit
        print("good fit:", is_good_fit)

        # begin power-law fit:
        # for now we have no real error data, so just use '1's:
        ar_flux_err = np.ones(self.ar_flux.size)

        spec = spectrum.Spectrum(self.ar_flux, ar_flux_err, self.ar_wavelength)
        qso_line_mask.mask_qso_lines(spec, qso_z)

        # mask the Ly-alpha part of the spectrum
        qso_line_mask.mask_ly_absorption(spec, qso_z)

        # fit the power-law to unmasked part of the spectrum
        # amp, index = continuum_fit.fit_powerlaw(
        #     spec.ma_wavelength.compressed(),
        #     spec.ma_flux.compressed(),
        #     spec.ma_flux_err.compressed())

        if os.path.exists(settings.get_mean_transmittance_npy()):
            m = mean_transmittance.MeanTransmittance.from_file(settings.get_mean_transmittance_npy())
            ar_mean_flux_lookup = m.get_weighted_mean()
            self.ar_z = self.ar_wavelength / lya_center - 1
            self.ar_mean_flux_for_z_range = np.asarray(np.interp(self.ar_z, m.ar_z, ar_mean_flux_lookup))
            self.fitted_mean = (self.fit_spectrum * self.ar_mean_flux_for_z_range)[self.ar_z < qso_z]

    def set_flux_range(self, flux_min, flux_max):
        if flux_max > flux_min:
            self.flux_range = (flux_min, flux_max)
        else:
            self.flux_range = None

    def set_wavelength_range(self, wavelength_min, wavelength_max):
        if wavelength_max > wavelength_min:
            self.wavelength_range = (wavelength_min, wavelength_max)
        else:
            self.wavelength_range = None

    def plot_spectrum(self):
        assert self.qso_data_, "QSO data not loaded"

        # Define function for calculating a power law
        power_law = lambda x, amp, index: amp * (x ** index)

        if self.flux_range:
            plt.ylim(self.flux_range[0], self.flux_range[1])

        if self.wavelength_range:
            plt.xlim(self.wavelength_range[0], self.wavelength_range[1])
        else:
            plt.xlim(3e3, 1e4)

        ar_flux_err = np.reciprocal(np.sqrt(self.ar_ivar_correct))
        plt.fill_between(self.ar_wavelength, self.ar_flux_correct - ar_flux_err,
                         self.ar_flux_correct + ar_flux_err, color='lightgray', linewidth=.3)

        box_size = 1
        window_func = signal.boxcar(box_size)
        # convolve and divide by box_size to keep the same scale
        # ar_ivar_smoothed = signal.convolve(ar_ivar, window_func, mode='same')
        # ar_flux_smoothed = signal.convolve(ar_flux_correct * ar_ivar, window_func, mode='same') / (
        #     ar_ivar_smoothed)
        ar_flux_smoothed = rolling_weighted_median(self.ar_flux, self.ar_ivar, box_size)

        # ar_flux_smoothed = signal.medfilt(ar_flux_correct, 15)
        # b, a = signal.butter(N=3, Wn=0.02, analog=False)
        # ar_flux_smoothed = signal.filtfilt(b=b, a=a, x=ar_flux_correct)
        # ar_flux_smoothed = ap_convolve(ar_flux_correct, Gaussian1DKernel(1), boundary='extend')

        # plt.plot(ar_wavelength, ar_flux_smoothed, ms=2, color='blue')

        # plt.plot(ar_wavelength, ar_flux, ms=2, linewidth=.3, color='cyan')
        plt.plot(self.ar_wavelength, self.ar_flux_correct, ms=2, linewidth=.3, color='blue', label='Observed flux')
        # plt.loglog(spec.ma_wavelength.compressed(),
        # spec.ma_flux.compressed(), ',', ms=2, color='darkblue')
        plt.plot(self.ar_wavelength, self.fit_spectrum, color='darkorange', label='Continuum fit')

        qso_z = self.qso_z

        if self.fitted_mean is not None:
            plt.plot(self.ar_wavelength[self.ar_z < qso_z], self.fitted_mean, color='red',
                     label='Mean transmission flux')

        plt.axvspan(redshift(1040, qso_z), redshift(1200, qso_z),
                    alpha=0.2, facecolor='yellow', edgecolor='yellow')

        # for l in qso_line_mask.SpecLines:
        #     plot_v_mark(redshift(l.wavelength, qso_z))
        #     plt.axvspan(redshift(l.wavelength / l.width_factor, qso_z),
        #                 redshift(l.wavelength * l.width_factor, qso_z),
        #                 alpha=0.02, facecolor='cyan', edgecolor='none')

        plt.xlabel(r"$\lambda [{\rm \AA}]$", fontsize=12)
        plt.ylabel(r"$f(\lambda)$ $[{\rm 10^{-17}erg/s/cm^{2}/\AA}]$", fontsize=12)

        # create a predicted flux array, based on fitted power_law
        # noinspection PyTypeChecker
        # power_law_array = np.vectorize(power_law, excluded=['amp', 'index'])

        # ar_flux / power_law_array(ar_wavelength, amp, index)
        # plt.loglog(ar_wavelength,
        # ar_flux/power_law_array(ar_wavelength,amp,index),'.',ms=2)
        # plt.plot(ar_wavelength,
        # power_law_array(ar_wavelength, amp=amp, index=index), color='r')

        # draw vertical fill for masked values
        ar_flux_mask = np.isnan(ar_flux_err) | ~np.isfinite(ar_flux_err)
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        plt.fill_between(self.ar_wavelength, y_min, y_max, where=ar_flux_mask,
                         linewidth=.5, color='lightgray', alpha=1)

        plt.legend(loc='upper right', prop={'size': 9})

    def plot_transmittance(self):
        ar_mean_flux_for_z_range = None
        if os.path.exists(settings.get_mean_transmittance_npy()):
            m = mean_transmittance.MeanTransmittance.from_file(settings.get_mean_transmittance_npy())
            ar_mean_flux_lookup = m.get_weighted_mean()
            ar_mean_flux_for_z_range = np.interp(self.ar_z, m.ar_z, ar_mean_flux_lookup)

        stats = Counter(
            {'bad_fit': 0, 'empty_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'accepted': 0})
        lya_forest_transmittance = calc_mean_transmittance.qso_transmittance(self.qso_data_, self.fit_spectrum, stats)
        ar_transmittance_err = np.reciprocal(np.sqrt(lya_forest_transmittance.ar_ivar))
        ar_transmittance_mask = np.isnan(ar_transmittance_err) | ~np.isfinite(ar_transmittance_err)
        ar_transmittance_lower = lya_forest_transmittance.ar_transmittance - ar_transmittance_err
        ar_transmittance_higher = lya_forest_transmittance.ar_transmittance + ar_transmittance_err
        plt.fill_between(lya_forest_transmittance.ar_z, ar_transmittance_lower,
                         ar_transmittance_higher, linewidth=.5, color='lightgray')
        plt.plot(lya_forest_transmittance.ar_z, lya_forest_transmittance.ar_transmittance, linewidth=.5)
        # draw vertical fill for masked values
        axes = plt.gca()
        axes.set_ylim(-1, 2)
        y_min, y_max = axes.get_ylim()
        plt.fill_between(lya_forest_transmittance.ar_z, y_min, y_max, where=ar_transmittance_mask,
                         linewidth=.5, color='lightgray', alpha=1)
        if ar_mean_flux_for_z_range is not None:
            plt.plot(self.ar_z[self.ar_z < self.qso_z], ar_mean_flux_for_z_range[self.ar_z < self.qso_z], color='red')
        plt.xlabel(r"$z$")
        # F(lambda)/Cq(lambda) is the same as F(z)/Cq(z)
        plt.ylabel(r"$f_q(z)/C_q(z)$")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        i = int(sys.argv[1])

    spec_sample_1 = read_spectrum_fits.enum_spectra(qso_record_table[[i]])
    for qso_data_1 in spec_sample_1:
        ps = PlotSpectrum(qso_data_1)
        plt.subplot(2, 1, 1)
        ps.plot_spectrum()
        plt.subplot(2, 1, 2)
        ps.plot_transmittance()
        plt.tight_layout()
        plt.show()

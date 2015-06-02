import os.path

import matplotlib.pyplot as plt
import astropy.table as table
import numpy as np
from scipy import signal
# from astropy.convolution import convolve as ap_convolve, Gaussian1DKernel
import weighted

from data_access import read_spectrum_fits
import common_settings
import continuum_fit_pca
import spectrum
import qso_line_mask
import continuum_fit
import calc_mean_transmittance
import mean_transmittance
from physics_functions.deredden_func import deredden_spectrum
import sys

i = 233
flux_range = None
wavelength_range = None


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


def rolling_weighted_median(ar_data, ar_weights, box_size):
    ar_flux_smoothed = np.zeros_like(ar_data)
    box_size_lower = - (box_size // 2)
    box_size_upper = box_size // 2 + (box_size & 1)
    for j in xrange(ar_data.size):
        start = max(j + box_size_lower, 0)
        end = min(j + box_size_upper, ar_data.size)
        ar_flux_smoothed[j] = weighted.median(ar_data[start:end], ar_weights[start:end])
    return ar_flux_smoothed


def set_flux_range(flux_min, flux_max):
    global flux_range
    if flux_max > flux_min:
        flux_range = (flux_min, flux_max)
    else:
        flux_range = None


def set_wavelength_range(wavelength_min, wavelength_max):
    global wavelength_range
    if wavelength_max > wavelength_min:
        wavelength_range = (wavelength_min, wavelength_max)
    else:
        wavelength_range = None


def plot_fits_spectra(spec_sample):
    for qso_data_ in spec_sample:
        qso_rec = qso_data_.qso_rec
        qso_z = qso_rec.z
        print "Plate, FiberID, MJD:", qso_rec.plate, qso_rec.fiberID, qso_rec.mjd
        print "Z:", qso_z

        fit_pca_files = settings.get_pca_continuum_tables()
        fit_pca = continuum_fit_pca.ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])

        # create the wavelength series for the measurements
        ar_wavelength = qso_data_.ar_wavelength
        # use selected spectrum
        ar_flux = qso_data_.ar_flux
        ar_ivar = qso_data_.ar_ivar
        # we assume the wavelength range in the input file is correct
        assert ar_wavelength.size == ar_flux.size

        # correct extinction:
        ar_flux_correct = deredden_spectrum(ar_wavelength, ar_flux, qso_data_.qso_rec.extinction_g)

        # begin PCA fit:
        ar_wavelength_rest = ar_wavelength / (1 + qso_z)
        fit_spectrum, fit_normalization_factor, is_good_fit = \
            fit_pca.fit(ar_wavelength_rest, ar_flux_correct, ar_ivar, qso_z,
                        boundary_value=np.nan)
        print "good fit:", is_good_fit

        lya_forest_transmittance = calc_mean_transmittance.qso_transmittance(qso_data_, fit_spectrum)

        # begin power-law fit:
        # for now we have no real error data, so just use '1's:
        ar_flux_err = np.ones(ar_flux.size)

        spec = spectrum.Spectrum(ar_flux, ar_flux_err, ar_wavelength)
        qso_line_mask.mask_qso_lines(spec, qso_z)

        # mask the Ly-alpha part of the spectrum
        qso_line_mask.mask_ly_absorption(spec, qso_z)

        # fit the power-law to unmasked part of the spectrum
        amp, index = continuum_fit.fit_powerlaw(
            spec.ma_wavelength.compressed(),
            spec.ma_flux.compressed(),
            spec.ma_flux_err.compressed())

        # Define function for calculating a power law
        power_law = lambda x, amp, index: amp * (x ** index)

        plt.subplot(2, 1, 1)

        if flux_range:
            plt.ylim(flux_range[0], flux_range[1])

        if wavelength_range:
            plt.xlim(wavelength_range[0], wavelength_range[1])
        else:
            plt.xlim(3e3, 1e4)

        ar_flux_err = np.reciprocal(np.sqrt(ar_ivar))
        plt.fill_between(ar_wavelength, ar_flux_correct - ar_flux_err,
                         ar_flux_correct + ar_flux_err, color='gray', linewidth=.3)

        box_size = 1
        window_func = signal.boxcar(box_size)
        # convolve and divide by box_size to keep the same scale
        # ar_ivar_smoothed = signal.convolve(ar_ivar, window_func, mode='same')
        # ar_flux_smoothed = signal.convolve(ar_flux_correct * ar_ivar, window_func, mode='same') / (
        #     ar_ivar_smoothed)
        ar_flux_smoothed = rolling_weighted_median(ar_flux, ar_ivar, box_size)

        # ar_flux_smoothed = signal.medfilt(ar_flux_correct, 15)
        # b, a = signal.butter(N=3, Wn=0.02, analog=False)
        # ar_flux_smoothed = signal.filtfilt(b=b, a=a, x=ar_flux_correct)
        # ar_flux_smoothed = ap_convolve(ar_flux_correct, Gaussian1DKernel(1), boundary='extend')

        # plt.plot(ar_wavelength, ar_flux_smoothed, ms=2, color='blue')

        # plt.plot(ar_wavelength, ar_flux, ms=2, linewidth=.3, color='cyan')
        plt.plot(ar_wavelength, ar_flux_correct, ms=2, linewidth=.3, color='blue', label='Observed flux')
        # plt.loglog(spec.ma_wavelength.compressed(),
        # spec.ma_flux.compressed(), ',', ms=2, color='darkblue')
        plt.plot(ar_wavelength, fit_spectrum, color='orange', label='Continuum fit')

        if os.path.exists(settings.get_mean_transmittance_npy()):
            m = mean_transmittance.MeanTransmittance.from_file(settings.get_mean_transmittance_npy())
            ar_mean_flux_lookup = m.get_weighted_mean()
            ar_z = ar_wavelength / lya_center - 1
            ar_mean_flux_for_z_range = np.interp(ar_z, m.ar_z, ar_mean_flux_lookup)
            fitted_mean = (fit_spectrum * ar_mean_flux_for_z_range)[ar_z < qso_z]
            plt.plot(ar_wavelength[ar_z < qso_z], fitted_mean, color='red', label='Mean transmission flux')

        plt.axvspan(redshift(1040, qso_z), redshift(1200, qso_z),
                    alpha=0.3, facecolor='yellow', edgecolor='red')

        for l in qso_line_mask.SpecLines:
            plot_v_mark(redshift(l.wavelength, qso_z))
            plt.axvspan(redshift(l.wavelength / l.width_factor, qso_z),
                        redshift(l.wavelength * l.width_factor, qso_z),
                        alpha=0.02, facecolor='cyan', edgecolor='none')

        plt.xlabel(r"$\lambda [\AA]$")
        plt.ylabel(r"$f(\lambda)$ $[10^{-17}erg/s/cm^{2}/\AA]$")

        # create a predicted flux array, based on fitted power_law
        # noinspection PyTypeChecker
        power_law_array = np.vectorize(power_law, excluded=['amp', 'index'])

        # ar_flux / power_law_array(ar_wavelength, amp, index)
        # plt.loglog(ar_wavelength,
        # ar_flux/power_law_array(ar_wavelength,amp,index),'.',ms=2)
        # plt.plot(ar_wavelength,
        # power_law_array(ar_wavelength, amp=amp, index=index), color='r')

        # draw vertical fill for masked values
        ar_flux_mask = np.isnan(ar_flux_err) | ~np.isfinite(ar_flux_err)
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        plt.fill_between(ar_wavelength, y_min, y_max, where=ar_flux_mask,
                         linewidth=.5, color='red', alpha=0.1)

        plt.legend(loc='best')

        plt.subplot(2, 1, 2)

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
                         linewidth=.5, color='red', alpha=0.1)

        plt.plot(ar_z[ar_z < qso_z], ar_mean_flux_for_z_range[ar_z < qso_z], color='red')

        plt.xlabel(r"$z$")
        # F(lambda)/Cq(lambda) is the same as F(z)/Cq(z)
        plt.ylabel(r"$f_q(z)/C_q(z)$")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        i = int(sys.argv[1])

    spec_sample_1 = read_spectrum_fits.enum_spectra([qso_record_table[i]])
    plot_fits_spectra(spec_sample_1)

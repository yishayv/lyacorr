

import read_spectrum_fits
import matplotlib.pyplot as plt
import common_settings
import astropy.table as table
import numpy as np
import continuum_fit_pca
from qso_data import QSOData
import spectrum
import qso_line_mask
import continuum_fit

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
spec_sample = read_spectrum_fits.return_spectra_2([qso_record_table[243]])

for qso_data_ in spec_sample:
    qso_z = qso_data_.qso_rec.z
    print qso_z

    fit_pca_files = settings.get_pca_continuum_tables()
    fit_pca = continuum_fit_pca.ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])

    # create the wavelength series for the measurements
    ar_wavelength = qso_data_.ar_wavelength
    # use selected spectrum
    ar_flux = qso_data_.ar_flux
    ar_ivar = qso_data_.ar_ivar
    # we assume the wavelength range in the input file is correct
    assert ar_wavelength.size == ar_flux.size

    # begin PCA fit:
    ar_wavelength_rest = ar_wavelength / (1 + qso_z)
    fit_spectrum, fit_normalization_factor = \
        fit_pca.fit(ar_wavelength_rest, ar_flux, normalized=False)

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
    plt.fill_between(ar_wavelength, ar_flux - np.reciprocal(np.sqrt(ar_ivar)),
                     ar_flux + np.reciprocal(np.sqrt(ar_ivar)), color='gray', linewidth=.3)
    plt.plot(ar_wavelength, ar_flux, ms=2, linewidth=.3)
    # plt.loglog(spec.ma_wavelength.compressed(),
    #           spec.ma_flux.compressed(), ',', ms=2, color='darkblue')
    plt.plot(ar_wavelength,
               fit_spectrum, color='orange')
    plt.axvspan(3817, redshift_to_lya_center(qso_z),
                alpha=0.3, facecolor='yellow', edgecolor='red')

    for l in qso_line_mask.SpecLines:
        plot_v_mark(redshift(l.wavelength, qso_z))
        plt.axvspan(redshift(l.wavelength / l.width_factor, qso_z),
                    redshift(l.wavelength * l.width_factor, qso_z),
                    alpha=0.2, facecolor='cyan', edgecolor='none')

    plt.xlim(3e3, 1e4)
    plt.xlabel(r"$\lambda [\AA]$")
    plt.ylabel(r"$f(\lambda)$ $[10^{-17}erg/s/cm^{2}/\AA]$")

    # create a predicted flux array, based on fitted power_law
    # noinspection PyTypeChecker
    power_law_array = np.vectorize(power_law, excluded=['amp', 'index'])

    # ar_flux / power_law_array(ar_wavelength, amp, index)
    # plt.loglog(ar_wavelength,
    # ar_flux/power_law_array(ar_wavelength,amp,index),'.',ms=2)
    # plt.plot(ar_wavelength,
    #            power_law_array(ar_wavelength, amp=amp, index=index), color='r')

    plt.subplot(2, 1, 2)

    forest_indexes = np.logical_and(ar_wavelength < lya_center * (1 + qso_z),
                                    ar_wavelength > fit_pca.BLUE_START * (1 + qso_z))
    forest_wavelength = ar_wavelength[forest_indexes]
    forest_z = forest_wavelength / lya_center - 1
    forest_flux = ar_flux[forest_indexes]
    forest_continuum = fit_spectrum[forest_indexes]
    plt.plot(forest_z, forest_flux / forest_continuum, linewidth=.5)
    plt.xlabel(r"$z$")
    # F(lambda)/Cq(lambda) is the same as F(z)/Cq(z)
    plt.ylabel(r"$f_q(z)/C_q(z)$")
    plt.tight_layout()
    plt.show()

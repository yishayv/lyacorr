import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import spectrum
import qso_line_mask
import continuum_fit
import continuum_fit_pca
import cProfile

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


def profile_main():
    # load spectra from CSV
    # spectra = np.genfromtxt('../../data/QSOs_spectra_for_yishay_2.csv',
    # delimiter=',', skip_footer=736, skip_header=0)

    # load a individual spectrum from CSV
    count = 740
    i = 405
    # interesting objects: 137, 402, 716, 536(z=3.46, bright!!)
    # problematic objects: 0, 712, 715, 538, 552(bad fit)

    spectra = np.load('../../data/QSOs_spectra_for_yishay_2.npy')

    spec_index = np.genfromtxt('../../data/MyResult_20141225.csv',
                               delimiter=',',
                               skip_header=1)

    fit_pca = continuum_fit_pca.ContinuumFitPCA('../../data/Suzuki/datafile4.txt',
                                                '../../data/Suzuki/datafile3.txt',
                                                '../../data/Suzuki/projection_matrix.csv')

    qso_z = spec_index[i][3]
    print qso_z

    # create the wavelength series for the measurements
    ar_wavelength = np.arange(3817, 9206, 0.5)
    # use selected spectrum
    ar_flux = spectra[i]
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
    plt.plot(ar_wavelength,
               power_law_array(ar_wavelength, amp=amp, index=index), color='r')

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


cProfile.run('profile_main()', sort=2)
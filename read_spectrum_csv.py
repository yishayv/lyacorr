import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import spectrum
import qso_line_mask
import continuum_fit
import continuum_fit_pca

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


# load spectra from CSV
# spectra = np.genfromtxt('../../data/QSOs_spectra_for_yishay_2.csv',
# delimiter=',', skip_footer=736, skip_header=0)

# load a individual spectrum from CSV
count = 740
i = 372
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

# BEGIN PCA TEST

ar_wavelength_rest = ar_wavelength / (1 + qso_z)
red_spectrum = ar_flux[(1216 <= ar_wavelength_rest) & (ar_wavelength_rest <= 1600)]
red_spectrum_rebinned = ndimage.zoom(red_spectrum, ((1600.-1216)*2 + 1) / red_spectrum.size)

#Suzuki 2004 normalizes flux according to 21 pixels around 1216
ly_a_peak_binned = (1216-1020)/0.5
red_spectrum_normalization_factor = red_spectrum_rebinned[ly_a_peak_binned-10:ly_a_peak_binned+11].mean()
red_spectrum_rebinned_normalized = red_spectrum_rebinned / red_spectrum_normalization_factor
red_spectrum_coefficients = fit_pca.project_red_spectrum(red_spectrum_rebinned_normalized)
full_spectrum_coefficients = fit_pca.red_to_full(red_spectrum_coefficients)
full_spectrum = fit_pca.full_spectrum(full_spectrum_coefficients)
ar_wavelength_rest_binned = np.arange(1020, 1600.1, 0.5)

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

plt.loglog(ar_wavelength, ar_flux, ms=2, linewidth=.3)
#plt.loglog(spec.ma_wavelength.compressed(),
#           spec.ma_flux.compressed(), ',', ms=2, color='darkblue')
plt.loglog(ar_wavelength_rest_binned*(1+qso_z),
           full_spectrum*red_spectrum_normalization_factor, color='orange')
plt.axvspan(3817, redshift_to_lya_center(qso_z),
            alpha=0.3, facecolor='yellow', edgecolor='red')

for l in qso_line_mask.SpecLines:
    plot_v_mark(redshift(l.wavelength, qso_z))
    plt.axvspan(redshift(l.wavelength / l.width_factor, qso_z),
                redshift(l.wavelength * l.width_factor, qso_z),
                alpha=0.2, facecolor='cyan', edgecolor='none')

plt.xlim(3e3, 1e4)

# create a predicted flux array, based on fitted power_law
# noinspection PyTypeChecker
power_law_array = np.vectorize(power_law, excluded=['amp', 'index'])

ar_flux / power_law_array(ar_wavelength, amp, index)
# plt.loglog(ar_wavelength,
# ar_flux/power_law_array(ar_wavelength,amp,index),'.',ms=2)
plt.loglog(ar_wavelength,
           power_law_array(ar_wavelength, amp=amp, index=index), color='r')

plt.show()



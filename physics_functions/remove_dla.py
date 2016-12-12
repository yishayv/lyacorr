"""
Provides two ways to attempt to remove DLAs from QSO spectra.
RemoveDlaSimple uses a combination of linear filters to discard high absorption forest pixels.
    It is intended to work on the transmittance, and therefore cannot be applied before continuum fits.
RemoveDlaByCatalog uses a preexisting catalog by Roman Garnett et al. 2016, and can be applied to the original spectra.
"""

from collections import namedtuple

import numpy as np
from astropy import units as u
from astropy.constants import c, m_e, e
from astropy.units.quantity import Quantity
from scipy import signal

import common_settings

settings = common_settings.Settings()  # type: common_settings.Settings

AbsorptionLine = namedtuple('AbsorptionLine', ['transition_rate', 'wavelength', 'oscillator_strength'])

lya = AbsorptionLine(transition_rate=6.265e8 * u.Hz, wavelength=1215.67 * u.Angstrom, oscillator_strength=0.4164)
# We only use the forest region between Lyman Beta and Alpha.
# Lyman Beta can be relevant only in high column densities where the DLA redshift is very close to the QSO.
lyb = AbsorptionLine(transition_rate=1.6725e+08 * u.Hz, wavelength=1025.72 * u.Angstrom, oscillator_strength=7.9142e-02)

absorption_lines = [lya, lyb]

classical_electron_radius = np.square(e.gauss) / (m_e * np.square(c))


class RemoveDlaSimple(object):
    def __init__(self):
        self.detect_box_size = 30
        self.detect_boxcar = signal.boxcar(self.detect_box_size)
        self.mask_box_size = 60
        self.mask_boxcar = signal.boxcar(self.mask_box_size)

    def get_mask(self, ar_flux):
        """

        :type ar_flux: np.ndarray
        :rtype: np.ndarray
        """
        # detect low flux regions using threshold
        # convolve and divide by box_size to keep the same scale
        spec1_smooth = signal.convolve(ar_flux > -0.6, self.detect_boxcar, mode='same') / self.detect_box_size

        # expand the mask to nearby pixels by smoothing
        mask_thresh = np.array(spec1_smooth < 0.2)
        mask_smooth = signal.convolve(mask_thresh, self.mask_boxcar, mode='same') > 1

        return mask_smooth


def wavelength_to_rel_velocity(line_center, z, wavelength):
    rest_wavelength = wavelength / (1 + z)
    wavelength_ratio_sq = np.square(rest_wavelength / line_center)
    beta = - (1 - wavelength_ratio_sq) / (1 + wavelength_ratio_sq)
    return beta * c


def wavelength_to_small_velocity(line_center, z, wavelength):
    rest_wavelength = wavelength / (1 + z)
    wavelength_ratio = rest_wavelength / line_center
    beta = wavelength_ratio - 1
    return beta * c


def lorentzian_profile(center_wavelength, gamma, wavelength):
    freq = c / wavelength
    freq0 = c / center_wavelength
    return freq * 4 * gamma / (np.square(4 * np.pi * (freq - freq0)) + np.square(gamma))


def get_dla_transmittance(nhi, z, ar_wavelength):
    tau = np.zeros(shape=ar_wavelength.shape)
    for line in absorption_lines:
        current_tau = (nhi * np.pi * classical_electron_radius * line.oscillator_strength *
                       line.wavelength) * lorentzian_profile(line.wavelength,
                                                             line.transition_rate,
                                                             ar_wavelength / (1 + z)
                                                             )  # type: Quantity
        tau += current_tau.decompose()

    return np.exp(-tau)


def get_lorentz_width(nhi, line=lya):
    return np.sqrt(np.reciprocal(np.pi * np.log(2)) * classical_electron_radius * nhi * line.oscillator_strength *
                   np.square(line.wavelength) / c * line.transition_rate)


class RemoveDlaByCatalog(object):
    def __init__(self):
        file_structure = [['THINGID', 'i8'], ['SDSS', 'S18'], ['Plate', 'i4'], ['MJD', 'i4'], ['Fiber', 'i4'],
                          ['RAdeg', 'f8'], ['DEdeg', 'f8'], ['z_QSO', 'f8'], ['SNRSpec', 'f8'],
                          ['b_z_DLA', 'f8'], ['B_z_DLA', 'f8'], ['log.pnDLA', 'f8'], ['log.pDLA', 'f8'],
                          ['log.pDnDLA', 'f8'], ['log.pDDLA', 'f8'], ['pnDLAD', 'f8'], ['pDLAD', 'f8'],
                          ['z_DLA', 'f8'], ['log.NHI', 'f8']]
        names, formats = zip(*file_structure)
        dla_catalog = np.loadtxt(settings.get_qso_dla_catalog(),
                                 dtype={'names': names, 'formats': formats})
        self.dla_dict = {(i['Plate'], i['MJD'], i['Fiber']): i for i in dla_catalog}

    def get_mask(self, plate, mjd, fiber_id, ar_wavelength):
        trivial_mask = np.ones_like(ar_wavelength)
        # best effort matching - return a trivial mask if not found
        if (plate, mjd, fiber_id) not in self.dla_dict:
            return trivial_mask

        dla_record = self.dla_dict[plate, mjd, fiber_id]

        # take only DLAs with 50% confidence or higher.
        if dla_record['pDLAD'] < 0.5:
            return trivial_mask

        return get_dla_transmittance(10 ** dla_record['log.NHI'] * u.cm ** -2,
                                     dla_record['z_DLA'], ar_wavelength * u.Angstrom)

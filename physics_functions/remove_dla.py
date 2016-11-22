import numpy as np
from astropy import units as u
from astropy.constants import c, m_e, e
from scipy import signal

import common_settings

settings = common_settings.Settings()  # type: common_settings.Settings

transition_rate = 6.265e8 * u.Hz
ly_alpha_wavelength = 1215.67 * u.Angstrom
oscillator_strength = 0.4164
classical_electron_radius = e.gauss ** 2 / (m_e * c ** 2)


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
    wavelength_ratio_sq = (rest_wavelength / line_center) ** 2
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
    return freq * 4 * gamma / ((4 * np.pi * (freq - freq0)) ** 2 + gamma ** 2)


def get_dla_transmittance(nhi, z, ar_wavelength):
    tau = (nhi * np.pi * classical_electron_radius * oscillator_strength *
           ly_alpha_wavelength) * lorentzian_profile(ly_alpha_wavelength,
                                                     transition_rate,
                                                     ar_wavelength / (1 + z)
                                                     )

    return np.exp(-tau)


def get_lorentz_width(nhi):
    return np.sqrt(np.reciprocal(np.pi * np.log(2)) * classical_electron_radius * nhi * oscillator_strength *
                   ly_alpha_wavelength ** 2 / c * transition_rate)


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
        # best effort matching - return a trivial mask if not found
        if (plate, mjd, fiber_id) not in self.dla_dict:
            return np.ones_like(ar_wavelength)

        dla_record = self.dla_dict[plate, mjd, fiber_id]

        return get_dla_transmittance(10 ** dla_record['log.NHI'] * u.cm ** -2,
                                     dla_record['z_DLA'], ar_wavelength * u.Angstrom)

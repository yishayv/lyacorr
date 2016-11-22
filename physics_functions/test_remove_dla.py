from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity

from physics_functions.remove_dla import get_lorentz_width, get_dla_transmittance, lyb


class TestGetLorentzWidth(TestCase):
    def test_get_lorentz_width(self):
        z_test = 2.6147
        nhi_test = 10 ** 21.194 * u.cm ** -2

        ar_wavelength_test = (1 + z_test) * np.linspace(900, 1300, 400) * u.Angstrom  # type: Quantity

        plt.plot(ar_wavelength_test, get_dla_transmittance(nhi_test, z_test, ar_wavelength_test))
        lya_center_test = 1215.67 * (1 + z_test)
        lyb_center_test = 1025.72 * (1 + z_test)

        lya_sigma_test = get_lorentz_width(nhi_test)
        lyb_sigma_test = get_lorentz_width(nhi_test, lyb)
        plt.axvline(lya_center_test * (1 + lya_sigma_test / 2), linestyle='dotted')
        plt.axvline(lya_center_test / (1 + lya_sigma_test / 2), linestyle='dotted')
        plt.axvline(lyb_center_test * (1 + lyb_sigma_test / 2), linestyle='dotted')
        plt.axvline(lyb_center_test / (1 + lyb_sigma_test / 2), linestyle='dotted')
        plt.show()

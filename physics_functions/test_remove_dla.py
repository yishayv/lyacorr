from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity

from physics_functions.remove_dla import get_lorentz_width, get_dla_transmittance


class TestGetLorentzWidth(TestCase):
    def test_get_lorentz_width(self):
        z_test = 2.6147
        nhi_test = 10 ** 21.194 * u.cm ** -2

        ar_wavelength_test = (1 + z_test) * np.linspace(1100, 1300, 100) * u.Angstrom  # type: Quantity

        plt.plot(ar_wavelength_test, get_dla_transmittance(nhi_test, z_test, ar_wavelength_test))
        line_center_test = 1215.67 * (1 + z_test)
        sigma_test = get_lorentz_width(nhi_test)
        plt.axvline(line_center_test * (1 + sigma_test / 2))
        plt.axvline(line_center_test / (1 + sigma_test / 2))
        plt.show()

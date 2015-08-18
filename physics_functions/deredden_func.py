from scipy.interpolate import interp1d
import numpy


class DereddenSpectrum:
    def __init__(self):
        # dust model
        self.wls = numpy.array([2600, 2700, 4110, 4670, 5470, 6000, 12200, 26500])
        self.a_l = numpy.array([6.591, 6.265, 4.315, 3.806, 3.055, 2.688, 0.829, 0.265])
        self.f_interp = interp1d(self.wls, self.a_l, kind="cubic")

    def apply_correction(self, ar_wavelength, ar_flux, ar_ivar, extinction_g):
        """
        function dereddens a spectrum based on the given extinction_g value and Fitzpatric99 model
        IMPORTANT: the spectrum should be in the observer frame (do not correct for redshift)
        """

        a_l_all = self.f_interp(ar_wavelength)
        e_bv = extinction_g / 3.793
        a_lambda = e_bv * a_l_all
        correction_factor = 10 ** (a_lambda / 2.5)
        ar_flux_corrected = ar_flux * correction_factor
        ar_ivar_corrected = ar_ivar / correction_factor ** 2

        return ar_flux_corrected, ar_ivar_corrected

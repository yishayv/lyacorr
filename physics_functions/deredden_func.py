from scipy.interpolate import interp1d
import numpy


def deredden_spectrum(wl, spec, extinction_g):
    """
    function dereddens a spectrum based on the given extinction_g value and Fitzpatric99 model
    IMPORTANT: the spectrum should be in the observer frame (do not correct for redshift)
    """
    # dust model
    wls = numpy.array([2600, 2700, 4110, 4670, 5470, 6000, 12200, 26500])
    a_l = numpy.array([6.591, 6.265, 4.315, 3.806, 3.055, 2.688, 0.829, 0.265])
    f_interp = interp1d(wls, a_l, kind="cubic")

    a_l_all = f_interp(wl)
    E_bv = extinction_g / 3.793
    A_lambda = E_bv * a_l_all
    spec_real = spec * 10 ** (A_lambda / 2.5)

    return spec_real
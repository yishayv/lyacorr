import numpy as np
import numpy.ma as ma

class Spectrum:
    ar_flux = ma.empty
    ar_wavelength = ma.empty
    def __init__(self, ar_flux, ar_wavelength):
        assert len(ar_flux)==len(ar_wavelength)
        self.ar_flux=ar_flux
        self.ar_wavelength=ar_wavelength


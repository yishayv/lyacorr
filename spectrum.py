import numpy.ma as ma


class Spectrum:
    ma_flux = ma.empty
    ma_flux_err = ma.empty
    ma_wavelength = ma.empty

    def __init__(self, ar_flux, ar_flux_err, ar_wavelength):
        assert len(ar_flux) == len(ar_wavelength)
        self.ma_flux = ma.array(ar_flux)
        self.ma_flux_err = ma.array(ar_flux_err)
        self.ma_wavelength = ma.array(ar_wavelength)

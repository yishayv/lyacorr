import os.path

import numpy as np


MAX_SPECTRA = 1000
MAX_WAVELENGTH_COUNT = 4992


class NpSpectrumContainer(object):
    """
    Holds spectra in a numpy memory mapped file.
    """

    def __init__(self, filename, readonly, num_spectra=-1):
        self.filename = filename
        self.readonly = readonly
        mode = 'r' if readonly else 'w+'
        self.np_array = np.memmap(filename, 'f8', mode=mode, shape=(MAX_SPECTRA, 2, MAX_WAVELENGTH_COUNT))
        self.num_spectra = os.path.getsize(filename) // (2 * MAX_WAVELENGTH_COUNT * 8)
        if not readonly:
            assert self.num_spectra == num_spectra

    def get_wavelength(self, n):
        return self._get_array(n, 0)

    def get_flux(self, n):
        return self._get_array(n, 1)

    def set_wavelength(self, n, data):
        return self._set_array(n, data, 0)

    def set_flux(self, n, data):
        return self._set_array(n, data, 1)

    def _set_array(self, n, data, i):
        assert data.size < MAX_WAVELENGTH_COUNT
        np.copyto(self.np_array[n, i, :data.size], data)

    def _get_array(self, n, i):
        assert n < self.num_spectra
        ar = self.np_array[n][i]
        # trim zeros
        return ar[ar != 0]


class NpSpectrumIterator(object):
    """
    Iterate over all spectra
    """

    def __init__(self, np_spectrum_container):
        """

        :type np_spectrum_container: NpSpectrumContainer
        """
        self._n = 0
        self._np_spectrum_container = np_spectrum_container

    def __iter__(self):
        return self

    def next(self):
        if self._n >= self._np_spectrum_container.num_spectra:
            raise StopIteration
        else:
            self._n += 1
            return self

    def get_flux(self):
        return self._np_spectrum_container.get_flux(self._n)

    def get_wavelength(self):
        return self._np_spectrum_container.get_wavelength(self._n)

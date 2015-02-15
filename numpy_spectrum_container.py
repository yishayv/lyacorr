import os.path

import numpy as np


MAX_SPECTRA = 200000
MAX_WAVELENGTH_COUNT = 4992


class NpSpectrumContainer(object):
    """
    Holds spectra in a numpy memory mapped file, or a memory based array
    """

    def __init__(self, readonly, num_spectra=-1, filename=None):
        assert num_spectra <= MAX_SPECTRA
        self.filename = filename
        self.readonly = readonly
        if filename:
            mode = 'r' if readonly else 'w+'
            self.np_array = np.memmap(filename, 'f8', mode=mode, shape=(num_spectra, 2, MAX_WAVELENGTH_COUNT))
            if readonly:
                self.num_spectra = os.path.getsize(filename) // (2 * MAX_WAVELENGTH_COUNT * 8)
                if num_spectra != -1:
                    assert self.num_spectra == num_spectra
        else:
            self.np_array = np.ndarray(shape=(num_spectra, 2, MAX_WAVELENGTH_COUNT))
            self.num_spectra = num_spectra

    def get_wavelength(self, n):
        return self._get_array(n, 0)

    def get_flux(self, n):
        return self._get_array(n, 1)

    def set_wavelength(self, n, data):
        return self._set_array(n, data, 0)

    def set_flux(self, n, data):
        return self._set_array(n, data, 1)

    def _set_array(self, n, data, i):
        assert data.size <= MAX_WAVELENGTH_COUNT, "data size too large: %d" % data.size
        np.copyto(self.np_array[n, i, :data.size], data)

    def _get_array(self, n, i):
        assert n <= self.num_spectra
        ar_wavelength = self.np_array[n][0]
        # trim zeros, always according to wavelength
        return self.np_array[n][i][ar_wavelength != 0]


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
        self._n += 1
        if self._n >= self._np_spectrum_container.num_spectra:
            raise StopIteration
        else:
            return self

    def get_flux(self):
        return self._np_spectrum_container.get_flux(self._n)

    def get_wavelength(self):
        return self._np_spectrum_container.get_wavelength(self._n)

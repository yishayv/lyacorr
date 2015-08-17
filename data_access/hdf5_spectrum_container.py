import h5py
import numpy as np


INITIAL_SPECTRA = 1000
MAX_SPECTRA = 220000
MAX_WAVELENGTH_COUNT = 4992
CHUNK_COUNT = 20
NUM_FIELDS = 3


class Hdf5SpectrumContainer(object):
    def __init__(self, filename, readonly, create_new, num_spectra=-1):
        self.filename = filename
        self.readonly = readonly
        mode = 'r' if readonly else 'w' if create_new else 'r+'
        self.f = h5py.File(filename, mode=mode)
        if num_spectra != -1:
            assert num_spectra <= MAX_SPECTRA
        if create_new and not readonly:
            self.data_set = self.f.create_dataset('spectra',
                                                  shape=(INITIAL_SPECTRA, NUM_FIELDS, MAX_WAVELENGTH_COUNT),
                                                  maxshape=(MAX_SPECTRA, NUM_FIELDS, MAX_WAVELENGTH_COUNT),
                                                  dtype='f8',
                                                  chunks=(CHUNK_COUNT, NUM_FIELDS, MAX_WAVELENGTH_COUNT))
        else:
            self.data_set = self.f['spectra']

    def __del(self):
        self.f.flush()
        self.f.close()

    def get_wavelength(self, n):
        """

        :rtype : np.multiarray.ndarray
        """
        return self._get_array(n, 0)

    def get_flux(self, n):
        """

        :rtype : np.multiarray.ndarray
        """
        return self._get_array(n, 1)

    def get_ivar(self, n):
        """

        :rtype : np.multiarray.ndarray
        """
        return self._get_array(n, 2)

    def set_wavelength(self, n, data):
        return self._set_array(n, data, 0)

    def set_flux(self, n, data):
        return self._set_array(n, data, 1)

    def set_ivar(self, n, data):
        return self._set_array(n, data, 2)

    def _set_array(self, n, data, i):
        assert data.size < MAX_WAVELENGTH_COUNT
        if n >= self.data_set.shape[0]:
            self.data_set.resize((n + CHUNK_COUNT, NUM_FIELDS, MAX_WAVELENGTH_COUNT))
        self.data_set[n, i, :data.size] = data

    def _get_array(self, n, i):
        """
        helper function for returning flux, ivar, or wavelength arrays.
        :type n: int
        :type i: int
        :rtype: np.multiarray.ndarray
        """
        assert n < self.data_set.shape[0]
        ar = self.data_set[n][i]
        # trim zeros, according to wavelength
        return ar[self.data_set[n][0] != 0]


class Hdf5SpectrumIterator(object):
    """
    Iterate over all spectra
    """

    def __init__(self, spectrum_container):
        """

        :type spectrum_container: Hdf5SpectrumContainer
        """
        self._n = -1
        self.spectrum_container = spectrum_container

    def __iter__(self):
        return self

    def next(self):
        if self._n >= self.spectrum_container.data_set.shape[0]:
            raise StopIteration
        else:
            self._n += 1
            return self

    def get_flux(self):
        return self.spectrum_container.get_flux(self._n)

    def get_wavelength(self):
        return self.spectrum_container.get_wavelength(self._n)

    def get_ivar(self):
        return self.spectrum_container.get_ivar(self._n)
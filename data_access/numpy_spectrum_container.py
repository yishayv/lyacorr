import os.path

import numpy as np


MAX_SPECTRA = 200000
MAX_WAVELENGTH_COUNT = 4992
NUM_FIELDS = 3


class NpSpectrumContainer(object):
    """
    Holds spectra in a numpy memory mapped file, or a memory based array
    """

    def __init__(self, readonly, create_new=True, num_spectra=-1, filename=None,
                 max_wavelength_count=MAX_WAVELENGTH_COUNT):
        assert num_spectra <= MAX_SPECTRA
        self.filename = filename
        self.readonly = readonly
        self.max_wavelength_count = max_wavelength_count
        if filename:
            mode = 'r' if readonly else 'w+' if create_new else 'r+'
            if readonly:
                existing_file_num_spectra = os.path.getsize(filename) // (NUM_FIELDS * self.max_wavelength_count * 8)
                if num_spectra != -1:
                    assert num_spectra == existing_file_num_spectra
                else:
                    num_spectra = existing_file_num_spectra
            self.np_array = np.memmap(filename, 'f8', mode=mode,
                                      shape=(num_spectra, NUM_FIELDS, self.max_wavelength_count))
        else:
            self.np_array = np.ndarray(shape=(num_spectra, NUM_FIELDS, self.max_wavelength_count))
        self.num_spectra = num_spectra

    def get_wavelength(self, n):
        return self._get_array(n, 0)

    def get_flux(self, n):
        return self._get_array(n, 1)

    def get_ivar(self, n):
        return self._get_array(n, 2)

    def set_wavelength(self, n, data):
        return self._set_array(n, data, 0)

    def set_flux(self, n, data):
        return self._set_array(n, data, 1)

    def set_ivar(self, n, data):
        return self._set_array(n, data, 2)

    def _set_array(self, n, data, i):
        assert data.size <= self.max_wavelength_count, "data size too large: %d" % data.size
        np.copyto(self.np_array[n, i, :data.size], data)

    def _get_array(self, n, i):
        assert n <= self.num_spectra
        ar_wavelength = self.np_array[n][0]
        # trim zeros, always according to wavelength
        return self.np_array[n][i][ar_wavelength != 0]

    def zero(self):
        self.np_array[:] = 0

    def as_np_array(self):
        return self.np_array

    def as_object(self):
        pass

    @classmethod
    def from_np_array(cls, np_array, readonly):
        assert np_array.ndim == 3 & np_array.shape[1] == NUM_FIELDS
        num_spectra = np_array.shape[0]
        max_wavelength_count = np_array.shape[2]
        # create a similar object with an empty array
        # this might not be the best way to do that, but it saves adding an argument and extra logic to the initializer.
        new_obj = cls(readonly=readonly, create_new=False, num_spectra=0)
        # replace the empty array with the one supplied as an argument.
        new_obj.np_array = np_array
        # update instance variables.
        new_obj.num_spectra = num_spectra
        new_obj.max_wavelength_count = max_wavelength_count
        return new_obj


class NpSpectrumIterator(object):
    """
    Iterate over all spectra
    """

    def __init__(self, np_spectrum_container):
        """

        :type np_spectrum_container: NpSpectrumContainer
        """
        self._n = -1
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

    def get_ivar(self):
        return self._np_spectrum_container.get_ivar(self._n)

import h5py


INITIAL_SPECTRA = 1000
MAX_SPECTRA = 220000
MAX_WAVELENGTH_COUNT = 4992
CHUNK_COUNT = 1000


class Hdf5SpectrumContainer(object):
    def __init__(self, filename, readonly, num_spectra=-1):
        self.filename = filename
        self.readonly = readonly
        mode = 'r' if readonly else 'w'
        self.f = h5py.File(filename, mode=mode)
        if readonly:
            self.data_set = self.f['spectra']
        else:
            self.data_set = self.f.create_dataset('spectra',
                                                  shape=(INITIAL_SPECTRA, 2, MAX_WAVELENGTH_COUNT),
                                                  maxshape=(MAX_SPECTRA, 2, MAX_WAVELENGTH_COUNT),
                                                  dtype='f8',
                                                  chunks=(CHUNK_COUNT, 2, MAX_WAVELENGTH_COUNT))

    def __del(self):
        self.f.flush()
        self.f.close()

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
        if n >= self.data_set.shape[0]:
            self.data_set.resize(n, 2, MAX_WAVELENGTH_COUNT)
        self.data_set[n, i, :data.size] = data

    def _get_array(self, n, i):
        assert n < self.data_set.shape[0]
        ar = self.data_set[n][i]
        # trim zeros
        return ar[ar != 0]


class Hdf5SpectrumIterator(object):
    """
    Iterate over all spectra
    """

    def __init__(self, spectrum_container):
        """

        :type spectrum_container: Hdf5SpectrumContainer
        """
        self._n = 0
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

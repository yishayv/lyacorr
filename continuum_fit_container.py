import numpy as np
from astropy import table as table

from continuum_fit_pca import settings
from data_access.numpy_spectrum_container import NpSpectrumContainer


class ContinuumFitContainer(object):
    def __init__(self, num_spectra=-1):
        self.num_spectra = num_spectra
        self.np_spectrum = NpSpectrumContainer(readonly=False, num_spectra=num_spectra)
        self.continuum_fit_metadata = table.Table()
        self.continuum_fit_metadata.add_columns(
            [table.Column(name='index', dtype='i8', unit=None, length=num_spectra),
             table.Column(name='is_good_fit', dtype='b', unit=None, length=num_spectra),
             table.Column(name='goodness_of_fit', dtype='f8', unit=None, length=num_spectra),
             table.Column(name='snr', dtype='f8', unit=None, length=num_spectra)])

        # initialize array
        self.np_spectrum.zero()

    def get_wavelength(self, n):
        return self.np_spectrum.get_wavelength(n)

    def get_flux(self, n):
        return self.np_spectrum.get_flux(n)

    def set_wavelength(self, n, data):
        self.np_spectrum.set_wavelength(n, data)

    def set_flux(self, n, data):
        self.np_spectrum.set_flux(n, data)

    def set_metadata(self, n, is_good_fit, goodness_of_fit, snr):
        self.continuum_fit_metadata[n] = [n, is_good_fit, goodness_of_fit, snr]

    def copy_metadata(self, n, metadata):
        self.continuum_fit_metadata[n] = metadata

    def get_metadata(self, n):
        return self.continuum_fit_metadata[n]

    def get_is_good_fit(self, n):
        return self.get_metadata(n)['is_good_fit']

    def get_goodness_of_fit(self, n):
        return self.get_metadata(n)['goodness_of_fit']

    def get_snr(self, n):
        return self.get_metadata(n)['snr']

    @classmethod
    def from_np_array_and_object(cls, np_array, obj):
        # TODO: consider refactoring.
        np_spectrum = NpSpectrumContainer.from_np_array(np_array, readonly=True)
        new_instance = cls(num_spectra=np_spectrum.num_spectra)
        # replace spectrum container with existing data
        new_instance.np_spectrum = np_spectrum
        # replace metadata with existing metadata object
        assert type(new_instance.continuum_fit_metadata) == type(obj)
        new_instance.continuum_fit_metadata = obj
        return new_instance

    def as_object(self):
        return self.continuum_fit_metadata

    def as_np_array(self):
        return self.np_spectrum.as_np_array()


class ContinuumFitContainerFiles(ContinuumFitContainer):
    # noinspection PyMissingConstructor
    def __init__(self, create_new=False, num_spectra=-1):
        # note: do NOT call super(ContinuumFitContainer, self)
        # we don't want to initialize a very large object in memory.

        if create_new:
            self.num_spectra = num_spectra
            self.np_spectrum = NpSpectrumContainer(readonly=False, num_spectra=num_spectra,
                                                   filename=settings.get_continuum_fit_npy())
            self.continuum_fit_metadata = table.Table()
            self.continuum_fit_metadata.add_columns(
                [table.Column(name='index', dtype='i8', unit=None, length=num_spectra),
                 table.Column(name='is_good_fit', dtype='b', unit=None, length=num_spectra),
                 table.Column(name='goodness_of_fit', dtype='f8', unit=None, length=num_spectra),
                 table.Column(name='snr', dtype='f8', unit=None, length=num_spectra)])

            # initialize file
            self.np_spectrum.zero()
        else:
            self.np_spectrum = NpSpectrumContainer(readonly=True, filename=settings.get_continuum_fit_npy())
            self.continuum_fit_metadata = np.load(settings.get_continuum_fit_metadata_npy())
            self.num_spectra = self.np_spectrum.num_spectra

    def save(self):
        np.save(settings.get_continuum_fit_metadata_npy(), self.continuum_fit_metadata)

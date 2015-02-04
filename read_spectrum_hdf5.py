import itertools

from read_spectrum_fits import QSORecord
from hdf5_spectrum_container import Hdf5SpectrumContainer, Hdf5SpectrumIterator

FILENAME = '/mnt/gastro/yishay/sdss_QSOs/spectra.hdf5'
MAX_WAVELENGTH_COUNT = 4992


def return_spectra_2(qso_record_table, spectra_file=FILENAME):
    spectra = Hdf5SpectrumContainer(spectra_file, True)

    # we assume that the order of spectra is the same as in the QSO list
    for i, j in itertools.izip(Hdf5SpectrumIterator(spectra), qso_record_table):
        qso_rec = QSORecord.from_row(j)
        ar_wavelength = i.get_wavelength()
        ar_flux = i.get_flux()
        if ar_wavelength.size == 0:
            continue
        yield ar_wavelength, ar_flux, qso_rec

# TODO: refactor this or remove completely
class SpectraWithMetadata:
    def __init__(self, qso_record_table, spectra_file=FILENAME):
        self.spectra = Hdf5SpectrumContainer(spectra_file, True)
        self.qso_record_table = qso_record_table

    def return_spectrum(self, n):
        # we assume that the order of spectra is the same as in the QSO list
        qso_rec = QSORecord.from_row(self.qso_record_table[n])
        ar_wavelength = self.spectra.get_wavelength(n)
        ar_flux = self.spectra.get_flux(n)
        return ar_wavelength, ar_flux, qso_rec
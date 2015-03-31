import itertools

from hdf5_spectrum_container import Hdf5SpectrumContainer, Hdf5SpectrumIterator
from qso_data import QSORecord, QSOData

MAX_WAVELENGTH_COUNT = 4992


def return_spectra_2(qso_record_table, spectra_file):
    spectra = Hdf5SpectrumContainer(spectra_file, readonly=True, create_new=False)

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
    def __init__(self, qso_record_table, spectra_file, table_offset=0):
        self.spectra = Hdf5SpectrumContainer(spectra_file, readonly=True, create_new=False)
        self.qso_record_table = qso_record_table
        self.table_offset = table_offset

    def return_spectrum(self, n):
        # we assume that the order of spectra is the same as in the QSO list
        """

        :rtype : QSOData
        """
        qso_rec = QSORecord.from_row(self.qso_record_table[n - self.table_offset])
        ar_wavelength = self.spectra.get_wavelength(n)
        ar_flux = self.spectra.get_flux(n)
        ar_ivar = self.spectra.get_ivar(n)
        return QSOData(qso_rec, ar_wavelength, ar_flux, ar_ivar)
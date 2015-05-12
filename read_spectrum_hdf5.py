from hdf5_spectrum_container import Hdf5SpectrumContainer
from qso_data import QSORecord, QSOData

MAX_WAVELENGTH_COUNT = 4992


# TODO: refactor this or remove completely
class SpectraWithMetadata:
    def __init__(self, qso_record_table, spectra_file):
        self.spectra = Hdf5SpectrumContainer(spectra_file, readonly=True, create_new=False)
        self.qso_record_table = qso_record_table

    def return_spectrum(self, n):
        """
        return a spectrum based on the index in the corresponding qso_record_table
        :rtype : QSOData
        """
        qso_rec = QSORecord.from_row(self.qso_record_table[n])
        index = qso_rec.index
        ar_wavelength = self.spectra.get_wavelength(index)
        ar_flux = self.spectra.get_flux(index)
        ar_ivar = self.spectra.get_ivar(index)
        return QSOData(qso_rec, ar_wavelength, ar_flux, ar_ivar)
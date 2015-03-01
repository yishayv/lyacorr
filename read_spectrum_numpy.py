import itertools

from numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from qso_data import QSORecord, QSOData

MEM_MAP_FILE = '/mnt/gastro/yishay/sdss_QSOs/spectra.npy'
MAX_WAVELENGTH_COUNT = 4992


def return_spectra_2(qso_record_table, mem_map_file=MEM_MAP_FILE):
    spectra = NpSpectrumContainer(True, mem_map_file)

    # we assume that the order of spectra is the same as in the QSO list
    for i, j in itertools.izip(NpSpectrumIterator(spectra), qso_record_table):
        qso_rec = QSORecord.from_row(j)
        ar_wavelength = i.get_wavelength()
        ar_flux = i.get_flux()
        if ar_wavelength.size == 0:
            continue
        yield QSOData(qso_rec, ar_wavelength, ar_flux)

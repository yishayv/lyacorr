import numpy as np
import os.path
import itertools

from read_spectrum_fits import QSORecord


MEM_MAP_FILE = '/mnt/gastro/yishay/sdss_QSOs/spectra.npy'
MAX_WAVELENGTH_COUNT = 4992


def return_spectra_2(qso_record_table, mem_map_file=MEM_MAP_FILE):
    num_spectra = os.path.getsize(mem_map_file) // (2 * MAX_WAVELENGTH_COUNT * 8)

    fp = np.memmap(mem_map_file, dtype='f8', mode='r', shape=(num_spectra, 2, MAX_WAVELENGTH_COUNT))

    # we assume that the order of spectra is the same as in the QSO list
    for i, j in itertools.izip(fp, qso_record_table):
        qso_rec = QSORecord.from_row(j)
        ar_wavelength = i[0][i[0] != 0]
        ar_flux = i[1][i[1] != 0]
        if ar_wavelength.size == 0:
            continue
        yield ar_wavelength, ar_flux, qso_rec

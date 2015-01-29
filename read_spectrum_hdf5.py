import numpy as np
import os.path
import itertools

from read_spectrum_fits import QSORecord
from hdf5_spectrum_container import Hdf5SpectrumContainer, Hdf5SpectrumIterator

FILENAME = '/mnt/gastro/yishay/sdss_QSOs/spectra.hdf5'
MAX_WAVELENGTH_COUNT = 4992


def return_spectra_2(qso_record_table, mem_map_file=FILENAME):

    spectra = Hdf5SpectrumContainer(mem_map_file, True)

    # we assume that the order of spectra is the same as in the QSO list
    for i, j in itertools.izip(Hdf5SpectrumIterator(spectra), qso_record_table):
        qso_rec = QSORecord.from_row(j)
        ar_wavelength = i.get_wavelength()
        ar_flux = i.get_flux()
        if ar_wavelength.size == 0:
            continue
        yield ar_wavelength, ar_flux, qso_rec

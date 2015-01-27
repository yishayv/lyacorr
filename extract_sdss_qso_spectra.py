import multiprocessing
import itertools
import random

import numpy as np
import astropy.table as table

import read_spectrum_fits
from numpy_spectrum_container import NpSpectrumContainer


MAX_SPECTRA = 1000
MAX_WAVELENGTH_COUNT = 4992

FORCE_SINGLE_PROCESS = 0

index = 0

fp = None


def save_spectrum(qso_spec_obj):
    qso_rec = qso_spec_obj[2]
    z = qso_rec.z
    ar_wavelength = qso_spec_obj[0]
    ar_flux = qso_spec_obj[1]
    return [ar_wavelength, ar_flux]


sample_fraction = 1

qso_record_table = table.Table(np.load('../../data/QSO_table.npy'))

spec_sample = read_spectrum_fits.return_spectra_2(qso_record_table)

output_spectra = NpSpectrumContainer('/mnt/gastro/yishay/sdss_QSOs/spectra.npy', readonly=False,
                                     num_spectra=MAX_SPECTRA)

pool = multiprocessing.Pool()

if 1 == FORCE_SINGLE_PROCESS:
    result_enum = itertools.imap(save_spectrum,
                                 itertools.ifilter(lambda x: random.random() < sample_fraction, spec_sample))
else:
    result_enum = pool.imap(save_spectrum,
                            itertools.ifilter(lambda x: random.random() < sample_fraction, spec_sample), 100)

for i in result_enum:
    output_spectra.set_wavelength(index, i[0])
    output_spectra.set_flux(index, i[1])
    index += 1

pool.close()
pool.join()
del fp
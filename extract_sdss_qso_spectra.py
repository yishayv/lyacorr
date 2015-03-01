import multiprocessing
import itertools
import random
import cProfile

import numpy as np
import astropy.table as table

import read_spectrum_fits
from hdf5_spectrum_container import Hdf5SpectrumContainer
import common_settings
from qso_data import QSORecord, QSOData


MAX_SPECTRA = 220000
MAX_WAVELENGTH_COUNT = 4992

settings = common_settings.Settings()


def save_spectrum(qso_spec_obj):
    """

    :type qso_spec_obj: QSOData
    :return:
    """
    qso_rec = qso_spec_obj.qso_rec
    z = qso_rec.z
    ar_wavelength = qso_spec_obj.ar_wavelength
    ar_flux = qso_spec_obj.ar_flux
    return [ar_wavelength, ar_flux]


sample_fraction = 1


def profile_main():
    index = 0

    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))

    spec_sample = read_spectrum_fits.return_spectra_2(qso_record_table)

    mean_qso_spectra_hdf5 = settings.get_mean_qso_spectra_hdf5()
    output_spectra = Hdf5SpectrumContainer(mean_qso_spectra_hdf5, readonly=False,
                                           num_spectra=MAX_SPECTRA)

    if settings.get_single_process():
        result_enum = itertools.imap(save_spectrum,
                                     itertools.ifilter(lambda x: random.random() < sample_fraction, spec_sample))
    else:
        pool = multiprocessing.Pool()
        result_enum = pool.imap(save_spectrum,
                                itertools.ifilter(lambda x: random.random() < sample_fraction, spec_sample), 100)

    for i in result_enum:
        output_spectra.set_wavelength(index, i[0])
        output_spectra.set_flux(index, i[1])
        index += 1

    if not settings.get_single_process():
        pool.close()
        pool.join()


if settings.get_profile():
    cProfile.run('profile_main()', sort=2, filename='extract_sdss_qso_spectra.prof')
else:
    profile_main()

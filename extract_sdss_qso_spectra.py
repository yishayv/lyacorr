import itertools

import cProfile

import numpy as np
import astropy.table as table

from data_access import read_spectrum_fits

from pixel_flags import FlagStats
from data_access.hdf5_spectrum_container import Hdf5SpectrumContainer
import common_settings
from data_access.qso_data import QSOData


MAX_SPECTRA = 220000
MAX_WAVELENGTH_COUNT = 4992

settings = common_settings.Settings()


def save_spectrum(qso_spec_obj):
    """

    :type qso_spec_obj: QSOData
    :return:
    """
    qso_rec = qso_spec_obj.qso_rec
    index = qso_rec.index
    ar_wavelength = qso_spec_obj.ar_wavelength
    ar_flux = qso_spec_obj.ar_flux
    ar_ivar = qso_spec_obj.ar_ivar
    return [index, ar_wavelength, ar_flux, ar_ivar]


def profile_main():

    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))

    flag_stats = FlagStats()

    # assume qso_record_table is already sorted
    spec_sample = read_spectrum_fits.enum_spectra(qso_record_table, pre_sort=False, flag_stats=flag_stats)

    qso_spectra_hdf5 = settings.get_qso_spectra_hdf5()
    output_spectra = Hdf5SpectrumContainer(qso_spectra_hdf5, readonly=False, create_new=True,
                                           num_spectra=MAX_SPECTRA)

    if settings.get_single_process():
        result_enum = itertools.imap(save_spectrum, spec_sample)
    else:
        assert False, "Not supported"

    for i in result_enum:
        index = i[0]
        output_spectra.set_wavelength(index, i[1])
        output_spectra.set_flux(index, i[2])
        output_spectra.set_ivar(index, i[3])

    for bit in xrange(0, 32):
        print(flag_stats.to_string(bit))

    print('Total count: ' + str(flag_stats.pixel_count))

if settings.get_profile():
    cProfile.run('profile_main()', sort=2, filename='extract_sdss_qso_spectra.prof')
else:
    profile_main()

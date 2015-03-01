import multiprocessing
import itertools
import random
import cProfile

import numpy as np
import astropy.table as table
import astropy.units as u

import read_spectrum_fits
from qso_data import QSORecord
from read_spectrum_fits import QSO_fields_dict
import common_settings


settings = common_settings.Settings()


def create_rec(i):
    # make sure we have no QSOs with warning bits set (other than bit #4)
    assert not i[QSO_fields_dict['zWarning']] & ~0x10
    return QSORecord(i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']],
                     i[QSO_fields_dict['ra']], i[QSO_fields_dict['dec']],
                     i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
                     i[QSO_fields_dict['fiberID']])


def create_rec_2(i):
    # make sure we have no QSOs with warning bits set (other than bit #4)
    assert not i[QSO_fields_dict['zWarning']] & ~0x10
    return [i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']],
            i[QSO_fields_dict['ra']], i[QSO_fields_dict['dec']],
            i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
            i[QSO_fields_dict['fiberID']]]


def create_qso_table():
    t = table.Table()
    t.add_columns([table.Column(name='specObjID', dtype='i8', unit=None),
                   table.Column(name='z', unit=u.dimensionless_unscaled),
                   table.Column(name='ra', unit=u.degree),
                   table.Column(name='dec', unit=u.degree),
                   table.Column(name='plate', dtype='i4', unit=None),
                   table.Column(name='mjd', dtype='i4', unit=None),
                   table.Column(name='fiberID', dtype='i4', unit=None)])
    return t


def fill_qso_table(t):
    pool = multiprocessing.Pool()

    if settings.get_single_process():
        qso_record_list = map(create_rec_2, itertools.ifilter(lambda x: random.random() < 1,
                                                              read_spectrum_fits.generate_qso_details()))
    else:
        qso_record_list = pool.map(create_rec_2, itertools.ifilter(lambda x: random.random() < 1,
                                                                   read_spectrum_fits.generate_qso_details()), 200)
    # remove None values
    qso_record_list = [i for i in qso_record_list if i is not None]

    for i in qso_record_list:
        t.add_row(i)

    return t


def profile_main():
    t_ = create_qso_table()
    fill_qso_table(t_)
    np.save(settings.get_qso_metadata_npy(), t_)


if settings.get_profile():
    cProfile.run('profile_main()', sort=2)
else:
    profile_main()
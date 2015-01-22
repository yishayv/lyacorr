import numpy as np
import astropy.table as table
import astropy.units as u
import multiprocessing
import itertools
import random
import cPickle

import read_spectrum_fits
from read_spectrum_fits import QSO_fields_dict


def create_rec(i):
    if i[QSO_fields_dict['zWarning']]:
        return None
    return read_spectrum_fits.QSORecord(i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']],
                                        i[QSO_fields_dict['ra']], i[QSO_fields_dict['dec']],
                                        i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
                                        i[QSO_fields_dict['fiberID']])


def create_rec_2(i):
    if i[QSO_fields_dict['zWarning']]:
        return None
    return [i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']],
            i[QSO_fields_dict['ra']], i[QSO_fields_dict['dec']],
            i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
            i[QSO_fields_dict['fiberID']]]


def create_QSO_table():
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
    qso_record_list = pool.map(create_rec_2, itertools.ifilter(lambda x: random.random() < 1,
                                                               read_spectrum_fits.generate_qso_details()), 50)
    # remove None values
    qso_record_list = [i for i in qso_record_list if i is not None]

    for i in qso_record_list:
        t.add_row(i)

    return t


t_ = create_QSO_table()
fill_qso_table(t_)
np.save('../../data/QSO_table.npy', t_)

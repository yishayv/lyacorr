import cProfile

import astropy.table as table
import astropy.units as u
import numpy as np

import common_settings
from data_access import read_spectrum_fits
from data_access.read_spectrum_fits import QSO_fields_dict
from python_compat import range

settings = common_settings.Settings()

galaxy_file = '../../data/GalaxySpectra_test.fit'


def create_record(i):
    # make sure we have no QSOs with warning bits set (other than bit #4 #2 and #0)
    assert not i[QSO_fields_dict['zWarning']] & ~0x15
    # add a zero value for the index, since we sort the table later and overwrite it anyway.
    return [0,
            i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']],
            i[QSO_fields_dict['ra']], i[QSO_fields_dict['dec']],
            i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
            i[QSO_fields_dict['fiberID']], i[QSO_fields_dict['extinction_g']]]


def create_qso_table():
    t = table.Table()
    t.add_columns([table.Column(name='index', dtype='i8', unit=None),
                   table.Column(name='specObjID', dtype='i8', unit=None),
                   table.Column(name='z', unit=u.dimensionless_unscaled),
                   table.Column(name='ra', unit=u.degree),
                   table.Column(name='dec', unit=u.degree),
                   table.Column(name='plate', dtype='i4', unit=None),
                   table.Column(name='mjd', dtype='i4', unit=None),
                   table.Column(name='fiberID', dtype='i4', unit=None),
                   table.Column(name='extinction_g', unit=u.dimensionless_unscaled)])
    return t


def fill_qso_table(t):
    qso_record_list = map(create_record, read_spectrum_fits.generate_qso_details(galaxy_file))

    # remove None values
    qso_record_list = [i for i in qso_record_list if i is not None]

    for i in qso_record_list:
        t.add_row(i)

    return t


def profile_main():
    t_ = create_qso_table()
    fill_qso_table(t_)
    t_.sort(['plate'])

    # add indices after sort
    t_['index'] = range(len(t_))

    np.save('../../data/galaxy_metadata.npy', t_)


if settings.get_profile():
    cProfile.run('profile_main()', filename='write_extinction_bins.prof', sort=2)
else:
    profile_main()

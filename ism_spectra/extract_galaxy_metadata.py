import cProfile

import astropy.table as table
import astropy.units as u
import numpy as np
from astropy.io import fits
from mpi4py import MPI

import common_settings
from python_compat import range

comm = MPI.COMM_WORLD

settings = common_settings.Settings()  # type: common_settings.Settings

galaxy_file_fits = settings.get_galaxy_metadata_fits()
galaxy_file_npy = settings.get_galaxy_metadata_npy()


def convert_fits_columns(fits_data):
    t = table.Table()
    t.add_columns([table.Column(range(len(fits_data)), name='index', dtype='i8', unit=None),
                   table.Column(fits_data['specObjID'], name='specObjID', dtype='i8', unit=None),
                   table.Column(fits_data['z'], name='z', unit=u.dimensionless_unscaled),
                   table.Column(fits_data['ra'], name='ra', unit=u.degree),
                   table.Column(fits_data['dec'], name='dec', unit=u.degree),
                   table.Column(fits_data['plate'], name='plate', dtype='i4', unit=None),
                   table.Column(fits_data['mjd'], name='mjd', dtype='i4', unit=None),
                   table.Column(fits_data['fiberID'], name='fiberID', dtype='i4', unit=None),
                   table.Column(fits_data['extinction_g'], name='extinction_g', unit=u.dimensionless_unscaled)])
    return t


def fill_galaxy_table():
    fits_data = fits.getdata(galaxy_file_fits)

    return convert_fits_columns(fits_data)


def profile_main():
    t_ = fill_galaxy_table()

    if comm.rank == 0:
        t_.sort(['plate'])

        # add indices after sort
        t_['index'] = range(len(t_))

        np.save(galaxy_file_npy, t_)


if settings.get_profile():
    cProfile.run('profile_main()', filename='extract_galaxy_metadata.prof', sort=2)
else:
    profile_main()

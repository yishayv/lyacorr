import cProfile

import astropy.table as table
import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from mpi4py import MPI

import common_settings
from python_compat import range

comm = MPI.COMM_WORLD
ar_map_nside = 2048

settings = common_settings.Settings()  # type: common_settings.Settings

galaxy_file_fits = settings.get_galaxy_metadata_fits()
galaxy_file_npy = settings.get_galaxy_metadata_npy()

ar_dust_map = hp.fitsfunc.read_map(settings.get_planck_extinction_fits(), field=0)


def ra_dec2ang(ra, dec):
    return (90. - dec) * np.pi / 180., ra / 180. * np.pi


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
                   table.Column(fits_data['extinction_g'], name='extinction_g', unit=u.dimensionless_unscaled),
                   table.Column(fits_data['class'], name='class', unit=u.dimensionless_unscaled)
                   ])
    return t


def fill_galaxy_table():
    fits_data = fits.getdata(galaxy_file_fits)

    return convert_fits_columns(fits_data)


def profile_main():
    if comm.rank == 0:
        t = fill_galaxy_table()

        t.sort(['plate', 'mjd', 'fiberID'])

        # add indices after sort
        t['index'] = range(len(t))

        ar_ra, ar_dec = t['ra'], t['dec']
        coordinates_icrs = SkyCoord(ra=ar_ra, dec=ar_dec)
        coordinates_galactic = coordinates_icrs.galactic

        theta, phi = ra_dec2ang(coordinates_galactic.l.value, coordinates_galactic.b.value)
        ar_pix = hp.ang2pix(ar_map_nside, theta, phi)

        t['extinction_v_planck'] = ar_dust_map[ar_pix]

        np.save(galaxy_file_npy, t)


if settings.get_profile():
    cProfile.run('profile_main()', filename='extract_galaxy_metadata.prof', sort=2)
else:
    profile_main()

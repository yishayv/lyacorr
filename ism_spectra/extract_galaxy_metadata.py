import cProfile
from collections import namedtuple

import astropy.table as table
import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from mpi4py import MPI

import common_settings
from ism_spectra.sfd_lookup import SFDLookUp
from python_compat import range, zip

comm = MPI.COMM_WORLD

settings = common_settings.Settings()  # type: common_settings.Settings

galaxy_file_fits = settings.get_galaxy_metadata_fits()
galaxy_file_npy = settings.get_galaxy_metadata_npy()

HealPixMapEntry = namedtuple('HealPixMapEntry', ['data', 'nside', 'filename', 'column_name'])
column_names = settings.get_custom_column_names()
file_names = settings.get_custom_healpix_maps()
fields = settings.get_custom_healpix_data_fields()


def make_heal_pix_map_entry(filename, column_name, field):
    print("Loading: {0}:{2} as column '{1}'".format(filename, column_name, field))
    data = hp.fitsfunc.read_map(filename, field=field)
    nside = hp.npix2nside(data.size)
    return HealPixMapEntry(data=data, nside=nside, filename=filename, column_name=column_name)


healpix_maps = [make_heal_pix_map_entry(filename, column_name, field)
                for filename, column_name, field in zip(file_names, column_names, fields)]


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

        # add a column for extinction from the full resolution SFD map:
        sfd = SFDLookUp(*settings.get_sfd_maps_fits())
        t['extinction_sfd_hires'] = sfd.lookup_bilinear(coordinates_galactic.l.to(u.rad).value,
                                                        coordinates_galactic.b.to(u.rad).value)

        # add custom columns from healpix map lookup, based on the common settings.
        theta, phi = ra_dec2ang(coordinates_galactic.l.value, coordinates_galactic.b.value)

        for healpix_map in healpix_maps:
            # lookup values in current map
            map_lookup_results = hp.ang2pix(healpix_map.nside, theta, phi)
            # add a new column to the table
            t[healpix_map.column_name] = healpix_map.data[map_lookup_results]

        np.save(galaxy_file_npy, t)


if settings.get_profile():
    cProfile.run('profile_main()', filename='extract_galaxy_metadata.prof', sort=2)
else:
    profile_main()

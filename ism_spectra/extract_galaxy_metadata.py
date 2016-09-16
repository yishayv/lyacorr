import cProfile
import itertools

import astropy.table as table
import astropy.units as u
import numpy as np
import pyfits
from mpi4py import MPI

import common_settings
from data_access.read_spectrum_fits import QSO_fields_dict
from mpi_helper import get_chunks
from mpi_helper import r_print, l_print
from python_compat import range, map, zip

comm = MPI.COMM_WORLD

settings = common_settings.Settings()  # type: common_settings.Settings

galaxy_file_fits = settings.get_galaxy_metadata_fits()
galaxy_file_npy = settings.get_galaxy_metadata_npy()

num_progress_updates = 100


def get_update_mask(num_updates, num_items):
    mask = np.zeros(num_items, dtype=bool)
    for i in range(num_updates):
        mask[int((i + 1) * num_items / num_updates) - 1] = True
    return mask


def create_record(i):
    # make sure we have no Galaxies/QSOs with warning bits set (other than bit #4 #2 and #0)
    assert not i[QSO_fields_dict['zWarning']] & ~0x15
    # add a zero value for the index, since we sort the table later and overwrite it anyway.
    return [0,
            i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']],
            i[QSO_fields_dict['ra']], i[QSO_fields_dict['dec']],
            i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
            i[QSO_fields_dict['fiberID']], i[QSO_fields_dict['extinction_g']]]


def create_galaxy_table(data=None):
    t = table.Table()
    t.add_columns([table.Column(data[0], name='index', dtype='i8', unit=None),
                   table.Column(data[1], name='specObjID', dtype='i8', unit=None),
                   table.Column(data[2], name='z', unit=u.dimensionless_unscaled),
                   table.Column(data[3], name='ra', unit=u.degree),
                   table.Column(data[4], name='dec', unit=u.degree),
                   table.Column(data[5], name='plate', dtype='i4', unit=None),
                   table.Column(data[6], name='mjd', dtype='i4', unit=None),
                   table.Column(data[7], name='fiberID', dtype='i4', unit=None),
                   table.Column(data[8], name='extinction_g', unit=u.dimensionless_unscaled)])
    return t


def fill_galaxy_table():
    fits_data = pyfits.getdata(galaxy_file_fits)

    chunk_sizes, chunk_offsets = get_chunks(len(fits_data), comm.size)

    local_start_index = chunk_offsets[comm.rank]
    local_end_index = local_start_index + chunk_sizes[comm.rank]
    update_gather_mask = get_update_mask(num_progress_updates, chunk_sizes[comm.rank])

    galaxy_record_list = []
    for n, record in enumerate(fits_data[local_start_index:local_end_index]):
        galaxy_record_list.append(create_record(record))
        if update_gather_mask[n]:
            l_print(n)
            list_n = comm.gather(n)
            if comm.rank == 0:
                r_print(sum(list_n))

    # remove None values
    galaxy_record_list = [i for i in galaxy_record_list if i is not None]

    galaxy_record_list_of_lists = comm.gather(galaxy_record_list)

    if comm.rank == 0:
        galaxy_record_list = itertools.chain(*galaxy_record_list_of_lists)

        # transpose the list so that we add columns rather than
        column_list = list(map(list, zip(*galaxy_record_list)))

        # create an astropy table
        t = create_galaxy_table(column_list)
        return t

    return None


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

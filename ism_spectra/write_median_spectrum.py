import cProfile

import numpy as np
from astropy import table
from mpi4py import MPI
from scipy.signal import savgol_filter

import common_settings
from data_access.qso_data import QSOData
from data_access.read_spectrum_fits import enum_spectra
from mpi_helper import get_chunks
from mpi_helper import l_print_no_barrier, r_print
from python_compat import range

comm = MPI.COMM_WORLD

settings = common_settings.Settings()

num_bins = int(2e3)
spec_res = 0.5
spec_start = 3566
spec_end = 7200
ar_wavelength = np.arange(spec_start, spec_end, spec_res)
spec_size = int(ar_wavelength.size)
# window length must be odd
detrend_window = int(150 / spec_res / 2) * 2 + 1

flux_min = 0.50
flux_max = 1.5
flux_range = flux_max - flux_min

histogram = np.zeros(shape=(num_bins, spec_size))
global_histogram = np.zeros(shape=(num_bins, spec_size))

num_update_gather = 20

galaxy_metadata_file_npy = settings.get_galaxy_metadata_npy()
histogram_output_npz = settings.get_ism_histogram_npz()


def reduce_and_save():
    comm.Reduce(
        [histogram, MPI.DOUBLE],
        [global_histogram, MPI.DOUBLE],
        op=MPI.SUM, root=0)
    if comm.rank == 0:
        np.savez(histogram_output_npz, histogram=global_histogram, ar_wavelength=ar_wavelength,
                 flux_range=[flux_min, flux_max])


def get_update_mask(num_updates, num_items):
    mask = np.zeros(num_items, dtype=bool)
    for i in range(num_updates):
        mask[int((i + 1) * num_items / num_updates) - 1] = True
    return mask


def profile_main():
    galaxy_record_table = table.Table(np.load(galaxy_metadata_file_npy))
    galaxy_record_table.sort(['plate'])

    chunk_sizes, chunk_offsets = get_chunks(len(galaxy_record_table), comm.size)
    local_start_index = chunk_offsets[comm.rank]
    local_end_index = local_start_index + chunk_sizes[comm.rank]
    update_gather_mask = get_update_mask(num_update_gather, chunk_sizes[comm.rank])

    spectrum_iterator = enum_spectra(qso_record_table=galaxy_record_table[local_start_index:local_end_index],
                                     pre_sort=False)
    for n, spectrum in enumerate(spectrum_iterator):  # type: int,QSOData
        ar_flux = np.interp(ar_wavelength, spectrum.ar_wavelength, spectrum.ar_flux, left=np.nan, right=np.nan)
        ar_ivar = np.interp(ar_wavelength, spectrum.ar_wavelength, spectrum.ar_ivar, left=np.nan, right=np.nan)

        ar_trend = savgol_filter(ar_flux, detrend_window, polyorder=2)

        # de-trend the spectrum
        ar_flux /= ar_trend

        ar_flux_int = np.empty(shape=spec_size, dtype=np.int)
        ar_flux_int[:] = ((ar_flux - flux_min) * num_bins / flux_range).astype(np.int)
        ar_flux_int[ar_flux_int >= num_bins] = num_bins - 1
        ar_flux_int[ar_flux_int < 0] = 0

        # noinspection PyArgumentList
        mask = np.logical_and.reduce((np.isfinite(ar_flux), ar_ivar > 0, ar_trend > 2.))

        x = ar_flux_int[mask]
        y = np.arange(spec_size)[mask]
        c = np.ones_like(y)

        histogram[x, y] += c

        if update_gather_mask[n]:
            reduce_and_save()
            l_print_no_barrier(n)
            list_n = comm.gather(n)
            if comm.rank == 0:
                r_print(sum(list_n))

    r_print('------------')
    reduce_and_save()


if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_median_spectrum.prof', sort=2)
else:
    profile_main()

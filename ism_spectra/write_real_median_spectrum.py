import cProfile
from os.path import splitext

import numpy as np
from astropy import table
from mpi4py import MPI
from scipy.signal import savgol_filter

import common_settings
from data_access.qso_data import QSOData
from data_access.read_spectrum_fits import enum_spectra
from mpi_helper import get_chunks
from mpi_helper import r_print, l_print
from python_compat import range

comm = MPI.COMM_WORLD

settings = common_settings.Settings()  # type: common_settings.Settings

histogram_properties = settings.get_histogram_properties()
spec_start = histogram_properties['spec_start']
spec_end = histogram_properties['spec_end']
spec_res = histogram_properties['spec_res']
flux_min = histogram_properties['flux_min']
flux_max = histogram_properties['flux_max']
num_bins = histogram_properties['num_flux_bins']

ar_wavelength = np.arange(spec_start, spec_end, spec_res)
spec_size = int(ar_wavelength.size)
# window length must be odd
detrend_window = int(int(settings.get_detrend_window()) / spec_res / 2) * 2 + 1

flux_range = flux_max - flux_min

num_update_gather = 1


def save(output_file, ar_median, group_parameters):
    np.savez(output_file, ar_wavelength=ar_wavelength,
             ism_spec=ar_median, group_parameters=group_parameters)


def get_update_mask(num_updates, num_items):
    mask = np.zeros(num_items, dtype=bool)
    for i in range(num_updates):
        mask[int((i + 1) * num_items / num_updates) - 1] = True
    return mask


def calc_median_spectrum(galaxy_record_table, histogram_output_npz, group_parameters):
    num_spectra = len(galaxy_record_table)
    # allocate a very big array:
    spectra = np.zeros(shape=(num_spectra, spec_size))

    spectrum_iterator = enum_spectra(qso_record_table=galaxy_record_table,
                                     pre_sort=False, and_mask=np.uint32(0), or_mask=np.uint32(0))
    for n, spectrum in enumerate(spectrum_iterator):  # type: int,QSOData
        ar_flux = np.interp(ar_wavelength, spectrum.ar_wavelength, spectrum.ar_flux, left=np.nan, right=np.nan)
        ar_ivar = np.interp(ar_wavelength, spectrum.ar_wavelength, spectrum.ar_ivar, left=np.nan, right=np.nan)

        ar_trend = savgol_filter(ar_flux, detrend_window, polyorder=2)

        # de-trend the spectrum
        ar_flux /= ar_trend

        # noinspection PyArgumentList
        mask = np.logical_and.reduce((np.isfinite(ar_flux), ar_ivar > 0, ar_trend > 0.5))

        ar_flux[~mask] = np.nan

        spectra[n] = ar_flux

    l_print('Starting Median Calculation')
    # calculate the median of the entire array
    ar_median = np.nanmedian(spectra, axis=0)

    r_print('------------')
    save(output_file=histogram_output_npz, ar_median=ar_median, group_parameters=group_parameters)


def profile_main():
    galaxy_metadata_file_npy = settings.get_galaxy_metadata_npy()
    histogram_output_npz = settings.get_ism_real_median_npz()

    galaxy_record_table = table.Table(np.load(galaxy_metadata_file_npy))

    num_extinction_bins = settings.get_num_extinction_bins()

    # group results into extinction bins with roughly equal number of spectra.
    galaxy_record_table.sort(['extinction_g'])

    # remove objects with unknown extinction
    galaxy_record_table = galaxy_record_table[np.where(np.isfinite(galaxy_record_table['extinction_g']))]

    chunk_sizes, chunk_offsets = get_chunks(len(galaxy_record_table), num_extinction_bins)
    for i in range(num_extinction_bins):
        extinction_bin_start = chunk_offsets[i]
        extinction_bin_end = extinction_bin_start + chunk_sizes[i]

        extinction_bin_record_table = galaxy_record_table[extinction_bin_start:extinction_bin_end]

        # this should be done before plate sort
        group_parameters = {'extinction_bin_number': i,
                            'extinction_minimum': extinction_bin_record_table['extinction_g'][0],
                            'extinction_maximum': extinction_bin_record_table['extinction_g'][-1],
                            'extinction_average': np.mean(extinction_bin_record_table['extinction_g']),
                            'extinction_median': np.median(extinction_bin_record_table['extinction_g']),
                            }

        # sort by plate to avoid constant switching of fits files (which are per plate).
        extinction_bin_record_table.sort(['plate', 'mjd', 'fiberID'])

        base_filename, file_extension = splitext(histogram_output_npz)
        histogram_output_filename = '{}_{:02d}{}'.format(base_filename, i, file_extension)

        r_print('Starting extinction bin {}'.format(i))
        calc_median_spectrum(extinction_bin_record_table, histogram_output_filename, group_parameters=group_parameters)
        r_print('Finished extinction bin {}'.format(i))


if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_median_spectrum.prof', sort=2)
else:
    profile_main()

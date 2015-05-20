from astropy import table as table
import numpy as np

import common_settings
from data_access.numpy_spectrum_container import NpSpectrumContainer
from mpi_accumulate import comm


settings = common_settings.Settings()


def update_mean(delta_t_file):
    n = 0
    ar_z = np.arange(1.9, 3.3, 0.001)
    ar_delta_t_sum = np.zeros_like(ar_z)
    ar_delta_t_count = np.zeros_like(ar_z)
    ar_delta_t_weighted = np.zeros_like(ar_z)
    ar_ivar_total = np.zeros_like(ar_z)
    # calculate the weighted sum of the delta transmittance per redshift bin.
    for i in xrange(delta_t_file.num_spectra):
        ar_wavelength = delta_t_file.get_wavelength(i)
        ar_flux = delta_t_file.get_flux(i)
        ar_ivar = delta_t_file.get_ivar(i)
        if ar_wavelength.size:
            ar_delta_t = np.interp(ar_z, ar_wavelength, ar_flux, 0, 0)
            ar_ivar = np.interp(ar_z, ar_wavelength, ar_ivar, 0, 0)
            ar_delta_t_sum += ar_delta_t
            ar_delta_t_weighted += ar_delta_t * ar_ivar
            ar_delta_t_count += ar_delta_t != 0
            ar_ivar_total += ar_ivar
            n += 1

    # save intermediate result (the mean delta_t before removal)
    np.save(settings.get_mean_delta_t_npy(), np.vstack((ar_z,
                                                        ar_delta_t_weighted, ar_ivar_total,
                                                        ar_delta_t_sum, ar_delta_t_count)))
    return ar_delta_t_weighted, ar_ivar_total, ar_z, n


def remove_mean():
    """
    Remove the mean of the delta transmittance per redshift bin.
    The change is made in-place.

    :return:
    """

    # execute only on rank 0, since this is a simple IO-bound operation.
    comm.Barrier()
    if comm.rank != 0:
        return

    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    delta_t_file = NpSpectrumContainer(readonly=False, create_new=False, num_spectra=len(qso_record_table),
                                       filename=settings.get_delta_t_npy(), max_wavelength_count=1000)

    ar_delta_t_weighted, ar_ivar_total, ar_z, n = update_mean(delta_t_file)

    # remove nan values (redshift bins with a total weight of 0)
    mask = ar_ivar_total != 0

    # calculate the mean of the delta transmittance per redshift bin.
    ar_weighted_mean_no_nan = ar_delta_t_weighted[mask] / ar_ivar_total[mask]
    ar_z_no_nan = ar_z[mask]

    empty_array = np.array([])

    n = 0
    # remove the mean (in-place)
    for i in xrange(delta_t_file.num_spectra):
        ar_wavelength = delta_t_file.get_wavelength(i)
        ar_flux = delta_t_file.get_flux(i)
        ar_ivar = delta_t_file.get_ivar(i)
        if ar_wavelength.size:
            ar_delta_t_correction = np.interp(ar_wavelength, ar_z_no_nan, ar_weighted_mean_no_nan, 0, 0)
            delta_t_file.set_wavelength(i, ar_wavelength)
            delta_t_file.set_flux(i, ar_flux - ar_delta_t_correction)
            delta_t_file.set_ivar(i, ar_ivar)
            n += 1
        else:
            delta_t_file.set_wavelength(i, empty_array)
            delta_t_file.set_flux(i, empty_array)
            delta_t_file.set_ivar(i, empty_array)


def get_weighted_mean_from_file():
    ar_mean_delta_t_table = np.load(settings.get_mean_delta_t_npy())
    ar_z, ar_delta_t_weighted, ar_ivar_total, ar_delta_t_sum, ar_delta_t_count = np.vsplit(ar_mean_delta_t_table,5)
    mask = ar_ivar_total != 0

    return ar_z[mask], ar_delta_t_weighted[mask] / ar_ivar_total[mask]

if __name__ == '__main__':
    remove_mean()
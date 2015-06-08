from astropy import table as table
import numpy as np
from scipy import interpolate

import common_settings
from data_access.numpy_spectrum_container import NpSpectrumContainer
from mpi_accumulate import comm

settings = common_settings.Settings()


def update_mean(delta_t_file):
    n = 0
    ar_z = np.arange(1.9, 3.3, 0.001)

    # weighted mean
    ar_delta_t_sum = np.zeros_like(ar_z)
    ar_delta_t_count = np.zeros_like(ar_z)
    ar_delta_t_weighted = np.zeros_like(ar_z)

    # histogram median
    delta_t_min, delta_t_max = (0, 1)
    delta_t_num_buckets = 1000
    ar_delta_t_histogram = np.zeros(shape=(ar_z.size, delta_t_num_buckets))

    ar_ivar_total = np.zeros_like(ar_z)
    # calculate the weighted sum of the delta transmittance per redshift bin.
    for i in xrange(delta_t_file.num_spectra):
        ar_z_unbinned = delta_t_file.get_wavelength(i)
        ar_delta_t_unbinned = delta_t_file.get_flux(i)
        ar_ivar_unbinned = delta_t_file.get_ivar(i)
        if ar_z_unbinned.size:
            f_delta_t = interpolate.interp1d(ar_z_unbinned, ar_delta_t_unbinned,
                                             kind='nearest', bounds_error=False,
                                             fill_value=0, assume_sorted=True)
            ar_delta_t = f_delta_t(ar_z)
            f_ivar = interpolate.interp1d(ar_z_unbinned, ar_ivar_unbinned,
                                          kind='nearest', bounds_error=False,
                                          fill_value=0, assume_sorted=True)
            ar_ivar = f_ivar(ar_z)

            ar_delta_t_sum += ar_delta_t
            ar_delta_t_weighted += ar_delta_t * ar_ivar
            ar_delta_t_count += ar_delta_t != 0
            ar_ivar_total += ar_ivar

            ar_delta_t_clipped = np.clip(ar_delta_t, delta_t_min, delta_t_max)
            ar_delta_t_buckets = (ar_delta_t_clipped * np.reciprocal(delta_t_max - delta_t_min)
                                  * delta_t_num_buckets).astype(np.int32)
            ar_delta_t_buckets = np.clip(ar_delta_t_buckets, 0, delta_t_num_buckets - 1)
            ar_delta_t_histogram_current = np.bincount(ar_delta_t_buckets, ar_ivar, minlength=delta_t_num_buckets)
            ar_delta_t_histogram += ar_delta_t_histogram_current
            n += 1

    # save intermediate result (the mean delta_t before removal)
    np.save(settings.get_mean_delta_t_npy(), np.vstack((ar_z,
                                                        ar_delta_t_weighted, ar_ivar_total,
                                                        ar_delta_t_sum, ar_delta_t_count)))

    np.save(settings.get_median_delta_t_npy(), np.hstack((np.atleast_2d(ar_z).T, ar_delta_t_histogram)))
    return ar_delta_t_weighted, ar_ivar_total, ar_z, n


# noinspection PyShadowingNames
def remove_mean(delta_t, ar_delta_t_weighted, ar_ivar_total, ar_z):
    """
    Remove the mean of the delta transmittance per redshift bin.
    The change is made in-place.

    :return:
    """

    # remove nan values (redshift bins with a total weight of 0)
    mask = ar_ivar_total != 0

    # calculate the mean of the delta transmittance per redshift bin.
    ar_weighted_mean_no_nan = ar_delta_t_weighted[mask] / ar_ivar_total[mask]
    ar_z_no_nan = ar_z[mask]

    empty_array = np.array([])

    n = 0
    # remove the mean (in-place)
    for i in xrange(delta_t.num_spectra):
        ar_wavelength = delta_t.get_wavelength(i)
        ar_flux = delta_t.get_flux(i)
        ar_ivar = delta_t.get_ivar(i)
        if ar_wavelength.size:
            ar_delta_t_correction = np.interp(ar_wavelength, ar_z_no_nan, ar_weighted_mean_no_nan, 0, 0)
            delta_t.set_wavelength(i, ar_wavelength)
            delta_t.set_flux(i, ar_flux - ar_delta_t_correction)
            delta_t.set_ivar(i, ar_ivar)
            n += 1
        else:
            delta_t.set_wavelength(i, empty_array)
            delta_t.set_flux(i, empty_array)
            delta_t.set_ivar(i, empty_array)


def get_weighted_mean_from_file():
    ar_mean_delta_t_table = np.load(settings.get_mean_delta_t_npy())
    ar_z, ar_delta_t_weighted, ar_ivar_total, ar_delta_t_sum, ar_delta_t_count = np.vsplit(ar_mean_delta_t_table, 5)
    mask = ar_ivar_total != 0

    return ar_z[mask], ar_delta_t_weighted[mask] / ar_ivar_total[mask]


if __name__ == '__main__':
    # execute only on rank 0, since this is a simple IO-bound operation.
    comm.Barrier()
    if comm.rank != 0:
        exit()

    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    delta_t_file = NpSpectrumContainer(readonly=False, create_new=False, num_spectra=len(qso_record_table),
                                       filename=settings.get_delta_t_npy(), max_wavelength_count=1000)

    ar_delta_t_weighted, ar_ivar_total, ar_z, n = update_mean(delta_t_file)

    # remove_mean(delta_t_file, ar_delta_t_weighted, ar_ivar_total, ar_z)

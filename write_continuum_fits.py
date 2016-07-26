import cProfile
import itertools
import pprint
from collections import Counter

import numpy as np
from mpi4py import MPI

import common_settings
import continuum_goodness_of_fit
import median_transmittance
from continuum_fit_container import ContinuumFitContainerFiles, ContinuumFitContainer
from continuum_fit_pca import ContinuumFitPCA
from data_access import read_spectrum_hdf5
from delta_transmittance_remove_mean import get_weighted_mean_from_file
from mpi_accumulate import accumulate_over_spectra
from mpi_helper import l_print_no_barrier, r_print
from physics_functions.pre_process_spectrum import PreProcessSpectrum

MAX_WAVELENGTH_COUNT = 4992

comm = MPI.COMM_WORLD

settings = common_settings.Settings()
fit_pca = ContinuumFitPCA()
l_print_no_barrier('Continuum fit SNR selection Power-law: {0}'.format(
    continuum_goodness_of_fit.power_law_to_string(fit_pca.power_law_fit_result)))

z_range = (1.9, 3.5, 0.0001)
local_stats = Counter(
    {'bad_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'no_flux_calibration': 0, 'no_mw_lines': 0,
     'accepted': 0})
pre_process_spectrum = PreProcessSpectrum()


class ContinuumAccumulator:
    def __init__(self, num_spectra):
        self.num_spectra = num_spectra
        self.continuum_fit_container = ContinuumFitContainerFiles(
            create_new=True, num_spectra=self.num_spectra)
        self.n = 0

    def accumulate(self, result_enum, ar_qso_indices_list, object_all_results):
        for ar_continua, ar_qso_indices, object_result in itertools.izip(
                result_enum, ar_qso_indices_list, object_all_results):

            continua = ContinuumFitContainer.from_np_array_and_object(ar_continua, object_result)
            # array based mpi gather returns zeros at the end of the global array.
            # use the fact that the object based gather returns the correct number of elements:
            num_spectra = len(object_result)
            for n in xrange(num_spectra):
                index = ar_qso_indices[n]
                self.continuum_fit_container.set_wavelength(index, continua.get_wavelength(n))
                self.continuum_fit_container.set_flux(index, continua.get_flux(n))
                # TODO: refactor
                self.continuum_fit_container.copy_metadata(index, continua.get_metadata(n))
                self.n += 1
            l_print_no_barrier("n =", self.n)
        l_print_no_barrier("n =", self.n)
        return self.return_result()

    def return_result(self):
        return self.n

    def finalize(self):
        self.continuum_fit_container.save()


def do_continuum_fit_chunk(qso_record_table):
    start_offset = qso_record_table[0]['index']
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5())
    num_spectra = len(qso_record_table)
    continuum_chunk = ContinuumFitContainer(num_spectra)

    # DISABLED FOR NOW
    # use_existing_mean_transmittance = os.path.exists(settings.get_median_transmittance_npy()) and os.path.exists(
    #     settings.get_mean_delta_t_npy())
    use_existing_mean_transmittance = False

    median_flux_correction_func = None
    if use_existing_mean_transmittance:
        # m = mean_transmittance.MeanTransmittance.from_file(settings.get_mean_transmittance_npy())
        med = median_transmittance.MedianTransmittance.from_file(settings.get_median_transmittance_npy())
        # for debugging with a small data set:
        # ignore values with less than 20 sample points
        # ar_z_mean_flux, ar_mean_flux = m.get_weighted_mean_with_minimum_count(20)
        ar_z_mean_flux, ar_mean_flux = med.get_weighted_median_with_minimum_count(20)
        median_flux_func = lambda (ar_z): np.interp(ar_z, ar_z_mean_flux, ar_mean_flux)
        ar_z_mean_correction, ar_mean_correction = get_weighted_mean_from_file()
        median_flux_correction_func = lambda (ar_z): median_flux_func(ar_z) * (
            1 - np.interp(ar_z, ar_z_mean_correction, ar_mean_correction))

    for n in xrange(len(qso_record_table)):
        current_qso_data = spectra.return_spectrum(n)

        pre_processed_qso_data, result_string = pre_process_spectrum.apply(current_qso_data)

        if result_string != 'processed':
            # error during pre-processing. log statistics of error causes.
            local_stats[result_string] += 1
            continue

        ar_wavelength = pre_processed_qso_data.ar_wavelength
        ar_flux = pre_processed_qso_data.ar_flux
        ar_ivar = pre_processed_qso_data.ar_ivar
        qso_rec = pre_processed_qso_data.qso_rec
        z = qso_rec.z
        assert ar_flux.size == ar_ivar.size

        if not ar_ivar.sum() > 0 or not np.any(np.isfinite(ar_flux)):
            # no useful data
            local_stats['empty'] += 1
            continue

        fit_spectrum, fit_normalization_factor, is_good_fit = \
            fit_pca.fit(ar_wavelength / (1 + z), ar_flux, ar_ivar, z, boundary_value=np.nan,
                        mean_flux_constraint_func=median_flux_correction_func)

        if not is_good_fit:
            local_stats['bad_fit'] += 1
            l_print_no_barrier("skipped QSO (bad fit): ", qso_rec)
            continue

        continuum_chunk.set_wavelength(n, ar_wavelength)
        continuum_chunk.set_flux(n, fit_spectrum)
        # TODO: find a way to estimate error, or create a file without ivar values.

        local_stats['accepted'] += 1

    l_print_no_barrier("offset =", start_offset)
    return continuum_chunk.as_np_array(), continuum_chunk.as_object()


def profile_main():
    accumulate_over_spectra(do_continuum_fit_chunk, ContinuumAccumulator)
    l_print_no_barrier(pprint.pformat(local_stats))

    snr_stats_list = comm.gather(fit_pca.snr_stats)
    stats_list = comm.gather(local_stats)
    if comm.rank == 0:
        total_stats = sum(stats_list, Counter())
        r_print(pprint.pformat(total_stats))
        snr_stats = np.zeros_like(snr_stats_list[0])
        for i in snr_stats_list:
            snr_stats += i
        np.save(settings.get_fit_snr_stats(), snr_stats)


if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_continuum_fits.prof', sort=2)
else:
    profile_main()

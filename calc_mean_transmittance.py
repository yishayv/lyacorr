"""
    This module is responsible for calculating the mean/median transmittance, as well
    as the delta of each forest from that value.
    It uses MPI to select different spectra for different nodes, and then combines
    the results in Rank 0 (root).
    The spectra are processed in chunks, and gathered to the root rank after each sub-chunk.
"""
import itertools
import pprint
from collections import Counter

import numpy as np
from scipy import interpolate

from data_access import read_spectrum_hdf5
import mean_transmittance
import median_transmittance
from continuum_fit_pca import ContinuumFitContainerFiles, ContinuumFitPCA
from mpi_accumulate import accumulate_over_spectra, comm
import common_settings
from data_access.numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from data_access.qso_data import QSOData
from physics_functions import pixel_weight_coefficients
from lya_data_structures import LyaForestTransmittanceBinned, LyaForestTransmittance
from mpi_helper import l_print_no_barrier
from physics_functions.pre_process_spectrum import PreProcessSpectrum

lya_center = 1215.67

settings = common_settings.Settings()
force_single_process = settings.get_single_process()
fit_pca_files = settings.get_pca_continuum_tables()
fit_pca = ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])
z_range = (1.9, 3.5, 0.0004)
ar_z_range = np.arange(*z_range)
min_continuum_threshold = settings.get_min_continuum_threshold()
local_stats = Counter({'bad_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'accepted': 0})
pre_process_spectrum = PreProcessSpectrum()


class DeltaTransmittanceAccumulator:
    """
        Add delta transmittance data to a single memory mapped file.
        It is intended to be used as a helper object called by mpi_accumulate.accumulate_over_spectra
    """

    def __init__(self, num_spectra):
        self.num_spectra = num_spectra
        self.delta_t_file = NpSpectrumContainer(False, num_spectra=self.num_spectra,
                                                filename=settings.get_delta_t_npy(), max_wavelength_count=1000)
        self.n = 0
        # initialize file
        self.delta_t_file.zero()

    def accumulate(self, result_enum, ar_qso_indices_list, object_results):
        for ar_delta_t, ar_qso_indices in itertools.izip(result_enum, ar_qso_indices_list):
            delta_t = NpSpectrumContainer.from_np_array(ar_delta_t, readonly=True)
            for j, n in itertools.izip(NpSpectrumIterator(delta_t), ar_qso_indices):
                # if self.n >= self.num_spectra:
                # break
                self.delta_t_file.set_wavelength(n, j.get_wavelength())
                self.delta_t_file.set_flux(n, j.get_flux())
                self.delta_t_file.set_ivar(n, j.get_ivar())
                self.n += 1
            l_print_no_barrier("n =", self.n)
        l_print_no_barrier("n =", self.n)
        return self.return_result()

    def return_result(self):
        return self.n, None

    def finalize(self):
        pass


class MeanTransmittanceAccumulator:
    """
        Accumulate transmittance data into a total weighed mean and/or median.
        It is intended to be used as a helper object called by mpi_accumulate.accumulate_over_spectra
    """

    def __init__(self, num_spectra):
        self.m = mean_transmittance.MeanTransmittance(np.arange(*z_range))
        self.med = median_transmittance.MedianTransmittance(np.arange(*z_range))

    def accumulate(self, result_enum, qso_record_table, object_results):
        for ar_m_med in result_enum:
            l_print_no_barrier("--- mean accumulate ----")
            m = mean_transmittance.MeanTransmittance.from_np_array(ar_m_med[0:4])
            self.m.merge(m)
            med = median_transmittance.MedianTransmittance.from_np_array(ar_m_med[4:])
            self.med.merge(med)
        return self.return_result()

    def return_result(self):
        return self.m, self.med

    def finalize(self):
        pass


def qso_transmittance(qso_spec_obj, ar_fit_spectrum):
    """

    :type qso_spec_obj: QSOData
    :return:
    """

    empty_result = LyaForestTransmittance(np.array([]), np.array([]), np.array([]), np.array([]))

    qso_rec = qso_spec_obj.qso_rec
    z = qso_rec.z

    pre_processed_qso_data, result_string = pre_process_spectrum.apply(qso_spec_obj)

    if result_string != 'processed':
        # error during pre-processing. log statistics of error causes.
        local_stats[result_string] += 1
        return empty_result

    ar_wavelength = pre_processed_qso_data.ar_wavelength
    ar_flux = pre_processed_qso_data.ar_flux
    ar_ivar = pre_processed_qso_data.ar_ivar

    assert ar_flux.size == ar_ivar.size

    if not ar_fit_spectrum.size:
        local_stats['bad_fit'] += 1
        l_print_no_barrier("skipped QSO (bad fit): ", qso_rec)
        return empty_result

    assert ar_flux.size == ar_fit_spectrum.size

    if not ar_ivar.sum() > 0 or not np.any(np.isfinite(ar_flux)):
        # no useful data
        local_stats['empty'] += 1
        return empty_result

    # transmission is only meaningful in the ly_alpha range, and also requires a valid fit for that wavelength
    # use the same range as in 1404.1801 (2014)
    forest_mask = np.logical_and(ar_wavelength > 1040 * (1 + z),
                                 ar_wavelength < 1200 * (1 + z))
    fit_mask = ~np.isnan(ar_fit_spectrum)
    # since at high redshift the sample size becomes smaller,
    # discard all forest pixels that have a redshift greater/less than a globally defined value
    min_redshift = settings.get_min_forest_redshift()
    max_redshift = settings.get_max_forest_redshift()
    ar_redshift = ar_wavelength / lya_center - 1

    redshift_mask = (min_redshift < ar_redshift) & (ar_redshift < max_redshift)

    # combine all different masks
    effective_mask = forest_mask & fit_mask & redshift_mask
    ar_wavelength_masked = np.asarray(ar_wavelength[effective_mask])
    ar_fit_spectrum_masked = ar_fit_spectrum[effective_mask]

    # make sure we have any pixes before calling ar_fit_spectrum_masked.min()
    if ar_wavelength_masked.size < 50:
        local_stats['low_count'] += 1
        l_print_no_barrier("skipped QSO (low pixel count): ", qso_rec)
        return empty_result

    fit_min_value = ar_fit_spectrum_masked.min()
    if fit_min_value < min_continuum_threshold:
        local_stats['low_continuum'] += 1
        l_print_no_barrier("skipped QSO (low continuum) :", qso_rec)
        return empty_result

    local_stats['accepted'] += 1
    l_print_no_barrier("accepted QSO", qso_rec)

    ar_rel_transmittance = ar_flux / ar_fit_spectrum
    ar_rel_transmittance_masked = ar_rel_transmittance[effective_mask]
    ar_z_masked = ar_wavelength_masked / lya_center - 1
    assert ar_z_masked.size == ar_rel_transmittance_masked.size
    assert not np.isnan(ar_rel_transmittance_masked.sum())

    # calculate the weight of each point as a delta_t (without the mean transmittance part)
    ar_pipeline_ivar_masked = ar_ivar[effective_mask] * np.square(ar_fit_spectrum_masked)

    # effectively remove the points with very high positive or negative transmittance
    ar_pipeline_ivar_masked[np.logical_or(ar_rel_transmittance_masked > 5, ar_rel_transmittance_masked < -3)] = 0

    l_print_no_barrier("mean transmittance for QSO:", (ar_flux[effective_mask] / ar_fit_spectrum_masked).mean())

    return LyaForestTransmittance(ar_z_masked, ar_rel_transmittance_masked, ar_pipeline_ivar_masked,
                                  ar_fit_spectrum_masked)


def qso_transmittance_binned(qso_spec_obj, ar_fit_spectrum):
    ar_z = ar_z_range
    lya_forest_transmittance = qso_transmittance(qso_spec_obj, ar_fit_spectrum)
    if lya_forest_transmittance.ar_transmittance.size == 0:
        # no samples found, no need to interpolate, just return the empty result
        return LyaForestTransmittanceBinned(lya_forest_transmittance.ar_z,
                                            lya_forest_transmittance.ar_transmittance,
                                            lya_forest_transmittance.ar_ivar)

    # use nearest neighbor to prevent contamination of high accuracy flux, by nearby pixels with low ivar values,
    # with very high (or low) flux.

    f_flux = interpolate.interp1d(lya_forest_transmittance.ar_z, lya_forest_transmittance.ar_transmittance,
                                  kind='nearest', bounds_error=False, assume_sorted=True)
    ar_rel_transmittance_binned = f_flux(ar_z)
    f_ivar = interpolate.interp1d(lya_forest_transmittance.ar_z, lya_forest_transmittance.ar_ivar, bounds_error=False,
                                  kind='nearest', assume_sorted=True)
    ar_ivar_binned = f_ivar(ar_z)
    ar_mask_binned = ~np.isnan(ar_rel_transmittance_binned)
    return LyaForestTransmittanceBinned(ar_mask_binned, ar_rel_transmittance_binned, ar_ivar_binned)


def mean_transmittance_chunk(qso_record_table):
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5())
    continuum_fit_file = ContinuumFitContainerFiles(False)

    m = mean_transmittance.MeanTransmittance(np.arange(*z_range))
    med = median_transmittance.MedianTransmittance(np.arange(*z_range))
    for n in xrange(len(qso_record_table)):
        qso_spec_obj = spectra.return_spectrum(n)
        index = qso_spec_obj.qso_rec.index
        ar_fit_spectrum = continuum_fit_file.get_flux(index)

        lya_forest_transmittance_binned = qso_transmittance_binned(qso_spec_obj, ar_fit_spectrum)
        if lya_forest_transmittance_binned.ar_transmittance.size:
            # save mean and/or median according to common settings:
            if settings.get_enable_weighted_mean_estimator():
                m.add_flux_pre_binned(lya_forest_transmittance_binned.ar_transmittance,
                                      lya_forest_transmittance_binned.ar_mask,
                                      lya_forest_transmittance_binned.ar_ivar)
            if settings.get_enable_weighted_median_estimator():
                med.add_flux_pre_binned(lya_forest_transmittance_binned.ar_transmittance,
                                        lya_forest_transmittance_binned.ar_mask,
                                        lya_forest_transmittance_binned.ar_ivar)
            mean_transmittance_chunk.num_spec += 1

    l_print_no_barrier("finished chunk", mean_transmittance_chunk.num_spec)
    return np.vstack((m.as_np_array(), med.as_np_array())), None


def delta_transmittance_chunk(qso_record_table):
    start_offset = qso_record_table[0]['index']
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5())
    continuum_fit_file = NpSpectrumContainer(True, filename=settings.get_continuum_fit_npy())

    num_spectra = len(qso_record_table)
    delta_t = NpSpectrumContainer(False, num_spectra=num_spectra)
    # warning: np.ndarray is not initialized by default. zeroing manually.
    delta_t.zero()
    m = mean_transmittance.MeanTransmittance.from_file(settings.get_mean_transmittance_npy())
    # m = median_transmittance.MedianTransmittance.from_file(settings.get_median_transmittance_npy())
    # for debugging with a small data set:
    # ignore values with less than 20 sample points
    ar_z_mean_transmittance, ar_mean_transmittance = m.get_weighted_mean_with_minimum_count(20)
    # ar_z_mean_transmittance, ar_mean_transmittance = m.get_weighted_median_with_minimum_count(20, weighted=True)

    pixel_weight = pixel_weight_coefficients.PixelWeight(pixel_weight_coefficients.DEFAULT_WEIGHT_Z_RANGE)
    n = 0
    for n in xrange(len(qso_record_table)):
        qso_spec_obj = spectra.return_spectrum(n)
        index = qso_spec_obj.qso_rec.index
        ar_fit_spectrum = continuum_fit_file.get_flux(index)
        # we assume the fit spectrum uses the same wavelengths.

        lya_forest_transmittance = qso_transmittance(qso_spec_obj, ar_fit_spectrum)
        ar_z = lya_forest_transmittance.ar_z
        if ar_z.size:
            # prepare the mean transmittance for the z range of this QSO
            ar_mean_flux_for_z_range = np.asarray(np.interp(ar_z, ar_z_mean_transmittance, ar_mean_transmittance))

            # delta transmittance is the change in relative transmittance vs the mean
            # therefore, subtract 1.
            ar_delta_t = lya_forest_transmittance.ar_transmittance / ar_mean_flux_for_z_range - 1

            # finish the error estimation, and save it
            ar_delta_t_ivar = pixel_weight.eval(lya_forest_transmittance.ar_ivar,
                                                ar_mean_flux_for_z_range * lya_forest_transmittance.ar_fit,
                                                ar_z)

            # ignore nan or infinite values (in case m_mean has incomplete data because of a low sample size)
            # Note: using wavelength field to store redshift
            finite_mask = np.logical_and(np.isfinite(ar_delta_t), np.isfinite(ar_delta_t_ivar))
            finite_z = ar_z[finite_mask]
            finite_delta_t = ar_delta_t[finite_mask]
            finite_ivar = ar_delta_t_ivar[finite_mask]

            delta_t.set_wavelength(n, finite_z)
            delta_t.set_flux(n, finite_delta_t)
            delta_t.set_ivar(n, finite_ivar)
        else:
            # empty record
            pass
        n += 1

    l_print_no_barrier("chunk n =", n, "offset =", start_offset)
    return delta_t.as_np_array(), None


mean_transmittance_chunk.num_spec = 0


def calc_mean_transmittance():
    m, med = accumulate_over_spectra(mean_transmittance_chunk, MeanTransmittanceAccumulator)
    l_print_no_barrier("-------- END MEAN TRANSMITTANCE -------------")
    l_print_no_barrier(pprint.pformat(local_stats))
    comm.Barrier()

    stats_list = comm.gather(local_stats)
    if comm.rank == 0:
        total_stats = sum(stats_list, Counter())
        print(pprint.pformat(total_stats))
        # decide whether to save mean/median results based on common settings:
        if settings.get_enable_weighted_mean_estimator():
            m.save(settings.get_mean_transmittance_npy())
        if settings.get_enable_weighted_median_estimator():
            med.save(settings.get_median_transmittance_npy())


def calc_delta_transmittance():
    comm.Barrier()
    accumulate_over_spectra(delta_transmittance_chunk,
                            DeltaTransmittanceAccumulator)
    l_print_no_barrier(pprint.pformat(local_stats))

"""
    This module is responsible for calculating the mean/median transmittance, as well
    as the delta of each forest from that value.
    It uses MPI to select different spectra for different nodes, and then combines
    the results in Rank 0 (root).
    The spectra are processed in chunks, and gathered to the root rank after each sub-chunk.
"""
import pprint
from collections import Counter

import numpy as np
from scipy import interpolate

import common_settings
import mean_transmittance
import median_transmittance
from continuum_fit_container import ContinuumFitContainerFiles
from data_access import read_spectrum_hdf5
from data_access.numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from data_access.qso_data import QSOData
from lya_data_structures import LyaForestTransmittanceBinned, LyaForestTransmittance
from mpi_accumulate import accumulate_over_spectra, comm
from mpi_helper import l_print_no_barrier, r_print
from physics_functions import comoving_distance
from physics_functions import pixel_weight_coefficients
from physics_functions.pre_process_spectrum import PreProcessSpectrum
from physics_functions.remove_dla import RemoveDlaSimple
from python_compat import range, zip

lya_center = 1215.67

settings = common_settings.Settings()  # type: common_settings.Settings
force_single_process = settings.get_single_process()
z_range = (1.9, 3.5, 0.0005)
ar_z_range = np.arange(*z_range)
min_continuum_threshold = settings.get_min_continuum_threshold()
local_mean_stats = Counter(
    {'bad_fit': 0, 'empty_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'accepted': 0})
local_delta_stats = Counter(
    {'bad_fit': 0, 'empty_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'accepted': 0})
pre_process_spectrum = PreProcessSpectrum()

cd = comoving_distance.ComovingDistance()


def nu_boxcar(x, y, x_left_func, x_right_func, weights=None):
    y_boxcar = np.zeros_like(y)
    if weights is None:
        weights = np.ones_like(x)
    for n in range(x.size):
        x_left = np.searchsorted(x, x_left_func(x[n]))
        x_right = np.searchsorted(x, x_right_func(x[n]))
        box_weights = weights[x_left:x_right]
        if box_weights.sum() > 0:
            y_boxcar[n] = np.average(y[x_left:x_right], weights=weights[x_left:x_right])
        else:
            y_boxcar[n] = y[n]
    return y_boxcar


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
        del object_results
        for ar_delta_t, ar_qso_indices in zip(result_enum, ar_qso_indices_list):
            delta_t = NpSpectrumContainer.from_np_array(ar_delta_t, readonly=True)
            for j, n in zip(NpSpectrumIterator(delta_t), ar_qso_indices):
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
        del num_spectra
        self.m = mean_transmittance.MeanTransmittance(np.arange(*z_range))
        self.med = median_transmittance.MedianTransmittance(np.arange(*z_range))

    def accumulate(self, result_enum, qso_record_table, object_results):
        del qso_record_table, object_results
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


def downsample_spectrum(ar_wavelength, ar_flux, ar_ivar, scale):
    """
    :type ar_wavelength: np.ndarray
    :type ar_flux: np.ndarray
    :type ar_ivar: np.ndarray
    :type scale: int
    :return: (np.ndarray, np.ndarray, np.ndarray)
    """
    new_length = ar_wavelength.size // scale
    old_length_clipped = new_length * scale

    ar_wavelength_2d = ar_wavelength[:old_length_clipped].reshape((new_length, scale))
    ar_flux_2d = ar_flux[:old_length_clipped].reshape((new_length, scale))
    ar_ivar_2d = ar_ivar[:old_length_clipped].reshape((new_length, scale))
    ar_weighted_flux_2d = ar_flux_2d * ar_ivar_2d
    ar_wavelength_small = np.nanmean(ar_wavelength_2d, axis=1)
    ar_ivar_small = np.nansum(ar_ivar_2d, axis=1)
    with np.errstate(invalid='ignore'):
        ar_flux_small = np.nansum(ar_weighted_flux_2d, axis=1) / ar_ivar_small

    return ar_wavelength_small, ar_flux_small, ar_ivar_small


def qso_transmittance(qso_spec_obj, ar_fit_spectrum, stats, downsample_factor=1):
    """

    :type qso_spec_obj: QSOData
    :type ar_fit_spectrum: np.ndarray
    :type stats: Counter
    :type downsample_factor: int
    :return:
    """

    empty_result = LyaForestTransmittance(np.array([]), np.array([]), np.array([]), np.array([]))

    pre_processed_qso_data, result_string = pre_process_spectrum.apply(qso_spec_obj)

    # set z after pre-processing, because BAL QSOs have visually inspected redshift.
    qso_rec = qso_spec_obj.qso_rec
    z = qso_rec.z

    if result_string != 'processed':
        # error during pre-processing. log statistics of error causes.
        stats[result_string] += 1
        return empty_result

    ar_wavelength = pre_processed_qso_data.ar_wavelength
    ar_flux = pre_processed_qso_data.ar_flux
    ar_ivar = pre_processed_qso_data.ar_ivar

    assert ar_flux.size == ar_ivar.size

    if not ar_fit_spectrum.size:
        stats['empty_fit'] += 1
        l_print_no_barrier("skipped QSO (empty fit): ", qso_rec)
        return empty_result

    assert ar_flux.size == ar_fit_spectrum.size

    if not ar_ivar.sum() > 0 or not np.any(np.isfinite(ar_flux)):
        # no useful data
        stats['empty'] += 1
        return empty_result

    if downsample_factor != 1:
        # downsample the continuum (don't replace ar_wavelength and ar_ivar yet)
        _, ar_fit_spectrum, _ = downsample_spectrum(ar_wavelength, ar_fit_spectrum, ar_ivar, downsample_factor)
        # downsample the spectrum
        ar_wavelength, ar_flux, ar_ivar = downsample_spectrum(ar_wavelength, ar_flux, ar_ivar, downsample_factor)

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

    ivar_mask = ar_ivar > 0

    # combine all different masks
    effective_mask = forest_mask & fit_mask & redshift_mask & ivar_mask
    ar_wavelength_masked = np.asarray(ar_wavelength[effective_mask])
    ar_fit_spectrum_masked = ar_fit_spectrum[effective_mask]

    # make sure we have any pixes before calling ar_fit_spectrum_masked.min()
    if ar_wavelength_masked.size < (150 / downsample_factor):
        stats['low_count'] += 1
        l_print_no_barrier("skipped QSO (low pixel count): ", qso_rec)
        return empty_result

    fit_min_value = ar_fit_spectrum_masked.min()
    if fit_min_value < min_continuum_threshold:
        stats['low_continuum'] += 1
        l_print_no_barrier("skipped QSO (low continuum) :", qso_rec)
        return empty_result

    stats['accepted'] += 1
    l_print_no_barrier("accepted QSO", qso_rec)

    # suppress divide by zero: NaNs can be introduced by the downscale_spectrum method
    with np.errstate(divide='ignore'):
        ar_rel_transmittance = ar_flux / ar_fit_spectrum
    ar_rel_transmittance_masked = ar_rel_transmittance[effective_mask]
    ar_z_masked = ar_wavelength_masked / lya_center - 1
    assert ar_z_masked.size == ar_rel_transmittance_masked.size
    assert not np.isnan(ar_rel_transmittance_masked.sum())

    # calculate the weight of each point as a delta_t (without the mean transmittance part)
    ar_pipeline_ivar_masked = ar_ivar[effective_mask] * np.square(ar_fit_spectrum_masked)

    # optional: remove the weighted average of each forest
    # rel_transmittance_weighted_mean = np.average(ar_rel_transmittance_masked, weights=ar_pipeline_ivar_masked)
    # ar_rel_transmittance -= rel_transmittance_weighted_mean

    l_print_no_barrier("mean transmittance for QSO:", (ar_flux[effective_mask] / ar_fit_spectrum_masked).mean())

    return LyaForestTransmittance(ar_z_masked, ar_rel_transmittance_masked, ar_pipeline_ivar_masked,
                                  ar_fit_spectrum_masked)


def qso_transmittance_binned(qso_spec_obj, ar_fit_spectrum, stats):
    ar_z = ar_z_range
    lya_forest_transmittance = qso_transmittance(qso_spec_obj, ar_fit_spectrum, stats)
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
    start_offset = qso_record_table[0]['index']
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5())
    continuum_fit_file = ContinuumFitContainerFiles(False)

    m = mean_transmittance.MeanTransmittance(np.arange(*z_range))
    med = median_transmittance.MedianTransmittance(np.arange(*z_range))
    for n in range(len(qso_record_table)):
        qso_spec_obj = spectra.return_spectrum(n)
        index = qso_spec_obj.qso_rec.index
        ar_fit_spectrum = continuum_fit_file.get_flux(index)
        if not continuum_fit_file.get_is_good_fit(index):
            local_mean_stats['bad_fit'] += 1
            l_print_no_barrier("skipped QSO (bad fit): ", qso_spec_obj.qso_rec)
            continue

        lya_forest_transmittance_binned = qso_transmittance_binned(qso_spec_obj, ar_fit_spectrum, local_mean_stats)
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

    l_print_no_barrier("finished chunk, num spectra:", mean_transmittance_chunk.num_spec, " offset: ", start_offset)
    return np.vstack((m.as_np_array(), med.as_np_array())), None


def delta_transmittance_chunk(qso_record_table):
    start_offset = qso_record_table[0]['index']
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5())
    continuum_fit_file = ContinuumFitContainerFiles(False)

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
    remove_dla = RemoveDlaSimple()

    pixel_weight = pixel_weight_coefficients.PixelWeight(pixel_weight_coefficients.DEFAULT_WEIGHT_Z_RANGE)
    for n in range(len(qso_record_table)):
        qso_spec_obj = spectra.return_spectrum(n)
        index = qso_spec_obj.qso_rec.index

        if not continuum_fit_file.get_is_good_fit(index):
            local_delta_stats['bad_fit'] += 1
            l_print_no_barrier("skipped QSO (bad fit): ", qso_spec_obj.qso_rec)
            continue

        ar_fit_spectrum = continuum_fit_file.get_flux(index)
        # we assume the fit spectrum uses the same wavelengths.

        lya_forest_transmittance = qso_transmittance(qso_spec_obj, ar_fit_spectrum, local_delta_stats,
                                                     downsample_factor=settings.get_forest_downsample_factor())
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

            # simple DLA removal (without using a catalog)
            if settings.get_enable_simple_dla_removal():
                # remove DLA regions by setting the ivar of nearby pixels to 0
                ar_dla_mask = remove_dla.get_mask(ar_delta_t)
                if np.any(ar_dla_mask):
                    l_print_no_barrier("DLA(s) removed from QSO: ", qso_spec_obj.qso_rec)
                ar_delta_t_ivar[ar_dla_mask] = 0

            # ignore nan or infinite values (in case m_mean has incomplete data because of a low sample size)
            # Note: using wavelength field to store redshift
            finite_mask = np.logical_and(np.isfinite(ar_delta_t), np.isfinite(ar_delta_t_ivar))
            finite_z = ar_z[finite_mask]
            finite_delta_t = ar_delta_t[finite_mask]
            finite_ivar = ar_delta_t_ivar[finite_mask]

            # detrend forests with large enough range in comoving coordinates:
            finite_distances = cd.fast_comoving_distance(finite_z)
            if finite_distances[-1] - finite_distances[0] > 500:
                delta_t_boxcar = nu_boxcar(finite_distances, finite_delta_t, lambda c: c - 300, lambda c: c + 300,
                                           weights=finite_ivar)
                finite_delta_t = finite_delta_t - delta_t_boxcar

            delta_t.set_wavelength(n, finite_z)
            delta_t.set_flux(n, finite_delta_t)
            delta_t.set_ivar(n, finite_ivar)
        else:
            # empty record
            pass
            delta_transmittance_chunk.num_spec += 1

    l_print_no_barrier("finished chunk, num spectra:", delta_transmittance_chunk.num_spec, " offset: ", start_offset)
    return delta_t.as_np_array(), None


mean_transmittance_chunk.num_spec = 0
delta_transmittance_chunk.num_spec = 0


def calc_mean_transmittance():
    m, med = accumulate_over_spectra(mean_transmittance_chunk, MeanTransmittanceAccumulator)
    l_print_no_barrier("-------- END MEAN TRANSMITTANCE -------------")
    l_print_no_barrier(pprint.pformat(local_mean_stats))
    comm.Barrier()

    stats_list = comm.gather(local_mean_stats)
    if comm.rank == 0:
        total_stats = sum(stats_list, Counter())
        r_print(pprint.pformat(total_stats))
        # decide whether to save mean/median results based on common settings:
        if settings.get_enable_weighted_mean_estimator():
            m.save(settings.get_mean_transmittance_npy())
        if settings.get_enable_weighted_median_estimator():
            med.save(settings.get_median_transmittance_npy())


def calc_delta_transmittance():
    comm.Barrier()
    accumulate_over_spectra(delta_transmittance_chunk,
                            DeltaTransmittanceAccumulator)
    l_print_no_barrier(pprint.pformat(local_delta_stats))
    comm.Barrier()

    stats_list = comm.gather(local_delta_stats)
    if comm.rank == 0:
        total_stats = sum(stats_list, Counter())
        r_print(pprint.pformat(total_stats))

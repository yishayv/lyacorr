import itertools

import numpy as np
import astropy.table as table
from scipy import interpolate
from mpi4py import MPI

import mean_flux
import continuum_fit_pca
import read_spectrum_hdf5
import common_settings
from numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from qso_data import QSOData
import pixel_weight_coefficients
from lya_data_structures import LyaForestTransmittanceBinned, LyaForestTransmittance
import mpi_helper
from mpi_helper import l_print_no_barrier
from deredden_func import deredden_spectrum


comm = MPI.COMM_WORLD

lya_center = 1215.67

settings = common_settings.Settings()
force_single_process = settings.get_single_process()
fit_pca_files = settings.get_pca_continuum_tables()
fit_pca = continuum_fit_pca.ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])
z_range = (1.9, 3.5, 0.0001)
ar_z_range = np.arange(*z_range)
min_continuum_threshold = settings.get_min_continuum_threshold()


class DeltaTransmittanceAccumulator:
    def __init__(self, num_spectra):
        self.num_spectra = num_spectra
        self.delta_t_file = NpSpectrumContainer(False, self.num_spectra, settings.get_delta_t_npy(),
                                                max_wavelength_count=1000)
        self.ar_continuum_ivar = np.zeros(self.num_spectra)
        self.n = 0
        # initialize file
        self.delta_t_file.zero()

    def accumulate(self, result_enum, ar_qso_indices_list):
        for ar_delta_t, ar_qso_indices in itertools.izip(result_enum, ar_qso_indices_list):
            delta_t = NpSpectrumContainer.from_np_array(ar_delta_t, 1)
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
        return self.n


class MeanTransmittanceAccumulator:
    def __init__(self, num_spectra):
        self.m = mean_flux.MeanFlux(np.arange(*z_range))

    def accumulate(self, result_enum, qso_record_table):
        for ar_m in result_enum:
            l_print_no_barrier("--- mean accumulate ----")
            m = mean_flux.MeanFlux.from_np_array(ar_m)
            self.m.merge(m)
        return self.return_result()

    def return_result(self):
        return self.m


def qso_transmittance(qso_spec_obj):
    """

    :type qso_spec_obj: QSOData
    :return:
    """
    qso_rec = qso_spec_obj.qso_rec
    z = qso_rec.z
    ar_wavelength = qso_spec_obj.ar_wavelength
    ar_flux = qso_spec_obj.ar_flux
    # extinction correction:
    ar_flux = deredden_spectrum(ar_wavelength, ar_flux, qso_rec.extinction_g)
    # TODO: adjust pipeline variance for extinction
    ar_ivar = qso_spec_obj.ar_ivar
    assert ar_flux.size == ar_ivar.size
    empty_result = LyaForestTransmittance(np.array([]), np.array([]), np.array([]))

    fit_spectrum, fit_normalization_factor, is_good_fit = \
        fit_pca.fit(ar_wavelength / (1 + z), ar_flux, ar_ivar, z, boundary_value=np.nan)

    if not is_good_fit:
        l_print_no_barrier("skipped QSO (bad fit): ", qso_rec)
        return empty_result

    # transmission is only meaningful in the ly_alpha range, and also requires a valid fit for that wavelength
    # use the same range as in 1404.1801 (2014)
    forest_mask = np.logical_and(ar_wavelength > 1040 * (1 + z),
                                 ar_wavelength < 1200 * (1 + z))
    fit_mask = ~np.isnan(fit_spectrum)
    # since at high redshift the sample size becomes smaller,
    # discard all forest pixels that have a redshift greater/less than a globally defined value
    min_redshift = settings.get_min_forest_redshift()
    max_redshift = settings.get_max_forest_redshift()
    ar_redshift = ar_wavelength / lya_center - 1

    redshift_mask = (min_redshift < ar_redshift) & (ar_redshift < max_redshift)

    # combine all different masks
    effective_mask = forest_mask & fit_mask & redshift_mask
    ar_wavelength_masked = ar_wavelength[effective_mask]
    ar_fit_spectrum_masked = fit_spectrum[effective_mask]

    # make sure we have any pixes before calling ar_fit_spectrum_masked.min()
    if ar_wavelength_masked.size < 50:
        l_print_no_barrier("skipped QSO (low pixel count): ", qso_rec)
        return empty_result

    fit_min_value = ar_fit_spectrum_masked.min()
    if fit_min_value < min_continuum_threshold:
        l_print_no_barrier("skipped QSO (low continuum) :", qso_rec)
        return empty_result

    l_print_no_barrier("accepted QSO", qso_rec)

    ar_rel_transmittance = ar_flux / fit_spectrum
    ar_rel_transmittance_masked = ar_rel_transmittance[effective_mask]
    ar_z_masked = ar_wavelength_masked / lya_center - 1
    assert ar_z_masked.size == ar_rel_transmittance_masked.size
    assert not np.isnan(ar_rel_transmittance_masked.sum())

    # calculate the weight of each point as a delta_t (without the mean transmittance part)
    ar_pipeline_ivar_masked = ar_ivar[effective_mask] * np.square(ar_fit_spectrum_masked)

    # effectively remove the points with very high positive or negative transmittance
    ar_pipeline_ivar_masked[np.logical_or(ar_rel_transmittance_masked > 5, ar_rel_transmittance_masked < -3)] = 0

    l_print_no_barrier("mean flux:", (ar_flux[effective_mask] / ar_fit_spectrum_masked).mean())

    return LyaForestTransmittance(ar_z_masked, ar_rel_transmittance_masked, ar_pipeline_ivar_masked)


def qso_transmittance_binned(qso_spec_obj):
    ar_z = ar_z_range
    lya_forest_transmittance = qso_transmittance(qso_spec_obj)
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
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5(),
                                                     table_offset=qso_record_table[0]['index'])
    m = mean_flux.MeanFlux(np.arange(*z_range))
    for i in qso_record_table:
        lya_forest_transmittance_binned = qso_transmittance_binned(spectra.return_spectrum(i['index']))
        if lya_forest_transmittance_binned.ar_transmittance.size:
            m.add_flux_pre_binned(lya_forest_transmittance_binned.ar_transmittance,
                                  lya_forest_transmittance_binned.ar_mask,
                                  lya_forest_transmittance_binned.ar_ivar)
            mean_transmittance_chunk.num_spec += 1

    l_print_no_barrier("finished chunk", mean_transmittance_chunk.num_spec)
    return m


def delta_transmittance_chunk(qso_record_table):
    start_offset = qso_record_table[0]['index']
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5(),
                                                     table_offset=start_offset)
    num_spectra = len(qso_record_table)
    delta_t = NpSpectrumContainer(False, num_spectra)
    # warning: np.ndarray is not initialized by default. zeroing manually.
    delta_t.zero()
    m = mean_flux.MeanFlux.from_file(settings.get_mean_transmittance_npy())
    # for debugging with a small data set:
    # ignore values with less than 20 sample points
    ar_z_mean_flux, ar_mean_flux = m.get_low_pass_mean(20)

    pixel_weight = pixel_weight_coefficients.PixelWeight(pixel_weight_coefficients.DEFAULT_WEIGHT_Z_RANGE)
    chunk_weighted_delta_t = 0
    chunk_weight = 0
    n = 0
    for i in qso_record_table:
        lya_forest_transmittance = qso_transmittance(spectra.return_spectrum(i['index']))
        ar_z = lya_forest_transmittance.ar_z
        if ar_z.size:
            # prepare the mean flux for the z range of this QSO
            ar_mean_flux_for_z_range = np.interp(ar_z, ar_z_mean_flux, ar_mean_flux)

            # delta transmittance is the change in relative transmittance vs the mean
            # therefore, subtract 1.
            ar_delta_t = lya_forest_transmittance.ar_transmittance / ar_mean_flux_for_z_range - 1

            # finish the error estimation, and save it
            ar_delta_t_ivar = pixel_weight.eval(lya_forest_transmittance.ar_ivar, ar_mean_flux_for_z_range,
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

            # accumulate the total weight so that we can zero out the weight mean of delta_t.
            chunk_weight += finite_ivar.sum()
            chunk_weighted_delta_t += (finite_delta_t * finite_ivar).sum()
        else:
            # empty record
            pass
        n += 1

    l_print_no_barrier("chunk n =", n, "offset =", start_offset)
    return delta_t


mean_transmittance_chunk.num_spec = 0


def split_seq(size, iterable):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def accumulate_over_spectra(func, accumulator):
    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    qso_record_count = len(qso_record_table)

    chunk_sizes, chunk_offsets = mpi_helper.get_chunks(qso_record_count, comm.size)

    local_start_index = chunk_offsets[comm.rank]
    local_size = chunk_sizes[comm.rank]
    local_end_index = local_start_index + local_size
    if comm.rank == 0:
        global_acc = accumulator(qso_record_count)

    local_qso_record_table = itertools.islice(qso_record_table, local_start_index, local_end_index)
    l_print_no_barrier("-----", qso_record_count, local_start_index, local_end_index, local_size)
    slice_size = settings.get_file_chunk_size()
    for qso_record_table_chunk, slice_number in itertools.izip(split_seq(slice_size, local_qso_record_table),
                                                               itertools.count()):
        local_result = func(qso_record_table_chunk)
        ar_local_result = local_result.as_np_array()
        ar_all_results = np.zeros(shape=tuple([comm.size] + list(ar_local_result.shape)))
        comm.Gatherv(ar_local_result, ar_all_results, root=0)
        ar_qso_indices = np.zeros(shape=(comm.size, slice_size), dtype=int)
        comm.Gather(np.array([x['index'] for x in qso_record_table_chunk]), ar_qso_indices)

        # "reduce" results
        if comm.rank == 0:
            global_acc.accumulate(ar_all_results, ar_qso_indices)

    l_print_no_barrier("------------------------------")
    if comm.rank == 0:
        return global_acc.return_result()
    else:
        return


def mean_transmittance():
    m = accumulate_over_spectra(mean_transmittance_chunk, MeanTransmittanceAccumulator)
    l_print_no_barrier("-------- END MEAN TRANSMITTANCE -------------")

    if comm.rank == 0:
        m.save(settings.get_mean_transmittance_npy())


def delta_transmittance():
    accumulate_over_spectra(delta_transmittance_chunk,
                            DeltaTransmittanceAccumulator)

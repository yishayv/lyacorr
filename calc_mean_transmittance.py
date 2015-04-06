import itertools
import random

import numpy as np
import astropy.table as table
from scipy import interpolate

import mean_flux
import continuum_fit_pca
import read_spectrum_hdf5
import common_settings
from numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from qso_data import QSOData
import comoving_distance
import pixel_weight_coefficients


lya_center = 1215.67

settings = common_settings.Settings()
force_single_process = settings.get_single_process()
fit_pca_files = settings.get_pca_continuum_tables()
fit_pca = continuum_fit_pca.ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])
z_range = (2.1, 3.5, 0.0001)
ar_z_range = np.arange(*z_range)
cd = comoving_distance.ComovingDistance(2.0, 3.6, 0.001)
min_continuum_threshold = settings.get_min_continuum_threshold()


class DeltaTransmittanceAccumulator:
    def __init__(self, qso_record_table):
        self.num_spectra = len(qso_record_table)
        self.delta_t_file = NpSpectrumContainer(False, self.num_spectra, settings.get_delta_t_npy(),
                                                max_wavelength_count=1000)
        self.ar_continuum_ivar = np.zeros(self.num_spectra)

    def accumulate(self, result_enum):
        # initialize file
        self.delta_t_file.zero()
        n = 0
        total_weight = 0
        total_weighted_delta_t = 0
        for delta_t, chunk_weight, chunk_weighted_delta_t in result_enum:
            for j in NpSpectrumIterator(delta_t):
                self.delta_t_file.set_wavelength(n, j.get_wavelength())
                self.delta_t_file.set_flux(n, j.get_flux())
                self.delta_t_file.set_ivar(n, j.get_ivar())
                n += 1
            total_weight += chunk_weight
            total_weighted_delta_t += chunk_weighted_delta_t
        print "n =", n
        return n, total_weight, total_weighted_delta_t


class MeanTransmittanceAccumulator:
    def __init__(self, qso_record_table):
        self.m = mean_flux.MeanFlux(np.arange(*z_range))

    def accumulate(self, result_enum):
        for i in result_enum:
            self.m.merge(i)
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
    ar_ivar = qso_spec_obj.ar_ivar
    assert ar_flux.size == ar_ivar.size
    empty_result = (np.array([]), np.array([]), np.array([]))

    fit_spectrum, fit_normalization_factor = \
        fit_pca.fit(ar_wavelength / (1 + z), ar_flux, normalized=False, boundary_value=np.nan)

    # transmission is only meaningful in the ly_alpha range, and also requires a valid fit for that wavelength
    # use the same range as in 1404.1801 (2014)
    forest_mask = np.logical_and(ar_wavelength > 1040 * (1 + z),
                                 ar_wavelength < 1200 * (1 + z))
    fit_mask = ~np.isnan(fit_spectrum)
    # since at high redshift the sample size becomes smaller,
    # discard all forest pixels that have a redshift greater than a globally defined value
    redshift_mask = (ar_wavelength / lya_center - 1) < settings.get_max_forest_redshift()
    effective_mask = forest_mask & fit_mask & redshift_mask
    ar_wavelength_masked = ar_wavelength[effective_mask]
    ar_fit_spectrum_masked = fit_spectrum[effective_mask]

    fit_min_value = ar_fit_spectrum_masked.min()
    if fit_min_value < min_continuum_threshold:
        print "skipped QSO (low continuum) :", qso_rec
        return empty_result

    if ar_wavelength_masked.size < 50:
        print "skipped QSO (low pixel count): ", qso_rec
        return empty_result

    print "accepted QSO", qso_rec

    ar_rel_transmittance = ar_flux / fit_spectrum
    ar_rel_transmittance_masked = ar_rel_transmittance[effective_mask]
    ar_z_masked = ar_wavelength_masked / lya_center - 1
    assert ar_z_masked.size == ar_rel_transmittance_masked.size
    assert not np.isnan(ar_rel_transmittance_masked.sum())

    # calculate the weight of each point as a delta_t (without the mean transmittance part)
    ar_pipeline_ivar_masked = ar_ivar[effective_mask] * np.square(ar_fit_spectrum_masked)

    # effectively remove the points with very high positive or negative transmittance
    ar_pipeline_ivar_masked[np.logical_or(ar_rel_transmittance_masked > 5, ar_rel_transmittance_masked < -3)] = 0

    print "mean flux:", (ar_flux[effective_mask] / ar_fit_spectrum_masked).mean()

    return [ar_rel_transmittance_masked, ar_z_masked, ar_pipeline_ivar_masked]


def qso_transmittance_binned(qso_spec_obj):
    [ar_rel_transmittance_clipped, ar_z, ar_delta_t_ivar] = qso_transmittance(qso_spec_obj)
    if ar_rel_transmittance_clipped.size == 0:
        # no samples found, no need to interpolate, just return the empty result
        return [ar_rel_transmittance_clipped, ar_z, ar_delta_t_ivar]

    # use nearest neighbor to prevent contamination of high accuracy flux, by nearby pixels with low ivar values,
    # with very high (or low) flux.

    f_flux = interpolate.interp1d(ar_z, ar_rel_transmittance_clipped, bounds_error=False, assume_sorted=True)
    ar_rel_transmittance_binned = f_flux(ar_z_range)
    f_ivar = interpolate.interp1d(ar_z, ar_delta_t_ivar, bounds_error=False, assume_sorted=True)
    ar_ivar_binned = f_ivar(ar_z_range)
    ar_z_mask_binned = ~np.isnan(ar_rel_transmittance_binned)
    return [ar_rel_transmittance_binned, ar_z_mask_binned, ar_ivar_binned]


def mean_transmittance_chunk(qso_record_table_numbered):
    qso_record_table = [a for a, b in qso_record_table_numbered]
    qso_record_count = [b for a, b in qso_record_table_numbered]
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5(),
                                                     table_offset=qso_record_count[0])
    spec_iter = itertools.imap(spectra.return_spectrum, qso_record_count)
    m = mean_flux.MeanFlux(np.arange(*z_range))
    result_enum = itertools.imap(qso_transmittance_binned, spec_iter)
    for flux, mask, ar_delta_t_ivar in result_enum:
        if flux.size:
            m.add_flux_pre_binned(flux, mask, ar_delta_t_ivar)
            mean_transmittance_chunk.num_spec += 1

    print "finished chunk", mean_transmittance_chunk.num_spec
    return m


def delta_transmittance_chunk(qso_record_table_numbered):
    qso_record_table = [a for a, b in qso_record_table_numbered]
    qso_record_count = [b for a, b in qso_record_table_numbered]
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5(),
                                                     table_offset=qso_record_count[0])
    spec_iter = itertools.imap(spectra.return_spectrum, qso_record_count)
    num_spectra = len(qso_record_count)
    delta_t = NpSpectrumContainer(False, num_spectra)
    # warning: np.ndarray is not initialized by default. zeroing manually.
    delta_t.zero()
    result_enum = itertools.imap(qso_transmittance, spec_iter)
    m = mean_flux.MeanFlux.from_file(settings.get_mean_transmittance_npy())
    ar_mean_flux = m.get_weighted_mean_with_minimum_count(20)
    ar_z_mean_flux = m.get_z_with_minimum_count(20)
    pixel_weight = pixel_weight_coefficients.PixelWeight(pixel_weight_coefficients.DEFAULT_WEIGHT_Z_RANGE)
    n = 0
    chunk_weighted_delta_t = 0
    chunk_weight = 0
    for ar_flux, ar_z, ar_pipeline_ivar in result_enum:
        if ar_z.size:
            # prepare the mean flux for the z range of this QSO
            ar_mean_flux_for_z_range = np.interp(ar_z, ar_z_mean_flux, ar_mean_flux)

            # delta transmittance is the change in relative transmittance vs the mean
            # therefore, subtract 1.
            ar_delta_t = ar_flux / ar_mean_flux_for_z_range - 1

            # finish the error estimation, and save it
            ar_delta_t_ivar = pixel_weight.eval(ar_pipeline_ivar, ar_mean_flux_for_z_range, ar_z)

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
            chunk_weighted_delta_t += (finite_delta_t*finite_ivar).sum()
        else:
            # empty record
            pass
        n += 1

    print "chunk n =", n, "offset =", qso_record_count[0]
    return delta_t, chunk_weight, chunk_weighted_delta_t


mean_transmittance_chunk.num_spec = 0


def split_seq(size, iterable):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def accumulate_over_spectra(func, accumulator, sample_fraction):
    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    qso_record_table_numbered = itertools.izip(qso_record_table, itertools.count())
    acc = accumulator(qso_record_table)

    if force_single_process:
        result_enum = itertools.imap(func,
                                     split_seq(settings.get_file_chunk_size(),
                                               itertools.ifilter(lambda x: random.random() < sample_fraction,
                                                                 qso_record_table_numbered)))
        # "reduce" results
        acc_result = acc.accumulate(result_enum)
    else:
        assert False, "Not supported"

    return acc_result


def mean_transmittance(sample_fraction):
    return accumulate_over_spectra(mean_transmittance_chunk, MeanTransmittanceAccumulator, sample_fraction)


def delta_transmittance(sample_fraction):
    return accumulate_over_spectra(delta_transmittance_chunk, DeltaTransmittanceAccumulator, sample_fraction)

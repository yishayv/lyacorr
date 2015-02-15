import itertools
import random
import multiprocessing

import numpy as np
import astropy.table as table

import mean_flux
import continuum_fit_pca
import read_spectrum_hdf5
import common_settings
from numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
import comoving_distance


lya_center = 1215.67

settings = common_settings.Settings()
force_single_process = settings.get_single_process()
fit_pca_files = settings.get_pca_continuum_tables()
fit_pca = continuum_fit_pca.ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])
z_range = (2.1, 3.5, 0.0001)
ar_z_range = np.arange(*z_range)
cd = comoving_distance.ComovingDistance(2.1, 3.5, 0.001)


class DeltaTransmittanceAccumulator:
    def __init__(self, qso_record_table):
        self.delta_t_file = NpSpectrumContainer(False, len(qso_record_table), settings.get_delta_t_npy())

    def accumulate(self, result_enum):
        n = 0
        for i in result_enum:
            for j in NpSpectrumIterator(i):
                self.delta_t_file.set_wavelength(n, j.get_wavelength())
                self.delta_t_file.set_flux(n, j.get_flux())
                n += 1
        return n


class MeanTransmittanceAccumulator:
    def __init__(self, qso_record_table):
        self.m = mean_flux.MeanFlux(np.arange(*z_range))

    def accumulate(self, result_enum):
        for i in result_enum:
            self.m.merge(i)
        return self.m


def qso_transmittance(qso_spec_obj):
    qso_rec = qso_spec_obj[2]
    z = qso_rec.z
    ar_wavelength = qso_spec_obj[0]
    ar_flux = qso_spec_obj[1]
    empty_result = (np.array([]), np.array([]))

    fit_spectrum, fit_normalization_factor = \
        fit_pca.fit(ar_wavelength / (1 + z), ar_flux, normalized=False, boundary_value=np.nan)

    fit_min_value = fit_spectrum[~np.isnan(fit_spectrum)].min()
    if fit_min_value < 0.01:
        print "low continuum - rejected QSO:", qso_rec
        return empty_result

    # transmission is only meaningful in the ly_alpha range, and also requires a valid fit for that wavelength
    # use the same range as in 1404.1801 (2014)
    forest_mask = np.logical_and(ar_wavelength > 1040 * (1 + z),
                                 ar_wavelength < 1200 * (1 + z))
    fit_mask = ~np.isnan(fit_spectrum)
    ar_wavelength_clipped = ar_wavelength[forest_mask & fit_mask]

    if ar_wavelength_clipped.size < 50:
        print "skipped QSO: ", qso_rec
        return empty_result
    print "accepted QSO", qso_rec

    ar_rel_transmittance = ar_flux / fit_spectrum
    ar_rel_transmittance_clipped = ar_rel_transmittance[forest_mask & fit_mask]
    ar_z = ar_wavelength_clipped / lya_center - 1
    return [ar_rel_transmittance_clipped, ar_z]


def qso_transmittance_binned(qso_spec_obj):
    [ar_rel_transmittance_clipped, ar_z] = qso_transmittance(qso_spec_obj)
    if ar_rel_transmittance_clipped.size == 0:
        # no samples found, no need to interpolate, just return the empty result
        return [ar_rel_transmittance_clipped, ar_z]

    ar_rel_transmittance_binned = np.interp(ar_z_range, ar_z, ar_rel_transmittance_clipped, left=np.nan,
                                            right=np.nan)
    ar_z_mask_binned = ~np.isnan(ar_rel_transmittance_binned)
    return [ar_rel_transmittance_binned, ar_z_mask_binned]


def mean_transmittance_chunk(qso_record_table_numbered):
    qso_record_table = [a for a, b in qso_record_table_numbered]
    qso_record_count = [b for a, b in qso_record_table_numbered]
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, table_offset=qso_record_count[0])
    spec_iter = itertools.imap(spectra.return_spectrum, qso_record_count)
    m = mean_flux.MeanFlux(np.arange(*z_range))
    result_enum = itertools.imap(qso_transmittance_binned, spec_iter)
    for flux, mask in result_enum:
        if flux.size:
            m.add_flux_pre_binned(flux, mask)
            mean_transmittance_chunk.num_spec += 1

    print "finished chunk", mean_transmittance_chunk.num_spec
    return m


def delta_transmittance_chunk(qso_record_table_numbered):
    qso_record_table = [a for a, b in qso_record_table_numbered]
    qso_record_count = [b for a, b in qso_record_table_numbered]
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, table_offset=qso_record_count[0])
    spec_iter = itertools.imap(spectra.return_spectrum, qso_record_count)
    delta_t = NpSpectrumContainer(False, len(qso_record_count))
    result_enum = itertools.imap(qso_transmittance, spec_iter)
    m = mean_flux.MeanFlux.from_file(settings.get_mean_transmittance_npy())
    m_mean = m.get_mean()
    n = 0
    for flux, z in result_enum:
        if z.size:
            # Note: using wavelength field to store redshift
            delta_t.set_wavelength(n, z)
            # delta transmittance is the change in relative transmittance vs the mean
            # therefore, subtract 1
            m_mean_current = np.interp(z, m.ar_z, m_mean)
            delta_t.set_flux(n, flux / m_mean_current - 1.)
        n += 1

    return delta_t


mean_transmittance_chunk.num_spec = 0


def split_seq(size, iterable):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def accumulate_over_spectra(func, accumulator, sample_fraction=0.001):
    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    qso_record_table_numbered = itertools.izip(qso_record_table, itertools.count())
    acc = accumulator(qso_record_table)

    if force_single_process:
        result_enum = itertools.imap(func,
                                     split_seq(settings.get_chunk_size(),
                                               itertools.ifilter(lambda x: random.random() < sample_fraction,
                                                                 qso_record_table_numbered)))
        acc_result = acc.accumulate(result_enum)
    else:
        pool = multiprocessing.Pool()
        # TODO: is ordered imap efficient enough?
        result_enum = pool.imap(func,
                                split_seq(settings.get_chunk_size(),
                                          itertools.ifilter(lambda x: random.random() < sample_fraction,
                                                            qso_record_table_numbered)))
        acc_result = acc.accumulate(result_enum)
        pool.close()
        pool.join()

    return acc_result


def mean_transmittance(sample_fraction=0.001):
    return accumulate_over_spectra(mean_transmittance_chunk, MeanTransmittanceAccumulator, sample_fraction)


def delta_transmittance(sample_fraction=0.001):
    return accumulate_over_spectra(delta_transmittance_chunk, DeltaTransmittanceAccumulator, sample_fraction)

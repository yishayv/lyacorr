import itertools
import random
import multiprocessing

import numpy as np
import astropy.table as table

import mean_flux
import continuum_fit_pca
import read_spectrum_hdf5
import common_settings


FORCE_SINGLE_PROCESS = 0

lya_center = 1215.67

settings = common_settings.Settings()
fit_pca_files = settings.get_pca_continuum_tables()
fit_pca = continuum_fit_pca.ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])
z_range = (2.1, 3.5, 0.00001)
ar_z_range = np.arange(*z_range)
m = mean_flux.MeanFlux(z_range)


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
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table)
    spec_iter = itertools.imap(spectra.return_spectrum, qso_record_count)
    m = mean_flux.MeanFlux(z_range)
    result_enum = itertools.imap(qso_transmittance_binned, spec_iter)
    for i in result_enum:
        if i[0].size:
            m.add_flux_prebinned(i[0], i[1])
            mean_transmittance_chunk.numspec += 1

    print "finished chunk", mean_transmittance_chunk.numspec
    return m


mean_transmittance_chunk.numspec = 0


def chunks(n, iterable):
    iterable = iter(iterable)
    while True:
        yield itertools.chain([next(iterable)], itertools.islice(iterable, n - 1))


def split_seq(size, iterable):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def mean_transmittance(sample_fraction=0.001):
    pool = multiprocessing.Pool()

    qso_record_table = table.Table(np.load('../../data/QSO_table.npy'))
    qso_record_table_numbered = itertools.izip(qso_record_table, itertools.count())

    # spec_sample = read_spectrum_fits.return_spectra_2(qso_record_table)
    # spec_sample = read_spectrum_numpy.return_spectra_2(qso_record_table)
    # spec_sample = read_spectrum_hdf5.return_spectra_2(qso_record_table)

    if 1 == FORCE_SINGLE_PROCESS:
        result_enum = itertools.imap(mean_transmittance_chunk,
                                     split_seq(10000,
                                               itertools.ifilter(lambda x: random.random() < sample_fraction,
                                                                 qso_record_table_numbered)))
    else:
        result_enum = pool.imap_unordered(mean_transmittance_chunk,
                                          split_seq(10000,
                                                    itertools.ifilter(lambda x: random.random() < sample_fraction,
                                                                      qso_record_table_numbered)))

    for i in result_enum:
        m.merge(i)

    pool.close()
    pool.join()

    return m, ar_z_range

import numpy as np
import itertools
import mean_flux
import continuum_fit_pca
import read_spectrum_fits
from read_spectrum_fits import QSO_fields_dict, QSORecord
import random
import multiprocessing
import astropy.table as table

lya_center = 1215.67

fit_pca = continuum_fit_pca.ContinuumFitPCA('../../data/Suzuki/datafile4.txt',
                                            '../../data/Suzuki/datafile3.txt',
                                            '../../data/Suzuki/projection_matrix.csv')
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
    ar_rel_transmittance_binned = np.interp(ar_z_range, ar_z, ar_rel_transmittance_clipped, left=np.nan,
                                            right=np.nan)
    ar_wavelength_mask_binned = ~np.isnan(ar_rel_transmittance_binned)
    return [ar_rel_transmittance_binned, ar_wavelength_mask_binned]


def mean_transmittance():
    spec_sample = []
    qso_record_list = []
    pool = multiprocessing.Pool()

    qso_record_table = table.Table(np.load('../../data/QSO_table.npy'))

    spec_sample = read_spectrum_fits.return_spectra_2(qso_record_table)

    result_enum = pool.imap_unordered(qso_transmittance, spec_sample, 100)

    for i in result_enum:
        if i[0].size:
            m.add_flux_prebinned(i[0], i[1])

    pool.close()
    pool.join()

    return m, ar_z_range

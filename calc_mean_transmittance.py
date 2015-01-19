import numpy as np
import cProfile
import itertools
import matplotlib.pyplot as plt
import mean_flux
import continuum_fit_pca
import read_spectrum_fits
from read_spectrum_fits import QSO_fields_dict

lya_center = 1215.67


def profile_main():
    fit_pca = continuum_fit_pca.ContinuumFitPCA('../../data/Suzuki/datafile4.txt',
                                                '../../data/Suzuki/datafile3.txt',
                                                '../../data/Suzuki/projection_matrix.csv')

    spec_sample = []
    qso_record_list = []
    for i in itertools.islice(read_spectrum_fits.generate_qso_details(), 0, 200):
        rec = read_spectrum_fits.QSORecord(i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']],
                                           i[QSO_fields_dict['ra']], i[QSO_fields_dict['dec']],
                                           i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
                                           i[QSO_fields_dict['fiberID']])
        # print rec
        qso_record_list.append(rec)

    spec_sample = read_spectrum_fits.return_spectra_2(qso_record_list)

    z_range = (2.1, 3.5, 0.005)
    ar_z_range = np.arange(*z_range)
    m = mean_flux.MeanFlux(*z_range)

    print spec_sample.next()[1]
    for j in spec_sample:
        # TODO: why 3817?
        # TODO: better upper limit
        qso_rec = j[2]
        z = qso_rec.z
        ar_wavelength = j[0]
        ar_flux = j[1]
        freq_mask = np.logical_and(ar_wavelength > 3817, ar_wavelength < lya_center * (1 + z) / 1.05)
        ar_wavelength_clipped = ar_wavelength[freq_mask]
        if ar_wavelength_clipped.size < 100:
            print "skipped QSO: ", qso_rec
            continue
        print "accepted QSO", qso_rec

        fit_spectrum, fit_normalization_factor = \
            fit_pca.fit(ar_wavelength / (1 + z), ar_flux, normalized=False, boundary_value=np.nan)

        ar_rel_transmittance = ar_flux / fit_spectrum
        ar_rel_transmittance_clipped = ar_rel_transmittance[freq_mask]
        ar_z = ar_wavelength_clipped / lya_center - 1
        ar_rel_transmittance_binned = np.interp(ar_z_range, ar_z, ar_rel_transmittance_clipped, left=np.nan,
                                                right=np.nan)
        ar_wavelength_mask = ~np.isnan(ar_rel_transmittance_binned)
        # TODO: mean flux should be relative transmittance, not absolute flux
        m.add_flux_prebinned(ar_rel_transmittance_binned, ar_wavelength_mask)

    plt.plot(ar_z_range, m.get_mean() * 100)
    plt.plot(ar_z_range, m.ar_count)
    plt.plot(ar_z_range, m.ar_total_flux)
    plt.show()


cProfile.run('profile_main()', sort=2)

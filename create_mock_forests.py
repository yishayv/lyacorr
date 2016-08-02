import cProfile

import astropy.table as table
import numpy as np

import common_settings
import continuum_fit_pca
import mock_core_with_shell
from data_access.hdf5_spectrum_container import Hdf5SpectrumContainer
from data_access.qso_data import QSORecord
from physics_functions import comoving_distance

draw_graph = False

if draw_graph:
    from mayavi import mlab

MAX_SPECTRA = 220000
MAX_WAVELENGTH_COUNT = 4992

settings = common_settings.Settings()

fit_pca = continuum_fit_pca.ContinuumFitPCA()

z_start = 1.8
z_end = 3.6
z_step = 0.001

lya_center = 1215.67

cd = comoving_distance.ComovingDistance(z_start, z_end, z_step)
mock_forest = mock_core_with_shell.MockForest(settings.get_mock_resolution(), settings.get_mock_fractional_width(),
                                              settings.get_mock_shell_separation(), settings.get_mock_core_radius(),
                                              settings.get_mock_shell_radius())


def profile_main():
    qso_record_table = table.Table(np.load(settings.get_qso_metadata_npy()))
    qso_record_list = [QSORecord.from_row(i) for i in qso_record_table]

    qso_spectra_hdf5 = settings.get_qso_spectra_hdf5()
    output_spectra = Hdf5SpectrumContainer(qso_spectra_hdf5, readonly=False, create_new=False,
                                           num_spectra=MAX_SPECTRA)
    total_ar_x = np.array([])
    total_ar_y = np.array([])
    total_ar_z = np.array([])
    total_ar_c = np.array([])

    for n in xrange(len(qso_record_list)):
        qso_rec = qso_record_list[n]
        redshift = qso_rec.z

        # load data
        ar_wavelength = output_spectra.get_wavelength(n)
        ar_flux = output_spectra.get_flux(n)
        ar_ivar = output_spectra.get_ivar(n)

        # convert wavelength to redshift
        ar_redshift = ar_wavelength / lya_center - 1

        # fit continuum
        ar_rest_wavelength = ar_wavelength / (1 + redshift)

        fit_result = fit_pca.fit(ar_rest_wavelength, ar_flux, ar_ivar, qso_redshift=redshift,
                    boundary_value=np.nan, mean_flux_constraint_func=None)

        # transmission is only meaningful in the ly_alpha range, and also requires a valid fit for that wavelength
        # use the same range as in 1404.1801 (2014)
        forest_mask = np.logical_and(ar_wavelength > 1040 * (1 + redshift),
                                     ar_wavelength < 1200 * (1 + redshift))
        fit_mask = ~np.isnan(fit_result.spectrum)
        effective_mask = forest_mask & fit_mask
        ar_wavelength_masked = ar_wavelength[effective_mask]
        ar_fit_spectrum_masked = fit_result.spectrum[effective_mask]

        # convert redshift to distance
        ar_dist = np.asarray(cd.fast_comoving_distance(ar_redshift[effective_mask]))

        dec = qso_rec.dec * np.pi / 180
        ra = qso_rec.ra * np.pi / 180
        x_unit = np.cos(dec) * np.cos(ra)
        y_unit = np.cos(dec) * np.sin(ra)
        z_unit = np.sin(dec)

        scale = 1
        ar_x = x_unit * ar_dist * scale
        ar_y = y_unit * ar_dist * scale
        # Note: this is the geometric coordinate, not redshift
        ar_z = z_unit * ar_dist * scale

        ar_mock_forest_array = mock_forest.get_forest(ar_x, ar_y, ar_z)

        ar_delta_t = - ar_mock_forest_array

        ar_rel_transmittance = ar_delta_t + 1

        # set the forest part of the spectrum to the mock forest
        mock_fraction = 1
        ar_flux[effective_mask] = \
            ar_flux[effective_mask] * (1 - mock_fraction) + \
            ar_rel_transmittance * fit_result.spectrum[effective_mask] * mock_fraction

        if draw_graph:
            display_mask = ar_mock_forest_array > 0.
            total_ar_x = np.append(total_ar_x, ar_x[display_mask])
            total_ar_y = np.append(total_ar_y, ar_y[display_mask])
            total_ar_z = np.append(total_ar_z, ar_z[display_mask])
            total_ar_c = np.append(total_ar_c, ar_mock_forest_array[display_mask])

        # overwrite the existing forest
        output_spectra.set_flux(n, ar_flux)
        if n % 1000 == 0:
            print(n)

    if draw_graph:
        mlab.points3d(total_ar_x, total_ar_y, total_ar_z, total_ar_c,
                      mode='sphere', scale_mode='vector',
                      scale_factor=20, transparent=True, vmin=0, vmax=1, opacity=0.03)
        mlab.show()


if settings.get_profile():
    cProfile.run('profile_main()', sort=2, filename='create_mock_forests.prof')
else:
    profile_main()

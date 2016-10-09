import os.path

import numpy as np

import common_settings
import data_access.numpy_spectrum_container

settings = common_settings.Settings()  # type: common_settings.Settings

histogram_output_npz = settings.get_ism_histogram_npz()
base_filename, file_extension = os.path.splitext(histogram_output_npz)

real_median_list = []
for i in range(settings.get_num_extinction_bins()):
    real_median_output_npz = settings.get_ism_real_median_npz()
    base_filename, file_extension = os.path.splitext(real_median_output_npz)
    real_median_output_filename = '{}_{:02d}{}'.format(base_filename, i, file_extension)
    real_median_list += [np.load(real_median_output_filename)]

spectra = data_access.numpy_spectrum_container.NpSpectrumContainer(
    readonly=False, create_new=True, num_spectra=len(real_median_list), filename=settings.get_ism_extinction_spectra(),
    max_wavelength_count=real_median_list[0].f.ar_wavelength.size)

# iterate by order of extinction:
for real_median_item in real_median_list:
    spectra.set_wavelength(i, real_median_item.f.ar_wavelength)
    spectra.set_flux(i, real_median_item.f.ism_spec)

ar_extinction_levels = np.array([j.f.group_parameters.item()['extinction_mean'] for j in real_median_list])
np.save(settings.get_ism_extinction_levels(), ar_extinction_levels)

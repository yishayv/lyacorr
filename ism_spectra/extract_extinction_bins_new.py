import os.path

import numpy as np

import common_settings

settings = common_settings.Settings()  # type: common_settings.Settings

histogram_output_npz = settings.get_ism_histogram_npz()
base_filename, file_extension = os.path.splitext(histogram_output_npz)

ism_spectra_list = []
extinction_level_list = []
for i in range(settings.get_num_extinction_bins()):
    real_median_output_npz = settings.get_ism_real_median_npz()
    base_filename, file_extension = os.path.splitext(real_median_output_npz)
    real_median_output_filename = '{}_{:02d}{}'.format(base_filename, i, file_extension)
    real_median = np.load(real_median_output_filename)
    ism_spectra_list += [np.vstack((real_median['ar_wavelength'], real_median['ism_spec']))]
    extinction_level_list += [real_median['group_parameters'].item()['extinction_mean']]

ar_extinction_levels = np.array(extinction_level_list)
np.savez(settings.get_ism_extinction_spectra(), extinction_list=ar_extinction_levels, spectra_list=ism_spectra_list)

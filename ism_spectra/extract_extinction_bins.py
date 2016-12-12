"""
Convert extinction bins spectra from .csv to npz format.
(Deprecated: the csv spectra were produced from SDSS spectra, with limited range of the blue spectrograph.)
"""

import glob
import os.path

import numpy as np

import common_settings

settings = common_settings.Settings()  # type: common_settings.Settings

file_list = glob.glob("../../data/Extinction_Bins_20/*.csv")

# get extinction value from file name as float:
extinction_list = [float(os.path.splitext(os.path.basename(i))[0]) for i in file_list]
# map extinction back to its original file name
extinction_dict = {k: v for k, v in zip(extinction_list, file_list)}

spectra_list = []
max_wavelength_count = 0
# iterate by order of extinction:
for n, (extinction, filename) in enumerate(sorted(extinction_dict.items())):
    print(filename)
    ar_ism_spectrum = np.loadtxt(filename).T
    if ar_ism_spectrum.shape[0] != 2:
        raise Exception('Unexpected data format')
    max_wavelength_count = max(max_wavelength_count, ar_ism_spectrum.shape[1])
    spectra_list += [ar_ism_spectrum]

np.savez(settings.get_ism_extinction_spectra(), extinction_list=extinction_list, spectra_list=spectra_list)

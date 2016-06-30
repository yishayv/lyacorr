import glob
import os.path

import numpy as np

import data_access.numpy_spectrum_container

file_list = glob.glob("../../../data/Extinction_Bins_20/*.csv")
spectra = data_access.numpy_spectrum_container.NpSpectrumContainer(
    readonly=False, create_new=True, num_spectra=len(file_list), filename='../../../data/ExtinctionBins20.npy',
    max_wavelength_count=10880)

# get extinction value from file name as float:
extinction_list = [float(os.path.splitext(os.path.basename(i))[0]) for i in file_list]
# map extinction back to its original file name
extinction_dict = {k: v for k, v in zip(extinction_list, file_list)}

# iterate by order of extinction:
for n, (extinction, filename) in enumerate(sorted(extinction_dict.items())):
    print(filename)
    ar_ism = np.loadtxt(filename).T
    spectra.set_wavelength(n, ar_ism[0])
    spectra.set_flux(n, ar_ism[1])

np.save('../../../data/ExtinctionBins20_values.npy',np.array(extinction_list))
import glob
import itertools

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import data_access.numpy_spectrum_container
from python_compat import range

label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

file_list = glob.glob("../../../data/Extinction_Bins_20/*.csv")
spectra = data_access.numpy_spectrum_container.NpSpectrumContainer(
    readonly=True, create_new=False, num_spectra=len(file_list), filename='../../../data/ExtinctionBins20.npy',
    max_wavelength_count=10880)
ar_extinction = np.load('../../../data/ExtinctionBins20_values.npy')

# trim bad data in high extinction bins
if spectra.num_spectra != ar_extinction.size:
    quit(1)

max_extinction_bins = 21
ar_extinction = ar_extinction[:max_extinction_bins]

plt.plot(ar_extinction)
plt.show()

# assume same wavelengths for all bins
ar_wavelengths = spectra.get_wavelength(0)
ar_2d_plot = np.zeros(shape=(max_extinction_bins, ar_wavelengths.size))

for n in range(max_extinction_bins):
    ar_2d_plot[n] = spectra.get_flux(n)

# center around 0
ar_2d_plot -= 1
# scale by the extinction
# ar_2d_plot /= ar_extinction[:, np.newaxis]
# subtract by the lowest extinctions to show only differences
ar_2d_plot -= ar_2d_plot[9:12, :].mean(axis=0)


def pairwise(iterable):
    """
    transform an iterator as:
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def ism_2d_plot(ar_wavelengths_, ar_2d_plot_):
    split_at = [3000, 4500, 5500, 6500, 7500, 8500, 9500]
    num_subplots = len(split_at) - 1
    abs_range = np.max(np.abs(ar_2d_plot_[:, :]))
    plt.figure(figsize=(15, 8))

    for current_subplot, current_range in enumerate(pairwise(split_at)):
        ar_2d_subplot = ar_2d_plot_[:, np.logical_and(
            ar_wavelengths_ > current_range[0], ar_wavelengths_ < current_range[1])]
        gs = gridspec.GridSpec(num_subplots, 1)
        ax = plt.subplot(gs[current_subplot])
        # plt.subplots_adjust(left=None, bottom=0.001, right=None, top=0.999, wspace=None, hspace=0.0)
        extent = [max(current_range[0], ar_wavelengths_[0]),
                  min(current_range[1], ar_wavelengths_[-1]),
                  ar_2d_plot_.shape[0], 0]
        # ax.set_adjustable('box-forced')

        ax.imshow(ar_2d_subplot, cmap='gray', interpolation='nearest',
                  aspect='auto', extent=extent, vmin=-abs_range, vmax=+abs_range)
        # temp = tic.MaxNLocator(3)
        # ax.yaxis.set_major_locator(temp)
        # ax.set_xticklabels(())
        # ax.set_axis_off()
        # ax.title.set_visible(False)


ism_2d_plot(ar_wavelengths, ar_2d_plot)
plt.tight_layout()
plt.show()

# plt.imshow(np.arcsinh(ar_2d_plot[:, :] * 200), cmap='coolwarm', interpolation='nearest',
#            aspect=None, extent=[ar_wavelengths[0], ar_wavelengths[-1], 0, ar_2d_plot.shape[0]])
plt.show()
quit()

color_values = np.linspace(0, 1, spectra.num_spectra)
colors = [plt.get_cmap('coolwarm')(x) for x in color_values]
for n, color in zip(np.arange(spectra.num_spectra), colors)[::-1]:
    if ar_extinction[n] < 1.:
        plt.plot(spectra.get_wavelength(n)[:5000], spectra.get_flux(n)[:5000],
                 color=color, marker=',', linestyle='None')
plt.show()

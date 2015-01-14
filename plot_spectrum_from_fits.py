

import read_spectrum_fits
import matplotlib.pyplot as plt

for ogrid,spec in read_spectrum_fits.return_spectra('../../data/PlateList.csv'):
    plt.plot(ogrid,spec)

plt.show()
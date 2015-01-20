import calc_mean_transmittance
import matplotlib.pyplot as plt
import numpy as np


data = np.load('../../data/mean_transmittance.npy')
m = data[1]
m_count = data[2]
ar_z_range = data[0]

plt.plot(ar_z_range, m * m_count[~np.isnan(m_count)].mean())
plt.plot(ar_z_range, m_count)
plt.plot(ar_z_range, m * m_count)
plt.show()
import calc_mean_transmittance
import matplotlib.pyplot as plt
import numpy as np


data = np.load('../../data/mean_transmittance.npy')
m = data[1]
m_count = data[2]
ar_z_range = data[0]
mean_count = m_count[~np.isnan(m_count)].mean()

plt.subplot(2, 1, 1)
plt.plot(ar_z_range, m)
# plt.plot(ar_z_range, np.ones(m.size))
plt.ylabel(r"$\left< f_q(z)/C_q(z) \right> $")

plt.subplot(2, 1, 2)
plt.plot(ar_z_range, m_count)
plt.plot(ar_z_range, m * m_count)
plt.ylabel(r"$\sum_q f_q(z)/C_q(z)$")
plt.xlabel(r"$z$")
plt.show()
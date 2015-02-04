import matplotlib.pyplot as plt
import numpy as np

import common_settings


lya_center = 1215.67

settings = common_settings.Settings()
data = np.load(settings.get_mean_transmittance_npy())
m = data[1]
m_count = data[2]
ar_z_range = data[0]
mean_count = m_count[~np.isnan(m_count)].sum() / m_count.size

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = ax1.twiny()
ax1.plot(ar_z_range, m)
# plt.plot(ar_z_range, np.ones(m.size))
ax1.set_ylabel(r"$\left< f_q(z)/C_q(z) \right> $")
plt.ylim(1, 1.5)
# add wavelength tick marks on top
xlim2 = tuple([lya_center * (1 + z) for z in ax1.get_xlim()])
ax2.set_xlim(xlim2)
plt.axis()

ax3 = fig.add_subplot(2, 1, 2)
ax3.plot(ar_z_range, m_count)
ax3.plot(ar_z_range, m * m_count)
ax3.set_ylabel(r"$\sum_q f_q(z)/C_q(z)$")
ax3.set_xlabel(r"$z$")
plt.show()
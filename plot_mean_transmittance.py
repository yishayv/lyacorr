import calc_mean_transmittance
import matplotlib.pyplot as plt
import numpy as np
import cProfile


def profile_main(d):
    data = np.load('../../data/mean_transmittance.npy')
    m = data[1]
    ar_z_range =  data[0]

    d['m'] = m
    d['ar_z_range'] = ar_z_range


d_ = dict()
cProfile.run('profile_main(d_)', sort=2)
ar_z_range_ = d_['ar_z_range']
m_ = d_['m']

plt.plot(ar_z_range_, m_)
plt.show()
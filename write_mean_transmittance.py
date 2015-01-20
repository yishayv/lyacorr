import calc_mean_transmittance
import cProfile
import numpy as np


def profile_main(d):
    m, ar_z_range = calc_mean_transmittance.mean_transmittance()

    d['m'] = m
    d['ar_z_range'] = ar_z_range


d_ = dict()
cProfile.run('profile_main(d_)', sort=2)
ar_z_range_ = d_['ar_z_range']
m_ = d_['m']

np.save('../../data/mean_transmittance.npy',
        np.vstack((ar_z_range_, m_.get_mean(), m_.ar_count)))

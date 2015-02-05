import cProfile

import numpy as np

import calc_mean_transmittance
import common_settings


settings = common_settings.Settings()


def profile_main():
    m, ar_z_range = calc_mean_transmittance.mean_transmittance(sample_fraction=1)
    calc_mean_transmittance.delta_transmittance(sample_fraction=1)

    d['m'] = m
    d['ar_z_range'] = ar_z_range


d = dict()
if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_mean_transmittance.prof', sort=2)
else:
    profile_main()
ar_z_range_ = d['ar_z_range']
m_ = d['m']

np.save(settings.get_mean_transmittance_npy(),
        np.vstack((ar_z_range_, m_.get_mean(), m_.ar_count)))

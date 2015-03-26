import cProfile

import calc_mean_transmittance
import common_settings
import numpy as np

settings = common_settings.Settings()


def profile_main():
    m = calc_mean_transmittance.mean_transmittance(sample_fraction=1)
    m.save(settings.get_mean_transmittance_npy())
    n, total_weight, total_weighted_flux = calc_mean_transmittance.delta_transmittance(sample_fraction=1)
    np.save(settings.get_total_delta_t(), np.array([total_weight, total_weighted_flux]))
    print 'Total weight:', total_weight, 'Total weighted delta_t:', total_weighted_flux
    print 'Mean delta_t', total_weighted_flux/total_weight

    d['m'] = m


d = dict()
if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_mean_transmittance.prof', sort=2)
else:
    profile_main()
m_ = d['m']


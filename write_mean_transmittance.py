import cProfile

import calc_mean_transmittance
import common_settings


settings = common_settings.Settings()


def profile_main():
    m = calc_mean_transmittance.mean_transmittance(sample_fraction=1)
    calc_mean_transmittance.delta_transmittance(sample_fraction=1)

    d['m'] = m


d = dict()
if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_mean_transmittance.prof', sort=2)
else:
    profile_main()
m_ = d['m']

m_.save(settings.get_mean_transmittance_npy())

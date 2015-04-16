import cProfile

import calc_mean_transmittance
import common_settings

settings = common_settings.Settings()


def profile_main():
    calc_mean_transmittance.mean_transmittance()
    calc_mean_transmittance.delta_transmittance()


if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_mean_transmittance.prof', sort=2)
else:
    profile_main()


"""
    A launcher for calc_ism_transmittance.
"""
import cProfile

import calc_ism_transmittance
import common_settings

settings = common_settings.Settings()


def profile_main():
    calc_ism_transmittance.calc_ism_transmittance()


if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_ism_transmittance.prof', sort=2)
else:
    profile_main()

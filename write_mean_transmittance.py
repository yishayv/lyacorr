"""
    A launcher for calc_mean_transmittance.
    It runs in 2 steps:
    - calculate the mean/median transmittance of all spectra.
    - use the previous result to calculate the delta between each spectra and the mean/median.
"""
import cProfile

import calc_mean_transmittance
import common_settings

settings = common_settings.Settings()  # type: common_settings.Settings


def profile_main():
    calc_mean_transmittance.calc_mean_transmittance()
    calc_mean_transmittance.calc_delta_transmittance()


if settings.get_profile():
    cProfile.runctx('profile_main()', globals(), locals(), filename='write_mean_transmittance.prof', sort=2)
else:
    profile_main()

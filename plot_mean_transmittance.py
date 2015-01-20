import calc_mean_transmittance
import matplotlib.pyplot as plt
import cProfile


def profile_main(d):
    m, ar_z_range = calc_mean_transmittance.mean_transmittance()

    d['m'] = m
    d['ar_z_range'] = ar_z_range


d_ = dict()
cProfile.run('profile_main(d_)', sort=2)
ar_z_range_ = d_['ar_z_range']
m_ = d_['m']

plt.plot(ar_z_range_, m_.get_mean() * 100)
plt.plot(ar_z_range_, m_.ar_count)
plt.plot(ar_z_range_, m_.ar_total_flux)
plt.show()

import matplotlib.pyplot as plt

import common_settings
import mean_flux
import median_flux

lya_center = 1215.67

settings = common_settings.Settings()


def do_plot():
    m = mean_flux.MeanFlux.from_file(settings.get_mean_transmittance_npy())
    med = median_flux.MedianFlux.from_file(settings.get_median_transmittance_npy())

    ar_z, mean = m.get_weighted_mean_with_minimum_count(1)
    ar_z_med, ar_median = med.get_weighted_median_with_minimum_count(1)
    ar_z_med, ar_unweighted_median = med.get_weighted_median_with_minimum_count(1, weighted=False)
    low_pass_mean = m.get_low_pass_mean()[1]

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = ax1.twiny()
    ax1.plot(ar_z, mean)
    # ax1.plot(ar_z, low_pass_mean, color='red')
    ax1.plot(ar_z_med, ar_median, color='orange')
    ax1.plot(ar_z_med, ar_unweighted_median, color='green')

    ax1.set_ylabel(r"$\left< f_q(z)/C_q(z) \right> $")
    plt.ylim(0.0, 1.2)
    # add wavelength tick marks on top
    x_lim2 = tuple([lya_center * (1 + z) for z in ax1.get_xlim()])
    ax2.set_xlim(x_lim2)
    plt.axis()

    ax3 = fig.add_subplot(2, 1, 2)
    ax4 = ax3.twinx()
    ax4.set_ylabel(r"$N_{Spectra}$")
    ax4.plot(m.ar_z, m.ar_count, ':', color='red')
    ax3.plot(m.ar_z, m.ar_weights, ':', color='green')
    ax3.plot(m.ar_z, m.ar_total_flux, color='blue')
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_ylabel(r"$\sum_q f_q(z)/C_q(z)$")
    ax3.set_xlabel(r"$z$")

    plt.show()


if __name__ == '__main__':
    do_plot()


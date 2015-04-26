import matplotlib.pyplot as plt

import common_settings
import mean_flux


lya_center = 1215.67

settings = common_settings.Settings()

def do_plot():
    m = mean_flux.MeanFlux.from_file(settings.get_mean_transmittance_npy())
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = ax1.twiny()
    ax1.plot(m.ar_z, m.get_weighted_mean())
    # plt.plot(ar_z_range, np.ones(m.size))
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


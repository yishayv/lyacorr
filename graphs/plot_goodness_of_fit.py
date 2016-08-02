"""
Plot goodness-of-fit vs. SNR (as a 2D histogram)
note: this should be run from the directory containing the lyacorr.rc file.

"""

import matplotlib.pyplot as plt
import numpy as np
import weighted

import common_settings
import physics_functions.delta_f_snr_bins
from continuum_goodness_of_fit import calc_fit_power_law, fit_function, power_law_to_string

settings = common_settings.Settings()

delta_f_snr_bins_helper = physics_functions.delta_f_snr_bins.DeltaFSNRBins()

snr_stats = np.load(settings.get_fit_snr_stats())
snr_stats_total = snr_stats[0] + snr_stats[1]
# normalize values
# snr_stats_color = snr_stats / snr_stats_total.max(axis=1)[..., np.newaxis]
snr_stats_color = snr_stats / snr_stats_total.max()

fig = plt.figure(figsize=(9, 3.5))
ax1 = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(2,1,1)
gamma_corrected_image = (1 - snr_stats_color.T) ** (1 / 2.2)
gamma_corrected_image[:, :, 2] = gamma_corrected_image[:, :, 0]
log_factor = 1000.
log_color_image = 1 - np.clip(np.log(snr_stats_color.T * log_factor + 1) / np.log(log_factor + 1.), 0, 1)
# tweak the colors in the log plot to show high density areas linearly.
composite_image = log_color_image.copy()
composite_image[:, :, 0] = log_color_image[:, :, 1]
composite_image[:, :, 1] = log_color_image[:, :, 0]
composite_image[:, :, 2] = log_color_image[:, :, 1]
gamma_corrected_composite = composite_image ** (1 / 2.2)
# plt.imshow(gamma_corrected_image, extent=[0,30,0,1.], aspect = 30/1., origin='lower', interpolation='none')
ax1.imshow(gamma_corrected_composite, extent=[-1.6, 3.4, 0, 1.], aspect=3. / 1., origin='lower',
           interpolation='nearest')
# ax2.imshow(log_color_image, extent=[0,30,0,1.], aspect = 30/1., origin='lower', interpolation='nearest')
ax1.set_xlabel(r"${\rm log(SNR (red))}$", fontsize=14)
ax1.set_ylabel(r"${\rm \left|\delta F\right|}$", fontsize=14)

# ax2.set_xlabel(r"${\rm SNR (red)}$", fontsize=14)
# ax2.set_ylabel(r"${\rm \left|\delta F\right|}$", fontsize=14)
# plt.set_cmap('gray')
# plt.plot(x, y)
result, snr_bins, masked_snr_bins, y_quantile = calc_fit_power_law()

print(power_law_to_string(result))

ax1.plot(snr_bins, y_quantile, linewidth=1.)

fit = fit_function(params=result.params, data=0, x=masked_snr_bins)
fit = np.clip(fit, 0, 1)
ax1.plot(masked_snr_bins, fit, color='deepskyblue', linewidth=4.)  # , linestyle=(0., (8., 8.)))
plt.xticks(np.arange(min(snr_bins), max(masked_snr_bins) + 1, 1.0))
fig.tight_layout()

print("test fit", fit_function(params=result.params, data=0, x=[-1.5, -1., 0., 1., 2., 10.]))

plt.figure()
plt.plot(snr_bins, snr_stats_total.sum(axis=1))
total_snr_quantile_1 = weighted.quantile(snr_bins, snr_stats_total.sum(axis=1), 0.10)
total_snr_quantile_9 = weighted.quantile(snr_bins, snr_stats_total.sum(axis=1), 0.90)
print("SNR: 0.1 limit:", total_snr_quantile_1, "0.9 limit:", total_snr_quantile_9)
plt.show()

import lmfit
import numpy as np
import weighted

import common_settings
import physics_functions.delta_f_snr_bins

settings = common_settings.Settings()

delta_f_snr_bins_helper = physics_functions.delta_f_snr_bins.DeltaFSNRBins()


def fit_function(params, data, x):
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    d = params['d'].value
    y = ((x + d) ** a) * b + c
    diff = y - data
    return diff


snr_stats = np.load(settings.get_fit_snr_stats())
snr_stats_total = snr_stats[0] + snr_stats[1]


def calc_fit_powerlaw():
    snr_bins = delta_f_snr_bins_helper.get_log_snr_axis()
    y_quantile = np.zeros_like(snr_bins)
    y1 = delta_f_snr_bins_helper.get_delta_f_axis()
    for i in np.arange(50):
        y_quantile[i] = weighted.quantile(y1, snr_stats_total[i], .9)
    mask = [np.logical_and(-0 < snr_bins, snr_bins < 3)]
    masked_snr_bins = snr_bins[mask]
    print("x2:", masked_snr_bins)
    fit_params = lmfit.Parameters()
    fit_params.add('a', -2., min=-5, max=-1)
    fit_params.add('b', 1., min=0.1, max=20.)
    fit_params.add('c', 0.08, min=0, max=0.2)
    fit_params.add('d', 3, min=-5, max=5)
    fit_result = lmfit.minimize(fit_function, fit_params, kws={'data': y_quantile[mask], 'x': masked_snr_bins})
    return fit_result, snr_bins, masked_snr_bins, y_quantile

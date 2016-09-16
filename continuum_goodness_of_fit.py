import lmfit
import numpy as np
import weighted

import common_settings
import physics_functions.delta_f_snr_bins
from python_compat import range

settings = common_settings.Settings()  # type: common_settings.Settings

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


def calc_fit_power_law(delta_f_snr_bins=snr_stats_total):
    snr_bins = delta_f_snr_bins_helper.get_log_snr_axis()
    y_quantile = np.zeros_like(snr_bins)
    y1 = delta_f_snr_bins_helper.get_delta_f_axis()
    for i in range(50):
        y_quantile[i] = weighted.quantile(y1, delta_f_snr_bins[i], .9)
    mask = [np.logical_and(-0 < snr_bins, snr_bins < 3)]
    masked_snr_bins = snr_bins[mask]
    # print("x2:", masked_snr_bins)
    fit_params = lmfit.Parameters()
    fit_params.add('a', -2., min=-5, max=-1)
    fit_params.add('b', 1., min=0.1, max=20.)
    fit_params.add('c', 0.08, min=0, max=0.2)
    fit_params.add('d', 3, min=-5, max=5)
    fit_result = lmfit.minimize(fit_function, fit_params, kws={'data': y_quantile[mask], 'x': masked_snr_bins})
    return fit_result, snr_bins, masked_snr_bins, y_quantile


def max_delta_f_per_snr(snr, a, b, c, d):
    # approximate a fixed quantile of spectra as a function of SNR.
    mask = np.logical_and(np.exp(-0.5) < snr, snr < np.exp(4))
    x = np.log(snr[mask])
    max_delta_f = ((x + d) ** a) * b + c
    result = np.zeros_like(snr)
    result[mask] = np.minimum(max_delta_f, 1)
    return result


def get_max_delta_f_per_snr_func(fit_result):
    power_law_coefficients = {i.name: i.value for i in fit_result.params.values()}

    a = power_law_coefficients['a']
    b = power_law_coefficients['b']
    c = power_law_coefficients['c']
    d = power_law_coefficients['d']

    # max_delta_f = fit_function(params=fit_result.params(), data=0, x=np.log(snr))
    return lambda snr: max_delta_f_per_snr(snr, a, b, c, d)


def power_law_to_string(fit_result):
    power_law_coefficients = {i.name: i.value for i in fit_result.params.values()}
    return '(((np.log(snr) + ({d})) ** {a}) * {b}) + {c}'.format(**power_law_coefficients)

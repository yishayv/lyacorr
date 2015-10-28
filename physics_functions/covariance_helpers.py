import numpy as np


def jackknife_2d_weighted_test(ar_flux, ar_weights, out=None):
    if not out:
        out_shape = ar_flux.shape[1:] * 2
        out = np.empty(shape=out_shape)
    n = ar_flux.shape[0]
    global_weights = np.sum(ar_weights)
    global_flux = np.sum(ar_flux, axis=0)
    global_weighted_sum = global_flux / global_weights
    print("global weighted sum: shape:", global_weighted_sum.shape, "mean:", global_weighted_sum.mean())

    ar_estimators_except_i = (global_flux - ar_flux) / (global_weights - ar_weights)
    cov_term_mean = np.nansum(ar_estimators_except_i, axis=0) / n
    cov_term = ar_estimators_except_i - cov_term_mean
    cov_sum = np.einsum('ijk,ilm->jklm', cov_term, cov_term)
    print(cov_sum.shape)

    out[:, :, :, :] = float(n - 1) / n * cov_sum
    return out


def jackknife_2d(ar_flux, ar_weights, out=None):
    if not out:
        out_shape = ar_flux.shape[1:] * 2
        out = np.empty(shape=out_shape)
    n = ar_flux.shape[0]
    ar_estimators = np.nan_to_num(ar_flux / ar_weights)
    global_sum = np.nansum(ar_estimators, axis=0)
    print("global sum: shape:", global_sum.shape, "mean:", global_sum.mean())

    ar_estimators_except_i = (global_sum - ar_estimators) / (n - 1)
    cov_term_mean = np.nansum(ar_estimators_except_i, axis=0) / n
    cov_term = ar_estimators_except_i - cov_term_mean
    cov_sum = np.einsum('ijk,ilm->jklm', cov_term, cov_term)
    print(cov_sum.shape)

    out[:, :, :, :] = float(n - 1) / n * cov_sum
    return out


def subsample_2d(ar_flux, ar_weights, out=None):
    if not out:
        out_shape = ar_flux.shape[1:] * 2
        out = np.empty(shape=out_shape)
    n = ar_flux.shape[0]
    ar_estimators = np.nan_to_num(ar_flux / ar_weights)
    global_sum = np.nansum(ar_estimators, axis=0)
    print("global sum: shape:", global_sum.shape, "mean:", global_sum.mean())

    cov_term_mean = np.nansum(ar_estimators, axis=0) / n
    cov_term = ar_estimators - cov_term_mean
    cov_sum = np.einsum('ijk,ilm->jklm', cov_term, cov_term)
    print(cov_sum.shape)

    out[:, :, :, :] = 1 / float(n * (n - 1)) * cov_sum
    return out


def subsample_2d_weighted_test(ar_flux, ar_weights, out=None):
    if not out:
        out_shape = ar_flux.shape[1:] * 2
        out = np.empty(shape=out_shape)
    n = ar_flux.shape[0]
    global_weights = np.sum(ar_weights)
    global_flux = np.sum(ar_flux, axis=0)
    global_weighted_sum = global_flux / global_weights
    print("global weighted sum: shape:", global_weighted_sum.shape, "mean:", global_weighted_sum.mean())

    ar_estimators = np.nan_to_num(ar_flux / ar_weights)
    cov_term_mean = np.nansum(ar_estimators, axis=0) / n
    cov_term = ar_estimators - cov_term_mean
    cov_sum = np.einsum('ijk,ilm->jklm', cov_term, cov_term)
    print(cov_sum.shape)

    out[:, :, :, :] = 1 / float(n * (n - 1)) * cov_sum
    return out

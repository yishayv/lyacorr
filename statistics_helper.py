import numpy as np


def ivar_average(ar_values, ar_sigma):
    ar_weights = np.reciprocal(np.square(ar_sigma))
    return np.average(ar_values, weights=ar_weights)

"""
A simple module to simplify masking out certain observer frame wavelength ranges.
(currently only used for Hg I 4358)
"""

import numpy as np

lya_center = 1215.67


def wavelength_to_redshift(w):
    return w / lya_center - 1


absorption_line_pairs = [(wavelength_to_redshift(4356), wavelength_to_redshift(4364))]


def get_line_masks(ar_redshift):
    """
    return line masks according to predefined list.
    runtime is linear in the number of lines.
    :param ar_redshift:
    :return:
    """
    mask = np.ones(shape=ar_redshift.shape, dtype=bool)
    for min_val, max_val in absorption_line_pairs:
        mask &= ~ ((min_val < ar_redshift) & (ar_redshift < max_val))
    return mask

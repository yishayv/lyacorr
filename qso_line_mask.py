import collections

import numpy as np


# TODO: replace with a more accurate number
lya_center = 1215.67


def rydberg_ratio(m, n):
    return abs(1. / (m * m) - 1. / (n * n))


def ly_series_ratio(n):
    return rydberg_ratio(1, n)


def ly_series_wavelength(n):
    return lya_center * ly_series_ratio(2) / ly_series_ratio(n)


SpectralRange = collections.namedtuple(
    'SpectralRange', ['name', 'low', 'high'])

SpectralLine = collections.namedtuple(
    'SpectralLine', ['name', 'wavelength', 'width_factor'])

SpecRanges = [
    SpectralRange('Ly-alpha-absorption', 0, ly_series_wavelength(2))]

SpecLines = [
    # TODO: replace with a more accurate number
    SpectralLine('Ly-beta', ly_series_wavelength(3), 1.03),
    SpectralLine('Ly-alpha', ly_series_wavelength(2), 1.05),
    # CIV emission is actually a doublet:
    # CIV_line_1 = 1548.202 #f_lu=0.190
    # with a weaker line at:
    # CIV_line_2 = 1550.772 #f_lu=0.0962
    # TODO: figure out if we should do some kind of weighted average
    SpectralLine('CIV', 1548.202, 1.03),
    # the rest are from: http://astro.uchicago.edu/~subbarao/newWeb/line.html
    # note that their civ line is offset somewhat
    SpectralLine('SiIV_OIV', 1399.8, 1.03),
    SpectralLine('CIII', 1908.27, 1.03),
    SpectralLine('CII', 2326.0, 1.01),  # weak
    SpectralLine('MgII', 2800.32, 1.03)]


def is_masked_by_range(wavelength, range_low, range_high, z):
    return (1 + z) * range_high > wavelength > (1 + z) * range_low

# vectorize the previous function
vec_is_masked_by_range = np.vectorize(
    is_masked_by_range, excluded=['range_low', 'range_high', 'z'])


def is_masked_by_line(wavelength, line_wavelength, line_width_factor, z):
    return is_masked_by_range(wavelength, line_wavelength / line_width_factor,
                              line_wavelength * line_width_factor, z)

# vectorize the previous function
vec_is_masked_by_line = np.vectorize(
    is_masked_by_line, excluded=['line_wavelength', 'line_width_factor', 'z'])


def get_line_mask(ar_wavelength, spec_line):
    """

    :type ar_wavelength: np.multiarray.ndarray
    :type spec_line: SpectralLine
    :rtype: np.multiarray.ndarray
    """
    line_start = np.searchsorted(ar_wavelength, spec_line.wavelength / spec_line.width_factor)
    line_end = np.searchsorted(ar_wavelength, spec_line.wavelength * spec_line.width_factor)
    return np.concatenate((np.zeros(line_start),
                           np.ones(line_end - line_start),
                           np.zeros(ar_wavelength.size - line_end)))


def add_line_mask(ar_wavelength):
    m = ar_wavelength.mask
    for spec_line in SpecLines:
        current_mask = get_line_mask(ar_wavelength, spec_line)
        m = np.logical_or(m, current_mask)
    return m


def add_range_mask(ar_wavelength, z):
    m = ar_wavelength.mask
    for spec_range in SpecRanges:
        current_mask = vec_is_masked_by_range(
            ar_wavelength, spec_range.low, spec_range.high, z)
        m = np.logical_or(m, current_mask)
    return m


def mask_qso_lines(spec):
    m = add_line_mask(spec.ma_wavelength)
    spec.ma_wavelength.mask = m
    spec.ma_flux.mask = m
    spec.ma_flux_err.mask = m


def mask_ly_absorption(spec, z):
    m = add_range_mask(spec.ma_wavelength, z)
    spec.ma_wavelength.mask = m
    spec.ma_flux.mask = m
    spec.ma_flux_err.mask = m

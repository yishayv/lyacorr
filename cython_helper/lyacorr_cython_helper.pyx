# cython: profile=True
__version__ = '0.0.1'

import cython
import numpy as np
cimport numpy as np # for the special numpy stuff
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI


np.import_array()

@cython.profile(False)
@cython.boundscheck(False)
cdef inline double get_bin_x(double dist1, double dist2, double x_scale):
    #/* keep the result as double so that a boundary check can be done */
    #/* r|| = abs(r1 - r2) */
    return fabs(dist1 - dist2) * x_scale

@cython.profile(False)
@cython.boundscheck(False)
cdef inline double get_bin_y(double dist1, double dist2, double y_scale):
    #/* keep the result as double so that a boundary check can be done */
    #/* r_ = (r1 + r2)/2 * qso_angle */
    return (dist1 + dist2) * y_scale

@cython.profile(True)
@cython.boundscheck(False)
def find_largest_index(double max_dist_for_qso_angle, np.ndarray[double] in_array_dist, long dist_size):
    #/*
    # * find the largest index of dist2 for which a transverse distance to the other
    # * QSO is within range.
    # */
    cdef int j
    cdef double dist
    for j in range(dist_size):
        dist = in_array_dist[j]
        if (dist > max_dist_for_qso_angle):
            return j + 1
    #/* got to the end of the array. simply return the size of the array. */
    return dist_size


@cython.profile(True)
@cython.boundscheck(False)
def bin_pixel_pairs(np.ndarray[double] in_array_dist1, np.ndarray[double] in_array_dist2, 
                    np.ndarray[double] in_array_flux1, np.ndarray[double] in_array_flux2,
                    np.ndarray[double] in_array_weights1, np.ndarray[double] in_array_weights2, 
                    double qso_angle,
                    double x_bin_size, double y_bin_size, int x_bin_count, int y_bin_count, 
                    np.ndarray[double, ndim=3] out_array=None):

    cdef long i, j
    cdef long dist1_size, dist2_size
    cdef int bin_x, bin_y
    cdef long last_dist2_start, first_pair_dist2
    cdef long max_dist1_index, max_dist2_index
    cdef double f_bin_x, f_bin_y
    cdef double dist1, dist2, flux1, flux2, weight1, weight2
    cdef double weighted_flux1, weighted_flux2
    cdef double x_scale, y_scale
    cdef double max_dist_for_qso_angle

    if out_array is None:
        out_array = np.zeros(shape=(x_bin_count, y_bin_count, 3))

    # iterate over the arrays
    dist1_size = in_array_dist1.shape[0]
    dist2_size = in_array_dist2.shape[0]
    if  not dist1_size or not dist2_size:
        return

    # /* if dist1 is nearer, it is more efficient to run the function with 1 and 2 reversed. */
    dist1 = in_array_dist1[0]
    dist2 = in_array_dist2[0]
    if (dist1 < dist2):
        # MY_DEBUG_PRINT("SWAPPING 1,2\n");
        dist1, dist2 = dist2, dist1
        dist1_size, dist2_size = dist2_size, dist1_size
        in_array_dist1, in_array_dist2 = in_array_dist2, in_array_dist1
        in_array_flux1, in_array_flux2 = in_array_flux2, in_array_flux1
        in_array_weights1, in_array_weights2 = in_array_weights2, in_array_weights1

    #/* r|| = abs(r1 - r2) */
    x_scale = 1. / x_bin_size
    #/* r_ = (r1 + r2)/2 * qso_angle */
    y_scale = qso_angle / (2. * y_bin_size)

    #/*
    # * find the largest index of dist2 for which a transverse distance to the other
    # * QSO is within range.
    # */
    #/* set initial index to the end of the array. */
    max_dist_for_qso_angle = (y_bin_count + 1) * y_bin_size / sin(qso_angle)
    max_dist1_index = find_largest_index(max_dist_for_qso_angle, in_array_dist1, dist1_size)
    max_dist2_index = find_largest_index(max_dist_for_qso_angle, in_array_dist2, dist2_size)

    # MY_DEBUG_PRINT("max_dist1_index: %ld, dist1_size: %ld\n", max_dist1_index, dist1_size);
    # MY_DEBUG_PRINT("max_dist2_index: %ld, dist2_size: %ld\n", max_dist2_index, dist2_size);

    # MY_DEBUG_PRINT(":::::Before loop\n");

    last_dist2_start = 0
    for i in range(max_dist1_index):
        #/* MY_DEBUG_PRINT(":::::Outside iter, i=%d\n", i); */
        dist1 = in_array_dist1[i]
        flux1 = in_array_flux1[i]
        weight1 = in_array_weights1[i]

        weighted_flux1 = flux1 * weight1

        #/*
        # * distance values are ordered, so if any dist2 was too low to be close enough to the previous dist1,
        # * the same should hold for the current dist1.
        # */
        first_pair_dist2 = 0
        for j in range(last_dist2_start, max_dist2_index):
            dist2 = in_array_dist2[j]
            flux2 = in_array_flux2[j]
            weight2 = in_array_weights2[j]

            f_bin_x = get_bin_x(dist1, dist2, x_scale)
            f_bin_y = get_bin_y(dist1, dist2, y_scale)

            if (f_bin_x >= 0 and f_bin_x < x_bin_count):
                #/* pixel is in range of parallel separation */
                if (not first_pair_dist2):
                    first_pair_dist2 = j

                if (f_bin_y >= 0 and f_bin_y < y_bin_count):
                    #/* pixel is in range */

                    weighted_flux2 = flux2 * weight2

                    bin_x = int(f_bin_x)
                    bin_y = int(f_bin_y)

                    out_array[bin_x, bin_y, 0] += weighted_flux1 * weighted_flux2
                    out_array[bin_x, bin_y, 1] += 1
                    out_array[bin_x, bin_y, 2] += weight1 * weight2
            else:
                #/*
                # * in flat geometry we cannot move in and out of range more than once.
                # */
                if (first_pair_dist2):
                    break
        if (first_pair_dist2 > last_dist2_start):
            last_dist2_start = first_pair_dist2
    return out_array

@cython.profile(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def bin_pixel_pairs_histogram(np.ndarray[double] in_array_dist1, np.ndarray[double] in_array_dist2, 
                    np.ndarray[double] in_array_flux1, np.ndarray[double] in_array_flux2,
                    np.ndarray[double] in_array_weights1, np.ndarray[double] in_array_weights2, 
                    double qso_angle,
                    double x_bin_size, double y_bin_size, int x_bin_count, int y_bin_count, 
                    double f_min, double f_max, int f_bin_count,
                    np.ndarray[double, ndim=3] out_array=None):

    cdef long i, j
    cdef long dist1_size, dist2_size
    cdef int bin_x, bin_y, bin_f
    cdef long last_dist2_start, first_pair_dist2
    cdef long max_dist1_index, max_dist2_index
    cdef double f_bin_x, f_bin_y
    cdef double dist1, dist2, flux1, flux2, weight1, weight2
    cdef double weighted_flux1, weighted_flux2
    cdef double x_scale, y_scale
    cdef double max_dist_for_qso_angle

    if out_array is None:
        out_array = np.zeros(shape=(x_bin_count, y_bin_count, 3))

    # iterate over the arrays
    dist1_size = in_array_dist1.shape[0]
    dist2_size = in_array_dist2.shape[0]
    if  not dist1_size or not dist2_size:
        return

    # /* if dist1 is nearer, it is more efficient to run the function with 1 and 2 reversed. */
    dist1 = in_array_dist1[0]
    dist2 = in_array_dist2[0]
    if (dist1 < dist2):
        # MY_DEBUG_PRINT("SWAPPING 1,2\n");
        dist1, dist2 = dist2, dist1
        dist1_size, dist2_size = dist2_size, dist1_size
        in_array_dist1, in_array_dist2 = in_array_dist2, in_array_dist1
        in_array_flux1, in_array_flux2 = in_array_flux2, in_array_flux1
        in_array_weights1, in_array_weights2 = in_array_weights2, in_array_weights1

    #/* r|| = abs(r1 - r2) */
    x_scale = 1. / x_bin_size
    #/* r_ = (r1 + r2)/2 * qso_angle */
    y_scale = qso_angle / (2. * y_bin_size)

    #/*
    # * find the largest index of dist2 for which a transverse distance to the other
    # * QSO is within range.
    # */
    #/* set initial index to the end of the array. */
    max_dist_for_qso_angle = (y_bin_count + 1) * y_bin_size / sin(qso_angle)
    max_dist1_index = find_largest_index(max_dist_for_qso_angle, in_array_dist1, dist1_size)
    max_dist2_index = find_largest_index(max_dist_for_qso_angle, in_array_dist2, dist2_size)

    # MY_DEBUG_PRINT("max_dist1_index: %ld, dist1_size: %ld\n", max_dist1_index, dist1_size);
    # MY_DEBUG_PRINT("max_dist2_index: %ld, dist2_size: %ld\n", max_dist2_index, dist2_size);

    # MY_DEBUG_PRINT(":::::Before loop\n");

    last_dist2_start = 0
    for i in range(max_dist1_index):
        #/* MY_DEBUG_PRINT(":::::Outside iter, i=%d\n", i); */
        dist1 = in_array_dist1[i]
        flux1 = in_array_flux1[i]
        weight1 = in_array_weights1[i]

        weighted_flux1 = flux1 * weight1

        #/*
        # * distance values are ordered, so if any dist2 was too low to be close enough to the previous dist1,
        # * the same should hold for the current dist1.
        # */
        first_pair_dist2 = 0
        for j in range(last_dist2_start, max_dist2_index):
            dist2 = in_array_dist2[j]
            flux2 = in_array_flux2[j]
            weight2 = in_array_weights2[j]

            f_bin_x = get_bin_x(dist1, dist2, x_scale)
            f_bin_y = get_bin_y(dist1, dist2, y_scale)

            if (f_bin_x >= 0 and f_bin_x < x_bin_count):
                #/* pixel is in range of parallel separation */
                if (not first_pair_dist2):
                    first_pair_dist2 = j

                if (f_bin_y >= 0 and f_bin_y < y_bin_count):
                    #/* pixel is in range */

                    flux_product = flux1 * flux2

                    if flux_product > f_max:
                        bin_f = f_bin_count - 1
                    elif flux_product < f_min:
                        bin_f = 0
                    else:
                        bin_f = (int)(f_bin_count * (flux_product - f_min) / (f_max - f_min))

                    bin_x = int(f_bin_x)
                    bin_y = int(f_bin_y)

                    out_array[bin_x, bin_y, bin_f] +=  weight1 * weight2
            else:
                #/*
                # * in flat geometry we cannot move in and out of range more than once.
                # */
                if (first_pair_dist2):
                    break
        if (first_pair_dist2 > last_dist2_start):
            last_dist2_start = first_pair_dist2
    return out_array
def initlyacorr_cython_helper():
    pass

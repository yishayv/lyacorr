from collections import namedtuple

import numpy as np
import healpy as hp
from numpy import deg2rad, rad2deg

const_deg2rad = np.pi / 180
const_rad2deg = 180 / np.pi


def deg360(ar_deg):
    """
    Normalize a numpy vector of  degrees to the [0,360) range
    :type ar_deg: np.multiarray.ndarray
    :rtype: np.multiarray.ndarray
    """
    return ar_deg - np.floor(ar_deg / 360) * 360


def find_spherical_mean_rad(ar_ra, ar_dec, axis=0):
    assert ar_ra.shape == ar_dec.shape
    ar_mean_x = np.mean(np.cos(ar_dec) * np.cos(ar_ra), axis)
    ar_mean_y = np.mean(np.cos(ar_dec) * np.sin(ar_ra), axis)
    ar_mean_z = np.mean(np.sin(ar_dec), axis)

    ar_mean_radius = np.sqrt(np.square(ar_mean_x) + np.square(ar_mean_y) + np.square(ar_mean_z))
    ar_mean_ra = np.arctan2(ar_mean_y, ar_mean_x)
    ar_mean_dec = np.arcsin(ar_mean_z / ar_mean_radius)

    return ar_mean_ra, ar_mean_dec, ar_mean_radius


def find_spherical_mean_deg(ar_ra, ar_dec, axis=0):
    ar_mean_ra_rad, ar_mean_dec_rad, ar_mean_radius = find_spherical_mean_rad(deg2rad(ar_ra), deg2rad(ar_dec), axis)
    return deg360(rad2deg(ar_mean_ra_rad)), rad2deg(ar_mean_dec_rad), ar_mean_radius


group_types = namedtuple('group_types', ('plate', 'healpix'))


class SkyGroups:
    def __init__(self, group_type=group_types.healpix):
        self.group_type = group_type

    def get_group_ids(self, ar_ra, ar_dec):
        assert self.group_type == group_types.healpix
        ar_theta = deg2rad(-ar_dec + 90)
        ar_phi = deg2rad(ar_ra % 360)
        return hp.ang2pix(32, ar_theta, ar_phi)

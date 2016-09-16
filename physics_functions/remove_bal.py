from collections import namedtuple

import numpy as np
import pyfits

import common_settings
from python_compat import range

settings = common_settings.Settings()  # type: common_settings.Settings

# CIV_LINE = 1545.86
LIGHT_SPEED_KM_S = 299792.458

MaskElement = namedtuple('MaskElement', ('start', 'end'))


def civ_velocity_to_wavelength(line_center, z, velocity):
    return line_center * (1. - velocity / LIGHT_SPEED_KM_S) * (1 + z)


def civ_rel_velocity_to_wavelength(line_center, z, velocity):
    beta = velocity / LIGHT_SPEED_KM_S
    return line_center * np.sqrt((1. - beta) / (1. + beta)) * (1 + z)


class RemoveBALSimple(object):
    def __init__(self):
        self.bal_fits = pyfits.open(settings.get_qso_bal_fits())
        self.data = self.bal_fits[1].data
        self.bal_dict = {}
        self.create_dict()
        self.line_centers = {'CIV': 1550.77, 'OVI': 1033.30, 'Lya': 1215.67, 'NV': 1239.42, 'SiIV+OIV': 1397.61}

    def create_dict(self):
        d = self.data
        for i, bal_entry in enumerate(self.data):
            self.bal_dict[(d.PLATE[i], d.MJD[i], d.FIBERID[i])] = i

    def get_mask_list(self, plate, mjd, fiber_id):
        qso_tuple = (plate, mjd, fiber_id)
        mask_list = []
        z_vi = None
        # if QSO is not in BAL list, return an empty list
        if qso_tuple in self.bal_dict:
            i = self.bal_dict[qso_tuple]
            d = self.data
            z_vi = d.Z_VI[i]
            for j in range(d.NCIV_450[i]):
                for line_center in self.line_centers.values():
                    # note that start<==>max
                    # add a safety margin
                    margin = 0.002
                    end = civ_rel_velocity_to_wavelength(line_center, z_vi, d.VMIN_CIV_450[i][j]) * (1 + margin)
                    start = civ_rel_velocity_to_wavelength(line_center, z_vi, d.VMAX_CIV_450[i][j]) * (1 - margin)
                    mask_list += [MaskElement(start, end)]

        return mask_list, z_vi

    def get_mask(self, plate, mjd, fiber_id, ar_wavelength):
        mask = np.zeros_like(ar_wavelength, dtype=bool)
        # if QSO is not in BAL list, return an empty mask
        mask_list, z_vi = self.get_mask_list(plate, mjd, fiber_id)
        for mask_element in mask_list:
            mask[np.logical_and(mask_element.start < ar_wavelength, ar_wavelength < mask_element.end)] = 1

        return mask, z_vi

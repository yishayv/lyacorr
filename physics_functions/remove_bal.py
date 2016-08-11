import numpy as np
import pyfits
import common_settings

settings = common_settings.Settings()

# CIV_LINE = 1545.86
LIGHT_SPEED_KM_S = 299792.458


def civ_velocity_to_wavelength(line_center, z, velocity):
    return line_center * (1. + velocity / LIGHT_SPEED_KM_S) * (1 + z)


class RemoveBALSimple(object):
    def __init__(self):
        self.bal_fits = pyfits.open(settings.get_qso_bal_fits())
        self.data = self.bal_fits[1].data
        self.bal_dict = {}
        self.create_dict()
        self.line_centers = {'CIV': 1550.77, 'Lya': 1215.67, 'NV': 1239.42}

    def create_dict(self):
        d = self.data
        for i, bal_entry in enumerate(self.data):
            self.bal_dict[(d.PLATE[i], d.MJD[i], d.FIBERID[i])] = i

    def get_mask(self, plate, mjd, fiber_id, ar_wavelength):
        qso_tuple = (plate, mjd, fiber_id)
        mask = np.zeros_like(ar_wavelength, dtype=bool)
        # if QSO is not in BAL list, return an empty map
        if qso_tuple in self.bal_dict:
            i = self.bal_dict[qso_tuple]
            d = self.data
            z = d.Z_PIPE[i]
            for j in np.arange(d.NCIV_450[i]):
                for line_center in self.line_centers.values():
                    start = civ_velocity_to_wavelength(line_center, z, d.VMIN_CIV_450[i][j])
                    end = civ_velocity_to_wavelength(line_center, z, d.VMAX_CIV_450[i][j])
                    mask[np.logical_and(start < ar_wavelength, ar_wavelength < end)] = 1

        return mask

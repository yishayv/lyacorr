import numpy as np

import lookup_table
import common_settings


settings = common_settings.Settings()


class SigmaSquaredLSS:
    def __init__(self, z_start, z_end, z_step):
        self.lookup_table = lookup_table.LinearInterpTable(self._sigma_function, z_start, z_end, z_step)

    @staticmethod
    def _sigma_function(ar_z):
        ar_sigma = np.loadtxt(settings.get_sigma_squared_lss(), skiprows=1)
        # interpolate results to a fixed grid for fast lookup
        return np.interp(ar_z, ar_sigma[:, 0], ar_sigma[:, 1])

    def evaluate(self, ar_z):
        return self.lookup_table.evaluate(ar_z)


class WeightsEta:
    def __init__(self, z_start, z_end, z_step):
        self.lookup_table = lookup_table.LinearInterpTable(self._sigma_function, z_start, z_end, z_step)

    @staticmethod
    def _sigma_function(ar_z):
        ar_sigma = np.loadtxt(settings.get_weights_eta(), skiprows=1)
        # interpolate results to a fixed grid for fast lookup
        return np.interp(ar_z, ar_sigma[:, 0], ar_sigma[:, 1])

    def evaluate(self, ar_z):
        return self.lookup_table.evaluate(ar_z)

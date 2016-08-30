import numpy as np

import common_settings
import lookup_table

settings = common_settings.Settings()

DEFAULT_WEIGHT_Z_RANGE = (1.9, 3.6, 0.01)


class PixelWeight:
    def __init__(self, z_range=DEFAULT_WEIGHT_Z_RANGE):
        z_start, z_end, z_step = z_range
        self.sigma_squared_lss = SigmaSquaredLSS(z_start, z_end, z_step)
        self.weight_eta = WeightEta(z_start, z_end, z_step)

    def eval(self, ar_pipeline_ivar, ar_mean_flux, ar_z):
        assert ar_pipeline_ivar.shape == ar_mean_flux.shape == ar_z.shape
        # equations (15) and (16) in Busca et al. 2013
        gamma_half = 3.8 / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            xi_squared = 1 / (ar_pipeline_ivar * np.square(ar_mean_flux) * self.weight_eta.eval(ar_z)) + \
                         self.sigma_squared_lss.eval(ar_z)
            weight = (1 + ar_z) ** gamma_half / xi_squared
        return weight


class SigmaSquaredLSS:
    def __init__(self, z_start, z_end, z_step):
        self.lookup_table = lookup_table.LinearInterpTable(self._sigma_function, z_start, z_end, z_step)

    @staticmethod
    def _sigma_function(ar_z):
        ar_sigma = np.loadtxt(settings.get_sigma_squared_lss(), skiprows=1)
        # interpolate results to a fixed grid for fast lookup
        return np.interp(ar_z, ar_sigma[:, 0], ar_sigma[:, 1])

    def eval(self, ar_z):
        return self.lookup_table.eval(ar_z)


class WeightEta:
    def __init__(self, z_start, z_end, z_step):
        self.lookup_table = lookup_table.LinearInterpTable(self._sigma_function, z_start, z_end, z_step)

    @staticmethod
    def _sigma_function(ar_z):
        ar_sigma = np.loadtxt(settings.get_weight_eta(), skiprows=1)
        # interpolate results to a fixed grid for fast lookup
        return np.interp(ar_z, ar_sigma[:, 0], ar_sigma[:, 1])

    def eval(self, ar_z):
        return self.lookup_table.eval(ar_z)

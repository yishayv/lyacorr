import numpy as np


class MeanFlux:
    def __init__(self, ar_z):
        self.ar_z = np.copy(ar_z)
        self.ar_total_flux = np.zeros_like(self.ar_z)
        self.ar_count = np.zeros_like(self.ar_z)
        self.ar_weights = np.zeros_like(self.ar_z)

    def add_flux_pre_binned(self, ar_flux, ar_mask, ar_weights):
        self.ar_total_flux[ar_mask] += ar_flux[ar_mask] * ar_weights[ar_mask]
        self.ar_count[ar_mask] += 1
        self.ar_weights[ar_mask] += ar_weights[ar_mask]

    def merge(self, mean_flux2):
        """

        :type mean_flux2: MeanFlux
        """
        self.ar_total_flux += mean_flux2.ar_total_flux
        self.ar_count += mean_flux2.ar_count
        self.ar_weights += mean_flux2.ar_weights

    def get_weighted_mean(self):
        ar_weights_no_zero = self.ar_weights
        ar_weights_no_zero[self.ar_weights == 0] = np.nan
        return self.ar_total_flux / ar_weights_no_zero

    def save(self, filename):
        np.save(filename, np.vstack((self.ar_z,
                                     self.ar_total_flux,
                                     self.ar_count,
                                     self.ar_weights)))

    def load(self, filename):
        stacked_array = np.load(filename)
        self.ar_z = stacked_array[0]
        self.ar_total_flux = stacked_array[1]
        self.ar_count = stacked_array[2]
        self.ar_weights = stacked_array[3]

    @classmethod
    def from_file(cls, filename):
        """

        :rtype : MeanFlux
        """
        m = cls(np.empty(1))
        m.load(filename)
        return m
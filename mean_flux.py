import numpy as np


class MeanFlux:
    def __init__(self, ar_z):
        self.ar_z = np.copy(ar_z)
        self.ar_total_flux = np.zeros_like(self.ar_z)
        self.ar_count = np.zeros(self.ar_total_flux.size)

    def add_flux_pre_binned(self, ar_flux, ar_mask):
        self.ar_total_flux[ar_mask] += ar_flux[ar_mask]
        self.ar_count[ar_mask] += 1

    def merge(self, mean_flux2):
        self.ar_total_flux += mean_flux2.ar_total_flux
        self.ar_count += mean_flux2.ar_count

    def get_mean(self):
        ar_count_no_zero = self.ar_count
        ar_count_no_zero[self.ar_count == 0] = np.nan
        return self.ar_total_flux / ar_count_no_zero

    def save(self, filename):
        np.save(filename, np.vstack((self.ar_z,
                                     self.ar_total_flux,
                                     self.ar_count)))

    def load(self, filename):
        stacked_array = np.load(filename)
        self.ar_z = stacked_array[0]
        self.ar_total_flux = stacked_array[1]
        self.ar_count = stacked_array[2]

    @classmethod
    def fromfile(cls, filename):
        """

        :rtype : MeanFlux
        """
        m = cls(np.empty(1))
        m.load(filename)
        return m
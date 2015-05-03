import numpy as np
from scipy import signal


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
        ar_weights_no_zero = np.copy(self.ar_weights)
        ar_weights_no_zero[self.ar_weights == 0] = np.nan
        return self.ar_total_flux / ar_weights_no_zero

    def get_weighted_mean_with_minimum_count(self, minimum_count):
        return self.get_z_with_minimum_count(minimum_count), self.get_weighted_mean()[self.ar_count >= minimum_count]

    def get_z_with_minimum_count(self, n):
        return self.ar_z[self.ar_count >= n]

    def get_low_pass_mean(self, minimum_count=1):
        assert minimum_count > 0
        ar_z, mean = self.get_weighted_mean_with_minimum_count(minimum_count)
        b, a = signal.butter(N=3, Wn=0.05, analog=False)
        low_pass_mean = signal.filtfilt(b=b, a=a, x=mean)
        return ar_z, low_pass_mean

    def as_np_array(self):
        return np.vstack((self.ar_z,
                          self.ar_total_flux,
                          self.ar_count,
                          self.ar_weights))

    @classmethod
    def from_np_array(cls, np_array):
        new_obj = cls(np.empty(1))
        new_obj.ar_z = np_array[0]
        new_obj.ar_total_flux = np_array[1]
        new_obj.ar_count = np_array[2]
        new_obj.ar_weights = np_array[3]
        return new_obj

    def save(self, filename):
        np.save(filename, self.as_np_array())

    @classmethod
    def load(cls, filename):
        stacked_array = np.load(filename)
        return cls.from_np_array(stacked_array)

    @classmethod
    def from_file(cls, filename):
        """

        :rtype : MeanFlux
        """
        return cls.load(filename)


import numpy as np
from scipy import signal
import weighted as weighted_module

class MedianTransmittance:
    def __init__(self, ar_z, flux_res=1000):
        self.ar_z = np.copy(ar_z)
        self.ar_flux_bins = np.zeros(shape=(ar_z.size, flux_res))
        self.ar_unweighted_flux_bins = np.zeros(shape=(ar_z.size, flux_res))
        self.flux_min = 0.
        self.flux_max = 1.
        assert self.flux_max > self.flux_min
        self.flux_range = self.flux_max - self.flux_min
        self.flux_offset = self.flux_min
        self.flux_res = flux_res

    def add_flux_pre_binned(self, ar_flux, ar_mask, ar_weights):
        for n in xrange(self.ar_z.size):
            if ar_mask[n]:
                ar_effective_weight = ar_weights[n]
                ar_effective_flux = ar_flux[n]
                ar_normalized_flux = ar_effective_flux / self.flux_range - self.flux_offset
                ar_flux_index_float = np.clip(ar_normalized_flux, self.flux_min, self.flux_max) * self.flux_res
                ar_flux_index = np.clip((ar_flux_index_float).astype(int), 0, self.flux_res - 1)
                self.ar_flux_bins[n, ar_flux_index] += ar_effective_weight
                self.ar_unweighted_flux_bins[n, ar_flux_index] += 1

    def merge(self, median_flux2):
        """

        :type median_flux2: MedianTransmittance
        """
        self.ar_flux_bins += median_flux2.ar_flux_bins
        self.ar_unweighted_flux_bins += median_flux2.ar_unweighted_flux_bins

    def get_weighted_median(self, weighted=True):
        ar_median_weights = self.ar_flux_bins if weighted else self.ar_unweighted_flux_bins
        res = np.zeros(self.ar_z.size)
        for n in xrange(self.ar_z.size):
            res[n] = weighted_module.median(np.arange(self.flux_res), ar_median_weights[n])

        return res / self.flux_res * self.flux_range + self.flux_offset

    def get_weighted_median_with_minimum_count(self, minimum_count, weighted=True):
        mask_minimum_count = self.get_minimum_count_mask(minimum_count)
        return self.ar_z[mask_minimum_count], self.get_weighted_median(weighted)[mask_minimum_count]

    def get_minimum_count_mask(self, n):
        return self.ar_unweighted_flux_bins.sum(axis=1) >= n

    def get_low_pass_median(self, minimum_count=1):
        assert minimum_count > 0
        ar_z, mean = self.get_weighted_median_with_minimum_count(minimum_count)
        b, a = signal.butter(N=3, Wn=0.05, analog=False)
        low_pass_mean = signal.filtfilt(b=b, a=a, x=mean)
        return ar_z, low_pass_mean

    def as_np_array(self):
        return np.vstack((self.ar_z,
                          self.ar_flux_bins.T,
                          self.ar_unweighted_flux_bins.T))

    # noinspection PyMethodMayBeStatic
    def as_object(self):
        """
        Return data that cannot be easily represented in an array.
        """
        pass

    @classmethod
    def from_np_array(cls, np_array):
        new_obj = cls(np.empty(1))
        new_obj.ar_z = np_array[0]
        # calculate the flux resolution based on array shape
        new_obj.flux_res = (np_array.shape[0] - 1) // 2
        new_obj.ar_flux_bins = np_array[np.arange(new_obj.flux_res) + 1].T
        new_obj.ar_unweighted_flux_bins = np_array[np.arange(new_obj.flux_res) + new_obj.flux_res + 1].T
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

        :rtype : MeanTransmittance
        """
        return cls.load(filename)


import numpy as np


class MeanFlux:
    def __init__(self, z_range):
        self.ar_total_flux = np.arange(*z_range)
        self.ar_count = np.zeros(self.ar_total_flux.size)

    def add_flux_prebinned(self, ar_flux, ar_mask):
        self.ar_total_flux[ar_mask] += ar_flux[ar_mask]
        self.ar_count[ar_mask] += 1

    def merge(self, mean_flux2):
        self.ar_total_flux += mean_flux2.ar_total_flux
        self.ar_count += mean_flux2.ar_count

    def get_mean(self):
        ar_count_no_zero = self.ar_count
        ar_count_no_zero[self.ar_count == 0] = np.nan
        return self.ar_total_flux / ar_count_no_zero
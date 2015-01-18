import numpy as np


class MeanFlux:
    def __init__(self, x_start, x_end, step):
        self.x_start = x_start
        self.x_end = x_end
        self.step = step
        self.ar_total_flux = np.zeros((x_end - x_start) / step)
        self.ar_count = np.zeros((x_end - x_start) / step)

    def add_flux_prebinned(self, ar_flux, ar_mask):
        self.ar_total_flux[ar_mask] += ar_flux[ar_mask]
        self.ar_count[ar_mask] += 1

    def get_mean(self):
        return self.ar_total_flux / self.ar_count
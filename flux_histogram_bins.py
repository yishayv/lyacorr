import numpy as np

from flux_accumulator import AccumulatorBase


class FluxHistogramBins(AccumulatorBase):
    def __init__(self, x_count, y_count, f_count, x_range, y_range, f_min, f_max, filename=''):
        self.ar_flux = np.zeros((x_count, y_count, f_count))
        self.x_count = x_count
        self.y_count = y_count
        self.f_count = f_count
        self.filename = filename
        self.max_range = np.sqrt(np.square(x_range) + np.square(y_range))
        self.x_range = x_range
        self.y_range = y_range
        self.f_min = f_min
        self.f_max = f_max
        self.x_bin_size = float(x_range) / x_count
        self.y_bin_size = float(y_range) / y_count

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask, ar_weights):
        assert False, "Not implemented"

    def merge(self, bins2):
        assert self.ar_flux.shape == bins2.ar_flux.shape
        assert self.x_range == bins2.x_range
        assert self.y_range == bins2.y_range
        assert self.x_count == bins2.x_count
        assert self.y_count == bins2.y_count
        assert self.f_count == bins2.f_count
        assert self.f_min == bins2.f_min
        assert self.f_max == bins2.f_max
        self.ar_flux += bins2.ar_flux
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_3d_array(self, stacked_array):
        self.ar_flux = stacked_array[:, :, :, 0]
        self.x_count = self.ar_flux.shape[0]
        self.y_count = self.ar_flux.shape[1]
        self.f_count = self.ar_flux.shape[2]

    def load(self, filename):
        # TODO: to static
        stacked_array = np.load(filename)
        self.from_3d_array(stacked_array)

    def __radd__(self, other):
        return self.merge(other)

    def __add__(self, other):
        return self.merge(other)

    @classmethod
    def init_as(cls, other):
        """

        :type other: FluxHistogramBins
        """
        return cls(other.x_count, other.y_count, other.x_range, other.y_range)

    @classmethod
    def from_other(cls, other):
        new_obj = cls.init_as(other)
        new_obj.merge(other)

    @classmethod
    def from_np_array(cls, ar_flux, x_range, y_range, f_min, f_max):
        """

        :type ar_flux: np.array
        """
        assert ar_flux.ndim == 3
        new_bins = cls(ar_flux.shape[0], ar_flux.shape[1], ar_flux.shape[2], x_range, y_range, f_min, f_max)
        new_bins.ar_flux = ar_flux
        return new_bins

    def set_filename(self, filename):
        self.filename = filename

    def as_3d_array(self):
        return self.ar_flux

    def flush(self):
        np.save(self.filename, self.as_3d_array())

    def get_max_range(self):
        return self.max_range

    def get_x_range(self):
        return self.x_range

    def get_y_range(self):
        return self.y_range

    def get_x_bin_size(self):
        return self.x_bin_size

    def get_y_bin_size(self):
        return self.y_bin_size

    def get_x_count(self):
        return self.x_count

    def get_y_count(self):
        return self.y_count

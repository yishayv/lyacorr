import numpy as np

from flux_accumulator import AccumulatorBase


class FluxHistogramBins(AccumulatorBase):
    def __init__(self, x_count, y_count, f_count, x_range, y_range, f_range, f_offset, filename=''):
        self.ar_flux = np.zeros((x_count, y_count, f_count))
        self.ar_weights = np.zeros((x_count, y_count, f_count))
        self.ar_count = np.zeros((x_count, y_count, f_count))
        self.x_count = x_count
        self.y_count = y_count
        self.f_count = f_count
        self.index_type = ''
        self.update_index_type()
        self.filename = filename
        self.max_range = np.sqrt(np.square(x_range) + np.square(y_range))
        self.x_range = x_range
        self.y_range = y_range
        self.f_range = f_range
        self.f_offset = f_offset
        self.x_bin_size = float(x_range) / x_count
        self.y_bin_size = float(y_range) / y_count

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask, ar_weights):
        assert False, "Not implemented"

    def merge(self, bins2):
        assert self.ar_flux.shape == bins2.ar_flux.shape
        assert self.ar_weights.shape == bins2.ar_weights.shape
        assert self.ar_count.shape == bins2.ar_count.shape
        assert self.x_range == bins2.x_range
        assert self.y_range == bins2.y_range
        assert self.x_count == bins2.x_count
        assert self.y_count == bins2.y_count
        assert self.f_count == bins2.f_count
        self.ar_flux += bins2.ar_flux
        self.ar_weights += bins2.ar_weights
        self.ar_count += bins2.ar_count
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_4d_array(self, stacked_array):
        self.ar_flux = stacked_array[:, :, :, 0]
        self.ar_count = stacked_array[:, :, :, 1]
        self.ar_weights = stacked_array[:, :, :, 2]
        self.x_count = self.ar_count.shape[0]
        self.y_count = self.ar_count.shape[1]
        self.f_count= self.ar_count.shape[2]
        self.update_index_type()

    def load(self, filename):
        # TODO: to static
        stacked_array = np.load(filename)
        self.from_4d_array(stacked_array)

    def update_index_type(self):
        # choose integer type according to number of bins
        self.index_type = 'int32' if self.x_count * self.y_count > 32767 else 'int16'

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
    def from_np_arrays(cls, ar_count, ar_flux, ar_weights, x_range, y_range):
        """

        :type ar_count: np.array
        :type ar_flux: np.array
        :type ar_weights: np.array
        """
        assert ar_count.ndim == ar_flux.ndim == ar_weights.ndim == 3
        assert ar_count.shape == ar_flux.shape == ar_weights.shape
        new_bins = cls(ar_count.shape[0], ar_count.shape[1], ar_count.shape[2], x_range, y_range)
        new_bins.ar_count = ar_count
        new_bins.ar_flux = ar_flux
        new_bins.ar_weights = ar_weights
        return new_bins

    def set_filename(self, filename):
        self.filename = filename

    def to_4d_array(self):
        return np.concatenate((self.ar_flux, self.ar_count, self.ar_weights), axis=3)

    def flush(self):
        np.save(self.filename, self.to_4d_array())

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

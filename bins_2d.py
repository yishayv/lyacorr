import numpy as np

from flux_accumulator import AccumulatorBase


class Bins2D(AccumulatorBase):
    def __init__(self, x_count, y_count, x_range, y_range, filename=''):
        self.ar_flux = np.zeros((x_count, y_count))
        self.ar_weights = np.zeros((x_count, y_count))
        self.ar_count = np.zeros((x_count, y_count))
        self.x_count = x_count
        self.y_count = y_count
        self.index_type = ''
        self.update_index_type()
        self.filename = filename
        self.max_range = np.sqrt(np.square(x_range) + np.square(y_range))
        self.x_range = x_range
        self.y_range = y_range
        self.x_bin_size = float(x_range) / x_count
        self.y_bin_size = float(y_range) / y_count

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask, ar_weights):
        """
        add flux to x,y bins with weights and a filter mask.
        note: the x,y values should be rescaled prior to calling this method.
        :type ar_flux: np.ndarray
        :param ar_x: np.ndarray
        :param ar_y: np.ndarray
        :param mask: np.ndarray
        :param ar_weights: np.ndarray
        """
        ar_x_int = ar_x.astype(self.index_type)
        ar_y_int = ar_y.astype(self.index_type)
        m = np.logical_and(np.logical_and(np.logical_and(ar_x_int >= 0, ar_y_int >= 0),
                                          np.logical_and(ar_x_int < self.x_count, ar_y_int < self.y_count)),
                           mask)
        ar_flux_masked = ar_flux[m]
        ar_weights_masked = ar_weights[m]
        ar_indices_x = ar_x_int[m]
        ar_indices_y = ar_y_int[m]
        # make sure we don't invert x an y
        # represent bins in 1D. this is faster than a 2D numpy histogram
        ar_indices_xy = ar_indices_y + (self.y_count * ar_indices_x)
        # bin data according to x,y values
        flux_hist_1d = np.bincount(ar_indices_xy, weights=ar_flux_masked * ar_weights_masked,
                                   minlength=self.y_count * self.x_count)
        weights_hist_1d = np.bincount(ar_indices_xy, weights=ar_weights_masked,
                                      minlength=self.y_count * self.x_count)
        count_hist_1d = np.bincount(ar_indices_xy, weights=None,
                                    minlength=self.y_count * self.x_count)
        # return from 1D to a 2d array
        flux_hist = flux_hist_1d.reshape((self.x_count, self.y_count))
        count_hist = count_hist_1d.reshape((self.x_count, self.y_count))
        weights_hist = weights_hist_1d.reshape((self.x_count, self.y_count))
        # accumulate new data
        self.ar_flux += flux_hist
        self.ar_weights += weights_hist
        self.ar_count += count_hist

    def merge(self, bins_2d_2):
        assert self.ar_flux.shape == bins_2d_2.ar_flux.shape
        assert self.ar_weights.shape == bins_2d_2.ar_weights.shape
        assert self.ar_count.shape == bins_2d_2.ar_count.shape
        assert self.x_range == bins_2d_2.x_range
        assert self.y_range == bins_2d_2.y_range
        assert self.x_count == bins_2d_2.x_count
        assert self.y_count == bins_2d_2.y_count
        self.ar_flux += bins_2d_2.ar_flux
        self.ar_weights += bins_2d_2.ar_weights
        self.ar_count += bins_2d_2.ar_count
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def load(self, filename):
        # TODO: to static
        stacked_array = np.load(filename)
        self.ar_flux = stacked_array[:, :, 0]
        self.ar_count = stacked_array[:, :, 1]
        self.ar_weights = stacked_array[:, :, 2]
        self.x_count = self.ar_count.shape[0]
        self.y_count = self.ar_count.shape[1]
        self.update_index_type()

    def update_index_type(self):
        # choose integer type according to number of bins
        self.index_type = 'int32' if self.x_count * self.y_count > 32767 else 'int16'

    def __radd__(self, other):
        return self.init_as(self).merge(self).merge(other)

    def __add__(self, other):
        return self.init_as(self).merge(self).merge(other)

    @classmethod
    def init_as(cls, other):
        """

        :type other: Bins2D
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
        assert ar_count.ndim == ar_flux.ndim == ar_weights.ndim == 2
        assert ar_count.shape == ar_flux.shape == ar_weights.shape
        new_bins = cls(ar_count.shape[0], ar_count.shape[1], x_range, y_range)
        new_bins.ar_count = ar_count
        new_bins.ar_flux = ar_flux
        new_bins.ar_weights = ar_weights
        return new_bins

    def set_filename(self, filename):
        self.filename = filename

    def flush(self):
        np.save(self.filename, np.dstack((self.ar_flux, self.ar_count, self.ar_weights)))

    def get_max_range(self):
        return self.max_range

    def get_x_bin_size(self):
        return self.x_bin_size

    def get_y_bin_size(self):
        return self.y_bin_size


class Expandable1DArray(object):
    def __init__(self, *args, **kwargs):
        self.ar = np.copy(np.array(*args, **kwargs))
        self.size = self.ar.size

    def get_array_view(self):
        return self.ar[:self.size]

    def add_array(self, ar):
        new_required_size = self.size + ar.size
        # if a resize is needed, resize to the next power of 2
        if self.ar.size < new_required_size:
            self.ar.resize([self._new_size(new_required_size)])
            # old_ar = self.ar
            # self.ar = np.zeros(self._new_size(new_required_size))
            # np.copyto(self.get_array_view(), old_ar)
        np.copyto(self.ar[self.size:self.size + ar.size], ar)
        self.size += ar.size
        return self.get_array_view()

    @classmethod
    def _new_size(cls, n):
        return max(cls.get_next_power_of_2(n), 0)

    @classmethod
    def get_next_power_of_2(cls, n):
        return 1 << n.bit_length()


class Bins2DLists:
    def __init__(self):
        self.bins = [Expandable1DArray() for i in 2500]


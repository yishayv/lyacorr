import numpy as np

from flux_accumulator import AccumulatorBase


class Bins2D(AccumulatorBase):
    def __init__(self, x_count, y_count, x_range, y_range, ar_existing_data=None, filename=''):
        if ar_existing_data is not None:
            assert (x_count, y_count, 3) == ar_existing_data.shape, "incompatible shape:{0}".format(
                ar_existing_data.shape)
            self.ar_data = ar_existing_data
        else:
            self.ar_data = np.zeros((x_count, y_count, 3))
        self.ar_flux = None
        self.ar_weights = None
        self.ar_count = None
        self.update_array_slices()
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
        :type ar_flux: np.multiarray.ndarray
        :param ar_x: np.multiarray.ndarray
        :param ar_y: np.multiarray.ndarray
        :param mask: np.multiarray.ndarray
        :param ar_weights: np.multiarray.ndarray
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
        assert self.ar_data.shape == bins_2d_2.ar_data.shape
        assert self.x_range == bins_2d_2.x_range
        assert self.y_range == bins_2d_2.y_range
        assert self.x_count == bins_2d_2.x_count
        assert self.y_count == bins_2d_2.y_count
        self.ar_data += bins_2d_2.ar_data
        return self

    def merge_array(self, ar_data):
        assert self.ar_data.shape == ar_data.shape
        self.ar_data += ar_data
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_3d_array(self, stacked_array):
        self.ar_data = stacked_array
        self.update_array_slices()
        self.x_count = self.ar_count.shape[0]
        self.y_count = self.ar_count.shape[1]
        self.update_index_type()

    def load(self, filename):
        # TODO: to static
        stacked_array = np.load(filename)
        self.from_3d_array(stacked_array)

    def update_index_type(self):
        # choose integer type according to number of bins
        self.index_type = 'int32' if self.x_count * self.y_count > 32767 else 'int16'

    def update_array_slices(self):
        self.ar_flux = self.ar_data[:, :, 0]
        self.ar_count = self.ar_data[:, :, 1]
        self.ar_weights = self.ar_data[:, :, 2]

    def __radd__(self, other):
        return self.merge(other)

    def __add__(self, other):
        return self.merge(other)

    @classmethod
    def init_as(cls, other):
        """

        :type other: Bins2D
        """
        new_obj = cls(other.x_count, other.y_count, other.x_range, other.y_range, filename=other.filename)
        return new_obj

    @classmethod
    def from_other(cls, other):
        new_obj = cls.init_as(other)
        new_obj.merge(other)

    def set_filename(self, filename):
        self.filename = filename

    def to_3d_array(self):
        return self.ar_data

    def flush(self):
        np.save(self.filename, self.to_3d_array())

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

    def get_pair_count(self):
        return self.ar_count.sum()

    def get_data_as_array(self):
        return self.to_3d_array()

    def get_array_shape(self):
        return self.ar_data.shape

    def get_metadata(self):
        return [self.x_count, self.y_count, self.index_type,
                self.filename, self.max_range,
                self.x_range, self.y_range,
                self.x_bin_size, self.y_bin_size]

    @classmethod
    def load_from(cls, ar, metadata):
        new_bins = cls(1, 1, 1, 1)
        (new_bins.x_count, new_bins.y_count, new_bins.index_type, new_bins.filename, new_bins.max_range,
         new_bins.x_range, new_bins.y_range, new_bins.x_bin_size, new_bins.y_bin_size) = metadata
        new_bins.ar_data = ar
        new_bins.update_array_slices()
        return new_bins

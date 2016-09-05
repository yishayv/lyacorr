import numpy as np

from flux_accumulator import AccumulatorBase


class Bins3D(AccumulatorBase):
    def __init__(self, dims, ranges, ar_existing_data=None, filename=''):
        """
        :param dims: the shape of the bins
        :param ranges: a 2-by-3 array representing the minimum and maximum ranges of the 3 coordinates
        :type dims: np.ndarray
        :type ranges np.ndarray
        :type ar_existing_data np.ndarray
        :type filename str
        """
        if ar_existing_data is not None:
            expected_shape = (dims[0], dims[1], dims[2], 3)
            ravelled_shape = (dims[0] * dims[1] * dims[2] * 3,)
            if ar_existing_data.shape == ravelled_shape:
                self.ar_data = ar_existing_data.reshape(expected_shape)
            else:
                assert ar_existing_data.shape == expected_shape, "incompatible shape:{0}".format(
                    ar_existing_data.shape)
                self.ar_data = ar_existing_data
        else:
            self.ar_data = np.zeros((dims[0], dims[1], dims[2], 3))
        self.ar_flux = None
        self.ar_weights = None
        self.ar_count = None
        self.update_array_slices()
        self.dims = dims
        self.index_type = ''
        self.update_index_type()
        self.filename = filename
        max_x = ranges[1, 0]
        max_y = ranges[1, 1]
        self.max_range = np.sqrt(np.square(max_x) + np.square(max_y))
        self.ranges = ranges
        self.bin_sizes = np.abs(ranges[1] - ranges[0]) / dims

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, ar_z, mask, ar_weights):
        """
        add flux to x,y bins with weights and a filter mask.
        note: the x,y values should be rescaled prior to calling this method.
        :type ar_flux: np.multiarray.ndarray
        :param ar_x: np.multiarray.ndarray
        :param ar_y: np.multiarray.ndarray
        :param ar_z: np.multiarray.ndarray
        :param mask: np.multiarray.ndarray
        :param ar_weights: np.multiarray.ndarray
        """
        ar_x_int = ar_x.astype(self.index_type)
        ar_y_int = ar_y.astype(self.index_type)
        ar_z_int = ar_z.astype(self.index_type)
        # restrict the mask to pixels inside the bin range.
        m = np.any((ar_x_int >= 0, ar_y_int >= 0,
                    ar_x_int < self.dims[0], ar_y_int < self.dims[1], mask), axis=0)
        ar_flux_masked = ar_flux[m]
        ar_weights_masked = ar_weights[m]
        ar_indices_x = ar_x_int[m]
        ar_indices_y = ar_y_int[m]
        ar_indices_z = ar_z_int[m]
        # make sure we don't invert x, y and z
        # z is the innermost coordinate, x is the outermost.
        # represent bins in 1D. this is faster than a 2D numpy histogram
        ar_indices_xyz = ar_indices_z + self.dims[2] * (ar_indices_y + (self.dims[1] * ar_indices_x))
        # bin data according to x,y values
        flux_hist_1d = np.bincount(ar_indices_xyz, weights=ar_flux_masked * ar_weights_masked,
                                   minlength=self.dims[1] * self.dims[0])
        weights_hist_1d = np.bincount(ar_indices_xyz, weights=ar_weights_masked,
                                      minlength=self.dims[1] * self.dims[0])
        count_hist_1d = np.bincount(ar_indices_xyz, weights=None,
                                    minlength=self.dims[1] * self.dims[0])
        # return from 1D to a 2d array
        flux_hist = flux_hist_1d.reshape(self.dims)
        count_hist = count_hist_1d.reshape(self.dims)
        weights_hist = weights_hist_1d.reshape(self.dims)
        # accumulate new data
        self.ar_flux += flux_hist
        self.ar_weights += weights_hist
        self.ar_count += count_hist

    def merge(self, bins_3d_2):
        assert self.ar_data.shape == bins_3d_2.ar_data.shape
        assert np.all(self.ranges == bins_3d_2.ranges)
        assert np.all(self.dims == bins_3d_2.dims)
        self.ar_data += bins_3d_2.ar_data
        return self

    def merge_array(self, ar_data):
        assert self.ar_data.shape == ar_data.shape
        self.ar_data += ar_data
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_4d_array(self, stacked_array):
        self.ar_data = stacked_array
        self.update_array_slices()
        self.dims = self.ar_count.shape
        self.update_index_type()

    def load(self, filename):
        # TODO: to static
        stacked_array = np.load(filename)
        self.from_4d_array(stacked_array)

    def update_index_type(self):
        # choose integer type according to number of bins
        self.index_type = 'int32' if np.prod(self.dims) > 32767 else 'int16'

    def update_array_slices(self):
        self.ar_flux = self.ar_data[:, :, :, 0]
        self.ar_count = self.ar_data[:, :, :, 1]
        self.ar_weights = self.ar_data[:, :, :, 2]

    def __radd__(self, other):
        return self.merge(other)

    def __add__(self, other):
        return self.merge(other)

    @classmethod
    def init_as(cls, other):
        """

        :type other: Bins3D
        """
        new_obj = cls(other.dims, other.ranges, filename=other.filename)
        return new_obj

    @classmethod
    def from_other(cls, other):
        new_obj = cls.init_as(other)
        new_obj.merge(other)

    def set_filename(self, filename):
        self.filename = filename

    def to_4d_array(self):
        return self.ar_data

    def flush(self):
        np.save(self.filename, self.to_4d_array())

    def get_max_range(self):
        return self.max_range

    def get_ranges(self):
        return self.ranges

    def get_dims(self):
        return self.dims

    def get_bin_sizes(self):
        return self.bin_sizes

    def get_pair_count(self):
        return self.ar_count.sum()

    def get_data_as_array(self):
        return self.to_4d_array()

    def get_array_shape(self):
        return self.ar_data.shape

    def get_metadata(self):
        return [self.dims, self.index_type,
                self.filename, self.max_range,
                self.ranges, self.bin_sizes]

    @classmethod
    def load_from(cls, ar, metadata):
        new_bins = cls(dims=(1, 1, 1), ranges=((1, 1, 1), (1, 1, 1)))
        (new_bins.dims, new_bins.index_type, new_bins.filename, new_bins.max_range,
         new_bins.ranges, new_bins.bin_sizes) = metadata
        new_bins.ar_data = ar
        new_bins.update_array_slices()
        return new_bins

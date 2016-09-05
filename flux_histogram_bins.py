import numpy as np

from flux_accumulator import AccumulatorBase


class FluxHistogramBins(AccumulatorBase):
    def __init__(self, dims, ranges, filename=''):
        self.ar_flux = np.zeros(dims)
        self.dims = dims
        self.filename = filename
        self.max_range = np.sqrt(np.square(ranges.x) + np.square(ranges.y))
        self.ranges = ranges
        self.bin_sizes = np.abs(ranges[1] - ranges[0]) / dims
        self.pair_count = 0

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, ar_z, mask, ar_weights):
        assert False, "Not implemented"

    def merge(self, bins2):
        assert self.ar_flux.shape == bins2.ar_flux.shape
        assert self.ranges == bins2.ranges
        assert self.dims == bins2.dims
        self.ar_flux += bins2.ar_flux
        self.pair_count += bins2.pair_count
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_3d_array(self, array):
        self.ar_flux = array
        self.dims = self.ar_flux.shape

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
        new_obj = cls(other.dims, other.ranges)
        new_obj.set_filename(other.filename)
        return new_obj

    @classmethod
    def from_other(cls, other):
        new_obj = cls.init_as(other)
        new_obj.merge(other)

    @classmethod
    def from_np_array(cls, ar_flux, ranges):
        """

        :type ar_flux: np.array
        :type ranges: BinRange
        """
        assert ar_flux.ndim == 3
        new_bins = cls(ar_flux.shape, ranges)
        new_bins.ar_flux = ar_flux
        return new_bins

    def set_filename(self, filename):
        self.filename = filename

    def get_data_as_array(self):
        return self.ar_flux

    def get_array_shape(self):
        return self.ar_flux.shape

    def flush(self):
        np.save(self.filename, self.get_data_as_array())

    def get_max_range(self):
        return self.max_range

    def get_ranges(self):
        return self.ranges

    def get_bin_sizes(self):
        return self.bin_sizes

    def get_dims(self):
        return self.dims

    def get_pair_count(self):
        return self.pair_count

    def get_metadata(self):
        return [self.dims,
                self.filename, self.max_range,
                self.ranges,
                self.bin_sizes,
                self.pair_count]

    @classmethod
    def load_from(cls, ar, metadata):
        new_bins = cls(dims=np.array((1, 1, 1)), ranges=np.array(((0,0,0),(1, 1, 1))))
        (new_bins.dims, new_bins.filename, new_bins.max_range,
         new_bins.ranges, new_bins.bin_size, new_bins.pair_count) = metadata
        new_bins.ar_flux = ar
        return new_bins

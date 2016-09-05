import collections
from collections import namedtuple

import numpy as np

import bins_3d
from flux_accumulator import AccumulatorBase

BinDims = namedtuple('BinDims', ['x', 'y', 'z'])
BinRange = namedtuple('BinRange', ['x', 'y', 'z'])


class Bins3DWithGroupID(AccumulatorBase):
    def __init__(self, dims, ranges, filename=''):
        self.dims = dims
        self.ranges = ranges
        self.filename = filename
        self.max_range = np.sqrt(np.square(ranges.x) + np.square(ranges.y))
        self.bin_sizes = BinRange([float(range_i) / dim_i for range_i, dim_i in zip(ranges, dims)])
        self.dict_bins_3d_data = collections.defaultdict(self._bins_creator)

    def _bins_creator(self, ar=None):
        return bins_3d.Bins3D(self.dims, self.ranges, ar_existing_data=ar)

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, ar_z, mask, ar_weights):
        assert False, "Not implemented"

    def add_to_group_id(self, group_id, bins_3d_data):
        """
        Add flux (and weights) from an existing bins_2d object into the specified group_id
        :type group_id: int
        :type bins_3d_data: bins_3d.Bins3D
        """
        self.dict_bins_3d_data[group_id] += bins_3d_data

    def add_array_to_group_id(self, group_id, ar_data):
        """
        Add flux (and weights) from an existing numpy array into the specified group_id
        :type group_id: int
        :type ar_data: np.multiarray.ndarray
        """
        self.add_to_group_id(group_id, self._bins_creator(ar_data))

    def merge(self, other):
        """
        Merge data from another object of the same type, adding fluxes, weights and counts.
        The data from each group_id in 'other' is added to the corresponding group_id of this instance,
        adding new group_ids as necessary.
        :type other: Bins3DWithGroupID
        :rtype: Bins3DWithGroupID
        """
        assert self.ranges == other.ranges
        assert self.dims == other.dims
        for group_id, bins_2d_data in other.dict_bins_3d_data.items():
            self.dict_bins_3d_data[group_id] += bins_2d_data
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_4d_array(self, stacked_array, group_ids):
        """

        :type stacked_array: np.multiarray.ndarray
        :param group_ids: collections.Iterable[int]
        """
        self.dict_bins_3d_data.clear()
        for index, group_id in enumerate(group_ids):
            self.dict_bins_3d_data[group_id] = self._bins_creator(stacked_array[index])
        self.dims = stacked_array.shape

    def load(self, filename):
        # TODO: to static
        npz_file = np.load(filename)
        stacked_array = npz_file['ar_data']
        group_ids = npz_file['group_ids']
        self.from_4d_array(stacked_array, group_ids)

    def __radd__(self, other):
        return self.merge(other)

    def __add__(self, other):
        return self.merge(other)

    @classmethod
    def init_as(cls, other):
        """
        Return an instance with empty data, similar to 'other'
        :type other: Bins3DWithGroupID
        :rtype : Bins3DWithGroupID
        """
        new_obj = cls(other.dims, other.ranges, filename=other.filename)
        return new_obj

    @classmethod
    def from_other(cls, other):
        new_obj = cls.init_as(other)
        new_obj.merge(other)

    def set_filename(self, filename):
        self.filename = filename

    def to_5d_array(self):
        if self.dict_bins_3d_data:
            return np.concatenate([np.expand_dims(i.ar_data, axis=0) for i in self.dict_bins_3d_data.values()], axis=0)
        else:
            return np.zeros(shape=(0,) + self.dims + (3,))

    def to_2d_array(self):
        return self.to_5d_array().reshape((-1, np.prod(self.dims) * 3,))

    def flush(self):
        np.savez(self.filename, ar_data=self.to_5d_array(), group_ids=np.array(list(self.dict_bins_3d_data.keys())))

    def get_max_range(self):
        return self.max_range

    def get_ranges(self):
        return self.ranges

    def get_bin_sizes(self):
        return self.bin_sizes

    def get_dims(self):
        return self.dims

    def get_pair_count(self):
        return np.sum([i.ar_count.sum() for i in self.dict_bins_3d_data.values()])

    def get_data_as_array(self):
        return self.to_5d_array()

    def get_array_shape(self):
        return self.dict_bins_3d_data.shape

    def get_metadata(self):
        return [self.dims,
                self.filename, self.max_range,
                self.ranges,
                self.bin_sizes,
                list(self.dict_bins_3d_data.keys())]

    @classmethod
    def load_from(cls, ar, metadata):
        """
        Load
        :type ar: np.multiarray.ndarray
        :type metadata: list
        :rtype : Bins3DWithGroupID
        """
        new_bins = cls(1, 1, 1, 1)
        (new_bins.x_count, new_bins.y_count, new_bins.filename, new_bins.max_range,
         new_bins.x_range, new_bins.y_range, new_bins.x_bin_size, new_bins.y_bin_size,
         group_ids) = metadata
        for index, group_id in enumerate(group_ids):
            new_bins.dict_bins_3d_data[group_id] = new_bins._bins_creator(ar[index])
        return new_bins

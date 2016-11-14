import collections

import numpy as np

import bins_3d
from flux_accumulator import AccumulatorBase


class Bins3DWithGroupID(AccumulatorBase):
    def __init__(self, dims, ranges, filename=''):
        """

        :type dims: np.ndarray
        :type ranges: np.ndarray
        :type filename: str
        """
        self.dims = dims
        self.ranges = ranges
        self.filename = filename
        self.bin_sizes = np.abs(ranges[1] - ranges[0]) / dims
        self.dict_bins_3d_data = collections.defaultdict(self._bins_creator)

    def _bins_creator(self, ar=None):
        return bins_3d.Bins3D(self.dims, self.ranges, ar_existing_data=ar)

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, ar_z, mask, ar_weights):
        assert False, "Not implemented"

    def add_to_group_id(self, group_id, bins_3d_data):
        """
        Add flux (and weights) from an existing bins_2d object into the specified group_id
        :type group_id: int64
        :type bins_3d_data: bins_3d.Bins3D
        """
        self.dict_bins_3d_data[group_id] += bins_3d_data

    def add_array_to_group_id(self, group_id, ar_data):
        """
        Add flux (and weights) from an existing numpy array into the specified group_id
        :type group_id: int64
        :type ar_data: Optional[np.multiarray.ndarray]
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
        assert np.all(self.ranges == other.ranges)
        assert np.all(self.dims == other.dims)
        for group_id, bins_2d_data in other.dict_bins_3d_data.items():
            self.dict_bins_3d_data[group_id] += bins_2d_data
        return self

    def get_group_view(self, group_id):
        if group_id not in self.dict_bins_3d_data:
            # create a new group_id key with a zeros array
            self.add_array_to_group_id(group_id, None)
        return self.dict_bins_3d_data[group_id]

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_5d_array(self, stacked_array, group_ids):
        """

        :type stacked_array: np.multiarray.ndarray
        :param group_ids: collections.Iterable[int]
        """
        self.dict_bins_3d_data.clear()
        for index, group_id in enumerate(group_ids):
            self.dict_bins_3d_data[group_id] = self._bins_creator(stacked_array[index])
        self.dims = stacked_array.shape[1:-1]

    def load(self, filename=None):
        # TODO: to static
        if not filename:
            filename = self.filename
        npz_file = np.load(filename)
        stacked_array = npz_file['ar_data']
        group_ids = npz_file['group_ids']
        metadata = npz_file['metadata']
        (self.dims, self.filename,
         self.ranges, self.bin_sizes) = metadata
        self.from_5d_array(stacked_array, group_ids)

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
            return np.zeros(shape=(0,) + tuple(self.dims) + (3,))

    def to_2d_array(self):
        return self.to_5d_array().reshape((-1, np.prod(self.dims) * 3,))

    def flush(self):
        np.savez(self.filename,
                 ar_data=self.to_5d_array(),
                 group_ids=np.array(list(self.dict_bins_3d_data.keys())),
                 # save all metadata except group_ids which is handled separately
                 metadata=self.get_metadata()[:-1])

    def get_ranges(self):
        return self.ranges

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
                self.filename,
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
        new_bins = cls(dims=np.array([1, 1, 1]), ranges=np.array([[1, 1, 1], [1, 1, 1]]))  # type: Bins3DWithGroupID
        (new_bins.dims, new_bins.filename,
         new_bins.ranges, new_bins.bin_sizes, group_ids) = metadata
        for index, group_id in enumerate(group_ids):
            new_bins.dict_bins_3d_data[group_id] = new_bins._bins_creator(ar[index])
        return new_bins

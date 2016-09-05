import collections

import numpy as np

import bins_3d
from flux_accumulator import AccumulatorBase


class Bins2DWithGroupID(AccumulatorBase):
    def __init__(self, x_count, y_count, x_range, y_range, filename=''):
        self.x_count = x_count
        self.y_count = y_count
        self.filename = filename
        self.max_range = np.sqrt(np.square(x_range) + np.square(y_range))
        self.x_range = x_range
        self.y_range = y_range
        self.x_bin_size = float(x_range) / x_count
        self.y_bin_size = float(y_range) / y_count
        self.dict_bins_2d_data = collections.defaultdict(self._bins_creator)

    def _bins_creator(self, ar=None):
        return bins_3d.Bins2D(self.x_count, self.y_count, self.x_range, self.y_range, ar_existing_data=ar)

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask, ar_weights):
        assert False, "Not implemented"

    def add_to_group_id(self, group_id, bins_2d_data):
        """
        Add flux (and weights) from an existing bins_2d object into the specified group_id
        :type group_id: int
        :type bins_2d_data: bins_3d.Bins2D
        """
        self.dict_bins_2d_data[group_id] += bins_2d_data

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
        :type other: Bins2DWithGroupID
        :rtype: Bins2DWithGroupID
        """
        assert self.x_range == other.x_range
        assert self.y_range == other.y_range
        assert self.x_count == other.x_count
        assert self.y_count == other.y_count
        for group_id, bins_2d_data in other.dict_bins_2d_data.items():
            self.dict_bins_2d_data[group_id] += bins_2d_data
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_4d_array(self, stacked_array, group_ids):
        """

        :type stacked_array: np.multiarray.ndarray
        :param group_ids: collections.Iterable[int]
        """
        self.dict_bins_2d_data.clear()
        for index, group_id in enumerate(group_ids):
            self.dict_bins_2d_data[group_id] = self._bins_creator(stacked_array[index])
        self.x_count = stacked_array.shape[1]
        self.y_count = stacked_array.shape[2]

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
        :type other: Bins2DWithGroupID
        :rtype : Bins2DWithGroupID
        """
        new_obj = cls(other.x_count, other.y_count, other.x_range, other.y_range, filename=other.filename)
        return new_obj

    @classmethod
    def from_other(cls, other):
        new_obj = cls.init_as(other)
        new_obj.merge(other)

    def set_filename(self, filename):
        self.filename = filename

    def to_4d_array(self):
        if self.dict_bins_2d_data:
            return np.concatenate([np.expand_dims(i.ar_data, axis=0) for i in self.dict_bins_2d_data.values()], axis=0)
        else:
            return np.zeros(shape=(0, self.x_count, self.y_count, 3))

    def to_2d_array(self):
        return self.to_4d_array().reshape((-1, self.x_count * self.y_count * 3,))

    def flush(self):
        np.savez(self.filename, ar_data=self.to_4d_array(), group_ids=np.array(list(self.dict_bins_2d_data.keys())))

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
        return np.sum([i.ar_count.sum() for i in self.dict_bins_2d_data.values()])

    def get_data_as_array(self):
        return self.to_4d_array()

    def get_array_shape(self):
        return self.dict_bins_2d_data.shape

    def get_metadata(self):
        return [self.x_count, self.y_count,
                self.filename, self.max_range,
                self.x_range, self.y_range,
                self.x_bin_size, self.y_bin_size,
                list(self.dict_bins_2d_data.keys())]

    @classmethod
    def load_from(cls, ar, metadata):
        """
        Load
        :type ar: np.multiarray.ndarray
        :type metadata: list
        :rtype : Bins2DWithGroupID
        """
        new_bins = cls(1, 1, 1, 1)
        (new_bins.x_count, new_bins.y_count, new_bins.filename, new_bins.max_range,
         new_bins.x_range, new_bins.y_range, new_bins.x_bin_size, new_bins.y_bin_size,
         group_ids) = metadata
        for index, group_id in enumerate(group_ids):
            new_bins.dict_bins_2d_data[group_id] = new_bins._bins_creator(ar[index])
        return new_bins

import collections
import numpy as np
import bins_2d

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
        self.dict_ar_data = collections.defaultdict(lambda: bins_2d.Bins2D(x_count, y_count, xrange, y_range))

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask, ar_weights):
        assert False, "Not implemented"

    def merge(self, other):
        assert isinstance(other, self)
        assert self.x_range == other.x_range
        assert self.y_range == other.y_range
        assert self.x_count == other.x_count
        assert self.y_count == other.y_count
        for group_id, ar_data in other.dict_ar_data.items():
            self.dict_ar_data[group_id] += ar_data
        return self

    def save(self, filename):
        self.filename = filename
        self.flush()

    def from_4d_array(self, stacked_array, group_ids):
        """

        :type stacked_array: np.multiarray.ndarray
        :param group_ids: collections.Iterable[int]
        """
        self.dict_ar_data.clear()
        for index, group_id in enumerate(group_ids):
            self.dict_ar_data[group_id] = stacked_array[index]
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

        :type other: Bins2DWithGroupID
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
        np.vstack([np.expand_dims(i, axis=0) for i in self.dict_ar_data.values()])

    def flush(self):
        np.savez(self.filename, ar_data=self.to_4d_array(), group_ids=np.array(self.dict_ar_data.keys()))

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
        bins_instance = bins_2d.Bins2D(self.x_count, self.y_count, self.x_range, self.y_range)
        return np.sum([bins_instance.from_3d_array(i).ar_count.sum() for i in self.dict_ar_data.values()])

    def get_data_as_array(self):
        return self.to_4d_array()

    def get_array_shape(self):
        return self.dict_ar_data.shape

    def get_metadata(self):
        return [self.x_count, self.y_count,
                self.filename, self.max_range,
                self.x_range, self.y_range,
                self.x_bin_size, self.y_bin_size,
                self.dict_ar_data.keys()]

    def load_from(self, ar, metadata):
        new_bins = self.init_as(self)
        (new_bins.x_count, new_bins.y_count, new_bins.filename, new_bins.max_range,
         new_bins.x_range, new_bins.y_range, new_bins.x_bin_size, new_bins.y_bin_size,
         group_ids) = metadata
        for index, group_id in enumerate(group_ids):
            new_bins.dict_ar_data[group_id] = ar[index]
        return new_bins

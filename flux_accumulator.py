from abc import ABCMeta, abstractmethod


class AccumulatorBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def flush(self):
        """
        Save intermediate results to file.

        :return:
        """
        return

    @abstractmethod
    def set_filename(self, filename):
        """
        Set filename for saving results.
        :return:
        """
        return

    @abstractmethod
    def add_array_with_mask(self, ar_flux, ar_x, ar_y, ar_z, mask, ar_weights):
        """
        Add flux with two corresponding coordinates, and a mask.
        :type ar_flux: np.multiarray.ndarray
        :type ar_x: np.multiarray.ndarray
        :type ar_y: np.multiarray.ndarray
        :type ar_z: np.multiarray.ndarray
        :type mask: np.multiarray.ndarray
        :type ar_weights: np.multiarray.ndarray
        :return:
        """
        return

    @abstractmethod
    def __radd__(self, other):
        return

    @abstractmethod
    def __add__(self, other):
        return

    @abstractmethod
    def get_ranges(self):
        return

    @abstractmethod
    def get_dims(self):
        return

    @abstractmethod
    def get_pair_count(self):
        return

    @abstractmethod
    def get_data_as_array(self):
        return

    @abstractmethod
    def get_array_shape(self):
        return

    @abstractmethod
    def get_metadata(self):
        return

    @abstractmethod
    def load_from(self, ar, metadata):
        return

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
    def set_filename(self):
        """
        Set filename for saving results.
        :return:
        """
        return

    @abstractmethod
    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask, ar_weights):
        """
        Add flux with two corresponding coordinates, and a mask.
        :type ar_flux: np.ndarray
        :type ar_x: np.ndarray
        :type ar_y: np.ndarray
        :type mask: np.ndarray
        :type ar_weights: np.ndarray
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
    def get_max_range(self):
        return

    @abstractmethod
    def get_x_range(self):
        return

    @abstractmethod
    def get_y_range(self):
        return

    @abstractmethod
    def get_x_bin_size(self):
        return

    @abstractmethod
    def get_y_bin_size(self):
        return

    @abstractmethod
    def get_x_count(self):
        return

    @abstractmethod
    def get_y_count(self):
        return